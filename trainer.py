import math, collections, torch, os, time, contextlib
from logger import log
from dataclasses import asdict
from torch.nn.parallel import DistributedDataParallel as DDP
from parallel.dist_utils import is_main, all_reduce_mean


def get_lr(step, train_config):
    # warmup 阶段，lr 从 0 线性上升到 max_lr
    # 过了warmup，lr 按余弦缓慢下降至 min_lr
    warmup_steps = train_config.warmup_steps
    max_steps = train_config.max_steps
    min_lr, max_lr = train_config.min_lr, train_config.max_lr
    if step >= max_steps:
        return min_lr
    elif step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    else:
        ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        c = 0.5 * (1 + math.cos(math.pi*ratio))
        return min_lr + c * (max_lr - min_lr)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


class Trainer:
    def __init__(self, model, optimizer, train_config, task, ckpt_dir=None, optimizer_name=None, config=None):
        self.model = model
        self.optimizer = optimizer
        self.train_config = train_config
        self.task = task
        self.ckpt_dir = ckpt_dir
        self.optimizer_name = optimizer_name
        self.tokens_per_step = getattr(self.task, "tokens_per_step", None)
        self.log_formats = getattr(self.task, "log_formats", {})
        self.config = config
        self.is_ddp = isinstance(model, DDP)
        self.raw_model = model.module if self.is_ddp else model

    def fit(self, start_step=0):
        records = collections.defaultdict(list)  # 记录返回值
        self._eval(start_step)  # 开始前的 eval

        for step in range(start_step, self.train_config.max_steps):
            t0 = time.perf_counter()
            # 拿数据
            parts, step_extra = self.task.get_parts(step)
            self.model.train()
            # 清梯度
            self.optimizer.zero_grad()
            loss_accum = 0.0
            part_extra_acc = {}
            # 前向反向
            for i, part in enumerate(parts):
                # 只在最后一个 part 做梯度 all-reduce
                if self.is_ddp and i < len(parts) - 1:
                    sync_ctx = self.model.no_sync()
                else:
                    sync_ctx = contextlib.nullcontext()
                with sync_ctx:
                    loss, part_extra = self.task.loss(part)
                    loss.backward()
                loss_accum += loss.detach()
                # 存额外参数，如 GRPO 的 kl
                for k, v in part_extra.items():
                    part_extra_acc[k] = part_extra_acc.get(k, 0.0) + v
            loss_accum = all_reduce_mean(loss_accum)  # step_extra, part_extra 没有跨进程平均，目前只有预训练用 DDP，没有额外数字
            # 梯度裁剪
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            # 设学习率
            lr = get_lr(step, self.train_config)
            set_lr(self.optimizer, lr)
            # step
            self.optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            dt = t1 - t0
            # 记录
            loss_value = loss_accum.item()
            norm_value = norm.item()
            extras = {}
            for k, v in {**step_extra, **part_extra_acc}.items():
                extras[k] = float(v)
            records["loss"].append(loss_value)
            records["norm"].append(norm_value)
            for k, v in extras.items():
                records[k].append(v)
            # 打印 log
            if step % self.train_config.log_interval == 0 or step == self.train_config.max_steps - 1:
                tmp = []
                for k, v in extras.items():
                    fmt = self.log_formats.get(k, ".4g")
                    tmp.append(f", {k}:{v:{fmt}}")
                extra_text = "".join(tmp)
                tok_text = f", tok/s:{self.tokens_per_step/dt:.0f}" if self.tokens_per_step is not None else ""
                log(f"step:{step}, loss:{loss_value:.4f}, norm:{norm_value:.4f}, lr:{lr:.4e}{extra_text}, dt:{dt*1000:.1f}ms{tok_text}")
            # 定期 eval
            if step == self.train_config.max_steps - 1:
                self._eval(step + 1, final=True)  # 最后一步，final = True
            elif (step + 1) % self.train_config.eval_interval == 0:
                self._eval(step + 1)
        return dict(records)

    
    def _eval(self, step, final=False):
        # 做 eval
        self.model.eval()
        metrics = self.task.evaluate()
        text = ", ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
        # 打印 log
        log(f"step:{step}, {text}")
        if self.ckpt_dir is not None:
            opt_state = self.optimizer.state_dict()  # ZeRO 时这是集合通信，每个进程都要调用
            if is_main():
                # 存 checkpoint
                filename = "final.pt" if final else "latest.pt"
                self._save(os.path.join(self.ckpt_dir, filename), step, metrics, opt_state)
        self.model.train()


    def _save(self, path, step, metrics, opt_state):
        checkpoint = {
            # 用模型需要的
            "model": self.raw_model.state_dict(),
            "model_config": asdict(self.raw_model.config),
            # 接着训需要的
            "optimizer": opt_state,
            "optimizer_name": self.optimizer_name,
            "step": step,  # 从这一步继续训
            # 留作记录的
            "config": asdict(self.config) if self.config is not None else None,
        }
        checkpoint = {**checkpoint, **self.task.state_dict(), **metrics}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
