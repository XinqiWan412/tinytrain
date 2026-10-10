import math, collections, torch
from logger import log


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


def set_lr(optimizers, lr):
    for optim in optimizers:
        for group in optim.param_groups:
            group["lr"] = lr


def zero_grad_all(optimizers):
    for optim in optimizers:
        optim.zero_grad()


def step_all(optimizers):
    for optim in optimizers:
        optim.step()


class Trainer:
    def __init__(self, model, optimizers, train_config, task):
        self.model = model
        self.optimizers = optimizers
        self.train_config = train_config
        self.task = task

    def fit(self, start_step=0):
        records = collections.defaultdict(list)  # 记录返回值
        self._eval(start_step)  # 开始前的 eval

        for step in range(start_step, self.train_config.max_steps):
            # 拿数据
            parts, step_extra = self.task.get_parts(step)
            self.model.train()
            # 清梯度
            zero_grad_all(self.optimizers)
            loss_accum = 0.0
            part_extra_acc = {}
            # 前向反向
            for part in parts:
                loss, part_extra = self.task.loss(part)
                loss.backward()
                loss_accum += loss.detach()
                # 存 GRPO 的额外参数，adv, rewards
                for k, v in part_extra.items():
                    part_extra_acc[k] = part_extra_acc.get(k, 0.0) + v
            # 梯度裁剪
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.grad_clip)
            # 设学习率
            lr = get_lr(step, self.train_config)
            set_lr(self.optimizers, lr)
            # step
            step_all(self.optimizers)
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
                extra_text = "".join(f", {k}:{v:.4g}" for k, v in extras.items())
                log(f"step:{step}, loss:{loss_value:.4f}, norm:{norm_value:.4f}, lr:{lr:.4e}{extra_text}")
            # 定期 eval
            if (step + 1) % self.train_config.eval_interval == 0 or step == self.train_config.max_steps - 1:
                self._eval(step + 1)
        return dict(records)

    
    def _eval(self, step):
        # 做 eval
        self.model.eval()
        metrics = self.task.evaluate()
        text = ", ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
        # 打印 log
        log(f"step:{step}, {text}")
        self.model.train()
