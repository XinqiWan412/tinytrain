import torch
import numpy as np
import model, dataloader
import time, math, os
from dataclasses import dataclass, asdict
import argparse
from logger import log, setup_log
from muon import Muon

# MODE = "overfit"
MODE = "train"
# MODE = "load"


@dataclass
class TrainConfig:
    data_dir: str = "./data/tokenized/bpe32k"
    B: int = 16
    T: int = 512
    global_batch_size: int = 65536
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 34000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 20
    eval_interval: int = 250
    eval_steps: int = 20


def overfit_single_batch(model, optimizers, loader: dataloader.DataLoader, train_config: TrainConfig):
    x, y = loader.next_batch()
    model.train()
    steps = train_config.max_steps
    for step in range(steps):
        zero_grad_all(optimizers)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        step_all(optimizers)
        if step % 20 == 0 or step == steps - 1:  # 每 20 步打印一次
            log(f"step {step} loss {loss.item()}")


def train(model: model.ModelForCausalLM, 
        optimizers: list[torch.optim.Optimizer], 
        loader: dataloader.DataLoader, 
        val_loader: dataloader.DataLoader,
        train_config: TrainConfig,
        optimizer_name,
        ckpt_dir,
        start_step=0):
    model.train()
    B, T = loader.B, loader.T
    use_cuda = torch.cuda.is_available()
    steps, log_interval = train_config.max_steps, train_config.log_interval
    global_batch_size = train_config.global_batch_size
    eval_interval = train_config.eval_interval
    assert global_batch_size % (B*T) == 0
    grad_accum_steps = global_batch_size // (B*T)
    for step in range(start_step, steps):
        # 每 eval_interval 个 step 就做一次 eval
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, train_config)
            log(f'step:{step}, val_loss:{val_loss:.4f}')
            save_checkpoint(os.path.join(ckpt_dir, "latest.pt"), model, optimizers, optimizer_name, loader, train_config, step, val_loss)
        t0 = time.perf_counter()
        zero_grad_all(optimizers)
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            x, y = loader.next_batch()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
            loss.backward()
            loss_accum += loss.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        lr = get_lr(step, train_config)
        set_lr(optimizers, lr)
        step_all(optimizers)
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = t1 - t0
        if step % log_interval == 0 or step == steps - 1:
            log(f"step:{step}, loss:{loss_accum.item():.4f}, dt:{dt*1000:.1f}ms, tok/s:{B*T/dt*grad_accum_steps:.0f}, norm:{norm.item():.4f}, lr:{lr:.4e}")

    # 最后做一次 eval
    val_loss = evaluate(model, val_loader, train_config)
    log(f'step:{steps}, val_loss:{val_loss:.4f}')
    save_checkpoint(os.path.join(ckpt_dir, "final.pt"), model, optimizers, optimizer_name, loader, train_config, steps, val_loss)


def configure_optimizer(model: model.ModelForCausalLM, train_config: TrainConfig, optimizer_name):
    lr, betas, weight_decay = train_config.max_lr, train_config.betas, train_config.weight_decay
    if optimizer_name == "adamw":
        wd, keep = [], []
        for p in model.parameters():
            if p.dim() >= 2:
                wd.append(p)
            else:
                keep.append(p)
        
        wd_numel = sum(p.numel() for p in wd)
        keep_numel = sum(p.numel() for p in keep)
        log(f"weight_decay: {len(wd)} tensors, {wd_numel} params | no_decay: {len(keep)} tensors, {keep_numel} params")
        params = [
            {"params": wd, "weight_decay": weight_decay},
            {"params": keep, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, fused=True)
        return [optimizer]
    elif optimizer_name == "muon":
        muon_params, wd, keep = [], [], []
        for name, p in model.named_parameters():
            if len(p.shape) == 1:
                keep.append(p)
            elif "embed_tokens" in name:
                wd.append(p)
            else:
                muon_params.append(p)
        
        muon_params_numel = sum(p.numel() for p in muon_params)
        wd_numel = sum(p.numel() for p in wd)
        keep_numel = sum(p.numel() for p in keep)
        assert muon_params_numel + wd_numel + keep_numel == sum(p.numel() for p in model.parameters())
        log(f"muon: {len(muon_params)} tensors, {muon_params_numel} params | weight_decay: {len(wd)} tensors, {wd_numel} params | no_decay: {len(keep)} tensors, {keep_numel} params")
        muon = Muon(muon_params, lr=lr, weight_decay=weight_decay)
        params = [
            {"params": wd, "weight_decay": weight_decay},
            {"params": keep, "weight_decay": 0.0},
        ]
        adamw = torch.optim.AdamW(params, lr=lr, betas=betas, fused=True)
        return [muon, adamw]
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name!r}, expected one of: adamw, muon")


def zero_grad_all(optimizers):
    for optim in optimizers:
        optim.zero_grad()


def set_lr(optimizers, lr):
    for optim in optimizers:
        for group in optim.param_groups:
            group["lr"] = lr


def step_all(optimizers):
    for optim in optimizers:
        optim.step()


def get_lr(step, train_config: TrainConfig):
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


def evaluate(model, val_loader: dataloader.DataLoader, train_config: TrainConfig):
    model.eval()
    val_loader.reset()
    eval_steps = train_config.eval_steps
    loss_accum = 0.0
    for step in range(eval_steps):
        x, y = val_loader.next_batch()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        loss_accum += loss.detach()
    loss_avg = loss_accum / eval_steps
    model.train()
    return loss_avg.item()


def save_checkpoint(path, 
                    model: model.ModelForCausalLM, 
                    optimizers: list[torch.optim.Optimizer], 
                    optimizer_name,
                    loader: dataloader.DataLoader, 
                    train_config: TrainConfig, 
                    step, val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        # 用模型需要的
        "model": model.state_dict(),
        "model_config": asdict(model.config),
        # 接着训需要的
        "optimizers": [optim.state_dict() for optim in optimizers],
        "optimizer_name": optimizer_name,
        "step": step,  # 从这一步继续训
        "loader": loader.state_dict(),
        # 留作记录的
        "train_config": asdict(train_config),
        "val_loss": val_loss,
    }
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/pretrain.log")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_interval", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--ckpt_dir", default="checkpoints")
    args = parser.parse_args()

    # 写 log 的准备
    setup_log(args.log)
    log(f"args:{args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = TrainConfig()
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.eval_interval is not None:
        train_config.eval_interval = args.eval_interval
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    data_dir = train_config.data_dir

    config = model.ModelConfig()

    if MODE == "overfit":
        m = model.ModelForCausalLM(config)
        m.to(device)
        optimizers = configure_optimizer(m, train_config, args.optimizer)

        B, T = 4, 128
        loader = dataloader.DataLoader(data_dir, 'train', B, T, device)
        overfit_single_batch(m, optimizers, loader, train_config)
    elif MODE == "train":
        m = model.ModelForCausalLM(config)
        m.to(device)
        optimizers = configure_optimizer(m, train_config, args.optimizer)

        B, T = train_config.B, train_config.T
        loader = dataloader.DataLoader(data_dir, 'train', B, T, device)
        val_loader = dataloader.DataLoader(data_dir, 'val', B, T, device)
        if args.resume:
            # 断点续训，读取训练中断前的模型、优化器、train loader 的 state_dict
            ckpt = torch.load(os.path.join(args.ckpt_dir, "latest.pt"), map_location=device)
            if ckpt["optimizer_name"] != args.optimizer:
                raise ValueError(f"上次训练使用的优化器是 {ckpt['optimizer_name']}，这次用的是 {args.optimizer}")
            m.load_state_dict(ckpt["model"])
            for optim, state_dict in zip(optimizers, ckpt["optimizers"]):
                optim.load_state_dict(state_dict)
            loader.load_state_dict(ckpt["loader"])
            start_step = ckpt["step"]
            del ckpt  # 注意要释放 ckpt，能省很多显存
            log(f"resume from step {start_step}")
            # 开始训练
            train(m, optimizers, loader, val_loader, train_config, args.optimizer, args.ckpt_dir, start_step=start_step)
        else:
            # 从头开始的训练走这里            
            train(m, optimizers, loader, val_loader, train_config, args.optimizer, args.ckpt_dir)
    elif MODE == "load":
        path = "checkpoints/final.pt"
        B, T = train_config.B, train_config.T
        val_loader = dataloader.DataLoader(data_dir, 'val', B, T, device)
        m = model.load_model(path, device)
        loss = evaluate(m, val_loader, train_config)

        ckpt = torch.load(path, map_location=device)
        ckpt_loss = ckpt["val_loss"]
        log(f"loss={loss:.6f}, ckpt_loss={ckpt_loss:.6f}")
