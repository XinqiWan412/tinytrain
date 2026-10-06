import torch
import numpy as np
import model, dataloader
import time, math, os
from dataclasses import dataclass, asdict
import argparse

# MODE = "overfit"
MODE = "train"
# MODE = "lr"
# MODE = "load"


@dataclass
class TrainConfig:
    data_dir: str = "./data/tokenized/bpe32k"
    B: int = 16
    T: int = 512
    global_batch_size: int = 65536
    max_lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 150
    max_steps: int = 3000
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 42
    log_interval: int = 20
    eval_interval: int = 250
    eval_steps: int = 20


def overfit_single_batch(model, optimizer, loader: dataloader.DataLoader, train_config: TrainConfig):
    x, y = loader.next_batch()
    model.train()
    steps = train_config.max_steps
    for step in range(steps):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        if step % 20 == 0 or step == steps - 1:  # 每 20 步打印一次
            log(f"step {step} loss {loss.item()}")


def train(model: model.ModelForCausalLM, 
        optimizer: torch.optim.Optimizer, 
        loader: dataloader.DataLoader, 
        val_loader: dataloader.DataLoader,
        train_config: TrainConfig):
    model.train()
    B, T = loader.B, loader.T
    use_cuda = torch.cuda.is_available()
    steps, log_interval = train_config.max_steps, train_config.log_interval
    global_batch_size = train_config.global_batch_size
    eval_interval = train_config.eval_interval
    assert global_batch_size % (B*T) == 0
    grad_accum_steps = global_batch_size // (B*T)
    for step in range(steps):
        # 每 eval_interval 个 step 就做一次 eval
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, train_config)
            log(f'step:{step}, val_loss:{val_loss:.4f}')
            save_checkpoint("checkpoints/latest.pt", model, optimizer, loader, train_config, step, val_loss)
        t0 = time.perf_counter()
        optimizer.zero_grad()
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
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        dt = t1 - t0
        if step % log_interval == 0 or step == steps - 1:
            log(f"step:{step}, loss:{loss_accum.item():.4f}, dt:{dt*1000:.1f}ms, tok/s:{B*T/dt*grad_accum_steps:.0f}, norm:{norm.item():.4f}, lr:{lr:.4e}")

    # 最后做一次 eval
    val_loss = evaluate(model, val_loader, train_config)
    log(f'step:{steps}, val_loss:{val_loss:.4f}')
    save_checkpoint("checkpoints/final.pt", model, optimizer, loader, train_config, steps, val_loss)


def configure_optimizer(model: model.ModelForCausalLM, train_config: TrainConfig):
    lr, betas, weight_decay = train_config.max_lr, train_config.betas, train_config.weight_decay
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
    return optimizer


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
                    optimizer: torch.optim.Optimizer, 
                    loader: dataloader.DataLoader, 
                    train_config: TrainConfig, 
                    step, val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        # 用模型需要的
        "model": model.state_dict(),
        "model_config": asdict(model.config),
        # 接着训需要的
        "optimizer": optimizer.state_dict(),
        "step": step,  # 从这一步继续训
        "loader_shard_idx": loader.shard_idx,
        "loader_pos": loader.pos,
        # 留作记录的
        "train_config": asdict(train_config),
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def log(text):
    print(text)
    log_file.write(text + "\n")
    log_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/pretrain.log")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    # 写 log 的准备
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log_file = open(args.log, "a")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config = TrainConfig()
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    data_dir = train_config.data_dir

    config = model.ModelConfig()

    if MODE == "overfit":
        m = model.ModelForCausalLM(config)
        m.to(device)
        optimizer = configure_optimizer(m, train_config)

        B, T = 4, 128
        loader = dataloader.DataLoader(data_dir, 'train', B, T, device)
        overfit_single_batch(m, optimizer, loader, train_config)
    elif MODE == "train":
        m = model.ModelForCausalLM(config)
        m.to(device)
        optimizer = configure_optimizer(m, train_config)

        B, T = train_config.B, train_config.T
        loader = dataloader.DataLoader(data_dir, 'train', B, T, device)
        val_loader = dataloader.DataLoader(data_dir, 'val', B, T, device)
        train(m, optimizer, loader, val_loader, train_config)
    elif MODE == "lr":
        for s in [0, 29, 30, 165, 299, 300]:
            log(f"s {s}, lr {get_lr(s, train_config):.6e}")
    elif MODE == "load":
        path = "checkpoints/final.pt"
        B, T = train_config.B, train_config.T
        val_loader = dataloader.DataLoader(data_dir, 'val', B, T, device)
        m = model.load_model(path, device)
        loss = evaluate(m, val_loader, train_config)

        ckpt = torch.load(path, map_location=device)
        ckpt_loss = ckpt["val_loss"]
        log(f"loss={loss:.6f}, ckpt_loss={ckpt_loss:.6f}")
