import torch
import numpy as np
import model, dataloader
import os
from dataclasses import dataclass
import argparse
from logger import log, setup_log
from muon import Muon
from trainer import Trainer, get_lr  # get_lr 在这里没用，但 test_pretrain 是从这里 import 的，所以要保留


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


class PretrainTask:
    def __init__(self, model, loader, val_loader, train_config):
        self.model = model
        self.loader = loader
        self.val_loader = val_loader
        self.train_config = train_config
        B, T = self.loader.B, self.loader.T
        global_batch_size = self.train_config.global_batch_size
        assert global_batch_size % (B*T) == 0
        self.grad_accum_steps = global_batch_size // (B*T)
        self.tokens_per_step = B*T*self.grad_accum_steps

    def get_parts(self, step):
        res = []
        for _ in range(self.grad_accum_steps):
            x, y = self.loader.next_batch()
            res.append((x, y))
        return res, {}

    def loss(self, part):
        x, y = part
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = self.model(x, y)
            loss = loss / self.grad_accum_steps
        return loss, {}

    def evaluate(self):
        loss_avg = evaluate(self.model, self.val_loader, self.train_config)
        return {"val_loss": loss_avg}

    def state_dict(self):
        return {"loader": self.loader.state_dict()}


def train(model: model.ModelForCausalLM, 
        optimizers: list[torch.optim.Optimizer], 
        loader: dataloader.DataLoader, 
        val_loader: dataloader.DataLoader,
        train_config: TrainConfig,
        optimizer_name,
        ckpt_dir,
        start_step=0):
    task = PretrainTask(model, loader, val_loader, train_config)
    trainer = Trainer(model, optimizers, train_config, task, ckpt_dir=ckpt_dir, optimizer_name=optimizer_name)
    records = trainer.fit(start_step)
    return records["loss"], records["norm"]


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


def load_checkpoint(path, model, optimizers, optimizer_name, loader, device):
    ckpt = torch.load(path, map_location=device)
    if ckpt["optimizer_name"] != optimizer_name:
        raise ValueError(f"上次训练使用的优化器是 {ckpt['optimizer_name']}，这次用的是 {optimizer_name}")
    model.load_state_dict(ckpt["model"])
    for optim, state_dict in zip(optimizers, ckpt["optimizers"]):
        optim.load_state_dict(state_dict)
    loader.load_state_dict(ckpt["loader"])
    start_step = ckpt["step"]
    log(f"resume from step {start_step}")
    return start_step


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

    m = model.ModelForCausalLM(config)
    m.to(device)
    optimizers = configure_optimizer(m, train_config, args.optimizer)

    B, T = train_config.B, train_config.T
    loader = dataloader.DataLoader(data_dir, 'train', B, T, device)
    val_loader = dataloader.DataLoader(data_dir, 'val', B, T, device)
    if args.resume:
        # 断点续训，读取训练中断前的模型、优化器、train loader 的 state_dict
        path = os.path.join(args.ckpt_dir, "latest.pt")
        start_step = load_checkpoint(path, m, optimizers, args.optimizer, loader, device)
        # 开始训练
        train(m, optimizers, loader, val_loader, train_config, args.optimizer, args.ckpt_dir, start_step=start_step)
    else:
        # 从头开始的训练走这里            
        train(m, optimizers, loader, val_loader, train_config, args.optimizer, args.ckpt_dir)
