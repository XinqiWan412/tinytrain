import torch
import numpy as np
import model, dataloader
import os
import argparse
from logger import log, setup_log
from muon import MuonWithAdamW
from trainer import Trainer
from config import TrainConfig, DataConfig


class PretrainTask:
    def __init__(self, model, loader, val_loader, data_config):
        self.model = model
        self.loader = loader
        self.val_loader = val_loader
        self.data_config = data_config
        B, T = self.loader.B, self.loader.T
        global_batch_size = self.data_config.global_batch_size
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
        loss_avg = evaluate(self.model, self.val_loader, self.data_config)
        return {"val_loss": loss_avg}

    def state_dict(self):
        return {"loader": self.loader.state_dict()}


def train(model: model.ModelForCausalLM, 
        optimizer: torch.optim.Optimizer, 
        loader: dataloader.DataLoader, 
        val_loader: dataloader.DataLoader,
        train_config: TrainConfig,
        data_config: DataConfig,
        optimizer_name,
        ckpt_dir,
        start_step=0):
    task = PretrainTask(model, loader, val_loader, data_config)
    trainer = Trainer(model, optimizer, train_config, task, ckpt_dir=ckpt_dir, optimizer_name=optimizer_name)
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
        return optimizer
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
        param_groups = [
            {"params": muon_params, "use_muon": True, "lr": lr, "weight_decay": weight_decay},
            {"params": wd, "use_muon": False, "lr": lr, "weight_decay": weight_decay, "betas": betas},
            {"params": keep, "use_muon": False, "lr": lr, "weight_decay": 0.0, "betas": betas},
        ]
        optimizer = MuonWithAdamW(param_groups)
        return optimizer
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name!r}, expected one of: adamw, muon")


def evaluate(model, val_loader: dataloader.DataLoader, data_config: DataConfig):
    model.eval()
    val_loader.reset()
    eval_steps = data_config.eval_steps
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


def load_checkpoint(path, model, optimizer, optimizer_name, loader, device):
    ckpt = torch.load(path, map_location=device)
    if ckpt["optimizer_name"] != optimizer_name:
        raise ValueError(f"上次训练使用的优化器是 {ckpt['optimizer_name']}，这次用的是 {optimizer_name}")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
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
    data_config = DataConfig()
    if args.max_steps is not None:
        train_config.max_steps = args.max_steps
    if args.eval_interval is not None:
        train_config.eval_interval = args.eval_interval
    train_config.optimizer = args.optimizer
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    data_dir = data_config.data_dir

    config = model.ModelConfig()

    m = model.ModelForCausalLM(config)
    m.to(device)
    optimizer = configure_optimizer(m, train_config, args.optimizer)

    B, T = data_config.B, data_config.T
    loader = dataloader.DataLoader(data_dir, 'train', B, T, device)
    val_loader = dataloader.DataLoader(data_dir, 'val', B, T, device)
    if args.resume:
        # 断点续训，读取训练中断前的模型、优化器、train loader 的 state_dict
        path = os.path.join(args.ckpt_dir, "latest.pt")
        start_step = load_checkpoint(path, m, optimizer, args.optimizer, loader, device)
        # 开始训练
        train(m, optimizer, loader, val_loader, train_config, data_config, args.optimizer, args.ckpt_dir, start_step=start_step)
    else:
        # 从头开始的训练走这里            
        train(m, optimizer, loader, val_loader, train_config, data_config, args.optimizer, args.ckpt_dir)
