from dataclasses import asdict
import json
import torch
import numpy as np
import model, dataloader
import os
import argparse
from logger import log, setup_log
from muon import MuonWithAdamW
from trainer import Trainer
from config import PretrainExperiment, TrainConfig, DataConfig, load_config
from torch.nn.parallel import DistributedDataParallel as DDP
from parallel.dist_utils import broadcast_params, setup, cleanup, get_rank, get_world_size, all_reduce_mean
from parallel.zero import ZeroOptimizer


class PretrainTask:
    def __init__(self, model, loader, val_loader, data_config):
        self.model = model
        self.loader = loader
        self.val_loader = val_loader
        self.data_config = data_config
        B, T = self.loader.B, self.loader.T
        global_batch_size = self.data_config.global_batch_size
        W = self.loader.world_size
        assert global_batch_size % (B*T*W) == 0
        self.grad_accum_steps = global_batch_size // (B*T*W)
        self.tokens_per_step = B*T*self.grad_accum_steps*W

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
        start_step=0,
        config=None):
    task = PretrainTask(model, loader, val_loader, data_config)
    trainer = Trainer(model, optimizer, train_config, task, ckpt_dir=ckpt_dir, optimizer_name=optimizer_name, config=config)
    records = trainer.fit(start_step)
    return records["loss"], records["norm"]


def configure_optimizer(model: model.ModelForCausalLM, train_config: TrainConfig, optimizer_name, zero_stage=0):
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
        param_groups = [
            {"params": wd, "weight_decay": weight_decay},
            {"params": keep, "weight_decay": 0.0},
        ]
        cls, kwargs = torch.optim.AdamW, dict(lr=lr, betas=betas, fused=True)
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
        cls, kwargs = MuonWithAdamW, {}
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name!r}, expected one of: adamw, muon")
    if zero_stage == 0:
        # 不做 Zero
        return cls(param_groups, **kwargs)
    elif zero_stage in (1, 2):
        optimizer = ZeroOptimizer(param_groups, cls, stage=zero_stage, **kwargs)
        log(f"optimizer.world_size:{optimizer.world_size}, optimizer.numel_per_rank:{optimizer.numel_per_rank}, zero_stage:{zero_stage}")
        return optimizer
    else:
        raise ValueError(f"zero_stage 只支持 0、1、2，收到 {zero_stage!r}")


def evaluate(model, val_loader: dataloader.DataLoader, data_config: DataConfig):
    model.eval()
    val_loader.reset()
    eval_steps = data_config.eval_steps
    W = val_loader.world_size
    assert eval_steps % W == 0, f"eval_steps % world_size = {eval_steps % W}, expected 0"
    loss_accum = 0.0
    per_rank_eval_steps = eval_steps // W
    for step in range(per_rank_eval_steps):
        x, y = val_loader.next_batch()
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
        loss_accum += loss.detach()
    loss_avg = loss_accum / per_rank_eval_steps
    model.train()
    loss_avg = all_reduce_mean(loss_avg)
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


def main(cfg: PretrainExperiment, resume, device):
    rank, world_size = get_rank(), get_world_size()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    data_dir = cfg.data.data_dir

    m = model.ModelForCausalLM(cfg.model)
    m.to(device)
    optimizer = configure_optimizer(m, cfg.train, cfg.train.optimizer, zero_stage=cfg.parallel.zero_stage)

    B, T = cfg.data.B, cfg.data.T
    loader = dataloader.DataLoader(data_dir, 'train', B, T, device, rank=rank, world_size=world_size)
    val_loader = dataloader.DataLoader(data_dir, 'val', B, T, device, rank=rank, world_size=world_size)
    start_step = 0
    if resume:
        # 断点续训，读取训练中断前的模型、优化器、train loader 的 state_dict
        path = os.path.join(cfg.run.ckpt_dir, "latest.pt")
        start_step = load_checkpoint(path, m, optimizer, cfg.train.optimizer, loader, device)
    if world_size > 1:
        if cfg.parallel.zero_stage == 2:
            # ZeRO-2 不能用 DDP 包
            broadcast_params(m)
            log(f"zero_stage=2：不包 DDP，参数已从 0 号卡广播")
        else:
            m = DDP(m, forward_sync_buffers=False)
    # 开始训练
    train(m, optimizer, loader, val_loader, cfg.train, cfg.data, cfg.train.optimizer, cfg.run.ckpt_dir, start_step=start_step, config=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, PretrainExperiment, args.set)
    
    # 写 log 的准备
    device = setup()
    setup_log(cfg.run.log)
    log(f"args:{args}")
    log(f"config:\n{json.dumps(asdict(cfg), indent=2)}")

    main(cfg, args.resume, device)
    cleanup()