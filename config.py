from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str = "./data/tokenized/bpe32k"
    B: int = 16
    T: int = 512
    global_batch_size: int = 65536
    eval_steps: int = 20


@dataclass
class TrainConfig:
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
    optimizer: str = "adamw"


@dataclass
class SFTConfig:
    B: int = 128


@dataclass
class GRPOConfig:
    B: int = 64
    G: int = 8
    beta: float = 0.2
    chunk_size: int = 128


@dataclass
class RunConfig:
    init: str = ""
    out: str = ""
    ckpt_dir: str = "checkpoints"
    log: str = "logs/train.log"
