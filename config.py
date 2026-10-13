from dataclasses import dataclass, fields
from model import ModelConfig
import tomllib


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


@dataclass
class PretrainExperiment:
    model: ModelConfig 
    data: DataConfig
    train: TrainConfig
    run: RunConfig


@dataclass
class SFTExperiment:
    train: TrainConfig
    sft: SFTConfig
    run: RunConfig


@dataclass
class GRPOExperiment:
    train: TrainConfig
    grpo: GRPOConfig
    run: RunConfig


def _convert(value, typ, where):
    if typ is bool:
        if type(value) is bool:
            return value
        raise ValueError(f"{where}: expected bool, got {value!r} ({type(value).__name__})")

    if typ is int:
        if type(value) is int:
            return value
        raise ValueError(f"{where}: expected int, got {value!r} ({type(value).__name__})")   

    if typ is float:
        if type(value) is int or type(value) is float:
            return float(value)
        raise ValueError(f"{where}: expected float, got {value!r} ({type(value).__name__})")

    if typ is str:
        if type(value) is str:
            return value
        raise ValueError(f"{where}: expected str, got {value!r} ({type(value).__name__})")

    if typ is tuple:
        if type(value) is list:
            return tuple(value)
        raise ValueError(f"{where}: expected list (for tuple field), got {value!r} ({type(value).__name__})")
    
    raise ValueError(f"{where}: unsupported field type {typ!r}")


def _build_section(cls, table, section_name):
    if type(table) is not dict:
        raise ValueError(f"{section_name}: expected a table, got {type(table).__name__}")

    valid = {f.name for f in fields(cls)}

    extra = set(table) - valid
    # table 里有不认识的 key 就报错
    if extra:
        raise ValueError(
            f"{section_name}: unknown keys {sorted(extra)},"
            f"valid keys are {sorted(valid)}"
        )
    
    result = {}
    for f in fields(cls):
        if f.name in table:
            where = f"{section_name}.{f.name}"
            result[f.name] = _convert(table[f.name], f.type, where)
    
    return cls(**result)


def load_config(path, cls, sets=()):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    
    # 遍历 sets，用 _parse_set 解析
    for s in sets:
        section, key, value = _parse_set(s)
        raw.setdefault(section, {})[key] = value

    valid_sections = {f.name for f in fields(cls)}
    # config 里有不认识的 key 就报错
    extra = set(raw) - valid_sections
    if extra:
        raise ValueError(
            f"unknown sections {sorted(extra)}, "
            f"valid sections are {sorted(valid_sections)}"
        )
    
    sections = {}
    for f in fields(cls):
        table = raw.get(f.name, {})
        sections[f.name] = _build_section(f.type, table, f.name)

    return cls(**sections)


def _parse_set(s):
    # 在第一个等号切开，没有就报错
    if "=" not in s:
        raise ValueError(f"invalid --set {s!r}: missing '='")
    left, right = s.split("=", 1)

    # 左边按点切开，必须两段
    parts = left.split(".")
    if len(parts) != 2:
        raise ValueError(f"invalid --set {s!r}: expected section.key, got {left!r}")
    section, key = parts

    # 右边拼成 "v = " + right
    try:
        v = tomllib.loads("v = " + right)["v"]
    except tomllib.TOMLDecodeError:
        # 解析失败就原样当字符串
        v = right
    
    return section, key, v

