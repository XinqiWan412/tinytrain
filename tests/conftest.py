from fileinput import filename
import pytest
from model import ModelConfig
import json
import numpy as np


@pytest.fixture
def tiny_config():
    config = ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=192,
        max_seq_len=32,
    )
    return config


@pytest.fixture
def make_data_dir(tmp_path):
    def _make(n_train, tokens_per_shard, val_same_as_train=False):
        data_dir = tmp_path / "data"      # 用子目录，checkpoint 可以另放 tmp_path / "ckpt"，互不干扰
        data_dir.mkdir(exist_ok=True)
        rng = np.random.default_rng(0)    # 固定种子，每次造的数据都一样
        shards = []                       # 用来收集 meta.json 里的 shards 列表

        # 1. 循环 n_train 次：生成随机 token，写成 train_xxxxxx.bin，往 shards 里加一项
        for i in range(n_train):
            tok = rng.integers(0, 256, size=tokens_per_shard, dtype=np.uint16)
            name = f"train_{i:06d}.bin"
            tok.tofile(data_dir / name)
            shards.append({"name": name, "tokens": tokens_per_shard})
            if i == 0 and val_same_as_train:
                first_tok = tok
        # 2. 写 val_000000.bin：val_same_as_train 为 True 时用第一片 train 的内容，否则另外生成一份
        tok = first_tok if val_same_as_train else rng.integers(0, 256, size=tokens_per_shard, dtype=np.uint16)
        name = "val_000000.bin"
        tok.tofile(data_dir / name)
        shards.append({"name": name, "tokens": tokens_per_shard})
        # 3. 写 meta.json
        with open(data_dir / "meta.json", "w") as f:
            json.dump({"dtype": "uint16", "shards": shards}, f)

        return data_dir
    return _make