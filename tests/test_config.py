import pytest
from config import load_config, SFTExperiment, TrainConfig, SFTConfig, RunConfig


def write_toml(tmp_path, text):
    path = tmp_path / "cfg.toml"
    path.write_text(text)
    return path


def test_load_ok(tmp_path):
    text = """[train]
max_lr = 3e-4
min_lr = 3e-5
warmup_steps = 50
max_steps = 2000
betas = [0.8, 0.9]

[sft]
B = 64"""
    path = write_toml(tmp_path, text)
    cfg = load_config(path, SFTExperiment)
    expected = SFTExperiment(
        train=TrainConfig(max_lr=3e-4, min_lr=3e-5, warmup_steps=50, max_steps=2000, betas=(0.8, 0.9)),
        sft=SFTConfig(B=64),
        run=RunConfig()
    )
    assert cfg == expected


def test_unknown_key_raises(tmp_path):
    # 拼错键
    text = """[train]
warmup_step = 50"""
    path = write_toml(tmp_path, text)
    # 必须抛 ValueError，才算通过测试
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path, SFTExperiment)
    
    # 拼错节名
    text = """[trian]
warmup_steps = 50"""
    path = write_toml(tmp_path, text)
    # 必须抛 ValueError，才算通过测试
    with pytest.raises(ValueError, match="trian"):
        load_config(path, SFTExperiment)


def test_wrong_type_raises(tmp_path):
    # 测试错误的数据格式
    text = """[train]
warmup_steps = 1e3"""
    path = write_toml(tmp_path, text)
    # 必须抛 ValueError，才算通过测试
    with pytest.raises(ValueError, match="expected int"):
        load_config(path, SFTExperiment)
    
    text = """[train]
warmup_steps = true"""
    path = write_toml(tmp_path, text)
    # 必须抛 ValueError，才算通过测试
    with pytest.raises(ValueError, match="expected int"):
        load_config(path, SFTExperiment)


def test_set_override(tmp_path):
    text = """[train]
max_steps = 2000"""
    path = write_toml(tmp_path, text)

    cfg = load_config(path, SFTExperiment, sets=[
        "train.max_steps=250",
        "train.max_lr=1e-3",
        "run.log=/tmp/x.log",
    ])

    assert cfg.train.max_steps == 250
    assert cfg.train.max_lr == 1e-3
    assert cfg.run.log == "/tmp/x.log"

    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path, SFTExperiment, sets=["train.max_step=250"])
