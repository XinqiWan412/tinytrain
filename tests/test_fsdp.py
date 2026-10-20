import torch, pytest
from model import ModelForCausalLM, load_model
from pretrain import configure_optimizer 
from config import TrainConfig
import os, sys, subprocess, shutil
from pathlib import Path


def write_config(tmp_path, data_path, optimizer_name):
    toml = f"""[model]
vocab_size = 256
hidden_size = 64
num_hidden_layers = 2
num_attention_heads = 4
num_key_value_heads = 2
head_dim = 64
intermediate_size = 192
max_seq_len = 32

[data]
data_dir = "{data_path}"
B = 2
T = 8
global_batch_size = 64
eval_steps = 4

[train]
warmup_steps = 3
max_steps = 10
eval_interval = 5
optimizer = "{optimizer_name}"

"""
    path = tmp_path / "cfg.toml"
    path.write_text(toml)
    return path


def run_subprocess(cmd):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def run_two(config_path, out_dir, zero_stage, resume=False):
    # 跑两进程的 pretrain.py
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node=2", "pretrain.py", "--config", config_path, "--set", f"parallel.zero_stage={zero_stage}",
        "--set", f"run.ckpt_dir={out_dir}", "--set", f"run.log={out_dir / 'two.log'}"]
    if resume:
        cmd.append("--resume")
    return run_subprocess(cmd)


def test_fsdp_matches_ddp(tmp_path, make_data_dir):
    optimizer_name = "adamw"
    zero_stage = 3
    # 造数据，写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 跑 ddp 和 fsdp
    run_two(config_path, tmp_path / "ddp", 0)
    result = run_two(config_path, tmp_path / "fsdp", zero_stage)
    # 加载两边的 final.pt
    ckpt_path1 = tmp_path / "ddp" / "final.pt"
    ckpt_path2 = tmp_path / "fsdp" / "final.pt"
    ckpt1 = torch.load(ckpt_path1, map_location="cpu")  # ddp
    ckpt2 = torch.load(ckpt_path2, map_location="cpu")  # fsdp
    assert "已用 fully_shard 分片" in result.stdout
    assert ckpt2["config"]["parallel"]["zero_stage"] == zero_stage
    # 比较参数
    sd1 = ckpt1["model"]
    sd2 = ckpt2["model"]
    assert set(sd1) == set(sd2)
    for k in sd1.keys():
        a = sd1[k]
        b = sd2[k]
        diff = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-5), f"{k}：最大差 {diff}"
    # 比较 val_loss
    assert ckpt2["val_loss"] == pytest.approx(ckpt1["val_loss"], rel=1e-5)
    # load_model 读回后，和 ckpt2["model"] 逐参数一致
    model = load_model(ckpt_path2, "cpu")
    sd = model.state_dict()
    for k in sd:
        assert torch.equal(sd[k], ckpt2["model"][k])
    

def test_fsdp_rejects_muon(tiny_config):
    model = ModelForCausalLM(tiny_config)
    with pytest.raises(ValueError, match="muon"):
        configure_optimizer(model, TrainConfig(), "muon", zero_stage=3)


def test_fsdp_resume_two(tmp_path, make_data_dir):
    optimizer_name = "adamw"
    zero_stage = 3
    # 造数据，写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 一口气跑完
    run_two(config_path, tmp_path / "full", zero_stage)
    # 把 latest.pt 复制到另一个目录
    (tmp_path / "resume").mkdir()
    shutil.copy(tmp_path / "full" / "latest.pt", tmp_path / "resume" / "latest.pt")
    # 续训
    result = run_two(config_path, tmp_path / "resume", zero_stage, resume=True)
    assert "resume from step 5" in result.stdout, "续训失败"
    # 加载两边的 final.pt
    ckpt_path1 = tmp_path / "full" / "final.pt"
    ckpt_path2 = tmp_path / "resume" / "final.pt"
    ckpt1 = torch.load(ckpt_path1, map_location="cpu")
    ckpt2 = torch.load(ckpt_path2, map_location="cpu")
    # 比较参数
    sd1 = ckpt1["model"]
    sd2 = ckpt2["model"]
    assert set(sd1) == set(sd2)
    for k in sd1.keys():
        a = sd1[k]
        b = sd2[k]
        diff = (a - b).abs().max().item()
        assert torch.equal(a, b), f"{k}：最大差 {diff}"
    # 比较 val_loss
    assert ckpt2["val_loss"] == ckpt1["val_loss"]

