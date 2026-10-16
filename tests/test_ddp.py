import os, sys, subprocess, torch, pytest, shutil
from pathlib import Path


def write_config(tmp_path, data_path):
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
optimizer = "adamw"

[run]
ckpt_dir = "{tmp_path}/single"
log = "{tmp_path}/single.log"
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


def test_ddp_matches_single(tmp_path, make_data_dir):
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir)
    # 跑单进程
    cmd = [sys.executable, "pretrain.py", "--config", config_path]
    run_subprocess(cmd)
    # 跑两进程
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=2", "pretrain.py", "--config", config_path,
            "--set", f"run.ckpt_dir={tmp_path / 'two'}", "--set", f"run.log={tmp_path / 'two.log'}"]
    run_subprocess(cmd)
    # 加载两边的 final.pt
    ckpt_path1 = tmp_path / "single" / "final.pt"
    ckpt_path2 = tmp_path / "two" / "final.pt"
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
        assert torch.allclose(a, b, atol=1e-5), f"{k}：最大差 {diff}"
    # 比较 val_loss
    assert ckpt2["val_loss"] == pytest.approx(ckpt1["val_loss"], rel=1e-5)


def test_ddp_resume(tmp_path, make_data_dir):
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir)
    # 两进程一次跑完
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=2", "pretrain.py", "--config", config_path,
            "--set", f"run.ckpt_dir={tmp_path / 'full'}", "--set", f"run.log={tmp_path / 'full.log'}"]
    run_subprocess(cmd)
    # 把 latest.pt 复制到另一个目录
    (tmp_path / "resume").mkdir()
    shutil.copy(tmp_path / "full" / "latest.pt", tmp_path / "resume" / "latest.pt")
    # 两进程续训
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
            "--nproc_per_node=2", "pretrain.py", "--config", config_path,
            "--set", f"run.ckpt_dir={tmp_path / 'resume'}", "--set", 
            f"run.log={tmp_path / 'resume.log'}", "--resume"]
    result = run_subprocess(cmd)
    assert "resume from step 5" in result.stdout  # 确保是续训而不是从头训
    # 比较两次 final.pt
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


