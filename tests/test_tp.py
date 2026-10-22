import shutil
from config import TrainConfig
from model import ModelForCausalLM, load_model
from parallel.dist_utils import ParallelLayout, compute_layout
import pytest, os, subprocess, sys, torch
from pathlib import Path

from pretrain import configure_optimizer


def test_compute_layout():
    # 4 卡，TP = 2
    assert compute_layout(0, 4, 2) == ParallelLayout(tp_rank=0, tp_size=2, dp_rank=0, dp_size=2)
    assert compute_layout(1, 4, 2) == ParallelLayout(tp_rank=1, tp_size=2, dp_rank=0, dp_size=2)
    assert compute_layout(2, 4, 2) == ParallelLayout(tp_rank=0, tp_size=2, dp_rank=1, dp_size=2)
    assert compute_layout(3, 4, 2) == ParallelLayout(tp_rank=1, tp_size=2, dp_rank=1, dp_size=2)
    # 2 卡，TP = 2
    assert compute_layout(0, 2, 2) == ParallelLayout(tp_rank=0, tp_size=2, dp_rank=0, dp_size=1)
    assert compute_layout(1, 2, 2) == ParallelLayout(tp_rank=1, tp_size=2, dp_rank=0, dp_size=1)
    # 4 卡，TP = 1
    assert compute_layout(0, 4, 1) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=0, dp_size=4)
    assert compute_layout(1, 4, 1) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=1, dp_size=4)
    assert compute_layout(2, 4, 1) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=2, dp_size=4)
    assert compute_layout(3, 4, 1) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=3, dp_size=4)
    # world_size 不能整除 tp_size
    with pytest.raises(ValueError):
        compute_layout(0, 3, 2)  # 3 卡，TP = 2


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


def test_tp_mlp_matches_full():
    path = Path(__file__).parent / "tp_worker.py"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
    "--nproc_per_node=2", str(path)]
    result = run_subprocess(cmd)
    assert "rank 0 OK" in result.stdout
    assert "rank 1 OK" in result.stdout


def test_tp_model_matches_full():
    path = Path(__file__).parent / "tp_model_worker.py"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
    "--nproc_per_node=2", str(path)]
    result = run_subprocess(cmd)
    assert "rank 0 OK" in result.stdout
    assert "rank 1 OK" in result.stdout


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


def run_tp(config_path, out_dir, nproc, tp_size, resume=False):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={nproc}", "pretrain.py", "--config", config_path, "--set", f"parallel.tp_size={tp_size}",
        "--set", f"run.ckpt_dir={out_dir}", "--set", f"run.log={out_dir / 'tp.log'}"]
    if resume:
        cmd.append("--resume")
    return run_subprocess(cmd)


def compare_state_dicts(sd1, sd2, atol=None):
    """atol 为 None 时要求逐位相同；失败时报出超标的参数个数和最大差"""
    assert set(sd1) == set(sd2), "两边的键不一致"
    diffs = {k: (sd1[k] - sd2[k]).abs().max().item() for k in sd1}
    worst = max(diffs, key=diffs.get)
    if atol is None:
        bad = [k for k in sd1 if not torch.equal(sd1[k], sd2[k])]
    else:
        bad = [k for k in sd1 if not torch.allclose(sd1[k], sd2[k], atol=atol)]
    assert not bad, f"{len(bad)} 个参数超出容差；最大差 {diffs[worst]:.2e}，在 {worst}"


def test_tp_matches_single(tmp_path, make_data_dir):
    optimizer_name = "adamw"
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # tp_size = 1
    run_tp(config_path, tmp_path / "single", 1, 1)
    # tp_size = 2
    result = run_tp(config_path, tmp_path / "tp", 2, 2)
    # 加载两边的 final.pt
    ckpt_path1 = tmp_path / "single" / "final.pt"
    ckpt_path2 = tmp_path / "tp" / "final.pt"
    ckpt1 = torch.load(ckpt_path1, map_location="cpu")
    ckpt2 = torch.load(ckpt_path2, map_location="cpu")
    # 检查 tp_size = 2 生效了
    assert "model 已切分" in result.stdout
    assert ckpt2["config"]["parallel"]["tp_size"] == 2
    # 比较 val_loss
    assert ckpt2["val_loss"] == pytest.approx(ckpt1["val_loss"], rel=1e-5)
    # 比较参数
    sd1 = ckpt1["model"]
    sd2 = ckpt2["model"]
    assert set(sd1) == set(sd2)  # 比较 key
    compare_state_dicts(sd1, sd2, atol=1e-4)
    # 检查 load_model
    model = load_model(ckpt_path2, "cpu")
    for name, p in model.named_parameters():
        assert torch.equal(p, ckpt2["model"][name])


def test_tp_resume_two(tmp_path, make_data_dir):
    optimizer_name = "adamw"
    nproc = 2
    tp_size = 2
    # 造数据，写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 一口气跑完
    run_tp(config_path, tmp_path / "full", nproc, tp_size)
    # 把 latest.pt 复制到另一个目录
    (tmp_path / "resume").mkdir()
    shutil.copy(tmp_path / "full" / "latest.pt", tmp_path / "resume" / "latest.pt")
    # 续训
    result = run_tp(config_path, tmp_path / "resume", nproc, tp_size, resume=True)
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
    compare_state_dicts(sd1, sd2)
    # 比较 val_loss
    assert ckpt2["val_loss"] == ckpt1["val_loss"]


def test_tp_rejects_muon_and_zero(tiny_config):
    model = ModelForCausalLM(tiny_config)
    with pytest.raises(ValueError, match="muon"):
        configure_optimizer(model, TrainConfig(), "muon", tp_size=2)
    with pytest.raises(ValueError, match="zero_stage"):
        configure_optimizer(model, TrainConfig(), "adamw", zero_stage=1, tp_size=2)

