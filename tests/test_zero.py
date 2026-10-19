from parallel.zero import partition_params, ZeroOptimizer
import torch, pytest
from model import ModelForCausalLM
from pretrain import configure_optimizer, load_checkpoint, train
from dataloader import DataLoader
from config import TrainConfig, DataConfig
import os, sys, subprocess, shutil
from pathlib import Path

B, T = 2, 8


def test_partition():
    assert partition_params([5, 9, 2, 9, 4], 2) == ([0, 0, 1, 1, 1], [14, 15])
    assert partition_params([5, 9, 2, 9, 4], 1) == ([0, 0, 0, 0, 0], [29])


def build(optimizer_name, tiny_config, make_data_dir, zero_stage, n_train=2, tokens_per_shard=None, val_same_as_train=False, max_steps=10):
    # 初始化模型、优化器、loader、train_config
    torch.manual_seed(400)
    device = "cpu"
    if tokens_per_shard is None:
        tokens_per_shard = 2*(B*T+1) 
    data_dir = make_data_dir(n_train=n_train, tokens_per_shard=tokens_per_shard, val_same_as_train=val_same_as_train)
    config = tiny_config
    model = ModelForCausalLM(config)
    train_config = TrainConfig(
        warmup_steps=3,
        max_steps=max_steps,
        eval_interval=5,
    )
    data_config = DataConfig(
        B=B,
        T=T,
        eval_steps=2,
        global_batch_size=32,
    )
    optimizer = configure_optimizer(model, train_config, optimizer_name, zero_stage=zero_stage)
    loader = DataLoader(data_dir, "train", B, T, device)
    val_loader = DataLoader(data_dir, "val", B, T, device)
    return model, optimizer, loader, val_loader, train_config, data_config


def run_train(optimizer_name, tmp_path, tiny_config, make_data_dir, n_train=2, tokens_per_shard=None, val_same_as_train=False, max_steps=10, zero_stage=1):
    model, optimizers, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage, n_train, tokens_per_shard, val_same_as_train, max_steps)
    losses, norms = train(model, optimizers, loader, val_loader, train_config, data_config, optimizer_name, tmp_path / "ckpt")
    return losses, norms, model


@pytest.mark.parametrize("zero_stage", [1, 2])
@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_zero1_matches_plain(optimizer_name, zero_stage, tiny_config, make_data_dir):
    # zero_stage 0 和 1 各跑一遍，对比
    model_0, optimizer, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage=0)
    losses_0, norms_0 = train(model_0, optimizer, loader, val_loader, train_config, data_config, optimizer_name, None)

    model_1, optimizer, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage)
    losses_1, norms_1 = train(model_1, optimizer, loader, val_loader, train_config, data_config, optimizer_name, None)
    assert isinstance(optimizer, ZeroOptimizer)
    assert optimizer.stage == zero_stage
    
    assert losses_0 == losses_1
    assert norms_0 == norms_1

    for p_0, p_1 in zip(model_0.parameters(), model_1.parameters(), strict=True):
        assert torch.equal(p_0, p_1)
    
        
@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_zero1_resume(optimizer_name, tmp_path, tiny_config, make_data_dir):
    # 测试断点续训
    ## 完整训练
    losses_A, norms_A, model_A = run_train(optimizer_name, tmp_path, tiny_config, make_data_dir, zero_stage=1)
    ## 断点续训
    model_B, optimizers, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage=1)  # 初始化
    start_step = load_checkpoint(tmp_path / "ckpt" / "latest.pt", model_B, optimizers, optimizer_name, loader, "cpu")  # 加载 checkpoint
    assert start_step == 5  # 默认配置，checkpoint 在 step==5 保存
    losses_B, norms_B = train(model_B, optimizers, loader, val_loader, train_config, data_config, optimizer_name, tmp_path / "ckpt_resume", start_step)  # 断点续训，注意存的路径要和 A 分开
    assert losses_A[start_step:] == losses_B
    assert norms_A[start_step:] == norms_B
    for p_A, p_B in zip(model_A.parameters(), model_B.parameters(), strict=True):
        assert torch.equal(p_A, p_B)


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


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_zero1_matches_ddp(optimizer_name, tmp_path, make_data_dir):
    # 造数据，写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 跑 ddp 和 zero
    run_two(config_path, tmp_path / "ddp", 0)
    run_two(config_path, tmp_path / "zero", 1)
    # 加载两边的 final.pt
    ckpt_path1 = tmp_path / "ddp" / "final.pt"
    ckpt_path2 = tmp_path / "zero" / "final.pt"
    ckpt1 = torch.load(ckpt_path1, map_location="cpu")  # ddp
    ckpt2 = torch.load(ckpt_path2, map_location="cpu")  # zero
    assert ckpt2["optimizer"]["world_size"] == 2, "zero_stage=1 没生效"
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
    # 比较优化器总参数大小
    total = state_bytes(ckpt1["optimizer"])
    per_rank = [state_bytes(sd) for sd in ckpt2["optimizer"]["per_rank"]]
    assert sum(per_rank) == total
    assert all(num < total * 0.75 for num in per_rank)


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_zero1_resume_two(optimizer_name, tmp_path, make_data_dir):
    # 造数据，写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 一口气跑完
    run_two(config_path, tmp_path / "full", 1)
    # 把 latest.pt 复制到另一个目录
    (tmp_path / "resume").mkdir()
    shutil.copy(tmp_path / "full" / "latest.pt", tmp_path / "resume" / "latest.pt")
    # 续训
    result = run_two(config_path, tmp_path / "resume", 1, resume=True)
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


def state_bytes(sd):
    return sum(v.numel() * v.element_size() for s in sd["state"].values() for v in s.values() if torch.is_tensor(v))


