import copy
import dataclasses
import shutil
import sys

import torch
from parallel.dist_utils import compute_layout, ParallelLayout, embed_group_ranks, pp_group_ranks, init_parallel
import pytest, os, subprocess
from model import ModelForCausalLM, load_model
from parallel.pp import forward_first, forward_last, split_model
from pathlib import Path


def test_compute_layout_pp():
    # 2 卡，PP = 2
    assert compute_layout(0, 2, 1, 2) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=0, dp_size=1, pp_rank=0, pp_size=2)
    assert compute_layout(1, 2, 1, 2) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=0, dp_size=1, pp_rank=1, pp_size=2)
    # 4 卡，PP = 2, DP = 2
    assert compute_layout(0, 4, 1, 2) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2)
    assert compute_layout(1, 4, 1, 2) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=1, dp_size=2, pp_rank=0, pp_size=2)
    assert compute_layout(2, 4, 1, 2) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=0, dp_size=2, pp_rank=1, pp_size=2)
    assert compute_layout(3, 4, 1, 2) == ParallelLayout(tp_rank=0, tp_size=1, dp_rank=1, dp_size=2, pp_rank=1, pp_size=2)
    # 8 卡，TP2*DP2*PP2
    assert compute_layout(0, 8, 2, 2) == ParallelLayout(tp_rank=0, tp_size=2, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2)
    assert compute_layout(1, 8, 2, 2) == ParallelLayout(tp_rank=1, tp_size=2, dp_rank=0, dp_size=2, pp_rank=0, pp_size=2)
    assert compute_layout(2, 8, 2, 2) == ParallelLayout(tp_rank=0, tp_size=2, dp_rank=1, dp_size=2, pp_rank=0, pp_size=2)
    assert compute_layout(3, 8, 2, 2) == ParallelLayout(tp_rank=1, tp_size=2, dp_rank=1, dp_size=2, pp_rank=0, pp_size=2)
    assert compute_layout(4, 8, 2, 2) == ParallelLayout(tp_rank=0, tp_size=2, dp_rank=0, dp_size=2, pp_rank=1, pp_size=2)
    assert compute_layout(5, 8, 2, 2) == ParallelLayout(tp_rank=1, tp_size=2, dp_rank=0, dp_size=2, pp_rank=1, pp_size=2)
    assert compute_layout(6, 8, 2, 2) == ParallelLayout(tp_rank=0, tp_size=2, dp_rank=1, dp_size=2, pp_rank=1, pp_size=2)
    assert compute_layout(7, 8, 2, 2) == ParallelLayout(tp_rank=1, tp_size=2, dp_rank=1, dp_size=2, pp_rank=1, pp_size=2)
    # 6 卡，TP2*PP2
    with pytest.raises(ValueError):
        compute_layout(0, 6, 2, 2)  # 6 卡，TP2*PP2


def test_pp_group_ranks():
    # 2 卡 PP2
    assert pp_group_ranks(2, 1, 2) == [[0, 1]]
    assert embed_group_ranks(2, 1, 2) == [[0, 1]]
    # 4 卡 PP2*DP2
    assert pp_group_ranks(4, 1, 2) == [[0, 2], [1, 3]]
    assert embed_group_ranks(4, 1, 2) == [[0, 2], [1, 3]]
    # 8 卡 TP2*DP2*PP2
    assert pp_group_ranks(8, 2, 2) == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert embed_group_ranks(8, 2, 2) == [[0, 4], [1, 5], [2, 6], [3, 7]]
    # 4 卡 PP4
    assert pp_group_ranks(4, 1, 4) == [[0, 1, 2, 3]]
    assert embed_group_ranks(4, 1, 4) == [[0, 3]]


def test_split_model(tiny_config):
    config = dataclasses.replace(tiny_config, num_hidden_layers=4)
    k = 3
    full = ModelForCausalLM(config)
    s0 = split_model(copy.deepcopy(full), 0, k)
    s1 = split_model(copy.deepcopy(full), 1, k)

    # 两段 state_dict() 的键不重叠
    keys0, keys1 = set(s0.state_dict()), set(s1.state_dict())
    assert not (keys0 & keys1)
    assert keys0 | keys1 == set(full.state_dict())

    # 每个键的张量和完整模型的张量相同
    sd_full = full.state_dict()
    for s in (s0, s1):
        for name, t in s.state_dict().items():
            assert torch.equal(t, sd_full[name]), name
    
    assert list(s0.pp_layers) == [0, 1, 2]
    assert list(s1.pp_layers) == [3]
    assert "model.layers.3.mlp.down_proj.weight" in keys1

    # 共享参数记号
    p0 = dict(s0.named_parameters())["model.embed_tokens.weight"]
    p1 = dict(s1.named_parameters())["lm_head.weight"]
    assert getattr(p0, "pp_shared", False)
    assert getattr(p1, "pp_shared", False)


def test_split_model_errors(tiny_config):
    config = dataclasses.replace(tiny_config, num_hidden_layers=4)
    full = ModelForCausalLM(config)

    # k = 0 时平分
    k = 0
    s0 = split_model(copy.deepcopy(full), 0, k)
    assert list(s0.pp_layers) == [0, 1]

    # k = 4 报错
    with pytest.raises(ValueError):
        split_model(copy.deepcopy(full), 0, 4)
    
    # stage = 2 报错
    with pytest.raises(ValueError):
        split_model(copy.deepcopy(full), 2, 2)


def test_stage_forward_matches_full(tiny_config):
    # pp 对比 完整模型，前向
    config = dataclasses.replace(tiny_config, num_hidden_layers=4)
    full = ModelForCausalLM(config)
    k = 3
    s0 = split_model(copy.deepcopy(full), 0, k)
    s1 = split_model(copy.deepcopy(full), 1, k)

    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    # full 的 loss
    _, loss_full = full(x, y)

    # pp 的 loss
    h = forward_first(s0, x)
    assert h.shape == (2, 8, config.hidden_size)
    loss = forward_last(s1, h, y)

    assert torch.equal(loss, loss_full), f"pp 切分前后 loss 值不同"


def test_stage_backward_matches_full(tiny_config):
    # pp 对比 完整模型，反向
    config = dataclasses.replace(tiny_config, num_hidden_layers=4)
    full = ModelForCausalLM(config)
    k = 3
    s0 = split_model(copy.deepcopy(full), 0, k)
    s1 = split_model(copy.deepcopy(full), 1, k)

    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    # full
    _, loss_full = full(x, y)
    loss_full.backward()

    # pp
    h = forward_first(s0, x)
    h_recv = h.detach().requires_grad_()
    loss = forward_last(s1, h_recv, y)
    loss.backward()
    h.backward(h_recv.grad)

    assert torch.equal(loss, loss_full)

    # 把 pp 两段梯度合并进一个字典
    g = {n: p.grad for n, p in s0.named_parameters()}
    g.update({n: p.grad for n, p in s1.named_parameters()})

    # 逐个比较梯度
    for name, p in full.named_parameters():
        if name == "model.embed_tokens.weight":
            assert torch.equal(p.grad, g["model.embed_tokens.weight"] + g["lm_head.weight"])
        else:
            assert torch.equal(p.grad, g[name]), name


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


def test_gpipe_matches_full():
    path = Path(__file__).parent / "pp_worker.py"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
    "--nproc_per_node=2", str(path)]
    result = run_subprocess(cmd)
    assert "rank 0 OK" in result.stdout
    assert "rank 1 OK" in result.stdout


def write_config(tmp_path, data_path, optimizer_name):
    toml = f"""[model]
vocab_size = 256
hidden_size = 64
num_hidden_layers = 4
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


def run_pp(config_path, out_dir, nproc, pp_size, resume=False):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={nproc}", "pretrain.py", "--config", config_path, "--set", f"parallel.pp_size={pp_size}",
        "--set", f"run.ckpt_dir={out_dir}", "--set", f"run.log={out_dir / 'pp.log'}"]
    if resume:
        cmd.append("--resume")
    return run_subprocess(cmd)


@pytest.mark.parametrize("optimizer_name, atol", [("adamw", 1e-5), ("muon", 1e-4)])
def test_pp_matches_single(tmp_path, make_data_dir, optimizer_name, atol):
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # pp_size = 1
    run_pp(config_path, tmp_path / "single", 1, 1)
    # pp_size = 2
    result = run_pp(config_path, tmp_path / "pp", 2, 2)
    # 加载两边的 final.pt
    ckpt_path1 = tmp_path / "single" / "final.pt"
    ckpt_path2 = tmp_path / "pp" / "final.pt"
    ckpt1 = torch.load(ckpt_path1, map_location="cpu")
    ckpt2 = torch.load(ckpt_path2, map_location="cpu")
    # 检查 pp_size = 2 生效了
    assert "模型已切分" in result.stdout
    assert ckpt2["config"]["parallel"]["pp_size"] == 2
    # 比较 val_loss
    assert ckpt2["val_loss"] == pytest.approx(ckpt1["val_loss"], rel=atol)
    # 比较参数
    sd1 = ckpt1["model"]
    sd2 = ckpt2["model"]
    assert set(sd1) == set(sd2)  # 比较 key
    compare_state_dicts(sd1, sd2, atol=atol)
    # 检查 load_model
    model = load_model(ckpt_path2, "cpu")
    for name, p in model.named_parameters():
        assert torch.equal(p, ckpt2["model"][name])


def test_pp_resume(tmp_path, make_data_dir):
    optimizer_name = "adamw"
    nproc = 2
    pp_size = 2
    # 造数据，写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 一口气跑完
    run_pp(config_path, tmp_path / "full", nproc, pp_size)
    # 把 latest.pt 复制到另一个目录
    (tmp_path / "resume").mkdir()
    shutil.copy(tmp_path / "full" / "latest.pt", tmp_path / "resume" / "latest.pt")
    # 续训
    result = run_pp(config_path, tmp_path / "resume", nproc, pp_size, resume=True)
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

