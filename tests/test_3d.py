from parallel.dist_utils import tp_group_ranks, dp_group_ranks, pp_group_ranks, embed_group_ranks
import subprocess, os, sys, pytest, shutil
from pathlib import Path
import torch


def test_group_ranks_3d ():
    tp = [[0, 1], [2, 3], [4, 5], [6, 7]]
    dp = [[0, 2], [1, 3], [4, 6], [5, 7]]
    pp = [[0, 4], [1, 5], [2, 6], [3, 7]]
    embed = [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert tp_group_ranks(8, 2, 2) == tp
    assert dp_group_ranks(8, 2, 2) == dp
    assert pp_group_ranks(8, 2, 2) == pp
    assert embed_group_ranks(8, 2, 2) == embed

    for w, t, p in [(8, 2, 2), (4, 2, 1), (4, 1, 2)]:
        tp_group = tp_group_ranks(w, t, p)
        dp_group = dp_group_ranks(w, t, p)
        pp_group = pp_group_ranks(w, t, p)
        # 每种组都把每张卡覆盖恰好一次
        assert sorted(r for g in tp_group for r in g) == list(range(w))
        assert sorted(r for g in dp_group for r in g) == list(range(w))
        assert sorted(r for g in pp_group for r in g) == list(range(w))
        # 对每张卡 r，它在任意两种组里的组交集应只有 {r}
        for r in range(w):
            tp_g = next(g for g in tp_group if r in g)
            dp_g = next(g for g in dp_group if r in g)
            pp_g = next(g for g in pp_group if r in g)
            assert set(tp_g) & set(dp_g) == {r}
            assert set(dp_g) & set(pp_g) == {r}
            assert set(tp_g) & set(pp_g) == {r}


def run_subprocess(cmd):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


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


def run_parallel(config_path, out_dir, tp_size, pp_size, nproc, resume=False):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc_per_node={nproc}", "pretrain.py", "--config", config_path, "--set", f"parallel.tp_size={tp_size}",
        "--set", f"run.ckpt_dir={out_dir}", "--set", f"run.log={out_dir / '3d.log'}", "--set", f"parallel.pp_size={pp_size}"]
    if resume:
        cmd.append("--resume")
    return run_subprocess(cmd)


@pytest.mark.parametrize("dp, tp, pp", [
    pytest.param(2, 2, 1, id="dp2_tp2", marks=pytest.mark.slow),
    pytest.param(2, 1, 2, id="dp2_pp2", marks=pytest.mark.slow),
    pytest.param(1, 2, 2, id="tp2_pp2", marks=pytest.mark.slow),
    pytest.param(2, 2, 2, id="dp2_tp2_pp2"),
])
def test_3d_matches_single(tmp_path, make_data_dir, dp, tp, pp):
    optimizer_name = "adamw"
    # 1. 造数据、写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 2. 对照组：单进程，也用 torchrun 起，保证两边 OMP_NUM_THREADS=1
    run_parallel(config_path, tmp_path / "single", 1, 1, nproc=1)
    # 3. 组合组：nproc = dp * tp * pp
    run_parallel(config_path, tmp_path / "combo", tp, pp, nproc=dp * tp * pp)
    # 4. 读两边的 final.pt
    ckpt1 = torch.load(tmp_path / "single" / "final.pt", map_location="cpu")
    ckpt2 = torch.load(tmp_path / "combo" / "final.pt", map_location="cpu")
    # 5. 检查组合真的生效了
    assert ckpt2["config"]["parallel"]["tp_size"] == tp
    assert ckpt2["config"]["parallel"]["pp_size"] == pp
    assert ckpt2["optimizer"]["dp_size"] == dp
    # 6. 比较 val_loss（容差同 TP 那一轮）
    assert ckpt2["val_loss"] == pytest.approx(ckpt1["val_loss"], rel=1e-5)
    # 7. 比较参数
    sd1, sd2 = ckpt1["model"], ckpt2["model"]
    assert set(sd1) == set(sd2)
    compare_state_dicts(sd1, sd2, atol=1e-4)


@pytest.mark.slow
def test_3d_resume(tmp_path, make_data_dir):
    optimizer_name = "adamw"
    tp = 2
    pp = 2
    nproc = 8
    # 造数据，写配置
    data_dir = make_data_dir(n_train=2, tokens_per_shard=129)
    config_path = write_config(tmp_path, data_dir, optimizer_name)
    # 一口气跑完
    run_parallel(config_path, tmp_path / "full", tp, pp, nproc)
    # 把 latest.pt 复制到另一个目录
    (tmp_path / "resume").mkdir()
    shutil.copy(tmp_path / "full" / "latest.pt", tmp_path / "resume" / "latest.pt")
    # 续训
    result = run_parallel(config_path, tmp_path / "resume", tp, pp, nproc, resume=True)
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

