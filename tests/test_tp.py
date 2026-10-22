from parallel.dist_utils import ParallelLayout, compute_layout
import pytest, os, subprocess, sys
from pathlib import Path


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

