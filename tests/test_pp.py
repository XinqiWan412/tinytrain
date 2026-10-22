from parallel.dist_utils import compute_layout, ParallelLayout, embed_group_ranks, pp_group_ranks, init_parallel
import pytest


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

