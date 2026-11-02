from parallel.dist_utils import tp_group_ranks, dp_group_ranks, pp_group_ranks, embed_group_ranks


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

