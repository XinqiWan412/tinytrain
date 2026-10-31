import copy
import dataclasses

import torch
from parallel.dist_utils import compute_layout, ParallelLayout, embed_group_ranks, pp_group_ranks, init_parallel
import pytest
from model import ModelForCausalLM
from parallel.pp import forward_first, forward_last, split_model


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
            assert torch.equal(p.grad, g[name])

