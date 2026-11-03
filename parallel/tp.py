import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from model import ModelForCausalLM
from parallel.dist_utils import get_layout, get_rank, get_tp_group, get_world_size
from parallel.pp import is_pp_model


class CopyToTP(torch.autograd.Function):
    """f: 前向原样传，反向把梯度在 TP 组内加起来"""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        dist.all_reduce(grad, group=get_tp_group())
        return grad


class ReduceFromTP(torch.autograd.Function):
    """g: 前向在组内加起来，反向原样传"""
    @staticmethod
    def forward(ctx, x):
        out = x.clone()
        dist.all_reduce(out, group=get_tp_group())
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad


copy_to_tp = CopyToTP.apply
reduce_from_tp = ReduceFromTP.apply


class ColumnParallelLinear(nn.Module):
    def __init__(self, full):
        super().__init__()
        layout = get_layout()
        tp_rank, tp_size = layout.tp_rank, layout.tp_size
        assert full.bias is None, "TP 线性层不支持 bias"
        assert full.weight.shape[0] % tp_size == 0
        self.weight = nn.Parameter(torch.chunk(full.weight.detach(), tp_size, dim=0)[tp_rank].clone())
        self.weight.tp_dim = 0
    
    def forward(self, x):
        return F.linear(x, self.weight)


class RowParallelLinear(nn.Module):
    def __init__(self, full):
        super().__init__()
        layout = get_layout()
        tp_rank, tp_size = layout.tp_rank, layout.tp_size
        assert full.bias is None, "TP 线性层不支持 bias"
        assert full.weight.shape[1] % tp_size == 0
        self.weight = nn.Parameter(torch.chunk(full.weight.detach(), tp_size, dim=1)[tp_rank].clone())
        self.weight.tp_dim = 1
    
    def forward(self, x):
        return reduce_from_tp(F.linear(x, self.weight))


class CopyToTPModule(nn.Module):
    """把 f 包成模块，用来替换 Attention、MLP 里的 tp_copy"""
    def forward(self, x):
        return copy_to_tp(x)


def apply_tp(model: ModelForCausalLM):
    """把 model 包成 TP"""
    tp_size = get_layout().tp_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    if num_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
        raise ValueError(f"tp_size={tp_size} 不能整除头数（Q 头 {num_heads}, KV 头 {num_kv_heads}）；KV 头数为 2 时 TP 最多切 2 份")
    for block in model.model.layers:
        if isinstance(block, nn.Identity):
            continue
        block.self_attn.q_proj = ColumnParallelLinear(block.self_attn.q_proj)
        block.self_attn.k_proj = ColumnParallelLinear(block.self_attn.k_proj)
        block.self_attn.v_proj = ColumnParallelLinear(block.self_attn.v_proj)
        block.self_attn.o_proj = RowParallelLinear(block.self_attn.o_proj)
        block.self_attn.tp_copy = CopyToTPModule()
        block.mlp.gate_proj = ColumnParallelLinear(block.mlp.gate_proj)
        block.mlp.up_proj = ColumnParallelLinear(block.mlp.up_proj)
        block.mlp.down_proj = RowParallelLinear(block.mlp.down_proj)
        block.mlp.tp_copy = CopyToTPModule()
        block.self_attn.q_norm.weight.tp_sum_grad = True
        block.self_attn.k_norm.weight.tp_sum_grad = True
    return model


def gather_full_state_dict(model):
    """所有进程都要调用；0 号进程返回完整的 state_dict（和单卡格式相同），其他进程返回 None"""
    layout = get_layout()
    world_size, rank = get_world_size(), get_rank()
    local = {name: (t.detach().cpu(), getattr(t, "tp_dim", None))
            for name, t in model.state_dict(keep_vars=True).items()}
    payload = (layout.tp_rank, local)
    if world_size == 1:
        gathered = [payload]
    else:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)
    if rank != 0:
        return None
    # 按名字分组
    pieces, tp_dims = {}, {}
    for tp_rank, sd in gathered:
        for name, (t, tp_dim) in sd.items():
            pieces.setdefault(name, {})[tp_rank] = t  # t 是这个 name 对应的张量
            tp_dims[name] = tp_dim
    full = {}
    for name, by_rank in pieces.items():
        if tp_dims[name] is None:
            full[name] = next(iter(by_rank.values()))
        else:
            full[name] = torch.cat([by_rank[r] for r in range(layout.tp_size)], dim=tp_dims[name])
    return full


def load_full_state_dict(model, full_sd):
    if set(model.state_dict()) != set(full_sd):
        missing = set(model.state_dict()) - set(full_sd)
        extras = set(full_sd) - set(model.state_dict())
        if missing:
            raise ValueError(f"gather 出来的 state_dict 和 model 的键不一致: 缺少 {missing}")
        if extras and not is_pp_model(model):
            raise ValueError(f"gather 出来的 state_dict 和 model 的键不一致: 多余 {extras}")
    layout = get_layout()
    tp_size, tp_rank = layout.tp_size, layout.tp_rank
    with torch.no_grad():
        for name, t in model.state_dict(keep_vars=True).items():
            tp_dim = getattr(t, "tp_dim", None)
            src = full_sd[name]
            if tp_dim is not None:
                src = torch.chunk(src, tp_size, dim=tp_dim)[tp_rank]
            t.copy_(src)


def is_tp_model(model):
    """判断是不是 tp model, 返回 bool"""
    return any(getattr(p, "tp_dim", None) is not None for p in model.parameters())

