import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from parallel.dist_utils import get_layout, get_tp_group


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

