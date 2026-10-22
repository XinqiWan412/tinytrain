from dataclasses import dataclass
import os, torch
import torch.distributed as dist

_layout = None
_tp_group = None
_pp_group = None
_embed_group = None


def setup():
    if "LOCAL_RANK" not in os.environ:
        # 环境变量里没有 LOCAL_RANK
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    else:
        # 环境变量里有 LOCAL_RANK
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
            backend = "nccl"
        else:
            device = "cpu"
            backend = "gloo"
        dist.init_process_group(backend=backend)
        return device


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def is_main():
    return get_rank() == 0


def all_reduce_mean(t):
    if get_world_size() == 1:
        return t
    else:
        t = t.detach().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t / get_world_size()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def broadcast_params(model, src=0):
    with torch.no_grad():
        for p in model.parameters():
            dist.broadcast(p, src=src)


def all_gather_list(obj):
    world_size = get_world_size()
    if world_size == 1:
        # 单进程
        return [obj]
    else:
        # 多进程
        out = [None] * world_size
        dist.all_gather_object(out, obj)
        return out


@dataclass(frozen=True)
class ParallelLayout:
    tp_rank: int
    tp_size: int
    dp_rank: int
    dp_size: int
    pp_rank: int = 0
    pp_size: int = 1


def compute_layout(rank, world_size, tp_size, pp_size=1):
    if world_size % (tp_size * pp_size) != 0:
        raise ValueError(f"world_size={world_size} 不能被 tp_size={tp_size} 整除")
    dp_size = world_size // (tp_size * pp_size)
    return ParallelLayout(
        tp_rank=rank % tp_size,
        tp_size=tp_size,
        dp_rank=(rank // tp_size) % dp_size,
        dp_size=dp_size,
        pp_rank=rank // (tp_size * dp_size),
        pp_size=pp_size,
    )


def init_parallel(tp_size, pp_size=1):
    global _layout, _tp_group, _pp_group, _embed_group
    rank = get_rank()
    world_size = get_world_size()
    _layout = compute_layout(rank, world_size, tp_size, pp_size)
    if tp_size > 1:
        for g in range(world_size // tp_size):
            ranks = list(range(g * tp_size, (g + 1) * tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                _tp_group = group
    if pp_size > 1:
        for ranks in pp_group_ranks(world_size, tp_size, pp_size):
            group = dist.new_group(ranks)
            if rank in ranks:
                _pp_group = group
        for ranks in embed_group_ranks(world_size, tp_size, pp_size):
            group = dist.new_group(ranks)
            if rank in ranks:
                _embed_group = group
    return _layout


def get_layout():
    if _layout is not None:
        return _layout
    else:
        return compute_layout(get_rank(), get_world_size(), 1)


def get_tp_group():
    if _tp_group is None:
        raise RuntimeError("先调用 init_parallel，且 tp_size > 1")
    return _tp_group


def pp_group_ranks(world_size, tp_size, pp_size):
    """8 卡时会返回 [[0, 4], [1, 5], [2, 6], [3, 7]]"""
    groups = {}
    for r in range(world_size):
        layout = compute_layout(r, world_size, tp_size, pp_size)
        groups.setdefault((layout.dp_rank, layout.tp_rank), []).append(r)
    return list(groups.values())


def embed_group_ranks(world_size, tp_size, pp_size):
    # 每一个 PP 组，都取首尾两个
    return [[g[0], g[-1]] for g in pp_group_ranks(world_size, tp_size, pp_size)]


def get_pp_group():
    if _pp_group is None:
        raise RuntimeError("先调用 init_parallel，且 pp_size > 1")
    return _pp_group


def get_embed_group():
    if _embed_group is None:
        raise RuntimeError("先调用 init_parallel，且 pp_size > 1")
    return _embed_group

