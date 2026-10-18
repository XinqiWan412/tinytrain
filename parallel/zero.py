import torch
import torch.distributed as dist
from parallel.dist_utils import get_rank, get_world_size


def partition_params(sizes: list[int], world_size: int):
    # 把 param 尽可能均匀地分给各个 rank
    owners = [None] * len(sizes)
    loads = [0] * world_size
    # 按 size 大小排序，例如 sizes=[3, 1, 2]，order=[0, 2, 1]
    order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
    for i in order:
        r = loads.index(min(loads))  # 找出当前负载最轻的 rank r
        owners[i] = r  # 这个权重分配给 r
        loads[r] += sizes[i]
    return owners, loads


class ZeroOptimizer:
    def __init__(self, param_groups, optimizer_cls, **defaults):
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.params = [p for g in param_groups for p in g["params"]]
        self.owners, self.numel_per_rank = partition_params([p.numel() for p in self.params], self.world_size)
        owner_of = dict(zip(self.params, self.owners))
        local_groups = []
        for g in param_groups:
            local_groups.append({**g, "params": [p for p in g["params"] if owner_of[p] == self.rank]})
        self.optimizer = optimizer_cls(local_groups, **defaults)

    @property
    def param_groups(self):
        return self.optimizer.param_groups
        
    def zero_grad(self):
        for p in self.params:
            p.grad = None
    
    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        if self.world_size > 1:
            for p, owner in zip(self.params, self.owners):
                dist.broadcast(p, src=owner)

    def state_dict(self):
        # 复制到 cpu 上
        local = self.optimizer.state_dict()
        cpu_state = {i: {k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()}
                    for i, s in local["state"].items()}
        local = {"state": cpu_state, "param_groups": local["param_groups"]}

        # 收集，返回
        if self.world_size == 1:
            per_rank = [local]
        else:
            # 0 号进程建空数组 per_rank，其他进程设 None
            if self.rank == 0:
                per_rank = [None] * self.world_size
            else:
                per_rank = None
            dist.gather_object(local, per_rank, dst=0)
        return {"world_size": self.world_size, "per_rank": per_rank}
    
    def load_state_dict(self, state):
        if "per_rank" not in state:
            raise ValueError("state 里没有 per_rank，该 checkpoint 不是 ZeRO-1 格式，可能是 zero_stage=0 时存的")
        if state["world_size"] != self.world_size:
            raise ValueError(f"checkpoint 要求 world_size={state["world_size"]}, 当前 world_size={self.world_size}")
        # 加载本卡的那一项 state_dict
        self.optimizer.load_state_dict(state["per_rank"][self.rank])

        
