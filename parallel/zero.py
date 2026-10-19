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
    def __init__(self, param_groups, optimizer_cls, stage=1, **defaults):
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.stage = stage
        if stage not in (1, 2): raise ValueError(f"stage is {stage}, expected: 1, 2")
        self.params = [p for g in param_groups for p in g["params"]]
        self.owners, self.numel_per_rank = partition_params([p.numel() for p in self.params], self.world_size)
        self.owner_of = dict(zip(self.params, self.owners))
        local_groups = []
        for g in param_groups:
            local_groups.append({**g, "params": [p for p in g["params"] if self.owner_of[p] == self.rank]})
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
            raise ValueError(f"checkpoint 要求 world_size={state['world_size']}, 当前 world_size={self.world_size}")
        # 加载本卡的那一项 state_dict
        self.optimizer.load_state_dict(state["per_rank"][self.rank])

    def reduce_grads(self):
        if self.stage == 1 or self.world_size == 1:
            # ZeRO-1 和单进程不需要做梯度规约
            return
        for p, owner in zip(self.params, self.owners):
            dist.reduce(p.grad, dst=owner)
            if self.rank == owner:
                # 如果是本卡的，就计算平均
                p.grad.div_(self.world_size)
            else:
                p.grad = None

    @torch.no_grad()
    def clip_grad_norm_(self, parameters, max_norm):
        if self.stage == 1:
            return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
        params = list(parameters)
        # 找出本卡对应的 p 和 g
        idx = [i for i, p in enumerate(params) if self.owner_of[p] == self.rank]
        grads = [params[i].grad for i in idx]
        # 算本卡每个梯度的范数
        own_norms = torch._foreach_norm(grads, 2.0)
        norms = torch.zeros(len(params), dtype=grads[0].dtype, device=grads[0].device)
        norms[idx] = torch.stack(own_norms)
        # 规约得到所有 g 的范数
        if self.world_size > 1:
            dist.all_reduce(norms)
        # 求全局范数 total
        total = torch.linalg.vector_norm(norms, 2.0)
        # 梯度裁剪
        torch.nn.utils.clip_grads_with_norm_([params[i] for i in idx], max_norm, total)
        return total

