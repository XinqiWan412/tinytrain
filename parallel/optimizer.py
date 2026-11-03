from parallel.dist_utils import get_dp_group, get_embed_group, get_layout, get_pp_group, get_rank, get_tp_group, get_world_size
import torch.distributed as dist
import torch


class ParallelOptimizer:
    def __init__(self, param_groups, optimizer_cls, **defaults):
        self.optimizer = optimizer_cls(param_groups, **defaults)
        self.params = [p for g in self.optimizer.param_groups for p in g["params"]]
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.layout = get_layout()
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
    
    def reduce_grads(self):
        for p in self.params:
            if p.grad is None:
                continue
            if getattr(p, "tp_sum_grad", False):
                dist.all_reduce(p.grad, group=get_tp_group())
            if getattr(p, "pp_shared", False):
                dist.all_reduce(p.grad, group=get_embed_group())
            if self.layout.dp_size > 1:
                dist.all_reduce(p.grad, group=get_dp_group())
                p.grad.div_(self.layout.dp_size)
    
    def clip_grad_norm_(self, parameters, max_norm):
        params = [p for p in parameters if p.grad is not None]
        device = params[0].grad.device
        sq = torch.zeros((), device=device)
        for p in params:
            if getattr(p, "pp_shared", False) and self.layout.pp_rank != 0:
                # 共享参数且不在第一段，跳过
                continue
            if getattr(p, "tp_dim", None) is not None:
                # 如果是 tp 切开的，每张卡都加
                sq += p.grad.detach().float().pow(2).sum()
            else:
                # 如果 tp 没切，只在 0 上加
                if self.layout.tp_rank == 0:
                    sq += p.grad.detach().float().pow(2).sum()
        if self.layout.tp_size > 1:
            dist.all_reduce(sq, group=get_tp_group())
        if self.layout.pp_size > 1:
            dist.all_reduce(sq, group=get_pp_group())
        total = sq.sqrt()
        torch.nn.utils.clip_grads_with_norm_(params, max_norm, total)
        return total

    def state_dict(self):
        local = self.optimizer.state_dict()
        cpu_state = {i: {k: v.cpu() if torch.is_tensor(v) else v for k, v in s.items()}
                    for i, s in local["state"].items()}
        local = {"state": cpu_state, "param_groups": local["param_groups"]}

        if self.world_size == 1:
            per_rank = [local]
        else:
            # 0 号进程建空数组 per_rank，其他进程设 None
            if self.rank == 0:
                per_rank = [None] * self.world_size
            else:
                per_rank = None
            dist.gather_object(local, per_rank, dst=0)
        return {
            "dp_size": self.layout.dp_size, 
            "tp_size": self.layout.tp_size,
            "pp_size": self.layout.pp_size, 
            "per_rank": per_rank
        }
    
    def load_state_dict(self, state):
        if "per_rank" not in state:
            raise ValueError("state 里没有 per_rank，该 checkpoint 不是 ParallelOptimizer 格式")
        if state["dp_size"] != self.layout.dp_size:
            raise ValueError(f"checkpoint 要求 dp_size={state['dp_size']}, 当前 dp_size={self.layout.dp_size}")
        if state["tp_size"] != self.layout.tp_size:
            raise ValueError(f"checkpoint 要求 tp_size={state['tp_size']}, 当前 tp_size={self.layout.tp_size}")
        if state["pp_size"] != self.layout.pp_size:
            raise ValueError(f"checkpoint 要求 pp_size={state['pp_size']}, 当前 pp_size={self.layout.pp_size}")
        # 加载本卡的那一项 state_dict
        self.optimizer.load_state_dict(state["per_rank"][self.rank])

