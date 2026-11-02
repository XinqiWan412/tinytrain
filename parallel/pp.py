import torch
from model import ModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from parallel.dist_utils import get_embed_group, get_layout, get_pp_group, get_rank, get_world_size


def split_model(model: ModelForCausalLM, stage, first_stage_layers):
    n = model.config.num_hidden_layers
    k = first_stage_layers or n // 2  # 左边为 0 时，取右边，就是平分
    if not 1 <= k <= n - 1:
        raise ValueError(f"当前模型, n={n}, k={k}, 每段至少要有一层")
    if stage not in (0, 1):
        raise ValueError(f"stage 必须为 0 或 1，当前值：{stage}")

    # 遍历所有层，不属于本段的就换成 nn.Identity()
    keep = range(0, k) if stage == 0 else range(k, n)  # 本段要保留的层号
    for i in range(n):
        if i not in keep:
            model.model.layers[i] = nn.Identity()
    
    if stage == 0:
        # 第一段不要 model.norm 和 lm_head
        model.model.norm = nn.Identity()
        model.lm_head = nn.Identity()
    else:
        # 第二段不要 embed_tokens
        model.model.embed_tokens = nn.Identity()
    
    # 在模型和共享参数上做记号
    model.pp_stage = stage
    model.pp_layers = keep
    
    if model.config.tie_word_embeddings:
        if stage == 0:
            model.model.embed_tokens.weight.pp_shared = True
        else:
            model.lm_head.weight.pp_shared = True
        
    return model


def forward_first(model, input_ids):
    _, T = input_ids.shape
    x = model.model.embed_tokens(input_ids)
    cos, sin = model.model.cos[:T], model.model.sin[:T]
    for i in model.pp_layers:
        x = model.model.layers[i](x, cos, sin)
    return x


def forward_last(model, x, targets):
    B, T, _ = x.shape
    cos, sin = model.model.cos[:T], model.model.sin[:T]
    for i in model.pp_layers:
        x = model.model.layers[i](x, cos, sin)
    x = model.model.norm(x)
    logits = model.lm_head(x)
    V = logits.shape[-1]
    loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
    return loss


def _act_buf(model, y):
    B, T = y.shape
    D = model.config.hidden_size
    buf = torch.empty((B, T, D), dtype=torch.float32, device=y.device)
    return buf


def gpipe_step(model, parts):
    m = len(parts)
    peer = dist.get_global_rank(get_pp_group(), 1 - model.pp_stage)
    if model.pp_stage == 0:
        hs = []
        # 前向传激活
        for x, _ in parts:
            # ① autocast 里算 h
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = forward_first(model, x)
            # ② 把 h 发给对方
            dist.send(h.detach(), dst=peer)
            # ③ 把 h 存进 hs
            hs.append(h)
        # 反向收梯度
        for h in hs:
            # ① 准备和 h 一样的空箱子
            buf = torch.empty_like(h)
            # ② 把梯度收进箱子
            dist.recv(buf, src=peer)
            # ③ 从 h 接着反向
            h.backward(buf)
        return None
    else:
        saved = []
        total = 0.0
        # 前向收激活
        for _, y in parts:
            # ① 准备 (B, T, D) 的 fp32 空箱子
            buf = _act_buf(model, y)
            # ② 把激活收进箱子
            dist.recv(buf, src=peer)
            # ③ 箱子 requires_grad_()
            buf.requires_grad_()
            # ④ autocast 里算 loss，除以 m
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = forward_last(model, buf, y) / m
            # ⑤ total 累加 loss.detach()
            total += loss.detach()
            # ⑥ 把 (箱子, loss) 存进 saved
            saved.append((buf, loss))
        # 反向传梯度
        for buf, loss in saved:
            # ① loss.backward()
            loss.backward()
            # ② 把 buf.grad 发给对方
            dist.send(buf.grad, dst=peer)
        return total


def is_pp_model(model):
    return hasattr(model, "pp_stage")


def broadcast_from_last(model, t):
    # 第一段用箱子接收，第二段直接发 t
    if model.pp_stage == 0:
        t = torch.zeros((), device=next(model.parameters()).device)
    dist.broadcast(t, src=dist.get_global_rank(get_pp_group(), 1), group=get_pp_group())
    return t


def pp_forward(model, x, y):
    # 给 eval 用的前向
    peer = dist.get_global_rank(get_pp_group(), 1 - model.pp_stage)
    if model.pp_stage == 0:
        h = forward_first(model, x)
        dist.send(h, peer)
        return None
    else:
        B, T = y.shape
        buf = torch.empty((B, T, model.config.hidden_size), dtype=torch.float32, device=y.device)
        dist.recv(buf, peer)
        loss = forward_last(model, buf, y)
        return loss


class PPOptimizer:
    def __init__(self, param_groups, optimizer_cls, **defaults):
        self.optimizer = optimizer_cls(param_groups, **defaults)
        self.params = [p for g in self.optimizer.param_groups for p in g["params"]]
        self.world_size = get_world_size()
        self.rank = get_rank()
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        self.optimizer.step()
    
    def reduce_grads(self):
        for p in self.params:
            if getattr(p, "pp_shared", False) and p.grad is not None:
                dist.all_reduce(p.grad, group=get_embed_group())
    
    def clip_grad_norm_(self, parameters, max_norm):
        params = [p for p in parameters if p.grad is not None]
        device = params[0].grad.device
        sq = torch.zeros((), device=device)
        for p in params:
            if getattr(p, "pp_shared", False) and get_layout().pp_rank != 0:
                # 共享参数只在第一段加
                continue
            sq += p.grad.detach().float().pow(2).sum()
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
        return {"world_size": self.world_size, "per_rank": per_rank}
    
    def load_state_dict(self, state):
        if "per_rank" not in state:
            raise ValueError("state 里没有 per_rank，该 checkpoint 不是 PP 格式")
        if state["world_size"] != self.world_size:
            raise ValueError(f"checkpoint 要求 world_size={state['world_size']}, 当前 world_size={self.world_size}")
        # 加载本卡的那一项 state_dict
        self.optimizer.load_state_dict(state["per_rank"][self.rank])


def _send_recv(t, buf, peer):
    ops = [dist.P2POp(dist.isend, t, peer), dist.P2POp(dist.irecv, buf, peer)]
    for req in dist.batch_isend_irecv(ops):
        req.wait()


def one_f_one_b_step(model, parts):
    m = len(parts)
    peer = dist.get_global_rank(get_pp_group(), 1 - model.pp_stage)
    if model.pp_stage == 0:
        # 灌满
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = forward_first(model, parts[0][0])
        dist.send(h.detach(), dst=peer)
        # 交替
        for i in range(1, m):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h_next = forward_first(model, parts[i][0])
            g = torch.empty_like(h)
            _send_recv(h_next.detach(), g, peer)
            h.backward(g)
            h = h_next
        # 排空
        g = torch.empty_like(h)
        dist.recv(g, src=peer)
        h.backward(g)
        return None
    else:
        # 准备箱子
        buf = _act_buf(model, parts[0][1])
        dist.recv(buf, src=peer)
        buf.requires_grad_()
        total = 0.0
        for i, (_, y) in enumerate(parts):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = forward_last(model, buf, y) / m
            total += loss.detach()
            loss.backward()
            # 如果不是最后一份
            if i != m - 1:
                nxt = torch.empty_like(buf)
                _send_recv(buf.grad, nxt, peer)
                buf = nxt.requires_grad_()
            # 如果是最后一份
            else:
                dist.send(buf.grad, peer)
        return total

