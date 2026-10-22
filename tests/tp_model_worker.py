import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import copy, torch
import torch.distributed as dist
from model import ModelConfig, ModelForCausalLM
from parallel.tp import TPOptimizer, apply_tp, gather_full_state_dict, load_full_state_dict
from parallel.dist_utils import setup, cleanup, get_layout, get_rank, get_tp_group, init_parallel


setup()
init_parallel(2)
torch.manual_seed(0)
rank = get_rank()
layout = get_layout()
tp_size, tp_rank = layout.tp_size, layout.tp_rank
cfg = ModelConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=64, intermediate_size=192, max_seq_len=32)
full = ModelForCausalLM(cfg)
tp = apply_tp(copy.deepcopy(full))
x = torch.randint(0, 256, (2, 8))
# full
_, loss_ref = full(x, x)
loss_ref.backward()
# TP
_, loss_tp = tp(x, x)
loss_tp.backward()

def check(name, a, b):
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=1e-6), f"rank {rank}：{name} 最大差 {diff}"

# 检查 loss 和 梯度
check("loss", loss_tp, loss_ref)
full_params = dict(full.named_parameters())
for name, p in tp.named_parameters():
    tp_dim = getattr(p, "tp_dim", None)
    if tp_dim is not None:
        chunk = torch.chunk(full_params[name].grad, tp_size, dim=tp_dim)[tp_rank]
        check(name, p.grad, chunk)
    elif "q_norm" in name or "k_norm" in name:
        grad = p.grad.clone()
        dist.all_reduce(grad, group=get_tp_group())
        check(name, grad, full_params[name].grad)
    else:
        check(name, p.grad, full_params[name].grad)

# 检查 gather 和 load 模型
sd = gather_full_state_dict(tp)
full_sd = full.state_dict()
if rank == 0:
    assert set(sd) == set(full_sd)
    for k, v in sd.items():
        assert torch.equal(sd[k], full_sd[k])
torch.manual_seed(1)
tp2 = apply_tp(ModelForCausalLM(cfg))
load_full_state_dict(tp2, full_sd)
for k, _ in full_sd.items():
    assert torch.equal(tp.state_dict()[k], tp2.state_dict()[k])

# 检查包了 TPOptimizer 后的梯度
full_opt = torch.optim.SGD(full.parameters(), lr=0.1)
tp_opt = TPOptimizer([{"params": list(tp.parameters())}], torch.optim.SGD, lr=0.1)
tp_opt.reduce_grads()
full_params = dict(full.named_parameters())
for name, p in tp.named_parameters():
    if getattr(p, "tp_sum_grad", False):
        check(name, p.grad, full_params[name].grad)

# 检查包了 TPOptimizer 后的 norm
norm_full = torch.nn.utils.clip_grad_norm_(full.parameters(), 0.1)
norm_tp = tp_opt.clip_grad_norm_(tp.parameters(), 0.1)
check("总范数", norm_tp, norm_full)

# 检查 step 后的权重
full_opt.step()
tp_opt.step()
full_params = dict(full.named_parameters())
for name, p in tp.named_parameters():
    tp_dim = getattr(p, "tp_dim", None)
    if tp_dim is not None:
        chunk = torch.chunk(full_params[name].detach(), tp_size, dim=tp_dim)[tp_rank]
        check(name, p.detach(), chunk)
    else:
        check(name, p.detach(), full_params[name].detach())

opt_sd = tp_opt.state_dict()
if rank == 0:
    assert opt_sd["world_size"] == 2 and len(opt_sd["per_rank"]) == 2


print(f"rank {rank} OK")
cleanup()
