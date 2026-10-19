import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.distributed as dist
from model import ModelConfig, ModelForCausalLM
from parallel.dist_utils import setup, cleanup, get_rank, get_world_size
from parallel.zero import ZeroOptimizer

setup()
rank, world_size = get_rank(), get_world_size()
torch.manual_seed(0)
cfg = ModelConfig(vocab_size=256, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=64, intermediate_size=192, max_seq_len=32)
model = ModelForCausalLM(cfg)
wd = [p for p in model.parameters() if p.dim() >= 2]
keep = [p for p in model.parameters() if p.dim() < 2]
groups = [{"params": wd, "weight_decay": 0.1}, {"params": keep, "weight_decay": 0.0}]
opt = ZeroOptimizer(groups, torch.optim.AdamW, stage=2, lr=1e-3, betas=(0.9, 0.95), fused=True)

torch.manual_seed(100 + rank)  # 每个进程的数据不同，本地梯度也不同
x = torch.randint(0, 256, (2, 8))
_, loss = model(x, x)
loss.backward()

# 参照：DDP 的做法，先各乘 1/卡数，再 all-reduce 求和
ref = [p.grad * (1.0 / world_size) for p in opt.params]
for r in ref:
    dist.all_reduce(r)

opt.reduce_grads()

own = [i for i, o in enumerate(opt.owners) if o == rank]
others = [i for i, o in enumerate(opt.owners) if o != rank]
same = all(torch.equal(opt.params[i].grad, ref[i]) for i in own)
freed = all(opt.params[i].grad is None for i in others)
assert same, f"rank {rank}：负责的梯度和 DDP 式平均不同"
assert freed, f"rank {rank}：不负责的梯度没有设成 None"
print(f"rank {rank} OK")
cleanup()
