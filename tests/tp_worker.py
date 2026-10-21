import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn.functional as F
from model import ModelConfig, MLP
from parallel.dist_utils import setup, cleanup, get_rank, init_parallel
from parallel.tp import ColumnParallelLinear, RowParallelLinear, copy_to_tp

setup()
init_parallel(2)
rank = get_rank()
torch.manual_seed(0)  # 两个进程建出同一个 MLP、同样的输入和上游梯度
# 单卡加载 MLP
full = MLP(ModelConfig(hidden_size=64, intermediate_size=192))
x = torch.randn(2, 8, 64, requires_grad=True)
dy = torch.randn(2, 8, 64)

def check(name, a, b):
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=1e-6), f"rank {rank}：{name} 最大差 {diff}"

y_ref = full(x)
y_ref.backward(dy)

# 以 TP 多卡的方式加载 MLP
gate = ColumnParallelLinear(full.gate_proj)
up = ColumnParallelLinear(full.up_proj)
down = RowParallelLinear(full.down_proj)
x2 = x.detach().clone().requires_grad_(True)

h = copy_to_tp(x2)
y_tp = down(F.silu(gate(h)) * up(h))
y_tp.backward(dy)

check("输出", y_tp, y_ref)
check("输入梯度", x2.grad, x.grad)
check("gate 的权重梯度", gate.weight.grad, torch.chunk(full.gate_proj.weight.grad, 2, dim=0)[rank])
check("up 的权重梯度", up.weight.grad, torch.chunk(full.up_proj.weight.grad, 2, dim=0)[rank])
check("down 的权重梯度", down.weight.grad, torch.chunk(full.down_proj.weight.grad, 2, dim=1)[rank])
print(f"rank {rank} OK")
cleanup()
