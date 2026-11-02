import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from parallel.pp import gpipe_step, split_model
import copy, torch
import torch.distributed as dist
from model import ModelConfig, ModelForCausalLM
from parallel.dist_utils import setup, cleanup, get_layout, get_rank, get_tp_group, init_parallel


setup()
init_parallel(1, 2)
rank = get_rank()
layout = get_layout()
torch.manual_seed(400)
# 模型只要 4 层
cfg = ModelConfig(vocab_size=256, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=64, intermediate_size=192, max_seq_len=32)
full = ModelForCausalLM(cfg)
stage = split_model(copy.deepcopy(full), layout.pp_rank, 3)  # 本进程 pp 切分后的模型
# 造 4 组数据
parts = [(torch.randint(0, cfg.vocab_size, (2, 8)), torch.randint(0, cfg.vocab_size, (2, 8))) for _ in range(4)]
# 用 full 算参考答案
ref = 0.0
for x, y in parts:
    _, loss = full(x, y)
    loss = loss / 4
    loss.backward()
    ref += loss.detach()

def check(name, a, b):
    diff = (a - b).abs().max().item()
    assert torch.equal(a, b), f"rank {rank}：{name} 最大差 {diff}"

loss = gpipe_step(stage, parts)
# 第二段的 loss 和 ref 比
if layout.pp_rank == 1:
    check("loss", ref, loss)
# 本段的每个梯度和 full 比
full_params = dict(full.named_parameters())
for name, p in stage.named_parameters():
    if getattr(p, "pp_shared", False):
        continue
    check(name, p.grad, full_params[name].grad)
print(f"rank {rank} OK")
cleanup()

