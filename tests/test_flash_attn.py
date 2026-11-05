import pytest
import torch
import torch.nn.functional as F
from kernels.flash_attn import _forward, _backward, flash_attn, flash_attn_fwd, flash_attn_bwd
from model import Attention, ModelForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")


@pytest.mark.parametrize("B, H, H_kv, T, D", [
    (2, 8, 2, 512, 128),    # tinytrain 实际情况，G=4
    (2, 8, 2, 500, 128),    # T 不是块大小整数倍，检查边界 mask
    (2, 4, 4, 256, 128),    # G=1，确认不分组也对
])
def test_fa_fwd_bwd(B, H, H_kv, T, D):
    torch.manual_seed(0)
    # 造输入
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    # 参考
    q_ref = q.float().detach().requires_grad_()
    k_ref = k.float().detach().requires_grad_()
    v_ref = v.float().detach().requires_grad_()
    o_ref = F.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, is_causal=True, enable_gqa=True
    )
    o_ref.backward(do.float())

    # 自己的 kernel
    o, M = _forward(q, k, v)
    dq, dk, dv = _backward(q, k, v, o, do, M)

    for name, got, want in [
        ("o", o, o_ref),
        ("dq", dq, q_ref.grad),
        ("dk", dk, k_ref.grad),
        ("dv", dv, v_ref.grad),
    ]:
        torch.testing.assert_close(
            got.float(), want, rtol=3e-2, atol=3e-2,
            msg=lambda s, n=name, sh=(B, H, H_kv, T, D):
            f"{n} 不对 shape={sh}\n{s}"
        )


def test_opcheck():
    B, H, H_kv, T, D = 2, 8, 2, 256, 128
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.library.opcheck(flash_attn_fwd, (q, k, v))

    o, M = flash_attn_fwd(q.detach(), k.detach(), v.detach())
    do = torch.randn_like(o)
    torch.library.opcheck(flash_attn_bwd, (q.detach(), k.detach(), v.detach(), o, do, M))


def test_autograd_matches_launcher():
    torch.manual_seed(0)
    B, H, H_kv, T, D = 2, 8, 2, 512, 128
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)

    # 路线一：经过 custom op 和 autograd
    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()
    o1 = flash_attn(q1, k1, v1)
    o1.backward(do)

    # 路线二：直接调用 launcher
    o2, M = _forward(q, k, v)
    dq2, dk2, dv2 = _backward(q, k, v, o2, do, M)

    # 验证逐位相同
    assert torch.equal(o1, o2)
    assert torch.equal(q1.grad, dq2)
    assert torch.equal(k1.grad, dk2)
    assert torch.equal(v1.grad, dv2)


def test_compile_fullgraph():
    torch.manual_seed(0)
    B, H, H_kv, T, D = 2, 8, 2, 512, 128
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

    def f(q, k, v, w):
        return (flash_attn(q, k, v).float() * w).sum()
    
    # 不 compile
    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()
    f(q1, k1, v1, w).backward()

    # compile，fullgraph=True：一旦断图就直接报错
    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()
    f_compiled = torch.compile(f, fullgraph=True)
    f_compiled(q2, k2, v2, w).backward()

    torch.testing.assert_close(q2.grad, q1.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k2.grad, k1.grad, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v2.grad, v1.grad, rtol=1e-3, atol=1e-3)


def test_model_triton_matches_sdpa(tiny_config):
    # 造模型和数据
    model = ModelForCausalLM(tiny_config)
    model.to("cuda")
    torch.manual_seed(0)
    x = torch.randint(0, model.config.vocab_size, (2, 32), device="cuda")
    targets = torch.randint(0, model.config.vocab_size, (2, 32), device="cuda")

    # 默认跑一次
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(x, targets)

    # 设置成 triton flash_attn，再跑一次
    for mod in model.modules():
        if isinstance(mod, Attention):
            mod.attn_impl = "triton"
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits2, loss2 = model(x, targets)

    # 比较
    torch.testing.assert_close(logits, logits2, rtol=1e-2, atol=1e-2)

