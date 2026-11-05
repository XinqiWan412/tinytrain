import triton.language as tl
import triton
import torch

# kernel 配置，按 D=128 调优
#           BLOCK_M, BLOCK_N, num_warps, num_stages
FWD_CFG     = (128,     64,     8,          2)
DKDV_CFG    = (64,      64,     8,          2)
DQ_CFG      = (128,     64,     8,          2)


@triton.jit
def fwd_kernel(
    Q, K, V, O, M,
    sm_scale, N_CTX,
    stride_z, stride_n, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    GROUP: tl.constexpr,
):
    start_m = tl.program_id(axis=0)
    off_z = tl.program_id(axis=1)
    base_q = off_z * stride_z
    base_kv = (off_z // GROUP) * stride_z

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    q = tl.load(Q + base_q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # for 循环的上边界 hi，比 q 更靠后的 kv 快可以跳过
    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX

    # 循环遍历 kv
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX
        k = tl.load(K + base_kv + offs_n[None, :] * stride_n + offs_d[:, None] * stride_d,
                    mask=mask_n[None, :], other=0.0)
        v = tl.load(V + base_kv + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, k) * sm_scale
        valid = mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + base_q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
             acc.to(O.dtype.element_ty), mask=mask_m[:, None])

    tl.store(M + off_z * N_CTX + offs_m, m_i + tl.log(l_i), mask=mask_m)


@triton.jit
def bwd_preprocess(
    O, DO, Delta, N_CTX,
    stride_z, stride_n, stride_d,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    # 唯一的计算就是 D = rowsum(O⊙dO)，后续求 dS = P⊙(dP-D)
    start_m = tl.program_id(axis=0)
    off_z = tl.program_id(axis=1)
    base = off_z * stride_z

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    o = tl.load(O + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask_m[:, None], other=0.0).to(tl.float32)
    do = tl.load(DO + base + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                 mask=mask_m[:, None], other=0.0).to(tl.float32)

    tl.store(Delta + off_z * N_CTX + offs_m, tl.sum(o * do, axis=1), mask=mask_m)


@triton.jit
def bwd_dkdv_kernel(
    Q, K, V, DO, DK, DV, M, Delta,
    sm_scale, N_CTX,
    stride_z, stride_n, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    GROUP: tl.constexpr,
):
    start_n = tl.program_id(axis=0)
    off_kv = tl.program_id(axis=1)
    base = off_kv * stride_z

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_n = offs_n < N_CTX

    k = tl.load(K + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask_n[:, None], other=0.0)
    v = tl.load(V + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask_n[:, None], other=0.0)

    # dk, dv 两个累加器
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # causal：第 j 批 key 只被 i >= j 的 query 看见，前面的 query 根本不用扫
    if CAUSAL:
        lo = start_n * BLOCK_N
    else:
        lo = 0

    for g in range(GROUP):
        off_z = off_kv * GROUP + g
        base_q = off_z * stride_z
        for start_m in range(lo, N_CTX, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N_CTX

            q = tl.load(Q + base_q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                        mask=mask_m[:, None], other=0.0)
            do = tl.load(DO + base_q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                        mask=mask_m[:, None], other=0.0)
            m = tl.load(M + off_z * N_CTX + offs_m, mask=mask_m, other=0.0)
            delta = tl.load(Delta + off_z * N_CTX + offs_m, mask=mask_m, other=0.0)

            valid = mask_m[:, None] & mask_n[None, :]
            if CAUSAL:
                valid = valid & (offs_m[:, None] >= offs_n[None, :])

            qk = tl.dot(q, tl.trans(k))
            p = tl.exp(qk * sm_scale - m[:, None])
            p = tl.where(valid, p, 0.0)

            dv = tl.dot(tl.trans(p).to(do.dtype), do, dv)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None])

            dk = tl.dot(tl.trans(ds).to(q.dtype), q, dk)
        
    dk *= sm_scale

    tl.store(DK + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
             dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(DV + base + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
             dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


@triton.jit
def bwd_dq_kernel(
    Q, K, V, DO, DQ, M, Delta,
    sm_scale, N_CTX,
    stride_z, stride_n, stride_d,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
    REVERSE: tl.constexpr, GROUP: tl.constexpr,
):
    num_m = tl.cdiv(N_CTX, BLOCK_M)
    if CAUSAL and REVERSE:
        start_m = num_m - 1 - tl.program_id(axis=0)
    else:
        start_m = tl.program_id(axis=0)
    off_z = tl.program_id(axis=1)
    base_q = off_z * stride_z
    base_kv = (off_z // GROUP) * stride_z

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    mask_m = offs_m < N_CTX

    q = tl.load(Q + base_q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                mask=mask_m[:, None], other=0.0)
    do = tl.load(DO + base_q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d,
                 mask=mask_m[:, None], other=0.0)
    m = tl.load(M + off_z * N_CTX + offs_m, mask=mask_m, other=0.0)
    delta = tl.load(Delta + off_z * N_CTX + offs_m, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    hi = (start_m + 1) * BLOCK_M if CAUSAL else N_CTX

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_CTX

        k = tl.load(K + base_kv + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=mask_n[:, None], other=0.0)
        v = tl.load(V + base_kv + offs_n[:, None] * stride_n + offs_d[None, :] * stride_d,
                    mask=mask_n[:, None], other=0.0)

        valid = mask_m[:, None] & mask_n[None, :]
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])

        p = tl.exp(tl.dot(q, tl.trans(k)) * sm_scale - m[:, None])
        p = tl.where(valid, p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])

        dq = tl.dot(ds.to(k.dtype), k, dq)

    tl.store(DQ + base_q + offs_m[:, None] * stride_n + offs_d[None, :] * stride_d, 
    (dq * sm_scale).to(DQ.dtype.element_ty), 
    mask=mask_m[:, None])


def _forward(q, k, v):
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()

    B, H, T, D = q.shape
    H_kv = k.shape[1]
    GROUP = H // H_kv
    q = q.view(B*H, T, D)
    k = k.view(B*H_kv, T, D)
    v = v.view(B*H_kv, T, D)
    o = torch.empty_like(q)
    M = torch.empty((B, H, T), device=q.device, dtype=torch.float32)

    BLOCK_M, BLOCK_N, num_warps, num_stages = FWD_CFG
    grid = (triton.cdiv(T, BLOCK_M), B*H)
    fwd_kernel[grid](
        q, k, v, o, M, 1.0 / (D ** 0.5), T,
        q.stride(0), q.stride(1), q.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, 
        GROUP=GROUP, CAUSAL=True,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o.view(B, H, T, D), M


def _backward(q, k, v, o, do, M):
    B, H, T, D = q.shape
    H_kv = k.shape[1]
    GROUP = H // H_kv
    q = q.view(B*H, T, D)
    k = k.view(B*H_kv, T, D)
    v = v.view(B*H_kv, T, D)
    o = o.view(B*H, T, D)
    do = do.contiguous()
    delta = torch.empty((B*H, T), device=q.device, dtype=torch.float32)
    dq = torch.empty_like(q); dk = torch.empty_like(k); dv = torch.empty_like(v)
    sm_scale = 1.0 / (D ** 0.5)
    strides = (q.stride(0), q.stride(1), q.stride(2))

    bwd_preprocess[(triton.cdiv(T, 64), B*H)](
        o, do, delta, T, *strides, BLOCK_M=64, HEAD_DIM=D)

    BLOCK_M, BLOCK_N, num_warps, num_stages = DKDV_CFG
    bwd_dkdv_kernel[(triton.cdiv(T, BLOCK_N), B*H_kv)](
        q, k, v, do, dk, dv, M, delta, sm_scale, T, *strides,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, 
        GROUP=GROUP, CAUSAL=True,
        num_warps=num_warps, num_stages=num_stages)

    BLOCK_M, BLOCK_N, num_warps, num_stages = DQ_CFG
    bwd_dq_kernel[(triton.cdiv(T, BLOCK_M), B*H)](
        q, k, v, do, dq, M, delta, sm_scale, T, *strides,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D, 
        GROUP=GROUP, CAUSAL=True, REVERSE=True,
        num_warps=num_warps, num_stages=num_stages)

    return dq.view(B, H, T, D), dk.view(B, H_kv, T, D), dv.view(B, H_kv, T, D)


# ==========================================================================
# custom op 注册：让 torch.compile 把整个 FA 当作一个不透明的节点
# ==========================================================================

@torch.library.custom_op("tinytrain::flash_attn_fwd", mutates_args=())
def flash_attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _forward(q, k, v)


@flash_attn_fwd.register_fake
def _(q, k, v):
    # 不运行 kernel，只告诉 compile 输出的形状、dtype、stride
    B, H, T, D = q.shape
    return torch.empty_like(q), q.new_empty((B, H, T), dtype=torch.float32)


@torch.library.custom_op("tinytrain::flash_attn_bwd", mutates_args=())
def flash_attn_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   o: torch.Tensor, do: torch.Tensor, M: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _backward(q, k, v, o, do, M)


@flash_attn_bwd.register_fake
def _(q, k, v, o, do, M):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup_context(ctx, inputs, output):
    # 前向结束时调用：决定反向需要保存什么
    q, k, v = inputs
    o, M = output
    ctx.save_for_backward(q, k, v, o, M)


def _backward_fn(ctx, do, dM):
    # dM 是 M 的梯度。M 只给反向用，不参与 loss，所以忽略
    q, k, v, o, M = ctx.saved_tensors
    return flash_attn_bwd(q, k, v, o, do, M)


flash_attn_fwd.register_autograd(_backward_fn, setup_context=_setup_context)


def flash_attn(q, k, v):
    """对外的接口：输入连续的 (B, H, T, D) 和 (B, H_kv, T, D)，因果注意力，返回 o"""
    o, _ = flash_attn_fwd(q, k, v)
    return o

