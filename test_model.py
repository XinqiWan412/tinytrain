import torch
import torch.nn as nn
import math
import model, dataloader


def test_rmsnorm():
    # RMSNorm 对拍
    my_rmsnorm = model.RMSNorm(256, eps=1e-6)
    torch_rmsnorm = nn.RMSNorm(256, eps=1e-6)
    with torch.no_grad():
        w = torch.randn(256)
        my_rmsnorm.weight.copy_(w)
        torch_rmsnorm.weight.copy_(w)
    x = torch.randn(2, 16, 256)
    y1 = my_rmsnorm(x)
    y2 = torch_rmsnorm(x)
    assert torch.allclose(y1, y2)
    print("rmsnorm ok")


def test_precompute_rope():
    max_seq_len = 512
    dh = head_dim = 64
    base = 1e6
    cos, sin = model.precompute_rope(head_dim, max_seq_len, base)  # (max_T, dh)
    assert cos.shape == (max_seq_len, head_dim)
    assert sin.shape == (max_seq_len, head_dim)
    assert torch.allclose(cos[0], torch.ones(head_dim))
    assert torch.allclose(sin[0], torch.zeros(head_dim))
    assert torch.allclose(cos[:, :dh//2], cos[:, dh//2:])
    assert torch.allclose(sin[:, :dh//2], sin[:, dh//2:])
    a, b = cos[1, 1], math.cos(base ** (-2 / dh))
    assert abs(a - b) < 1e-6
    print("precompute_rope ok")


def test_apply_rope():
    max_seq_len = 512
    dh = head_dim = 64
    base = 1e6
    B, H = 2, 4
    x = torch.randn(B, H, max_seq_len, dh)  # (B, H, T, dh)
    cos, sin = model.precompute_rope(head_dim, max_seq_len, base)  # (max_T, dh)
    out = model.apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), out.norm(dim=-1))  # 旋转不改变向量的长度
    assert torch.allclose(x[:, :, 0], out[:, :, 0])  # 位置 0 不旋转
    print("apply_rope ok")


def test_attention():
    config = model.ModelConfig()
    attn = model.Attention(config)
    head_dim, max_seq_len, base = config.head_dim, config.max_seq_len, config.rope_theta
    B, T, D = 2, 16, config.hidden_size
    cos, sin = model.precompute_rope(head_dim, max_seq_len, base)  # (max_T, dh)
    x = torch.randn(B, T, D)
    out = attn(x, cos, sin)
    x2 = x.clone()
    x2[:, -1] = torch.randn(B, D)
    out2 = attn(x2, cos, sin)

    assert out.shape == (B, T, D)
    assert torch.allclose(out[:, :-1], out2[:, :-1])
    assert not torch.allclose(out[:, -1], out2[:, -1])
    print('attention ok')


def test_mlp():
    config = model.ModelConfig()
    mlp = model.MLP(config)
    B, T, D = 2, 8, config.hidden_size
    x = torch.randn(B, T, D)
    out = mlp(x)
    numel = sum(p.numel() for p in mlp.parameters())

    assert x.shape == out.shape
    assert numel == config.hidden_size * config.intermediate_size * 3
    print('mlp ok')


def test_block():
    config = model.ModelConfig()
    block = model.Block(config)
    B, T, D = 2, 8, config.hidden_size
    x = torch.randn(B, T, D)
    cos, sin = model.precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
    cos, sin = cos[:T], sin[:T]
    out = block(x, cos, sin)
    numel = sum(p.numel() for p in block.parameters())
    expected_numel = (
        3 * config.hidden_size * config.intermediate_size
        + 2 * config.hidden_size * config.num_attention_heads * config.head_dim  # q_proj, o_proj
        + 2 * config.hidden_size * config.num_key_value_heads * config.head_dim  # k_proj, v_proj
        + 2 * config.hidden_size  # input_layernorm, post_attn_layernorm
        + 2 * config.head_dim  # q_norm, k_norm
    )

    assert x.shape == out.shape
    assert numel == expected_numel
    print('block ok')


def test_causal_model():
    config = model.ModelConfig()
    m = model.ModelForCausalLM(config)
    numel = sum(p.numel() for p in m.parameters())
    # 取一个 batch
    data_dir = "./data/tokenized/gpt2"
    B, T, device = 2, 16, 'cpu'
    dtloader = dataloader.DataLoader(data_dir, 'train', B, T, device)
    x, y = dtloader.next_batch()
    logits, loss = m(x, y)

    assert numel == 16_014_336
    assert logits.shape == (B, T, config.vocab_size)
    assert 10.8 <= loss.item() <= 11
    print(f'loss = {loss.item():.4f}')
    print('causal_model ok')


if __name__ == "__main__":
    test_rmsnorm()
    test_precompute_rope()
    test_apply_rope()
    test_attention()
    test_mlp()
    test_block()
    test_causal_model()
