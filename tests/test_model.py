import torch
import torch.nn as nn
import math
import model
from model import estimate_flops_per_token, ModelForCausalLM, Attention


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


def test_mlp():
    config = model.ModelConfig()
    mlp = model.MLP(config)
    B, T, D = 2, 8, config.hidden_size
    x = torch.randn(B, T, D)
    out = mlp(x)
    numel = sum(p.numel() for p in mlp.parameters())

    assert x.shape == out.shape
    assert numel == config.hidden_size * config.intermediate_size * 3


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


def test_causal_model():
    torch.manual_seed(400)
    config = model.ModelConfig()
    m = model.ModelForCausalLM(config)
    numel = sum(p.numel() for p in m.parameters())
    # 取一个 batch
    B, T = 2, 16
    x = torch.randint(0, config.vocab_size, (B, T))
    y = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = m(x, y)

    assert numel == 112_482_048
    assert logits.shape == (B, T, config.vocab_size)
    assert 10.3 <= loss.item() <= 10.7


def test_param_count():
    config = model.ModelConfig()
    # 参数
    hidden_size = config.hidden_size
    layers = config.num_hidden_layers
    num_head = config.num_attention_heads
    num_kv_head = config.num_key_value_heads
    head_dim = config.head_dim
    intermediate_state = config.intermediate_size
    vocab_size = config.vocab_size
    tie_embedding = config.tie_word_embeddings

    # Attention
    Wq = hidden_size * num_head * head_dim
    Wk = hidden_size * num_kv_head * head_dim
    Wv = Wk
    Wo = Wq
    q_norm = head_dim
    k_norm = q_norm
    attn_total = Wq + Wk + Wv + Wo + q_norm + k_norm

    # MLP
    Wup = hidden_size * intermediate_state
    Wgate = Wup
    Wdown = Wup
    mlp_total = Wup + Wgate + Wdown

    # RMSNorm
    rms_norm_total = 2 * hidden_size

    # Block
    block = attn_total + mlp_total + rms_norm_total

    # Embedding
    embedding = vocab_size * hidden_size

    # lm_head
    lm_head = 0 if tie_embedding else embedding

    # pre lm_head norm
    norm = hidden_size

    # model total
    model_total = embedding + block * layers + norm + lm_head

    m = model.ModelForCausalLM(config)
    assert sum(p.numel() for p in m.parameters()) == model_total
    

def test_flops_per_token(tiny_config):
    model = ModelForCausalLM(tiny_config)
    L = tiny_config.num_hidden_layers
    H = tiny_config.num_attention_heads
    dh = tiny_config.head_dim
    T = tiny_config.max_seq_len
    n = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    assert estimate_flops_per_token(tiny_config, T) == 6*n + 12*L*H*dh*T


def test_enable_gqa_matches(tiny_config):
    assert tiny_config.num_attention_heads != tiny_config.num_key_value_heads
    model = ModelForCausalLM(tiny_config)
    x = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    # 默认跑一遍，存梯度
    logits, loss = model(x, targets)
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.append(p.grad.detach().clone())
    
    # enable_gqa 打开，再跑一遍
    model.zero_grad()
    for mod in model.modules():
        if isinstance(mod, Attention):
            mod.enable_gqa = True
    logits2, loss = model(x, targets)
    loss.backward()
    grads2 = []
    for p in model.parameters():
        grads2.append(p.grad.detach().clone())
    
    # 验证 logits 和 grads 相等
    assert torch.allclose(logits, logits2, atol=1e-6)
    for g1, g2 in zip(grads, grads2):
        assert torch.allclose(g1, g2, atol=1e-6)
    

def test_chunked_ce_matches(tiny_config):
    model = ModelForCausalLM(tiny_config)
    x = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    # 默认跑一遍，存梯度
    logits, loss = model(x, targets)
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.append(p.grad.detach().clone())
    
    # chunked_ce 打开，再跑一遍
    model.zero_grad()
    model.ce_chunk_size = 10
    logits2, loss2 = model(x, targets)
    loss2.backward()
    grads2 = []
    for p in model.parameters():
        grads2.append(p.grad.detach().clone())
    
    # 验证 logits 和 grads 相等
    assert torch.allclose(loss, loss2, atol=1e-6)
    for g1, g2 in zip(grads, grads2):
        assert torch.allclose(g1, g2, atol=1e-6)


def test_act_ckpt_matches(tiny_config):
    model = ModelForCausalLM(tiny_config)
    x = torch.randint(0, model.config.vocab_size, (2, 16))
    targets = torch.randint(0, model.config.vocab_size, (2, 16))
    # 默认跑一遍，存梯度
    logits, loss = model(x, targets)
    loss.backward()
    grads = []
    for p in model.parameters():
        grads.append(p.grad.detach().clone())
    
    # activation checkpointing 打开，再跑一遍
    model.zero_grad()
    model.model.act_ckpt = True
    logits2, loss2 = model(x, targets)
    loss2.backward()
    grads2 = []
    for p in model.parameters():
        grads2.append(p.grad.detach().clone())
    
    # 验证 logits 和 grads 相等
    assert torch.equal(loss, loss2)
    for g1, g2 in zip(grads, grads2):
        assert torch.equal(g1, g2)

