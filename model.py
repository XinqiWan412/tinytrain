from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 768
    num_hidden_layers: int = 12  # layers
    num_attention_heads: int = 8  # num_head
    num_key_value_heads: int = 2  # num_kv_head
    head_dim: int = 128
    intermediate_size: int = 3 * 768
    max_seq_len: int = 512
    rope_theta: float = 1e6
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True  # embedding 和 lm_head 共用一份权重


class RMSNorm(nn.Module):   
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        r = torch.rsqrt(ms + self.eps)
        return (self.weight * x * r).to(input_dtype)


def precompute_rope(head_dim, max_seq_len, base):
    dh = head_dim
    step = torch.arange(0, dh, 2)  # 0, 2, ..., dh - 2
    theta = base ** (-step/dh)  # (dh/2,)
    pos = torch.arange(max_seq_len)  # (max_T,)
    angles = pos[:, None] * theta  # (max_T, dh/2)
    angles = torch.cat([angles, angles], dim=-1)  # (max_T, dh)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # 公式：
    # x_hat = x*cos - y*sin
    # y_hat = x*sin + y*cos
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    y = torch.cat([-x2, x1], dim=-1)
    return x * cos + y * sin


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads*self.head_dim, bias=False)  # (D, H*dh)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads*self.head_dim, bias=False)  # (D, H_kv*dh)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads*self.head_dim, bias=False)  # (D, H_kv*dh)
        self.o_proj = nn.Linear(self.num_heads*self.head_dim, self.hidden_size, bias=False)  # (H*dh, D)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # (dh,)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # (dh,)
        self.tp_copy = nn.Identity()
        self.enable_gqa = False
    
    def forward(self, x, cos, sin):
        # x 是 (B, T, D)
        B, T, _ = x.shape
        dh = self.head_dim
        H, H_kv = self.q_proj.weight.shape[0] // dh, self.k_proj.weight.shape[0] // dh
        g = H // H_kv
        x = self.tp_copy(x)  # 用 f 包一下 x，反向的时候才会做 TP 的规约
        q = self.q_proj(x).view(B, T, H, dh)  # (B, T, H, dh)
        k = self.k_proj(x).view(B, T, H_kv, dh)  # (B, T, H_kv, dh)
        v = self.v_proj(x).view(B, T, H_kv, dh)  # (B, T, H_kv, dh)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q.transpose(1, 2)  # (B, H, T, dh)
        k = k.transpose(1, 2)  # (B, H_kv, T, dh)
        v = v.transpose(1, 2)  # (B, H_kv, T, dh)
        cos, sin = cos[:T], sin[:T]  # 切前 T 行，对齐 qk
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if not self.enable_gqa:
            k = k.repeat_interleave(g, dim=1)
            v = v.repeat_interleave(g, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.enable_gqa)  # (B, H, T, dh)
        out = out.transpose(1, 2)  # (B, T, H, dh)
        out = out.reshape(B, T, H*dh)
        out = self.o_proj(out)  # (B, T, D)
        return out


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # D -> intermediate
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # D -> intermediate
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # intermediate -> D
        self.tp_copy = nn.Identity()
    
    def forward(self, x):
        x = self.tp_copy(x)  # 用 f 包一下 x，反向的时候才会做 TP 的规约
        gate = self.gate_proj(x)  # (B, T, intermediate)
        up = self.up_proj(x)  # (B, T, intermediate)
        silu_gate = F.silu(gate)  # (B, T, intermediate)
        o = self.down_proj(silu_gate * up)  # (B, T, D)
        return o


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
    
    def forward(self, x, cos, sin):
        h = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = h + self.mlp(self.post_attention_layernorm(h))
        return x


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        cos, sin = precompute_rope(config.head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    
    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"in Model.forward, T={T} > max_seq_len={self.config.max_seq_len}"
        x = self.embed_tokens(input_ids)  # (B, T, D)
        cos, sin = self.cos[:T], self.sin[:T]
        for block in self.layers:
            x = block(x, cos, sin)
        x = self.norm(x)
        return x


class ModelForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # D -> V
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
    
    def forward(self, input_ids, targets=None):
        x = self.model(input_ids)  # (B, T, D)
        logits = self.lm_head(x)  # (B, T, V)
        B, T, V = logits.shape
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, V), targets.view(B*T))
        else:
            loss = None
        return logits, loss


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    config = ModelConfig(**ckpt["model_config"])
    model = ModelForCausalLM(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model


def estimate_flops_per_token(config: ModelConfig, T):
    D = config.hidden_size
    L = config.num_hidden_layers
    H = config.num_attention_heads
    H_kv = config.num_key_value_heads
    dh = config.head_dim
    I = config.intermediate_size
    V = config.vocab_size

    matmul_params_per_layer = D*H*dh + 2*D*H_kv*dh + H*dh*D + 3*D*I  # 每层矩阵乘的参数量
    N = L * matmul_params_per_layer + D * V  # L 层，再加 lm_head
    # 矩阵乘是 6NT，注意力是 6LT²d，这里 12 是计算 mfu 的行业惯例，不扣 causal mask 省下的一半
    return 6*N + 12*L*H*dh*T

