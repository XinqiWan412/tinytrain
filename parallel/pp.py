from model import ModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F


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

