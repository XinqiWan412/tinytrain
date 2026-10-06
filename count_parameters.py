import model

cfg = model.ModelConfig()

# 参数
hidden_size = 768
layers = 12
num_head = 8
num_kv_head = 2
head_dim = 128
intermediate_state = 3 * hidden_size
vocab_size = 32768
tie_embedding = True


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

print(f"{model_total:,}")