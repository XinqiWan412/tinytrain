import torch
from model import ModelForCausalLM, load_model, ModelConfig
import torch.nn.functional as F
from tokenizer import enc


def generate(model: ModelForCausalLM, idx, max_new_tokens, temperature, top_k):
    # 定长批量推理
    # idx 是 (B, T) 的 token id
    with torch.no_grad():
        max_seq_len = model.config.max_seq_len
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -max_seq_len:]  # 超过 max_seq_len 的舍弃开头
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(idx_cond)  # (B, T, V)
            logits = logits[:, -1, :]  # 取序列里最后一个位置
            logits = logits.to(torch.float32)  # (B, V)
            if temperature != 0:
                # 正常采样
                logits = logits / temperature
                if top_k is not None:
                    v, ix = torch.topk(logits, top_k, dim=1)  # (B, k)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))  # 只保留前 k 大，其余置为 -inf
                probs = F.softmax(logits, dim=-1)  # (B, V)
                new_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                # temperatrue 为 0，贪心解码
                new_token = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, new_token], dim=-1)
    return idx  # (B, T + max_new_tokens)


if __name__ == "__main__":
    # 加载模型
    path = "checkpoints/final.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(path, device)
    model.eval()

    # 编码
    text = "The water cycle is"
    ids = [enc.eot_token] + enc.encode_ordinary(text)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    ids = ids.repeat(5, 1)  # dim=0 复制 5 份，dim=1 复制 1 份

    # 调用推理
    temperature, top_k, max_new_tokens = 1.0, 50, 100
    ids = generate(model, ids, max_new_tokens, temperature, top_k)

    # 解码
    ids = ids.tolist()
    for row in ids:
        row = row[1:]  # 去掉开头的 EOT
        if enc.eot_token in row:
            idx = row.index(enc.eot_token)
            row = row[:idx]
        print(enc.decode(row))
        print("-" * 40)
