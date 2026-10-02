import random
import torch
from generate import generate
from model import ModelForCausalLM, load_model
from sft_data import split_pairs, SEED, N_TEST, make_question, parse_answer
from tokenizer import render_prompt, IM_END, enc


def rollout(model: ModelForCausalLM, pairs, G, device):
    # 把一批问题重复推理 G 遍 
    prompts = []
    for a, b in pairs:
        question = make_question(a, b)
        prompt = render_prompt(question)
        prompts.append(prompt)
    x = torch.tensor(prompts, dtype=torch.long, device=device)  # (B, T)
    T = x.shape[1]
    x = x.repeat_interleave(G, dim=0)  # (B*G, T)
    idx = generate(model, x, max_new_tokens=5, temperature=1.0, top_k=None)  # (B*G, T + max_new_tokens)
    idx = idx[:, T:]  # 取出新生成的 token，(B*G, max_new_tokens)
    idx = idx.tolist()

    # 打分
    responses, rewards = [], []
    for i, row in enumerate(idx):
        # 截输出的 IM_END，存入 responses
        if IM_END in row:
            im_end_idx = row.index(IM_END)
            responses.append(row[:im_end_idx + 1])  # IM_END 也保留
        else:
            responses.append(row)  # 没有 IM_END，就保留全部
        # 打分
        a, b = pairs[i // G]
        if parse_answer(row) == a + b:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)  # (B*G,)
    
    # 算优势
    r = rewards.view(-1, G)  # (B, G)
    r_avg = r.mean(dim=-1, keepdim=True)
    r_std = r.std(dim=-1, keepdim=True)
    advantages = (r - r_avg) / (r_std + 1e-4)  # (B, G)
    advantages = advantages.view(-1)  # (B*G,)
    return responses, rewards, advantages


if __name__ == "__main__":
    train, test = split_pairs(SEED, N_TEST)
    lengths = set()
    # 验证 train 全部等长
    for a, b in train:
        question = make_question(a, b)
        prompt = render_prompt(question)
        lengths.add(len(prompt))
    assert len(lengths) == 1, f"prompt 长度不一致：{lengths}"

    pairs = random.Random(SEED).sample(train, 16)  # 随机抽 16 组

    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "checkpoints/sft_weak.pt"
    model = load_model(path, device)
    model.eval()
    G = 8  # 重复推理 G 次
    responses, rewards, advantages = rollout(model, pairs, G, device)

    # 平均奖励
    print(f"rewards_avg = {rewards.mean()}")
    # 有对有错的题占多少比例
    rewards = rewards.view(16, G)
    n_correct = rewards.sum(dim=-1)  # (16,) 每道题在 G 次采样里答对的次数
    mixed = (n_correct > 0) & (n_correct < G)  # (16,) 是否有对有错的 mask
    mixed_ratio = mixed.to(torch.float32).mean()
    print(f"有对有错的题占比为：{mixed_ratio.item()}")
    # 打印一道有对有错的题
    if not mixed.any():
        print("没有有对有错的题")
    else:
        q = mixed.nonzero()[0].item()
        print(f"q={q}")
        a, b = pairs[q]
        group = responses[q*G:(q+1)*G]
        group_rewards = rewards[q].tolist()
        group_advantages = advantages[q*G:(q+1)*G].tolist()

        print(f"题目：{a} + {b} = {a + b}")
        for reward, adv, resp in zip(group_rewards, group_advantages, group):
            if any(t >= enc.n_vocab for t in resp):  # 防止 response 里有空位 id
                body = f"tokens={resp}"
            else:
                body = repr(enc.decode(resp))
            print(f"reward={reward:.0f} | advantage={adv:+.3f} | {body}")
