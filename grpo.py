import random
from dataclasses import asdict
import torch
import torch.nn.functional as F
from generate import generate
from model import ModelForCausalLM, load_model
from sft_data import build_xy, split_pairs, SEED, N_TEST, make_question, parse_answer
from tokenizer import render_prompt, IM_END, enc
from sft import collate
from pretrain import TrainConfig, configure_optimizer, get_lr
from eval_sft import evaluate_accuracy


def rollout(model: ModelForCausalLM, pairs, G, device):
    # 把一批问题重复推理 G 遍 
    # 返回的 prompts 是 list[list[int]]，装的 token id
    # responses 是 list[list[int]]，装的推理输出的 token id（含im_end）
    # rewards 是得分，正确 1，错误 0，一维 tensor，(B*G,)
    # advantages 是优势，一维tensor，(B*G,)
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
    return prompts, responses, rewards, advantages


def compute_loss(model: ModelForCausalLM, prompts, responses, advantages, G, device):
    # 用 rollout 的结果，计算 loss
    examples = []
    for i, resp in enumerate(responses):
        prompt = prompts[i // G]
        x, y = build_xy(prompt, resp)
        examples.append((x, y))
    xs, ys = collate(examples, device)  # (B*G, L)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _ = model(xs)  # (B*G, L, V)
    assert logits.shape[:2] == xs.shape[:2]
    logits = logits.to(torch.float32)
    L, V = logits.shape[1:]
    # 这里算出 loss，得到的是 responses 里每一个 token 的 loss，注意 reduction=none，没有做平均
    nll = F.cross_entropy(logits.view(-1, V), ys.view(-1), reduction="none")  # (B*G*L,)
    nll = nll.view(-1, L)  # (B*G, L)
    mask = (ys != -100).to(torch.float32)  # (B*G, L)
    nll = (nll * mask).sum(dim=-1) / mask.sum(dim=-1)  # (B*G,)，每一行内部对 loss 做平均，得到这一行的 loss
    loss = (advantages * nll).mean()  # 每一行各自乘以自己的优势 advantages，然后整体求平均
    return loss


if __name__ == "__main__":
    train, test = split_pairs(SEED, N_TEST)
    lengths = set()
    # 验证 train 全部等长
    for a, b in train:
        question = make_question(a, b)
        prompt = render_prompt(question)
        lengths.add(len(prompt))
    assert len(lengths) == 1, f"prompt 长度不一致：{lengths}"

    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "checkpoints/sft_weak.pt"
    model = load_model(path, device)
    model.eval()
    G = 8  # 重复推理 G 次
    B = 64  # batch

    # GRPO 训练
    train_config = TrainConfig(
        max_lr=2e-5,
        min_lr=2e-6,
        warmup_steps=20,
        max_steps=500,
        eval_interval=50,
    )
    optimizer = configure_optimizer(model, train_config)
    rng = random.Random(SEED)
    model.eval()
    accuracy_rate = evaluate_accuracy(model, test, 100, device, show_wrong=False)
    print(f"step:0, accuracy_rate:{accuracy_rate:.4f}")

    # 训练循环
    for step in range(train_config.max_steps):
        pairs = rng.sample(train, B)  # 抽 B 道题
        model.eval()
        prompts, responses, rewards, advantages = rollout(model, pairs, G, device)
        model.train()
        optimizer.zero_grad()
        loss = compute_loss(model, prompts, responses, advantages, G, device)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        lr = get_lr(step, train_config)
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()

        # 打印 step、平均奖励、有对有错的比例、loss
        rewards = rewards.view(B, G)
        n_correct = rewards.sum(dim=-1)  # (B,) 每道题在 G 次采样里答对的次数
        mixed = (n_correct > 0) & (n_correct < G)  # (B,) 是否有对有错的 mask
        mixed_ratio = mixed.to(torch.float32).mean()
        print(f"step:{step+1}, rewards_avg:{rewards.mean():.4f}, mixed_ratio:{mixed_ratio.item():.4f}, loss:{loss.item():.4f}")
        # 定期用测试集评估
        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:
            model.eval()
            accuracy_rate = evaluate_accuracy(model, test, 100, device, show_wrong=False)
            print(f"step:{step+1}, accuracy_rate:{accuracy_rate:.4f}")

    # 保存模型
    save_path = "checkpoints/grpo.pt"
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.config),
    }
    torch.save(checkpoint, save_path)
