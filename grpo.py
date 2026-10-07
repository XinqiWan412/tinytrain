import random
from dataclasses import asdict
import torch
import torch.nn.functional as F
from generate import generate
from model import ModelForCausalLM, load_model
from sft_data import build_xy, split_pairs, SEED, N_TEST, make_question, parse_answer
from tokenizer import render_prompt, IM_END
from sft import collate
from pretrain import TrainConfig, configure_optimizer, get_lr
from eval_sft import evaluate_accuracy
from collections import defaultdict
import argparse
from logger import log, setup_log


def rollout(model: ModelForCausalLM, pairs, G, device):
    # 把一批问题重复推理 G 遍 
    # 返回的 prompts 是 list[list[int]]，装的 token id
    # responses 是 list[list[int]]，装的推理输出的 token id（含im_end）
    # rewards 是得分，正确 1，错误 0，一维 tensor，(B*G,)
    # advantages 是优势，一维tensor，(B*G,)

    # 按长度对 pairs 分组
    pairs_by_len = defaultdict(list)
    prompts = []
    for idx, (a, b) in enumerate(pairs):
        question = make_question(a, b)
        prompt = render_prompt(question)
        prompts.append(prompt)
        pairs_by_len[len(prompt)].append(idx)

    # 分组进行 rollout
    responses = [None] * (len(pairs) * G)
    for group in pairs_by_len.values():
        # 调用 generate 批量推理
        group_prompts = [prompts[idx] for idx in group]
        x = torch.tensor(group_prompts, dtype=torch.long, device=device)  # (n, T)
        T = x.shape[1]
        x = x.repeat_interleave(G, dim=0)  # (n*G, T)
        out_token = generate(model, x, max_new_tokens=5, temperature=1.0, top_k=None)  # (n*G, T + max_new_tokens)
        out_token = out_token[:, T:]  # 取出新生成的 token，(n*G, max_new_tokens)
        out_token = out_token.tolist()
        for k, row in enumerate(out_token):
            j = k // G  # 第几道题
            g = k % G  # 第几个采样回答
            i = group[j]  # 这道题在 pairs 里的下标
            # 截输出的 IM_END，存入 responses
            if IM_END in row:
                im_end_idx = row.index(IM_END)
                responses[i*G+g] = row[:im_end_idx + 1]  # IM_END 也保留
            else:
                responses[i*G+g] = row  # 没有 IM_END，就保留全部

    # 打分
    assert all(res is not None for res in responses)
    rewards = []
    for k, row in enumerate(responses):
        # 打分
        a, b = pairs[k // G]
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


def token_logprobs(model, xs, ys):
    # 返回逐 token 的 log(p)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # 前向
        logits, _ = model(xs)  # (B*G, L, V)
    assert logits.shape[:2] == xs.shape[:2]
    logits = logits.to(torch.float32)  # 转 fp32，因为要做 cross_entropy
    L, V = logits.shape[1:]
    # 这里算出 loss，得到的是 responses 里每一个 token 的 loss，注意 reduction=none，没有做平均
    nll = F.cross_entropy(logits.view(-1, V), ys.view(-1), reduction="none")  # (B*G*L,)
    nll = nll.view(-1, L)  # (B*G, L)
    return -nll  # 取负号是因为函数要返回 "logprobs"


def compute_loss(model: ModelForCausalLM, prompts, responses, advantages, G, device, ref_model: ModelForCausalLM, beta):
    # 用 rollout 的结果，计算 loss
    examples = []
    for i, resp in enumerate(responses):
        prompt = prompts[i // G]
        x, y = build_xy(prompt, resp)
        examples.append((x, y))
    xs, ys = collate(examples, device)  # (B*G, L)

    if beta > 0:
        # 启动 KL 散度
        with torch.no_grad():
            ref_logp = token_logprobs(ref_model, xs, ys)
    else:
        # 不启动 KL 散度
        ref_logp = None

    # 计算 GRPO 的 loss
    logp = token_logprobs(model, xs, ys)
    mask = (ys != -100).to(torch.float32)  # (B*G, L)
    nll = (-logp * mask).sum(dim=-1) / mask.sum(dim=-1)  # (B*G,)，行内求平均，得到这一行的 loss
    loss = (advantages * nll).mean()  # 每行乘以优势，再跨行求平均

    if ref_logp is not None:
        d = ref_logp - logp  # (B*G, L)
        kl_tok = torch.exp(d) - d - 1  # (B*G, L)
        kl = (kl_tok * mask).sum(dim=-1) / mask.sum(dim=-1)  # 先行内求平均
        kl = kl.mean()  # 再跨行求平均
        loss = loss + beta * kl  # loss 加上 kl
    else:
        kl = torch.zeros((), device=device)  # 占位，保证 kl.item() 的打印

    return loss, kl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--log", default="logs/grpo.log")
    args = parser.parse_args()
    setup_log(args.log)
    log(f"args: {args}")

    train, test = split_pairs(SEED, N_TEST)

    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.init, device)
    model.eval()
    G = 8  # 重复推理 G 次
    B = 64  # batch

    # 加载参考模型
    beta = 0.2
    ref_model = None
    if beta > 0:
        ref_model = load_model(args.init, device)
        ref_model.eval()
        ref_model.requires_grad_(False)

    # GRPO 训练
    train_config = TrainConfig(
        max_lr=2e-5,
        min_lr=2e-6,
        warmup_steps=20,
        max_steps=args.max_steps,
        eval_interval=50,
    )
    optimizer = configure_optimizer(model, train_config)
    rng = random.Random(SEED)
    model.eval()
    accuracy_rate = evaluate_accuracy(model, test, 100, device, show_wrong=False)
    log(f"step:0, accuracy_rate:{accuracy_rate:.4f}")

    # 训练循环
    for step in range(train_config.max_steps):
        pairs = rng.sample(train, B)  # 抽 B 道题
        model.eval()
        prompts, responses, rewards, advantages = rollout(model, pairs, G, device)  # rollout 得到 G 组结果
        model.train()
        optimizer.zero_grad()
        loss, kl = compute_loss(model, prompts, responses, advantages, G, device, ref_model, beta)  # GRPO 计算 loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)  # 梯度裁剪
        
        lr = get_lr(step, train_config)  # 每一步动态设置 lr
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()

        # 打印 step、平均奖励、有对有错的比例、loss
        rewards = rewards.view(B, G)
        n_correct = rewards.sum(dim=-1)  # (B,) 每道题在 G 次采样里答对的次数
        mixed = (n_correct > 0) & (n_correct < G)  # (B,) 是否有对有错的 mask
        mixed_ratio = mixed.to(torch.float32).mean()
        log(f"step:{step+1}, rewards_avg:{rewards.mean():.4f}, mixed_ratio:{mixed_ratio.item():.4f}, loss:{loss.item():.4f}, kl:{kl.item():.2e}")
        # 定期用测试集评估
        if (step + 1) % train_config.eval_interval == 0:
            model.eval()
            accuracy_rate = evaluate_accuracy(model, test, 100, device, show_wrong=False)
            log(f"step:{step+1}, accuracy_rate:{accuracy_rate:.4f}")

    # 保存模型
    save_path = args.out
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.config),
    }
    torch.save(checkpoint, save_path)
