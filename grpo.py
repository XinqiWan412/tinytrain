import random, os
from dataclasses import asdict
import torch
import torch.nn.functional as F
from generate import generate
from model import ModelForCausalLM, load_model
from sft_data import build_xy, split_pairs, SEED, N_TEST, make_question, parse_answer
from tokenizer import render_prompt, IM_END
from sft import collate
from trainer import Trainer
from pretrain import configure_optimizer
from eval_sft import evaluate_accuracy
from collections import defaultdict
import argparse
from logger import log, setup_log
from config import TrainConfig, GRPOConfig


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


def build_rows(prompts, responses, G, device):
    examples = []
    for i, resp in enumerate(responses):
        prompt = prompts[i // G]
        x, y = build_xy(prompt, resp)
        examples.append((x, y))
    xs, ys = collate(examples, device)  # (B*G, L)
    return xs, ys


def chunk_loss(model, ref_model, xs_c, ys_c, adv_c, N, beta):
    if beta > 0:
        # 启动 KL 散度
        with torch.no_grad():
            ref_logp = token_logprobs(ref_model, xs_c, ys_c)
    else:
        # 不启动 KL 散度
        ref_logp = None
        kl = 0.0

    # 计算 GRPO 的 loss
    logp = token_logprobs(model, xs_c, ys_c)
    mask = (ys_c != -100).to(torch.float32)  # (C, L)
    nll = (-logp * mask).sum(dim=-1) / mask.sum(dim=-1)  # (C,)，行内求平均，得到这一行的 loss
    loss = (adv_c * nll).sum() / N  # 每行乘以优势，再跨行求平均

    if ref_logp is not None:
        # ref_logp 非空，启动 kl 散度
        d = ref_logp - logp  # (C, L)
        kl_tok = torch.exp(d) - d - 1  # (C, L)
        kl = (kl_tok * mask).sum(dim=-1) / mask.sum(dim=-1)  # 先行内求平均
        kl = kl.sum() / N  # 再跨行求平均
        loss = loss + beta * kl  # loss 加上 kl
        kl = kl.item()
    return loss, kl


def compute_loss(model: ModelForCausalLM, prompts, responses, advantages, G, device, ref_model: ModelForCausalLM, beta, chunk_size):
    # 已经做完 rollout 了，用 rollout 的结果计算 loss
    xs, ys = build_rows(prompts, responses, G, device)
    N = xs.shape[0]  # 总行数
    loss_sum, kl_sum = 0.0, 0.0
    for i in range(0, N, chunk_size):
        # 切出一块
        xs_c = xs[i:i+chunk_size]
        ys_c = ys[i:i+chunk_size]
        adv_c = advantages[i:i+chunk_size]
        # 算 loss，kl
        loss, kl = chunk_loss(model, ref_model, xs_c, ys_c, adv_c, N, beta)
        kl_sum += kl
        loss.backward()
        loss_sum += loss.item()
    return loss_sum, kl_sum


class GRPOTask:
    def __init__(self, model, ref_model, train_pairs, test_pairs, B, G, beta, chunk_size, device, seed):
        self.model = model
        self.ref_model = ref_model
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs
        self.B = B
        self.G = G
        self.beta = beta
        self.chunk_size = chunk_size
        self.device = device
        self.rng = random.Random(seed)
        self.log_formats = {"kl": ".2e"}

    def get_parts(self, step):
        pairs = self.rng.sample(self.train_pairs, self.B)  # 抽 B 道题
        self.model.eval()
        prompts, responses, rewards, advantages = rollout(self.model, pairs, self.G, self.device)  # rollout 得到 G 组结果
        xs, ys = build_rows(prompts, responses, self.G, self.device)
        N, chunk_size = xs.shape[0], self.chunk_size
        parts = [(xs[i:i+chunk_size], ys[i:i+chunk_size], advantages[i:i+chunk_size], N) for i in range(0, N, chunk_size)]
        rewards = rewards.view(self.B, self.G)
        n_correct = rewards.sum(dim=-1)  # (B,) 每道题在 G 次采样里答对的次数
        mixed = (n_correct > 0) & (n_correct < self.G)  # (B,) 是否有对有错的 mask
        mixed_ratio = mixed.to(torch.float32).mean().item()
        rewards_avg = rewards.mean().item()
        return parts, {"rewards_avg": rewards_avg, "mixed_ratio": mixed_ratio}

    def loss(self, part):
        xs_c, ys_c, adv_c, N = part
        loss, kl = chunk_loss(self.model, self.ref_model, xs_c, ys_c, adv_c, N, self.beta)
        return loss, {"kl": kl}
    
    def evaluate(self):
        accuracy_rate = evaluate_accuracy(self.model, self.test_pairs, 100, self.device, show_wrong=False)
        return {"accuracy_rate": accuracy_rate}


def train_grpo(model, ref_model, optimizer, train_config, device,
                train_pairs, test_pairs,
                B, G, beta, chunk_size, seed):
    task = GRPOTask(model, ref_model, train_pairs, test_pairs, B, G, beta, chunk_size, device, seed)
    trainer = Trainer(model, optimizer, train_config, task)
    records = trainer.fit()
    return records["loss"], records["kl"], records["norm"], records["rewards_avg"], records["mixed_ratio"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--log", default="logs/grpo.log")
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    args = parser.parse_args()
    setup_log(args.log)
    log(f"args: {args}")
    torch.manual_seed(SEED)
    train, test = split_pairs(SEED, N_TEST)

    grpo_config = GRPOConfig(chunk_size=args.chunk_size)

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.init, device)
    model.eval()

    # 加载参考模型
    ref_model = None
    if grpo_config.beta > 0:
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
        log_interval=1,
        optimizer=args.optimizer,
    )
    optimizer = configure_optimizer(model, train_config, args.optimizer)
    losses, kls, norms, rewards_avgs, mixed_ratios = train_grpo(
        model, ref_model, optimizer, train_config, device,
        train, test, grpo_config.B, grpo_config.G, grpo_config.beta, grpo_config.chunk_size, SEED
    )

    # 保存模型
    save_path = args.out
    checkpoint = {
        "model": model.state_dict(),
        "model_config": asdict(model.config),
    }
    tmp_path = save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)
