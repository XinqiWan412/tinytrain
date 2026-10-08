import torch
import random
from tokenizer import enc
from dataclasses import asdict
import argparse, os

from sft_data import build_example, split_pairs, SEED, N_TEST
from model import load_model
from pretrain import TrainConfig, configure_optimizer, get_lr, zero_grad_all, set_lr, step_all
from logger import log, setup_log


def collate(examples, device):
    # 把 examples 变成 xs, ys 两组张量
    xs, ys = [], []
    # 找到最长样本的长度 L
    L = max(len(x) for x, _ in examples)
    
    # 每条样本补齐到 L
    for x, y in examples:
        x = x + [enc.eot_token] * (L - len(x))
        y = y + [-100] * (L - len(y))
        xs.append(x)
        ys.append(y)
    
    xs = torch.tensor(xs, dtype=torch.long).to(device)  # (B, L)
    ys = torch.tensor(ys, dtype=torch.long).to(device)  # (B, L)
    return xs, ys


class SFTLoader():
    def __init__(self, pairs, B, device, seed):
        self.pairs = list(pairs)
        self.rng = random.Random(seed)
        self.pos = 0
        self.epoch = 0
        self.B = B
        self.device = device

    def next_batch(self):
        # 如果不够了，进入下一个 epoch
        if self.pos + self.B > len(self.pairs):
            self.rng.shuffle(self.pairs)  # 打乱顺序
            self.pos = 0
            self.epoch += 1
        # 取这一批
        batch = self.pairs[self.pos:self.pos + self.B]
        self.pos += self.B
        examples = []  # 例如[(1, 3), (54, 32)]
        for a, b in batch:
            x, y = build_example(a, b)
            examples.append((x, y))
        xs, ys = collate(examples, self.device)
        return xs, ys


# 测试
def evaluate(model, test, B, device):
    model.eval()
    with torch.no_grad():
        loss_accum = 0
        for i in range(0, len(test), B):
            examples = []
            batch = test[i:i+B]
            for a, b in batch:
                x, y = build_example(a, b)
                examples.append((x, y))
            xs, ys = collate(examples, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(xs, ys)
            loss_accum += loss.item()
    n = (len(test) + B - 1) // B
    loss_accum /= n  # 对 loss 求平均
    model.train()
    return loss_accum


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--log", default="logs/sft.log")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    args = parser.parse_args()
    setup_log(args.log)
    log(f"args: {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    m = load_model(args.init, device)  # 加载模型
    m.train()
    log(f"vocab_size: {m.config.vocab_size}")
    # 配置 sft 训练参数
    train_config = TrainConfig(
        max_lr=3e-4,
        min_lr=3e-5,
        warmup_steps=50,
        max_steps=args.max_steps,
        eval_interval=100,
    )
    optimizers = configure_optimizer(m, train_config, args.optimizer)  # 加载优化器
    B = 128
    train, test = split_pairs(SEED, N_TEST)  # 取 pairs
    sftloader = SFTLoader(train, B, device, SEED)

    # 训练
    for step in range(train_config.max_steps):
        zero_grad_all(optimizers)
        x, y = sftloader.next_batch()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = m(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), train_config.grad_clip)
        lr = get_lr(step, train_config)  # 每步的学习率都在变
        set_lr(optimizers, lr)
        step_all(optimizers)
        if step % 10 == 0 or step == train_config.max_steps - 1:  # 定期打印 log
            log(f"step:{step}, loss:{loss.item():.4f}, lr:{lr:.4e}")
        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:  # 定期打印 val_loss
            val_loss = evaluate(m, test, B, device)
            log(f"step={step}, val_loss={val_loss:.4f}")
    
    # 保存
    save_path = args.out
    checkpoint = {
        "model": m.state_dict(),
        "model_config": asdict(m.config),
    }
    tmp_path = save_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, save_path)
