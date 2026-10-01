import torch
from sft_data import split_pairs, make_question, SEED, N_TEST
from tokenizer import render_prompt, enc, IM_END
from model import ModelForCausalLM, load_model
from generate import generate


def evaluate_accuracy(model: ModelForCausalLM, pairs, B, device):
    # 对于 a + b 计算题，用 generate 推理出答案，并计算正确率
    count, wrong_answer = 0, []
    for i in range(0, len(pairs), B):
        part_pairs = pairs[i:i + B]  # 每次做一个 batch
        prompts = []

        # 把 pairs 拼接成 prompts
        for a, b in part_pairs:
            prompt = render_prompt(make_question(a, b))
            prompts.append(prompt)

        # 变成张量，送入 generate 推理
        x = torch.tensor(prompts, dtype=torch.long, device=device)  # (B, T)
        y = generate(model, x, max_new_tokens=5, temperature=0, top_k=None)  # 贪心不用 top_k；新生成 5 个 token

        # 取出新生成的 5 个 token，答案就在里面
        T = x.shape[1]
        out = y[:, T:].tolist()  # (B, 5)

        # 逐行验证答案
        for row, (a, b) in zip(out, part_pairs):
            if IM_END in row:
                idx = row.index(IM_END)
                row = row[:idx]  # 把包含 IM_END 在内的后续 token 删掉
            text = enc.decode(row)
            # 提取答案
            try:
                ans = int(text.strip())
            except ValueError:
                ans = None
            # 如果答对，计数加一
            if ans == a + b:
                count += 1
            # 答错把 (a, b, text) 存入错题列表
            else:
                wrong_answer.append((a, b, text))

    # 打印错题表的前 10 条
    n = min(10, len(wrong_answer))
    for i in range(n):
        a, b, text = wrong_answer[i]
        print(f"{a} + {b} = {repr(text)}（正确 {a + b}）")
    return count / len(pairs)  # 返回正确率


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 拿到 500 个 test 题
    train, test = split_pairs(SEED, N_TEST)
    lengths = set()  # 收集长度，确保 500 道题一样长
    for a, b in test:
        question = make_question(a, b)
        prompt = render_prompt(question)
        lengths.add(len(prompt))
    assert len(lengths) == 1, f"prompt 长度不一致：{lengths}"

    model = load_model("checkpoints/sft.pt", device)
    model.eval()
    accuracy = evaluate_accuracy(model, test, 100, device)
    print(f"accuracy rate={accuracy}")