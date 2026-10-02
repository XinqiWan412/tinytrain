from tokenizer import IM_END, render_answer, render_prompt, enc
import random


SEED = 400
N_TEST = 500


def make_question(a, b):
    return f"What is {a} + {b}?"


def build_xy(prompt_ids, completion_ids):
    P = len(prompt_ids)
    ids = prompt_ids + completion_ids
    x = ids[:-1]
    y = ids[1:]
    y[:P-1] = [-100] * (P - 1)
    return x, y


def build_example(a, b):
    question = make_question(a, b)
    prompt = render_prompt(question)
    answer = str(a + b)
    return build_xy(prompt, render_answer(answer))


def split_pairs(seed, n_test=500):
    # 生成数字对的 list，例如 [(1, 2), (23, 41)]
    pairs = []
    for a in range(100):
        for b in range(100):
            pairs.append((a, b))
    random.Random(seed).shuffle(pairs)  # 打乱顺序
    test, train = pairs[:n_test], pairs[n_test:]
    return train, test


def parse_answer(token_ids):
    # 输入新生成的 token 列表，返回答案数字
    if IM_END not in token_ids:
        # 如果没有生成 IM_END，返回 None
        return None

    # 截掉 IM_END 及之后的部分
    idx = token_ids.index(IM_END)
    token_ids = token_ids[:idx]

    # 检查有没有空位 id
    if any(t >= enc.n_vocab for t in token_ids):
        return None
    
    # 解码，如果是数字，返回数字，不是返回 None
    text = enc.decode(token_ids)
    try:
        ans = int(text.strip())
        return ans
    except ValueError:
        return None


if __name__ == "__main__":
    n_test = N_TEST
    train, test = split_pairs(SEED, n_test)
    assert len(train) == 10000 - n_test and len(test) == n_test
    assert not set(train) & set(test)  # train 和 test 不能有交集

    a, b = test[0]
    x, y = build_example(a, b)
    print(f"sample: {a} + {b} = {a + b}\n")

    for i, (xi, yi) in enumerate(zip(x, y)):
        x_str = repr(enc.decode([xi]))
        y_str = "MASK" if yi == -100 else repr(enc.decode([yi]))
        print(f"{i:3d} | x={x_str:20s} | y={y_str}")

    assert len(x) == len(y)
    n_target = sum(1 for t in y if t != -100)
    print(f"\nlen={len(x)}, 参与 loss 的位置数={n_target}")
