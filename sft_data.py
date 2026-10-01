from tokenizer import render_answer, render_prompt, enc
import random


def build_example(a, b):
    question = f"What is {a} + {b}?"
    answer = str(a + b)
    prompt = render_prompt(question)
    P = len(prompt)
    ids = prompt + render_answer(answer)
    x = ids[:-1]
    y = ids[1:]
    y[:P-1] = [-100] * (P - 1)
    return x, y


def split_pairs(seed, n_test=500):
    pairs = []
    for a in range(100):
        for b in range(100):
            pairs.append((a, b))
    random.Random(seed).shuffle(pairs)  # 打乱顺序
    test, train = pairs[:n_test], pairs[n_test:]
    return train, test


if __name__ == "__main__":
    n_test = 500
    train, test = split_pairs(400, n_test)
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
