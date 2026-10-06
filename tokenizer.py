import tiktoken
import json
from pathlib import Path


# 读取 tokenizer 文件
path = Path(__file__).parent / "tokenizer_32k.json"
with open(path, encoding="utf-8") as f:
    _data = json.load(f)

# 重建 vocab，id -> 字节串
vocab = {i: bytes([i]) for i in range(256)}
for i, (a, b) in enumerate(_data['merges']):
    vocab[256 + i] = vocab[a] + vocab[b]

# 反转，字节串 -> id
mergeable_ranks = {}
for token_id, b in vocab.items():
    mergeable_ranks[b] = token_id

# 建 special_tokens 表，得到 字符串 -> id
special_tokens = {}
for j, sp_token in enumerate(_data["special_tokens"]):
    special_tokens[sp_token] = 256 + len(_data['merges']) + j

# 构造 enc
enc = tiktoken.Encoding(
    name="bpe32k",
    pat_str=_data["split_pattern"],
    mergeable_ranks=mergeable_ranks,
    special_tokens=special_tokens,
    explicit_n_vocab=32768,
)

EOT = enc.eot_token
IM_START, IM_END, THINK, END_THINK = enc.encode_single_token("<|im_start|>"), enc.encode_single_token("<|im_end|>"), enc.encode_single_token("<think>"), enc.encode_single_token("</think>")

def render_prompt(question: str) -> list[int]:
    # <|endoftext|><|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    return (
        [enc.eot_token]
        + [IM_START] + enc.encode_ordinary("user\n" + question) + [IM_END]
        + enc.encode_ordinary("\n")
        + [IM_START] + enc.encode_ordinary("assistant\n")
        + [THINK] + enc.encode_ordinary("\n\n") + [END_THINK]
        + enc.encode_ordinary("\n\n")
    )


def render_answer(answer: str) -> list[int]:
    # {answer}<|im_end|>
    return enc.encode_ordinary(answer) + [IM_END]


if __name__ == "__main__":
    # 下面是对 bpe32k 的检查
    assert enc.n_vocab == 32768 and enc.eot_token == 32763
    assert len(enc.encode_ordinary("12345")) == 5
    question = "What is 37 + 45?"
    answer = "82"
    tmp = render_prompt(question) + render_answer(answer)
    out = enc.decode(tmp)
    check_out = """<|endoftext|><|im_start|>user\nWhat is 37 + 45?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n82<|im_end|>"""
    print(repr(out))
    assert out == check_out

    # 拿到 1000 篇文档
    import itertools
    from train_tokenizer import iter_docs
    data_dir = "data/raw/fineweb-edu/sample/10BT"
    max_chars = 100_000_000_000
    docs = iter_docs(data_dir, max_chars)
    docs = itertools.islice(docs, 1000)

    # 加载 gpt2 作对照
    gpt2_enc = tiktoken.get_encoding("gpt2")

    # 对每篇文档做两件事
    total_bytes = tokens = gpt2_tokens = 0
    for doc in docs:
        ids = enc.encode_ordinary(doc)
        assert enc.decode(ids) == doc  # 检查 encode + decode 和原文相等
        ids_gpt2 = gpt2_enc.encode_ordinary(doc)
        total_bytes += len(doc.encode("utf-8"))  # 文档总字节数
        tokens += len(ids)  # 新分词器总 token 数
        gpt2_tokens += len(ids_gpt2)  # gpt2 的 token 数
    
    # 打印两个比值
    print(f"bpe32k 每 token 字节数：{total_bytes / tokens}")
    print(f"gpt2 每 token 字节数：{total_bytes / gpt2_tokens}")
