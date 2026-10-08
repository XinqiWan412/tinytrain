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
