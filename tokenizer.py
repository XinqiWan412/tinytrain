import tiktoken

_gpt2 = tiktoken.get_encoding("gpt2")

EOT = 50256
IM_START, IM_END, THINK, END_THINK = 50257, 50258, 50259, 50260

enc = tiktoken.Encoding(
    name="gpt2_qwen3_chat",
    pat_str=_gpt2._pat_str,
    mergeable_ranks=_gpt2._mergeable_ranks,
    special_tokens={
        **_gpt2._special_tokens,  # <|endoftext|>: 50256
        "<|im_start|>": IM_START,
        "<|im_end|>": IM_END,
        "<think>": THINK,
        "</think>": END_THINK,
    },
)


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
