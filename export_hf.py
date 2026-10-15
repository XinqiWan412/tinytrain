from transformers import Qwen3Config, AutoModelForCausalLM
from model import ModelConfig, ModelForCausalLM, load_model
import os
from safetensors.torch import save_file
import argparse, torch
import torch.nn.functional as F
from tokenizer import enc, render_prompt
from generate import generate
from sft_data import split_pairs, SEED, N_TEST, make_question, parse_answer

CHECK_TEXT = "The water cycle is a natural process that moves water between the Earth's surface, atmosphere, and underground areas through evaporation, condensation, and precipitation."


def make_hf_config(cfg: ModelConfig):
    return Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        rms_norm_eps=cfg.rms_norm_eps,
        tie_word_embeddings=cfg.tie_word_embeddings,
        max_position_embeddings=cfg.max_seq_len,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        use_sliding_window=False,
        architectures=["Qwen3ForCausalLM"],
    )


def export_hf(model: ModelForCausalLM, out_dir):
    # 把 .pt 格式导出成 .safetensors 格式
    # 导出 config 文件
    cfg = make_hf_config(model.config)
    cfg.save_pretrained(out_dir)

    # 导出模型权重文件
    state_dict = model.state_dict()
    if model.config.tie_word_embeddings:
        del state_dict["lm_head.weight"]
    new_state_dict = {k:v.cpu() for k, v in state_dict.items()}

    path = os.path.join(out_dir, "model.safetensors")
    save_file(new_state_dict, path, metadata={"format": "pt"})


def load_hf(out_dir):
    # 加载 HF 模型
    hf_model, info = AutoModelForCausalLM.from_pretrained(
        str(out_dir),
        dtype=torch.float32,
        output_loading_info=True
    )
    assert not info["missing_keys"], info["missing_keys"]
    assert not info["unexpected_keys"], info["unexpected_keys"]
    return hf_model


def compare_with_hf(model: ModelForCausalLM, hf_model, text):
    # model 和 hf_model 比较 loss 和 logits
    # 编码文本
    token_ids = [enc.eot_token] + enc.encode_ordinary(text)
    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # (1, T)
    assert x.shape[1] <= model.config.max_seq_len

    # 算两边的 logits
    with torch.no_grad():
        logits, _ = model(x)
        hf_logits = hf_model(x).logits
    
    # 算最大差
    max_abs_diff = (logits - hf_logits).abs().max().item()

    # 算两边的 loss
    loss = F.cross_entropy(logits[0, :-1], x[0, 1:]).item()
    hf_loss = F.cross_entropy(hf_logits[0, :-1], x[0, 1:]).item()

    # 打印结果
    print(f"tokens:{len(token_ids)}, logits_max_abs_diff:{max_abs_diff}")
    print(f"loss:{loss}, hf_loss:{hf_loss}, loss_diff:{loss - hf_loss}")


def compare_generate(model, hf_model, n=8):
    # model 和 hf_model 比较推理输出
    # 拼输入 x
    _, test = split_pairs(SEED, N_TEST)
    test = test[:n]
    same, correct = 0, 0
    for a, b in test:
        prompt = render_prompt(make_question(a, b))
        x = torch.tensor(prompt, dtype=torch.long).unsqueeze(0)  # (1, T)
        T = x.shape[1]

        # model 生成
        idx = generate(model, x, max_new_tokens=5, temperature=0, top_k=None)
        idx = idx[0, T:].tolist()
        ans = parse_answer(idx)

        # hf_model 生成
        hf_idx = hf_model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=5, do_sample=False)
        hf_idx = hf_idx[0, T:].tolist()
        hf_ans = parse_answer(hf_idx)

        # 比较
        if idx == hf_idx:
            same += 1
        if ans == a + b:
            correct += 1

        # 打印
        print(f"题目：{a} + {b} = ? model: {ans}, hf_model: {hf_ans}, is_equal: {idx == hf_idx}")
    
    print(f"一致 {same} / {n}，答对 {correct} / {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gen", action="store_true")
    args = parser.parse_args()

    m = load_model(args.ckpt, "cpu")
    export_hf(m, args.out)
    print(f"从 {args.ckpt} 导出到了 {args.out}")

    hf_model = load_hf(args.out)
    compare_with_hf(m, hf_model, CHECK_TEXT)
    if args.gen:
        compare_generate(m, hf_model)
    