from transformers import Qwen3Config
from model import ModelConfig, ModelForCausalLM, load_model
import os
from safetensors.torch import save_file
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    m = load_model(args.ckpt, "cpu")
    export_hf(m, args.out)
    print(f"从 {args.ckpt} 导出到了 {args.out}")
