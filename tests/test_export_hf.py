import pytest, torch
pytest.importorskip("transformers")
from transformers import AutoModelForCausalLM
from export_hf import export_hf
from model import ModelForCausalLM


def test_export_hf_matches_transformers(tiny_config, tmp_path):
    # 建一个小模型
    torch.manual_seed(400)
    model = ModelForCausalLM(tiny_config)

    # 让一维的参数随机
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 1:
                p.copy_(1 + 0.3 * torch.randn_like(p))

    # 导出
    path = tmp_path / "hf"
    export_hf(model, path)

    # 用 transformers 读回
    hf_model, info = AutoModelForCausalLM.from_pretrained(
        str(path), 
        dtype=torch.float32, 
        output_loading_info=True
    )
    assert hf_model.dtype == torch.float32
    assert not info["missing_keys"], info["missing_keys"]
    assert not info["unexpected_keys"], info["unexpected_keys"]

    # 造输入
    x = torch.randint(0, tiny_config.vocab_size, (2, tiny_config.max_seq_len))

    # 算两边的 logits
    with torch.no_grad():
        logits, _ = model(x)
        hf_logits = hf_model(x).logits
    
    # 比较
    cmp = (logits - hf_logits).abs().max().item()
    assert torch.allclose(logits, hf_logits, atol=1e-5), f"最大差值为 {cmp}"
