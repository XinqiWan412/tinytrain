import copy
from grpo import compute_loss, train_grpo
from pretrain import TrainConfig, configure_optimizer
from sft import train_sft
from tokenizer import render_prompt, render_answer, enc
from sft_data import SEED, make_question
from model import ModelForCausalLM
import pytest, torch

GRPO_PAIRS = [(3, 5), (37, 45), (9, 80), (12, 7)]
SFT_STEPS = 200  # 30/60/100 步时采样全错、mixed_ratio 全为 0，200 步才出现有对有错
GRPO_BASELINE = {
    "adamw": {
        "losses":       [-0.7848182320594788, -1.767622385174036, -0.38169796764850616, -1.145300004631281],
        "kls":          [0.0, 0.6124992445111275, 0.31461939960718155, 0.19988284539431334],
        "norms":        [2.1468207836151123, 5.698629856109619, 0.9374479055404663, 4.356701374053955],
        "rewards_avgs": [0.625, 0.3125, 0.5625, 0.375],
    },
    "muon": {
        "losses":       [-0.7848182320594788, -0.880949811078608, -0.2750803187489055, -2.506251037120819],
        "kls":          [0.0, 0.0039187169168144464, 0.01174536591861397, 0.04092317167669535],
        "norms":        [2.1468207836151123, 2.0313706398010254, 2.0404276847839355, 4.824938774108887],
        "rewards_avgs": [0.625, 0.875, 0.875, 0.75],
    },
}


def test_compute_loss_chunked_match_full(sft_config):
    config = sft_config
    torch.manual_seed(400)
    model = ModelForCausalLM(config)
    torch.manual_seed(401)
    ref_model = ModelForCausalLM(config)
    ref_model.requires_grad_(False)

    pairs = [(3, 5), (37, 45), (9, 80)]
    prompts = [render_prompt(make_question(a, b)) for a, b in pairs]
    
    G = 2
    N = len(prompts) * G
    
    responses = [
        render_answer("8"),
        render_answer("17"),
        render_answer("82"),
        enc.encode_ordinary("12345"),
        render_answer("89"),
        render_answer("9"),
    ]

    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
    beta = 0.2
    # 不分块跑一次
    model.zero_grad()
    loss_full, kl_full = compute_loss(model, prompts, responses, advantages, G, "cpu", ref_model, beta, chunk_size=N)
    grads_full = [p.grad.clone() for p in model.parameters()]
    
    # 分块跑一次
    model.zero_grad()
    ## 注意这里把 chunk_size 从 N 改成 4，就变成分块了
    loss_chunk, kl_chunk = compute_loss(model, prompts, responses, advantages, G, "cpu", ref_model, beta, chunk_size=4)

    # 比较
    assert loss_chunk == pytest.approx(loss_full, abs=1e-5)
    assert kl_chunk == pytest.approx(kl_full, abs=1e-5)
    assert kl_full > 0  # 如果 kl 为 0,说明传入的 model 和 ref_model 一样了
    for p, g in zip(model.parameters(), grads_full, strict=True):
        assert torch.allclose(p.grad, g, atol=1e-6)


def build_grpo_start(config, sft_steps):
    torch.manual_seed(400)
    model = ModelForCausalLM(config)
    sft_train_config = TrainConfig(warmup_steps=3, max_steps=sft_steps)
    optimizers = configure_optimizer(model, sft_train_config, "adamw")
    train_sft(model, optimizers, sft_train_config, "cpu", GRPO_PAIRS, GRPO_PAIRS, B=4, seed=SEED)
    return model


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_grpo_baseline(sft_config, optimizer_name):
    model = build_grpo_start(sft_config, SFT_STEPS)
    ref_model = copy.deepcopy(model)
    ref_model.requires_grad_(False)

    grpo_train_config = TrainConfig(warmup_steps=1, max_steps=4)
    optimizers = configure_optimizer(model, grpo_train_config, optimizer_name)
    torch.manual_seed(0)
    losses, kls, norms, rewards_avgs, mixed_ratios = train_grpo(model, ref_model, optimizers, grpo_train_config,
                                                            "cpu", GRPO_PAIRS, GRPO_PAIRS, B=4, G=4, beta=0.2, chunk_size=6, seed=SEED)
    assert losses == pytest.approx(GRPO_BASELINE[optimizer_name]['losses'], rel=1e-5)
    assert kls == pytest.approx(GRPO_BASELINE[optimizer_name]['kls'], rel=1e-5)
    assert norms == pytest.approx(GRPO_BASELINE[optimizer_name]['norms'], rel=1e-5)
    assert rewards_avgs == pytest.approx(GRPO_BASELINE[optimizer_name]['rewards_avgs'], rel=1e-5)
    assert any(r > 0 for r in mixed_ratios)  # 至少有一步出现有对有错的题
