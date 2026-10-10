from pretrain import get_lr
import math, pytest
import torch
from model import ModelForCausalLM, load_model
from pretrain import configure_optimizer, TrainConfig, train, load_checkpoint
from dataloader import DataLoader

B, T = 2, 8


def test_get_lr():
    train_config = TrainConfig(warmup_steps=30, max_steps=300)

    assert get_lr(0, train_config) == pytest.approx(1e-3/30)
    assert get_lr(29, train_config) == pytest.approx(1e-3)
    assert get_lr(30, train_config) == pytest.approx(1e-3)
    assert get_lr(165, train_config) == pytest.approx(5.5e-4)
    assert get_lr(299, train_config) == pytest.approx(1e-4 + 0.5*(1+math.cos(math.pi*269/270)) * (1e-3 - 1e-4))
    assert get_lr(300, train_config) == pytest.approx(1e-4)


def build(optimizer_name, tmp_path, tiny_config, make_data_dir, n_train=2, tokens_per_shard=None, val_same_as_train=False, max_steps=10):
    # 初始化模型、优化器、loader、train_config
    torch.manual_seed(400)
    device = "cpu"
    if tokens_per_shard is None:
        tokens_per_shard = 2*(B*T+1) 
    data_dir = make_data_dir(n_train=n_train, tokens_per_shard=tokens_per_shard, val_same_as_train=val_same_as_train)
    config = tiny_config
    model = ModelForCausalLM(config)
    train_config = TrainConfig(
        B=B,
        T=T,
        warmup_steps=3,
        max_steps=max_steps,
        eval_interval=5,
        eval_steps=2,
        global_batch_size=32,
    )
    optimizers = configure_optimizer(model, train_config, optimizer_name)
    loader = DataLoader(data_dir, "train", B, T, device)
    val_loader = DataLoader(data_dir, "val", B, T, device)
    return model, optimizers, loader, val_loader, train_config


def run_train(optimizer_name, tmp_path, tiny_config, make_data_dir, n_train=2, tokens_per_shard=None, val_same_as_train=False, max_steps=10):
    model, optimizers, loader, val_loader, train_config = build(optimizer_name, tmp_path, tiny_config, make_data_dir, n_train, tokens_per_shard, val_same_as_train, max_steps)
    losses, norms = train(model, optimizers, loader, val_loader, train_config, optimizer_name, tmp_path / "ckpt")
    return losses, norms, model


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_resume(optimizer_name, tmp_path, tiny_config, make_data_dir):
    # 测试断点续训
    ## 完整训练
    losses_A, norms_A, model_A = run_train(optimizer_name, tmp_path, tiny_config, make_data_dir)
    ## 断点续训
    model_B, optimizers, loader, val_loader, train_config = build(optimizer_name, tmp_path, tiny_config, make_data_dir)  # 初始化
    start_step = load_checkpoint(tmp_path / "ckpt" / "latest.pt", model_B, optimizers, optimizer_name, loader, "cpu")  # 加载 checkpoint
    assert start_step == 5  # 默认配置，checkpoint 在 step==5 保存
    losses_B, norms_B = train(model_B, optimizers, loader, val_loader, train_config, optimizer_name, tmp_path / "ckpt_resume", start_step)  # 断点续训，注意存的路径要和 A 分开
    assert losses_A[start_step:] == losses_B
    assert norms_A[start_step:] == norms_B
    for p_A, p_B in zip(model_A.parameters(), model_B.parameters(), strict=True):
        assert torch.equal(p_A, p_B)


BASELINE = {
    "adamw": {
        "losses": [5.551166534423828, 5.550848960876465, 5.225472927093506, 5.185811519622803, 4.972434043884277, 4.93494987487793, 4.8002214431762695, 4.780035018920898, 4.690847396850586, 4.7038679122924805],
        "norms":  [3.397036075592041, 3.4024150371551514, 2.2352688312530518, 2.2504138946533203, 1.8677551746368408, 1.9933675527572632, 1.7898682355880737, 1.8567321300506592, 1.7527062892913818, 1.7887133359909058],
    },
    "muon": {
        "losses": [5.551166534423828, 5.5482072830200195, 5.428950309753418, 5.371600151062012, 5.223235130310059, 5.184202671051025, 5.069838047027588, 5.0545854568481445, 4.982077598571777, 4.992427349090576],
        "norms":  [3.397036075592041, 3.3974556922912598, 3.158137798309326, 3.0395596027374268, 2.6792263984680176, 2.6725707054138184, 2.390488624572754, 2.479583501815796, 2.2621512413024902, 2.39917254447937],
    },
}


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_pretrain_baseline(optimizer_name, tmp_path, make_data_dir, tiny_config):
    losses, norms, _ = run_train(optimizer_name, tmp_path, tiny_config, make_data_dir)
    expected = BASELINE[optimizer_name]
    assert losses == pytest.approx(expected["losses"], rel=1e-5)
    assert norms == pytest.approx(expected['norms'], rel=1e-5)
    

@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_overfit_single_batch(optimizer_name, tmp_path, make_data_dir, tiny_config):
    _, _, _ = run_train(optimizer_name, tmp_path, tiny_config, make_data_dir, n_train=1, tokens_per_shard=B*T+1, val_same_as_train=True, max_steps=100)
    val_loss = torch.load(tmp_path / "ckpt" / "final.pt")["val_loss"]
    assert val_loss < 1.5  # 低于 ln16 ≈ 2.77，说明模型真的用了上下文


def test_load_model(tmp_path, make_data_dir, tiny_config):
    _, _, model_A = run_train("adamw", tmp_path, tiny_config, make_data_dir)
    model_B = load_model(tmp_path / "ckpt" / "final.pt", "cpu")

    a = torch.randint(0, tiny_config.vocab_size, (B, T))
    with torch.no_grad():
        logits_A, _ = model_A(a)
        logits_B, _ = model_B(a)
    assert torch.equal(logits_A, logits_B)
