from parallel.zero import partition_params, ZeroOptimizer
import torch, pytest
from model import ModelForCausalLM
from pretrain import configure_optimizer, load_checkpoint, train
from dataloader import DataLoader
from config import TrainConfig, DataConfig

B, T = 2, 8


def test_partition():
    assert partition_params([5, 9, 2, 9, 4], 2) == ([0, 0, 1, 1, 1], [14, 15])
    assert partition_params([5, 9, 2, 9, 4], 1) == ([0, 0, 0, 0, 0], [29])


def build(optimizer_name, tiny_config, make_data_dir, zero_stage, n_train=2, tokens_per_shard=None, val_same_as_train=False, max_steps=10):
    # 初始化模型、优化器、loader、train_config
    torch.manual_seed(400)
    device = "cpu"
    if tokens_per_shard is None:
        tokens_per_shard = 2*(B*T+1) 
    data_dir = make_data_dir(n_train=n_train, tokens_per_shard=tokens_per_shard, val_same_as_train=val_same_as_train)
    config = tiny_config
    model = ModelForCausalLM(config)
    train_config = TrainConfig(
        warmup_steps=3,
        max_steps=max_steps,
        eval_interval=5,
    )
    data_config = DataConfig(
        B=B,
        T=T,
        eval_steps=2,
        global_batch_size=32,
    )
    optimizer = configure_optimizer(model, train_config, optimizer_name, zero_stage=zero_stage)
    loader = DataLoader(data_dir, "train", B, T, device)
    val_loader = DataLoader(data_dir, "val", B, T, device)
    return model, optimizer, loader, val_loader, train_config, data_config


def run_train(optimizer_name, tmp_path, tiny_config, make_data_dir, n_train=2, tokens_per_shard=None, val_same_as_train=False, max_steps=10, zero_stage=1):
    model, optimizers, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage, n_train, tokens_per_shard, val_same_as_train, max_steps)
    losses, norms = train(model, optimizers, loader, val_loader, train_config, data_config, optimizer_name, tmp_path / "ckpt")
    return losses, norms, model


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_zero1_matches_plain(optimizer_name, tiny_config, make_data_dir):
    # zero_stage 0 和 1 各跑一遍，对比
    zero_stage = 0
    model_0, optimizer, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage)
    losses_0, norms_0 = train(model_0, optimizer, loader, val_loader, train_config, data_config, optimizer_name, None)
    zero_stage = 1

    model_1, optimizer, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage)
    losses_1, norms_1 = train(model_1, optimizer, loader, val_loader, train_config, data_config, optimizer_name, None)
    assert isinstance(optimizer, ZeroOptimizer)
    
    assert losses_0 == losses_1
    assert norms_0 == norms_1

    for p_0, p_1 in zip(model_0.parameters(), model_1.parameters(), strict=True):
        assert torch.equal(p_0, p_1)
    
        
@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_zero1_resume(optimizer_name, tmp_path, tiny_config, make_data_dir):
    # 测试断点续训
    ## 完整训练
    losses_A, norms_A, model_A = run_train(optimizer_name, tmp_path, tiny_config, make_data_dir, zero_stage=1)
    ## 断点续训
    model_B, optimizers, loader, val_loader, train_config, data_config = build(optimizer_name, tiny_config, make_data_dir, zero_stage=1)  # 初始化
    start_step = load_checkpoint(tmp_path / "ckpt" / "latest.pt", model_B, optimizers, optimizer_name, loader, "cpu")  # 加载 checkpoint
    assert start_step == 5  # 默认配置，checkpoint 在 step==5 保存
    losses_B, norms_B = train(model_B, optimizers, loader, val_loader, train_config, data_config, optimizer_name, tmp_path / "ckpt_resume", start_step)  # 断点续训，注意存的路径要和 A 分开
    assert losses_A[start_step:] == losses_B
    assert norms_A[start_step:] == norms_B
    for p_A, p_B in zip(model_A.parameters(), model_B.parameters(), strict=True):
        assert torch.equal(p_A, p_B)

