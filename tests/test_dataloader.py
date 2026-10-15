from dataloader import DataLoader
import torch


def test_next_batch_switch_shard(make_data_dir):
    B = 2
    T = 8
    data_dir = make_data_dir(n_train=2, tokens_per_shard=2*(B*T+1))
    device = "cpu"

    # 测试换页逻辑
    loader = DataLoader(data_dir, 'train', B, T, device)
    loader.pos = len(loader.tokens) - B*T
    loader.next_batch()
    assert loader.shard_idx == 1

    loader = DataLoader(data_dir, 'val', B, T, device)
    loader.pos = len(loader.tokens) - B*T
    loader.next_batch()
    assert loader.shard_idx ==0


def test_state_dict_roundtrip(make_data_dir):
    B = 2
    T = 8
    data_dir = make_data_dir(n_train=2, tokens_per_shard=2*(B*T+1))
    device = "cpu"
    loader_A = DataLoader(data_dir, "train", B, T, device)
    for _ in range(3):
        loader_A.next_batch()
    state = loader_A.state_dict()
    x_exp, y_exp = loader_A.next_batch()

    loader_B = DataLoader(data_dir, "train", B, T, device)
    loader_B.load_state_dict(state)
    x, y = loader_B.next_batch()
    assert torch.equal(x_exp, x) and torch.equal(y_exp, y)


def test_interleave_matches_single(make_data_dir):
    B = 2
    T = 8
    data_dir = make_data_dir(n_train=2, tokens_per_shard=2 * (2*B*T) + 1)
    device = "cpu"
    single = DataLoader(data_dir, "train", B, T, device)
    r0 = DataLoader(data_dir, "train", B, T, device, rank=0, world_size=2)
    r1 = DataLoader(data_dir, "train", B, T, device, rank=1, world_size=2)

    for _ in range(6):
        x1, y1 = r0.next_batch()
        x2, y2 = r1.next_batch()
        x3, y3 = single.next_batch()
        x4, y4 = single.next_batch()

        assert torch.equal(x1, x3) and torch.equal(y1, y3)
        assert torch.equal(x2, x4) and torch.equal(y2, y4)
        assert r0.shard_idx == r1.shard_idx and r1.shard_idx == single.shard_idx

        