from muon import newton_schulz, MuonWithAdamW
import torch
import torch.nn as nn

def test_newton_schulz():
    torch.manual_seed(400)
    a = torch.rand((128, 384))
    u = newton_schulz(a).float()
    s = torch.linalg.svdvals(u)
    assert s.min() >= 0.6 and s.max() <= 1.2


def test_adamw_branch_matches_torch():
    lr = 0.1
    weight_decay = 0.5
    eps = 1e-2
    betas = (0.9, 0.95)
    steps = 10

    torch.manual_seed(400)
    w_a = nn.Parameter(torch.randn(8, 16))
    v_a = nn.Parameter(torch.randn(16,))
    w_b = nn.Parameter(w_a.detach().clone())
    v_b = nn.Parameter(v_a.detach().clone())
    pairs = [(w_a, w_b), (v_a, v_b)]

    muon_optim = MuonWithAdamW([
        {"params": [w_a], "use_muon": False, "lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps},
        {"params": [v_a], "use_muon": False, "lr": lr, "weight_decay": 0.0, "betas": betas, "eps": eps},
    ])
    adamw_optim = torch.optim.AdamW([
        {"params": [w_b], "weight_decay": weight_decay, "lr": lr, "betas": betas, "eps": eps},
        {"params": [v_b], "weight_decay": 0.0, "lr": lr, "betas": betas, "eps": eps},
    ])
    
    for i in range(steps):
        for a, b in pairs:
            # 给两个优化器同样的梯度
            g = torch.randn_like(a)
            a.grad = g.clone()
            b.grad = g.clone()
        muon_optim.step()
        adamw_optim.step()
        for j, (a, b) in enumerate(pairs):
            assert torch.allclose(a, b, rtol=1e-6, atol=1e-6), f"step {i}, param {j}"
        