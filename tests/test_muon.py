from muon import newton_schulz
import torch

def test_newton_schulz():
    torch.manual_seed(400)
    a = torch.rand((128, 384))
    u = newton_schulz(a).float()
    s = torch.linalg.svdvals(u)
    assert s.min() >= 0.6 and s.max() <= 1.2
