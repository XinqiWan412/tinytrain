import torch


def newton_schulz(G: torch.Tensor, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transpose = G.size(-2) > G.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0.1, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    u = g + momentum * buf
                else:
                    u = buf
                u = newton_schulz(u)
                p.mul_(1 - lr * weight_decay)
                scale = 0.2 * (max(p.size(0), p.size(1)) ** 0.5)
                p.add_(u, alpha=-lr*scale)
                

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.rand((768, 2304), device=device)
    ns = newton_schulz(a)
    s1 = torch.linalg.svdvals(a)
    s2 = torch.linalg.svdvals(ns.float())
    print(f"原矩阵：min={s1.min().item():.4f}, max={s1.max().item():.4f}")
    print(f"正交化后：min={s2.min().item():.4f}, max={s2.max().item():.4f}")
    