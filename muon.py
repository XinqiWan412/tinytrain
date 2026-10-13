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


class MuonWithAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            assert "lr" in group and "weight_decay" in group
            group["params"] = list(group["params"])
            if group["use_muon"]:
                # muon
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                keys = group.keys()
                expected = {"params", "use_muon", "lr", "weight_decay", "momentum", "nesterov"}
                assert keys == expected, (
                    f"Muon param_group keys mismatch: "
                    f"expected {sorted(expected)}, got {sorted(keys)}; "
                    f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
                )
                assert all(len(param.shape) == 2 for param in group["params"])
            else:
                # adamw
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                keys = group.keys()
                expected = {"params", "use_muon", "lr", "weight_decay", "betas", "eps"}
                assert keys == expected, (
                    f"AdamW param_group keys mismatch: "
                    f"expected {sorted(expected)}, got {sorted(keys)}; "
                    f"missing={sorted(expected - keys)}, extra={sorted(keys - expected)}"
                )
        super().__init__(param_groups, {})

    @torch.no_grad()    
    def step(self):
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
    
    def _muon_step(self, group):
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

    def _adamw_step(self, group):
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        eps = group["eps"]
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            g = p.grad
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
            state["step"] += 1
            # p = p*(1 - lr*wd)
            p.mul_(1 - lr * weight_decay)
            # m = b1*m + (1-b1)*g
            state["exp_avg"].lerp_(g, 1-beta1)
            # v = b2*v + (1-b2)*g*g
            state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1-beta2)
            # bias_correction
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            # step_size = lr / (1 - b1^t)
            step_size = lr / bias_correction1
            # denom = (sqrt(v) / sqrt(1 - b2^t)) + eps
            bias_correction2_sqrt = bias_correction2 ** 0.5
            denom = (state["exp_avg_sq"].sqrt() / bias_correction2_sqrt).add_(eps)
            # p = p - step_size * m / denom，数学上等价原公式
            p.addcdiv_(state["exp_avg"], denom, value=-step_size)
            