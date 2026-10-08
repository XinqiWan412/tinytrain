from pretrain import get_lr, TrainConfig
import math, pytest


def test_get_lr():
    train_config = TrainConfig(warmup_steps=30, max_steps=300)

    assert get_lr(0, train_config) == pytest.approx(1e-3/30)
    assert get_lr(29, train_config) == pytest.approx(1e-3)
    assert get_lr(30, train_config) == pytest.approx(1e-3)
    assert get_lr(165, train_config) == pytest.approx(5.5e-4)
    assert get_lr(299, train_config) == pytest.approx(1e-4 + 0.5*(1+math.cos(math.pi*269/270)) * (1e-3 - 1e-4))
    assert get_lr(300, train_config) == pytest.approx(1e-4)
