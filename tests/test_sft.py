from sft import train_sft
from sft_data import SEED, N_TEST, split_pairs
from pretrain import configure_optimizer
from config import TrainConfig
from model import ModelForCausalLM
import torch, pytest


BASELINE = {
    "adamw": {
        "losses": [10.371084213256836, 10.182242393493652, 9.901473045349121, 9.844178199768066, 9.703195571899414, 9.621984481811523, 9.49390697479248, 9.384000778198242, 9.36064338684082, 9.380940437316895],
        "norms":  [6.238448619842529, 5.266232013702393, 4.40305757522583, 4.1445136070251465, 3.5362043380737305, 3.8555550575256348, 3.574216365814209, 3.9052352905273438, 3.8794147968292236, 3.6433818340301514],
    },
    "muon": {
        "losses": [10.371084213256836, 10.315914154052734, 10.141154289245605, 10.052567481994629, 9.890023231506348, 9.80479621887207, 9.66412353515625, 9.559350967407227, 9.525272369384766, 9.529617309570312],
        "norms":  [6.238448619842529, 6.384098052978516, 6.280553817749023, 5.261959552764893, 4.4076948165893555, 4.244746685028076, 4.235298156738281, 4.676079750061035, 4.646341323852539, 4.353753089904785],
    },
}


def run_sft(optimizer_name, config):
    torch.manual_seed(400)
    model = ModelForCausalLM(config)
    B = 4  # 16 题，每 4 步换一轮，10 步内打乱两次
    train_config = TrainConfig(warmup_steps=3, max_steps=10, eval_interval=5)
    optimizers = configure_optimizer(model, train_config, optimizer_name)
    
    train, test = split_pairs(SEED, N_TEST)
    train_pairs = train[:16]
    test_pairs = test[:4]
    losses, norms = train_sft(model, optimizers, train_config, "cpu", train_pairs, test_pairs, B, SEED)
    return losses, norms, model


@pytest.mark.parametrize("optimizer_name", ["adamw", "muon"])
def test_sft_baseline(sft_config, optimizer_name):
    losses, norms, _ = run_sft(optimizer_name, sft_config)
    expected = BASELINE[optimizer_name]
    assert losses == pytest.approx(expected["losses"], rel=1e-5)
    assert norms == pytest.approx(expected["norms"], rel=1e-5)
