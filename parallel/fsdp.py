import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict
from parallel.dist_utils import get_world_size



def shard_model(model, device):
    device_type = torch.device(device).type
    mesh = init_device_mesh(device_type, (get_world_size(),))
    for block in model.model.layers:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def get_full_state(model, optimizer):
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_sd = get_model_state_dict(model, options=options)
    if optimizer.state:
        optim_sd = get_optimizer_state_dict(model, optimizer, options=options)
    else:
        # 如果 optimizer.state 为空，get_optimizer_state_dict 会初始化，调用一次 step，会记入步数
        optim_sd = {}
    return model_sd, optim_sd


def load_full_state(model, optimizer, model_sd, optim_sd):
    options = StateDictOptions(full_state_dict=True)
    set_model_state_dict(model, model_sd, options=options)
    if optim_sd:
        set_optimizer_state_dict(model, optimizer, optim_sd, options=options)


