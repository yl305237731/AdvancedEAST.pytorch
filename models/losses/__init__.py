import copy
from .east_loss import QUADLoss

__all__ = ['build_loss']

support_loss = ['QUADLoss']


def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_type)(**copy_config)
    return criterion
