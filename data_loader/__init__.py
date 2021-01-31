import copy
import torch
from torch.utils.data import DataLoader


def get_dataset(module_name, dataset_args):
    """
    获取训练dataset
    """
    from . import dataset
    s_dataset = getattr(dataset, module_name)(**dataset_args)
    return s_dataset


class AdvancedEASTCollectFN:

    def __call__(self, batch):
        items = list(zip(*batch))
        img_tensors = list(items[0])
        annos = list(items[1])
        imgs = list(items[2])
        draw_gts = list(items[3])
        return {'img_tensor': torch.stack(img_tensors, 0), 'label': torch.stack(annos, 0), 'img': imgs,
                'draw_gt': draw_gts}


class AdvancedEASTCollectFN_eval:

    def __call__(self, batch):
        items = list(zip(*batch))
        img_tensors = list(items[0])
        img = list(items[1])
        annos = list(items[2])
        return {'img_tensor': torch.stack(img_tensors, 0), 'text_polys': annos, 'img': img}


def get_dataloader(module_config, distributed=False):
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    # 创建数据集
    dataset_name = config['dataset']['type']
    data_path = dataset_args['data_path']
    if data_path == None:
        raise Exception('data path is none')
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(
            config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn']['type'])()

    _dataset = get_dataset(module_name=dataset_name, dataset_args=dataset_args)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        # 3）使用DistributedSampler
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False
        config['loader']['pin_memory'] = True
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])
    return loader
