from __future__ import print_function

import argparse
import os
import anyconfig


def init_args():
    parser = argparse.ArgumentParser(description='AdvancedEAST.pytorch')
    parser.add_argument('--config_file', default='config/advanced_east.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')

    args = parser.parse_args()
    return args


def main(config):
    import torch
    from models import build_model, build_loss
    from data_loader import get_dataloader
    from scripts import Trainer
    from post_processing import get_post_processing
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(),
                                             rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank

    train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    assert train_loader is not None
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], False)
    else:
        validate_loader = None

    criterion = build_loss(config['loss'])

    config['arch']['backbone']['in_channels'] = 3
    model = build_model(config['arch'])

    post_p = get_post_processing(config['post_processing'])

    from utils import get_metric
    metric = get_metric(config['metric'])

    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      metric_cls=metric,
                      train_loader=train_loader,
                      validate_loader=validate_loader,
                      post_process=post_p)
    trainer.train()


if __name__ == '__main__':
    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    main(config)
