import time
from pprint import pformat

import torch
import os
import pathlib
import shutil
import anyconfig
import torchvision.utils as vutils
from torchvision import transforms
from utils.utils import setup_logger
from tqdm import tqdm
from utils import WarmupPolyLR


class Trainer():
    def __init__(self, config, model, criterion, metric_cls, train_loader, validate_loader, post_process=None):
        config['trainer']['output_dir'] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent),
                                                       config['trainer']['output_dir'])
        config['name'] = config['name'] + '_' + model.name
        self.save_dir = os.path.join(config['trainer']['output_dir'], config['name'])
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')

        if config['trainer']['resume_checkpoint'] == '' and config['trainer']['finetune_checkpoint'] == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = 0
        self.start_epoch = 0
        self.config = config
        self.model = model
        self.criterion = criterion
        self.metric_cls = metric_cls
        # logger and tensorboard
        self.epochs = self.config['trainer']['epochs']
        self.log_iter = self.config['trainer']['log_iter']
        self.tensorboard_enable = self.config['trainer']['tensorboard']
        if config['local_rank'] == 0:
            anyconfig.dump(config, os.path.join(self.save_dir, 'config.yaml'))
            self.logger = setup_logger(os.path.join(self.save_dir, 'train.log'))
            self.logger_info(pformat(self.config))

        # device
        torch.manual_seed(self.config['trainer']['seed'])  # 为CPU设置随机种子
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.config['trainer']['seed'])  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config['trainer']['seed'])  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.device = torch.device("cpu")
        self.logger_info('train with device {} and pytorch {}'.format(self.device, torch.__version__))

        self.optimizer = self._initialize('optimizer', torch.optim, model.parameters())

        # resume or finetune
        if self.config['trainer']['resume_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
        elif self.config['trainer']['finetune_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['finetune_checkpoint'], resume=False)

        if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
            self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.metrics = {'recall': 0, 'precision': 0, 'hmean': 0, 'train_loss': float('inf'), 'best_model_epoch': 0}
        self.model.to(self.device)

        # 分布式训练
        if torch.cuda.device_count() > 1:
            local_rank = config['local_rank']
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank],
                                                                   output_device=local_rank, broadcast_buffers=False,
                                                                   find_unused_parameters=True)

        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        if validate_loader is not None:
            assert post_process is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.train_loader_len = len(train_loader)
        if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
            warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
            if self.start_epoch > 1:
                self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
            self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                          warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
        if self.validate_loader is not None:
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset),
                    len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset),
                                                                                    self.train_loader_len))

        if self.tensorboard_enable and config['local_rank'] == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)
            try:
                dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
                self.writer.add_graph(self.model, dummy_input)
                torch.cuda.empty_cache()
            except:
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warn('add graph to tensorboard failed')

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            if self.config['distributed']:
                self.train_loader.sampler.set_epoch(epoch)
            self.epoch_result = self._train_epoch(epoch)
            if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
                self.scheduler.step()
            self._on_epoch_finish()
        if self.config['local_rank'] == 0 and self.tensorboard_enable:
            self.writer.close()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            cur_batch_size = batch['img_tensor'].size()[0]
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)

            preds = self.model(batch['img_tensor'])
            loss = self.criterion(preds, batch['label'])
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()

            train_loss += loss

            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                self.logger_info('[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, loss: {:.6f},lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, self.log_iter * cur_batch_size / batch_time, loss, lr, batch_time))
                batch_start = time.time()

            if self.tensorboard_enable and self.config['local_rank'] == 0:
                self.writer.add_scalar('TRAIN/LOSS/{}'.format('loss'), loss, self.global_step)
                self.writer.add_scalar('TRAIN/lr', lr, self.global_step)
                if self.global_step % self.show_images_iter == 0:
                    boxes, pred_imgs, act_imgs = self.post_process(batch['img'], preds, save_img=True)

                    show_act = vutils.make_grid([transforms.ToTensor()(x) for x in act_imgs], nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)

                    show_label = vutils.make_grid([transforms.ToTensor()(x) for x in batch['draw_gt']], nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)

                    show_pred = vutils.make_grid([transforms.ToTensor()(x) for x in pred_imgs], nrow=cur_batch_size, normalize=False, padding=20, pad_value=1)

                    self.writer.add_images('TRAIN/activation', show_act.unsqueeze(0), self.global_step)
                    self.writer.add_images('TRAIN/gt', show_label.unsqueeze(0), self.global_step)
                    self.writer.add_images('TRAIN/preds', show_pred.unsqueeze(0), self.global_step)

        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self):
        self.model.eval()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                for key, value in batch.items():
                    if value is not None:
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(self.device)
                start = time.time()
                preds = self.model(batch['img_tensor'])
                boxes, _, _ = self.post_process(batch['img'], preds)
                total_frame += batch['img_tensor'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, boxes)
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            save_best = False
            if self.validate_loader is not None and self.metric_cls is not None:  # 使用f1作为最优模型指标
                recall, precision, hmean = self._eval()
                self.logger_info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, hmean))

                if hmean >= self.metrics['hmean']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['hmean'] = hmean
                    self.metrics['precision'] = precision
                    self.metrics['recall'] = recall
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            else:
                if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
            best_str = 'current best, '
            for k, v in self.metrics.items():
                best_str += '{}: {:.6f}, '.format(k, v)
            self.logger_info(best_str)
            if save_best:
                import shutil
                shutil.copy(net_save_path, net_save_path_best)
                self.logger_info("Saving current best: {}".format(net_save_path_best))
            else:
                self.logger_info("Saving checkpoint: {}".format(net_save_path))

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        self.logger_info('finish train')

    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self.model.module.state_dict() if self.config['distributed'] else self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger_info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger_info("finetune from checkpoint {}".format(checkpoint_path))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def logger_info(self, s):
        if self.config['local_rank'] == 0:
            self.logger.info(s)
