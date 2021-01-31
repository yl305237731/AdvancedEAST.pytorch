import argparse
import time
import torch
from tqdm.auto import tqdm


class EVAL():
    def __init__(self, model_path, gpu_id=0):
        from models import build_model
        from data_loader import get_dataloader
        from post_processing import get_post_processing
        from utils import get_metric
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False

        self.validate_loader = get_dataloader(config['dataset']['validate'])

        self.model = build_model(config['arch'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)

        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

    def eval(self):
        self.model.eval()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='validate'):
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
        print('FPS:{}'.format(total_frame / total_time))
        print('recall: {}, precision: {}, fmeasure: {}'.format(metrics['recall'].avg, metrics['precision'].avg,
                                                               metrics['fmeasure'].avg))
        return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


def init_args():
    parser = argparse.ArgumentParser(description='Advanced_EAST.pytorch')
    parser.add_argument('--model_path', required=False, default='', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    eval = EVAL(args.model_path)
    result = eval.eval()
