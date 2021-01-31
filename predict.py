import os
import time
import torch
import pathlib
from tqdm import tqdm
from models import build_model
from torchvision import transforms
from PIL import Image
from post_processing import get_post_processing
from utils import get_file_list


class Infer:
    def __init__(self, model_path, gpu_id=None):
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    def predict(self, img_path: str, target_size: int = 640, show_img=False):
        assert os.path.exists(img_path), 'file is not exists'
        img = Image.open(img_path)
        im_ = img.resize((target_size, target_size), Image.NEAREST).convert('RGB')
        tensor = self.transform(im_)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            boxes, pred_img, actv_img = self.post_process([img], preds, show_img=show_img)
            t = time.time() - start
        return boxes, pred_img, actv_img, t


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='Advanced_EAST.pytorch')
    parser.add_argument('--model_path', default=r'model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='img path for output')
    parser.add_argument('--show', default=True, type=str, help='show result')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str('0')
    model = Infer(args.model_path, gpu_id=0)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_file_list(args.input_folder, p_postfix=['.jpg'])):
        boxes, pred_img, act_img, t = model.predict(img_path, show_img=args.show)
        print('infer time {}'.format(t))
        if args.show:
            os.makedirs(args.output_folder, exist_ok=True)
            img_path = pathlib.Path(img_path)
            act_path = os.path.join(args.output_folder, img_path.stem + '_activation.jpg')
            pred_path = os.path.join(args.output_folder, img_path.stem + '_pred.jpg')
            pred_img[0].save(pred_path)
            act_img[0].save(act_path)
