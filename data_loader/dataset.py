import os
import numpy as np
import torch
from PIL import Image
import copy
from torchvision import transforms
from torch.utils.data import Dataset
from data_loader.modules.make_label import make_gt, get_text_polys


class EASTDataSet(Dataset):

    def __init__(self, data_path='', img_dir='', label_dir='', shrink_ratio=0.4, shrink_side_ratio=0.6, target_size=640, ignore_tags=None, show_gt_img=True, redution=4, train=True):
        if ignore_tags is None:
            ignore_tags = ['###', '*']
        self.data_path = data_path
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.shrink_ratio = shrink_ratio
        self.shrink_side_ratio = shrink_side_ratio
        self.target_size = target_size
        self.redution = redution
        self.ignore_tags = ignore_tags
        self.show_gt_img = show_gt_img
        self.train = train
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.imgs_list = self.load_data()

    def load_data(self):
        imgs_list = os.listdir(os.path.join(self.data_path, self.img_dir))
        return imgs_list

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, item):
        try:
            img_name = copy.deepcopy(self.imgs_list[item])
            img_path = os.path.join(self.data_path, self.img_dir, img_name)
            assert os.path.exists(img_path)
            im = Image.open(img_path)
            label_path = os.path.join(self.data_path, self.label_dir, img_name[:-4] + '.txt')
            assert os.path.exists(label_path)
            label_file = open(label_path, 'r', encoding='utf-8')
            anno_lis = label_file.readlines()
            if self.train:
                im, gt, gt_img = make_gt(im, anno_list=anno_lis, shrink_ratio=self.shrink_ratio, shrink_side_ratio=self.shrink_side_ratio, draw_gt_quad=self.show_gt_img, ignore_tags=self.ignore_tags)
                im_tensor = self.transform(im)
                return im_tensor, torch.from_numpy(gt), im, gt_img
            else:
                im_ = im.resize((self.target_size, self.target_size), Image.NEAREST).convert('RGB')
                im_tensor = self.transform(im_)
                gt = get_text_polys(anno_lis, self.ignore_tags)
                return im_tensor, im, gt
        except:
            return self.__getitem__(np.random.randint(self.__len__()))