from PIL import Image, ImageDraw
import numpy as np
import torch
from utils.east_nms import nms
from utils.utils import resize_image


class EastRepresenter():
    def __init__(self, image_size, pixel_thresh=0.9, redution=4, side_vertex_pixel_threshold=0.9, trunc_threshold=0.1):
        self.image_size = image_size
        self.pixel_thresh = pixel_thresh
        self.redution = redution
        self.side_vertex_pixel_threshold = side_vertex_pixel_threshold
        self.trunc_threshold = trunc_threshold

    def __call__(self, imgs, preds, show_img=False):
        boxes = []
        pred_imgs = []
        activation_imgs = []
        for id, im in enumerate(imgs):
            box, pred, activation = self.call_image(imgs[id], preds[id, :, :, :], show_img)
            boxes.append(box)
            pred_imgs.append(pred)
            activation_imgs.append(activation)
        return boxes, pred_imgs, activation_imgs

    def call_image(self, img, pred, save_img):
        '''
        img: origin input image
        pred: network out [H, W, 7]
        :return
        box coords
        '''
        rescaled_geo_lists = []
        d_wight, d_height = resize_image(img, self.image_size)
        scale_ratio_w = d_wight / img.width
        scale_ratio_h = d_height / img.height
        pred[:, :, :3] = torch.sigmoid(pred[:, :, :3])
        pred = pred.cpu().detach().numpy()
        cond = np.greater_equal(pred[:, :, 0], self.pixel_thresh)
        activation_pixels = np.where(cond)
        quad_scores, quad_after_nms = nms(pred, activation_pixels)
        if save_img:
            quad_img = img.copy()
            quad_draw = ImageDraw.Draw(quad_img)

        for score, geo in zip(quad_scores, quad_after_nms):
            if np.amin(score) > 0:
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                if save_img:
                    quad_draw.line([tuple(rescaled_geo[0]),
                                    tuple(rescaled_geo[1]),
                                    tuple(rescaled_geo[2]),
                                    tuple(rescaled_geo[3]),
                                    tuple(rescaled_geo[0])], width=2, fill='blue')
                rescaled_geo = np.asarray(rescaled_geo.astype(int))
                rescaled_geo_lists.append({'points': rescaled_geo})

        if save_img:
            activation_img = self.draw_activation(img, pred, activation_pixels, self.redution, scale_ratio_w, scale_ratio_h)
            return rescaled_geo_lists, quad_img, activation_img
        return rescaled_geo_lists, None, None

    def draw_activation(self, im, pred, activation_pixels, redution, scale_ratio_w, scale_ratio_h):
        draw_img = im.copy()
        draw = ImageDraw.Draw(draw_img)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):
            px = (j + 0.5) * redution / scale_ratio_w
            py = (i + 0.5) * redution / scale_ratio_h
            line_width, line_color = 1, 'red'
            if pred[i, j, 1] >= self.side_vertex_pixel_threshold:
                if pred[i, j, 2] < self.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif pred[i, j, 2] >= 1 - self.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * redution, py - 0.5 * redution),
                       (px + 0.5 * redution, py - 0.5 * redution),
                       (px + 0.5 * redution, py + 0.5 * redution),
                       (px - 0.5 * redution, py + 0.5 * redution),
                       (px - 0.5 * redution, py - 0.5 * redution)],
                      width=line_width, fill=line_color)
        return draw_img
