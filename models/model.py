from addict import Dict
from torch import nn

from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head


class Model(nn.Module):
    def __init__(self, model_config: dict):
        """
        :param model_config:
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        head_type = model_config.head.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        y = y.view((-1, H // 4, W // 4, 7))
        return y


if __name__ == '__main__':
    import torch
    from torchsummary import summary

    device = torch.device('cpu')
    x = torch.zeros(2, 3, 640, 640).to(device)

    model_config = {
        'backbone': {'type': 'MobileNetV3', 'pretrained': False, "in_channels": 3},
        'neck': {'type': 'FPN', 'inner_channels': 256},
        'head': {'type': 'EASTHead', 'out_channels': 32},
    }
    model = Model(model_config=model_config).to(device)
    import time

    tic = time.time()
    y = model(x)
    print(y.shape)