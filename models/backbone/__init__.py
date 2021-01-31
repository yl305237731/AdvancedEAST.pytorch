from .mobilenet_v3 import MobileNetV3
from .resnet import *

__all__ = ['build_backbone']

support_backbone = ['MobileNetV3', 'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'deformable_resnet18', 'deformable_resnet50', 'resnet152']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
