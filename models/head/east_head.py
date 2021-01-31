import torch
from torch import nn


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class EASTHead(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        :param in_channels:
        :param kwargs:
        """
        super().__init__()
        self.before_output = ConvRelu(in_channels, out_channels, kernel_size=3, padding=1)
        self.inside_score = nn.Conv2d(out_channels, 1, kernel_size=1)
        self.side_v_code = nn.Conv2d(out_channels, 2, kernel_size=1)

        self.side_v_coord = nn.Conv2d(out_channels, 4, kernel_size=1)

    def forward(self, x):
        x = self.before_output(x)
        x1 = self.inside_score(x)
        x2 = self.side_v_code(x)
        x3 = self.side_v_coord(x)
        pred = torch.cat([x1, x2, x3], dim=1)
        return pred