"""论文的基础块：Conv -> InstanceNorm -> LeakyReLU，连续执行两次。"""

from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        norm = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = conv(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, bias=True)
        # affine/eps 是实现补全；不用 batch 统计，推理也按单个样本计算。
        self.norm = norm(out_channels, eps=1e-5, affine=True,
                         track_running_stats=False)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class TwoConv(nn.Module):
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.first = ConvBlock(dim, in_channels, out_channels, kernel_size, stride)
        self.second = ConvBlock(dim, out_channels, out_channels, kernel_size,
                                (1,) * dim)

    def forward(self, x):
        x = self.first(x)  # 编码器降采样只在这个卷积的 stride 中发生。
        return self.second(x)
