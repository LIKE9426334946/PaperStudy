"""Nature Methods nnU-Net 的可配置 2D/3D plain U-Net 模板。

网络接收 planning.py 生成的卷积核和步幅；它本身不负责自动规划。
每个空间维度必须能被该轴所有降采样 stride 的乘积整除。
"""

import torch
from torch import nn

from blocks import TwoConv


class NNUNet(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_sizes, strides,
                 base_channels=32, deep_supervision=True):
        super().__init__()
        dim = len(kernel_sizes[0])
        conv = nn.Conv2d if dim == 2 else nn.Conv3d
        upconv = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
        max_channels = 512 if dim == 2 else 320
        channels = [min(base_channels * 2 ** i, max_channels)
                    for i in range(len(kernel_sizes))]
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        self.encoders = nn.ModuleList()
        for i, out_channels in enumerate(channels):
            step = (1,) * dim if i == 0 else strides[i - 1]
            self.encoders.append(TwoConv(dim, in_channels, out_channels,
                                         kernel_sizes[i], step))
            in_channels = out_channels

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.heads = nn.ModuleDict({"0": conv(channels[0], num_classes, 1)})
        for i in range(len(channels) - 2, -1, -1):
            self.upconvs.append(upconv(channels[i + 1], channels[i],
                                      kernel_size=strides[i], stride=strides[i],
                                      bias=False))
            self.decoders.append(TwoConv(dim, channels[i] * 2, channels[i],
                                         kernel_sizes[i], (1,) * dim))
            # 全分辨率主输出；辅助输出排除最粗两个尺度（包含 bottleneck）。
            if i < max(1, len(channels) - 2):
                self.heads[str(i)] = conv(channels[i], num_classes, 1)

    def forward(self, x):
        # 2D: [B, C, H, W]；3D: [B, C, D, H, W]
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        if len(skips) == 1:  # 极小 patch 的单尺度退化情形。
            logits = self.heads["0"](x)
            return (logits,) if self.deep_supervision else logits

        # 最深层是 bottleneck，随后逐级恢复分辨率。
        outputs = []
        for j, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            i = len(skips) - 2 - j
            x = upconv(x)                   # 空间扩大；通道减少
            x = torch.cat((skips[i], x), 1)  # 通道拼接，不是相加
            x = decoder(x)                  # 两次 Conv -> IN -> LeakyReLU
            if str(i) in self.heads:
                outputs.append(self.heads[str(i)](x))

        # 输出 logits，不在网络内执行 softmax；从高分辨率到低分辨率排列。
        outputs.reverse()
        return tuple(outputs) if self.deep_supervision else outputs[0]
