"""论文 §3.2 和 Figure 1 的 CUP：逐级上采样，融合 CNN 的高分辨率特征。

优先遵循论文文字：每级一次 2x 上采样、一次 3x3 卷积、一次 ReLU。
官方发布代码每级使用两次 Conv-BN-ReLU，本参考实现未照搬该扩展。
"""

import torch
from torch import nn
from torch.nn import functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels + skip_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, skip=None):
        # 工程补全：双线性插值，align_corners=False；没有转置卷积。
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            # 两个特征的 H、W 相同，沿通道维拼接。
            # 第一级：[B,512,28,28] + [B,512,28,28] -> [B,1024,28,28]。
            x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return self.relu(x)


class CascadedUpsampler(nn.Module):
    """D -> 512 -> 256 -> 128 -> 64 -> 16，四次上采样恢复输入分辨率。"""

    def __init__(self, embed_dim=768, skip_scales=(8, 4, 2)):
        super().__init__()
        # 论文默认三条 skip；也允许论文消融中的 () 或仅 (4,) 等组合。
        self.skip_scales = tuple(skip_scales)
        if len(set(self.skip_scales)) != len(self.skip_scales):
            raise ValueError("skip_scales 不能包含重复尺度。")
        if any(scale not in (8, 4, 2) for scale in self.skip_scales):
            raise ValueError("skip_scales 只能从 (8, 4, 2) 中选择。")

        # Figure 1：Transformer 特征先投影成 512 通道，空间尺寸保持 /16。
        self.input_conv = nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1)
        self.input_relu = nn.ReLU(inplace=False)
        self.up_8 = DecoderBlock(512, 512 if 8 in self.skip_scales else 0, 256)
        self.up_4 = DecoderBlock(256, 256 if 4 in self.skip_scales else 0, 128)
        self.up_2 = DecoderBlock(128, 64 if 2 in self.skip_scales else 0, 64)
        self.up_1 = DecoderBlock(64, 0, 16)  # 最后一级没有 CNN skip。

    def forward(self, feature_map, skips):
        skip_8, skip_4, skip_2 = skips
        x = self.input_conv(feature_map)   # [B,D,14,14] -> [B,512,14,14]
        x = self.input_relu(x)
        x = self.up_8(x, skip_8 if 8 in self.skip_scales else None)  # [B,256,28,28]
        x = self.up_4(x, skip_4 if 4 in self.skip_scales else None)  # [B,128,56,56]
        x = self.up_2(x, skip_2 if 2 in self.skip_scales else None)  # [B,64,112,112]
        x = self.up_1(x)                                         # [B,16,224,224]
        return x


class SegmentationHead(nn.Module):
    """逐像素输出 K 类 logits；forward 中不做 softmax/argmax。"""

    def __init__(self, num_classes):
        super().__init__()
        # 主模型的 head 核尺寸在论文中没有明确写出，采用 1x1 是补全选择。
        # 论文明确写出的 1x1 分类层属于 naive baseline；不混淆这两处。
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, features):
        return self.classifier(features)  # [B,16,H,W] -> [B,K,H,W]
