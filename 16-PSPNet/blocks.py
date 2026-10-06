"""PSPNet 核心模块：展开残差计算、金字塔池化及逐像素分类。"""

import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    """ResNet 瓶颈块：1×1 降维 → 3×3 空间计算 → 1×1 升维 → 残差相加。

    stride 放在 3×3 卷积上，沿用作者后续 PyTorch 实现的工程选择。
    dilation=2/4 时，padding=2/4 保持 stride=1 时的空间大小。
    """

    expansion = 4

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, 3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 仅在形状变化时，用可学习的投影使两条路径可以逐元素相加。
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)           # [B, C_in, H, W] → [B, C, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)           # 3×3 标准卷积或膨胀卷积
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)           # C → 4C
        x = self.bn3(x)
        x = x + identity           # 残差相加，不是通道拼接
        return self.relu(x)


class PyramidPoolingBranch(nn.Module):
    """一个池化尺度：区域平均 → 1×1 降维 → BN → ReLU。"""

    def __init__(self, in_channels, out_channels, bins):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((bins, bins))
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool(x)            # [B, 2048, h, w] → [B, 2048, s, s]
        x = self.conv(x)            # → [B, 512, s, s]
        x = self.bn(x)              # BN/ReLU 来自作者后续实现，正文未逐层指定
        return self.relu(x)


class PyramidPoolingModule(nn.Module):
    """论文 §3.2：四级全局/区域上下文，与原始深层特征拼接。"""

    def __init__(self, in_channels=2048):
        super().__init__()
        self.bin_sizes = (1, 2, 3, 6)
        reduced_channels = in_channels // len(self.bin_sizes)
        self.branches = nn.ModuleList([
            PyramidPoolingBranch(in_channels, reduced_channels, s)
            for s in self.bin_sizes
        ])
        self.out_channels = in_channels + len(self.bin_sizes) * reduced_channels

    def forward(self, x):
        spatial_size = x.shape[-2:]
        features = [x]              # 保留原始特征 [B, 2048, h, w]
        for branch in self.branches:
            context = branch(x)    # 四个分支均读取同一个 x，彼此不串联
            context = F.interpolate(
                context, size=spatial_size, mode="bilinear", align_corners=True,
            )                      # [B, 512, s, s] → [B, 512, h, w]
            features.append(context)
        return torch.cat(features, dim=1)  # [B, 4096, h, w]


class SegmentationHead(nn.Module):
    """作者后续实现中的 3×3 融合 + 1×1 分类；输出未经 softmax 的 logits。"""

    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.classifier(x)   # [B, K, h, w]；每个位置有 K 个类别分数
