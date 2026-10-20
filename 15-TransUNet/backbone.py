"""CNN 编码器：展开 R50-ViT 混合骨干中的卷积、归一化和残差计算。

论文 §4.2 指定 R50-ViT，但未展开 CNN 的全部细节。这里采用官方混合
骨干的 (3, 4, 9) 个 bottleneck、Weight Standardization 和 GroupNorm。
它不是 torchvision 中输出步长为 32 的普通 ResNet-50。
"""

import torch
from torch import nn
from torch.nn import functional as F


class WeightStandardizedConv2d(nn.Conv2d):
    """每个输出卷积核分别标准化，再执行普通卷积；不是深度可分离卷积。"""

    def forward(self, x):
        # weight: [C_out, C_in, K_h, K_w]；沿每个卷积核的输入通道和空间维计算。
        variance, mean = torch.var_mean(
            self.weight, dim=(1, 2, 3), unbiased=False, keepdim=True
        )
        weight = (self.weight - mean) * torch.rsqrt(variance + 1e-5)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups,
        )


class HybridBottleneck(nn.Module):
    """1x1 降维 -> 3x3 空间计算 -> 1x1 升维 -> 残差相加。

名称不使用 PreAct，以免误认为本实现采用 GN-ReLU-Conv 顺序。
这里遵循官方代码实际执行的 Conv-GN-ReLU 顺序。
"""

    def __init__(self, in_channels, middle_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = WeightStandardizedConv2d(
            in_channels, middle_channels, kernel_size=1, bias=False
        )
        self.norm1 = nn.GroupNorm(32, middle_channels, eps=1e-6)
        self.conv2 = WeightStandardizedConv2d(
            middle_channels, middle_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.norm2 = nn.GroupNorm(32, middle_channels, eps=1e-6)
        self.conv3 = WeightStandardizedConv2d(
            middle_channels, out_channels, kernel_size=1, bias=False
        )
        self.norm3 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.relu = nn.ReLU(inplace=False)

        self.shortcut_conv = None
        self.shortcut_norm = None
        if in_channels != out_channels or stride != 1:
            self.shortcut_conv = WeightStandardizedConv2d(
                in_channels, out_channels, kernel_size=1,
                stride=stride, bias=False,
            )
            # 官方投影支路按每个通道分组；不是主支路的 32 组。
            self.shortcut_norm = nn.GroupNorm(out_channels, out_channels, eps=1e-5)

    def forward(self, x):
        residual = x
        if self.shortcut_conv is not None:
            residual = self.shortcut_conv(residual)
            residual = self.shortcut_norm(residual)

        # 主支路。stride=2 时，空间下采样发生在 3x3 卷积。
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)

        # 逐元素相加不改变通道数；与解码器的通道拼接不同。
        x = x + residual
        return self.relu(x)


class HybridCNNEncoder(nn.Module):
    """输入 RGB 或复制到三通道的灰度图，返回 /16 特征和三层 skip。

输入 H、W 必须是 16 的倍数且至少为 32，由 TransUNet 构造参数约束。
保留官方 max-pool 的 padding=0，以及 /4 skip 的右侧/下侧零填充。
"""

    def __init__(self):
        super().__init__()
        self.stem_conv = WeightStandardizedConv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.stem_norm = nn.GroupNorm(32, 64, eps=1e-6)
        self.stem_relu = nn.ReLU(inplace=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.stage1 = self._make_stage(64, 64, 256, count=3, stride=1)
        self.stage2 = self._make_stage(256, 128, 512, count=4, stride=2)
        self.stage3 = self._make_stage(512, 256, 1024, count=9, stride=2)

    @staticmethod
    def _make_stage(in_channels, middle_channels, out_channels, count, stride):
        # 每个 block 独立实例化，不共享参数；内部计算在 HybridBottleneck 展开。
        blocks = [HybridBottleneck(in_channels, middle_channels, out_channels, stride)]
        for _ in range(count - 1):
            blocks.append(HybridBottleneck(out_channels, middle_channels, out_channels))
        return nn.Sequential(*blocks)

    def forward(self, images):
        # 以下注释以 [B,3,224,224] 为例。
        x = self.stem_conv(images)
        x = self.stem_norm(x)
        skip_2 = self.stem_relu(x)           # [B,64,112,112]

        x = self.pool(skip_2)                # [B,64,55,55]，注意这里不是 56！
        x = self.stage1(x)                   # [B,256,55,55]
        # 对 16 的倍数输入，pool 恰好少一行一列。只对 skip 补零；
        # 继续送入 stage2 的仍然是未填充的 x，保持官方 CNN 主路计算。
        skip_4 = F.pad(x, (0, 1, 0, 1))     # [B,256,56,56]

        skip_8 = self.stage2(x)              # [B,512,28,28]
        deep_features = self.stage3(skip_8)  # [B,1024,14,14]
        return deep_features, (skip_8, skip_4, skip_2)
