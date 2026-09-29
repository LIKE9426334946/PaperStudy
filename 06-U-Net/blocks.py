"""U-Net 图 1 的基础模块：valid convolution、copy and crop、up-convolution。"""

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def center_crop(x: Tensor, output_size: tuple[int, int]) -> Tensor:
    """只裁剪最后两个空间维；也可用于 [B, H, W] 的标签或权重图。

    本实现采用中心裁剪来落实论文的 copy and crop，不进行插值。
    合规的 U-Net 输入会让各级 skip connection 的裁剪量都是偶数。
    """
    height, width = output_size
    top = (x.shape[-2] - height) // 2
    left = (x.shape[-1] - width) // 2
    if top < 0 or left < 0:
        raise ValueError("裁剪目标不能大于原 Tensor 的空间尺寸")
    return x[..., top:top + height, left:left + width]


class DoubleConv(nn.Module):
    """两个无填充 3x3 卷积，每个卷积后接 ReLU。

    [B, C_in, H, W] -> [B, C_out, H-4, W-4]。
    不包含 BatchNorm；论文没有描述该层。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)  # [B, C_out, H-2, W-2]
        x = F.relu(x)
        x = self.conv2(x)  # [B, C_out, H-4, W-4]
        x = F.relu(x)
        return x


class UpBlock(nn.Module):
    """上采样、裁剪 skip、沿通道拼接，再进行两次 valid convolution。

    PyTorch 映射选择：以 kernel_size=stride=2 的 ConvTranspose2d
    实现论文的可学习 up-convolution。论文没有给出 PyTorch API。
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2,
        )
        self.convs = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        # 以最深的一级解码为例：
        # x: [B, 1024, 28, 28]；skip: [B, 512, 64, 64]。
        x = self.up(x)  # [B, 512, 56, 56]；空间扩大 2 倍，通道减半。
        skip = center_crop(skip, x.shape[-2:])  # [B, 512, 56, 56]
        x = torch.cat([skip, x], dim=1)  # [B, 1024, 56, 56]
        x = self.convs(x)  # [B, 512, 52, 52]
        return x
