"""Residual building blocks from He et al. (CVPR 2016).

The code intentionally exposes the main tensor operations instead of hiding
them behind a high-level ResNet implementation.
"""

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """3x3 convolution that preserves size when stride=1."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    """1x1 projection used by shortcut options B and C."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class OptionAShortcut(nn.Module):
    """Parameter-free shortcut (option A in the paper).

    Spatial downsampling is performed by strided subsampling. New output
    channels are filled with zeros, so no trainable parameters are introduced.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be a positive integer")
        if out_channels < in_channels:
            raise ValueError("option A cannot reduce the channel dimension")

        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C_in, H, W]
        # Paper option A: identity subsampling + zero-filled new channels.
        identity = x[:, :, :: self.stride, :: self.stride]
        missing_channels = self.out_channels - identity.shape[1]

        if missing_channels > 0:
            # For NCHW, this six-value tuple pads W, H, then C.
            identity = F.pad(
                identity,
                pad=(0, 0, 0, 0, 0, missing_channels),
                mode="constant",
                value=0.0,
            )

        # identity: [B, C_out, H / stride, W / stride]
        return identity


class BasicBlock(nn.Module):
    """Two-layer residual function used by ResNet-18 and ResNet-34.

    Paper equation:
        y = F(x, {W_i}) + shortcut(x)

    The final ReLU is applied after the element-wise addition, matching the
    original ResNet v1 / post-activation design in Figures 2 and 5.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        shortcut: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        out_channels = channels * self.expansion

        self.conv1 = conv3x3(in_channels, channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut if shortcut is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C_in, H, W]
        identity = self.shortcut(x)
        # identity: [B, C_out, H / stride, W / stride]

        # Residual branch: W1 -> BN -> ReLU.
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = F.relu(residual, inplace=True)

        # Residual branch: W2 -> BN. No ReLU before the addition.
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # Equation (1) or (2): residual mapping + identity/projection shortcut.
        out = residual + identity
        out = F.relu(out, inplace=True)
        # out: [B, C_out, H / stride, W / stride]
        return out


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 bottleneck for ResNet-50/101/152.

    The first 1x1 convolution reduces channels, the 3x3 convolution processes
    the compact representation, and the last 1x1 convolution restores the
    output width. Figure 5 uses 64 -> 64 -> 256 as its example.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        shortcut: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        out_channels = channels * self.expansion

        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut if shortcut is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C_in, H, W]
        identity = self.shortcut(x)
        # identity: [B, 4 * channels, H / stride, W / stride]

        # Channel reduction: 1x1 convolution.
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = F.relu(residual, inplace=True)

        # Spatial processing: 3x3 convolution.
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = F.relu(residual, inplace=True)

        # Channel restoration: 1x1 convolution. No ReLU before addition.
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        # Equation (1) or (2): residual mapping + identity/projection shortcut.
        out = residual + identity
        out = F.relu(out, inplace=True)
        # out: [B, 4 * channels, H / stride, W / stride]
        return out
