"""Readable PyTorch reference implementation of the original ResNet paper.

Included architectures:
  * ImageNet: ResNet-18/34/50/101/152 from Table 1.
  * CIFAR-10: ResNet-20/32/44/56/110/1202 using depth = 6n + 2.

This file focuses on the model and forward data flow. Dataset loading, training,
optimization, checkpointing, and distributed systems are intentionally omitted.
"""

from typing import List, Literal, Sequence, Type, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from blocks import BasicBlock, Bottleneck, OptionAShortcut, conv1x1


BlockType = Type[Union[BasicBlock, Bottleneck]]
ShortcutOption = Literal["A", "B", "C"]


def make_shortcut(
    in_channels: int,
    out_channels: int,
    stride: int,
    option: ShortcutOption,
) -> nn.Module:
    """Construct one of the three shortcut choices studied in the paper.

    A: identity/subsampling plus zero padding when dimensions increase.
    B: projection only when dimensions change; identity otherwise.
    C: projection for every shortcut, including dimension-preserving blocks.
    """
    dimensions_change = stride != 1 or in_channels != out_channels

    if option == "A":
        if dimensions_change:
            return OptionAShortcut(in_channels, out_channels, stride)
        return nn.Identity()

    if option == "B":
        if dimensions_change:
            return nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        return nn.Identity()

    if option == "C":
        return nn.Sequential(
            conv1x1(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
        )

    raise ValueError(f"unknown shortcut option: {option!r}")


class ImageNetResNet(nn.Module):
    """ImageNet ResNet family described in Table 1 of the paper."""

    def __init__(
        self,
        block: BlockType,
        blocks_per_stage: Sequence[int],
        num_classes: int = 1000,
        shortcut_option: ShortcutOption = "B",
    ) -> None:
        super().__init__()
        if len(blocks_per_stage) != 4:
            raise ValueError("ImageNet ResNet requires four residual stages")

        self.in_channels = 64
        self.shortcut_option = shortcut_option

        # Table 1 stem: 224x224 -> 112x112 -> 56x56.
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x, conv3_x, conv4_x, conv5_x in Table 1.
        self.layer1 = self._make_stage(block, 64, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_stage(block, 128, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_stage(block, 256, blocks_per_stage[2], stride=2)
        self.layer4 = self._make_stage(block, 512, blocks_per_stage[3], stride=2)

        # Paper: 7x7 global average pool and a 1000-way fully connected layer.
        # Adaptive pooling is a shape-safe PyTorch equivalent for 224x224 input.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _make_stage(
        self,
        block: BlockType,
        channels: int,
        block_count: int,
        stride: int,
    ) -> nn.Sequential:
        out_channels = channels * block.expansion
        first_shortcut = make_shortcut(
            self.in_channels,
            out_channels,
            stride,
            self.shortcut_option,
        )

        blocks: List[nn.Module] = [
            block(
                self.in_channels,
                channels,
                stride=stride,
                shortcut=first_shortcut,
            )
        ]
        self.in_channels = out_channels

        for _ in range(1, block_count):
            shortcut = make_shortcut(
                self.in_channels,
                out_channels,
                stride=1,
                option=self.shortcut_option,
            )
            blocks.append(
                block(
                    self.in_channels,
                    channels,
                    stride=1,
                    shortcut=shortcut,
                )
            )

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        # PyTorch's Kaiming initialization implements the rectifier-aware
        # initialization cited by the paper (He et al., 2015).
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        # Input image: x [B, 3, 224, 224]

        # Stem: [B, 3, 224, 224] -> [B, 64, 112, 112]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        # Max pooling: [B, 64, 112, 112] -> [B, 64, 56, 56]
        x = self.maxpool(x)

        # conv2_x: [B, 64, 56, 56] -> [B, 64*expansion, 56, 56]
        x = self.layer1(x)

        # conv3_x: -> [B, 128*expansion, 28, 28]
        x = self.layer2(x)

        # conv4_x: -> [B, 256*expansion, 14, 14]
        x = self.layer3(x)

        # conv5_x: -> [B, 512*expansion, 7, 7]
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        # Convolutional feature extraction.
        x = self.forward_features(x)

        # Global average pooling: [B, C, 7, 7] -> [B, C, 1, 1]
        x = self.avgpool(x)

        # Flatten all non-batch dimensions: [B, C, 1, 1] -> [B, C]
        x = torch.flatten(x, start_dim=1)

        # Classification head: [B, C] -> [B, num_classes]
        logits = self.fc(x)

        # Return logits. During training, nn.CrossEntropyLoss applies the
        # required log-softmax internally; probabilities can use logits.softmax(1).
        return logits


class CifarResNet(nn.Module):
    """CIFAR ResNet with the paper's depth rule: depth = 6n + 2."""

    def __init__(
        self,
        depth: int,
        num_classes: int = 10,
        shortcut_option: ShortcutOption = "A",
    ) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("CIFAR ResNet depth must have the form 6n + 2")

        blocks_per_stage = (depth - 2) // 6
        self.in_channels = 16
        self.shortcut_option = shortcut_option

        # Paper CIFAR stem: one 3x3 convolution, preserving 32x32 resolution.
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        # 2n weighted layers at each feature-map size: 32, 16, and 8.
        self.layer1 = self._make_stage(16, blocks_per_stage, stride=1)
        self.layer2 = self._make_stage(32, blocks_per_stage, stride=2)
        self.layer3 = self._make_stage(64, blocks_per_stage, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        self._initialize_weights()

    def _make_stage(
        self,
        channels: int,
        block_count: int,
        stride: int,
    ) -> nn.Sequential:
        shortcut = make_shortcut(
            self.in_channels,
            channels,
            stride,
            self.shortcut_option,
        )
        blocks: List[nn.Module] = [
            BasicBlock(
                self.in_channels,
                channels,
                stride=stride,
                shortcut=shortcut,
            )
        ]
        self.in_channels = channels

        for _ in range(1, block_count):
            shortcut = make_shortcut(
                self.in_channels,
                channels,
                stride=1,
                option=self.shortcut_option,
            )
            blocks.append(
                BasicBlock(
                    self.in_channels,
                    channels,
                    shortcut=shortcut,
                )
            )

        return nn.Sequential(*blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 3, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        # x: [B, 16, 32, 32]

        x = self.layer1(x)
        # x: [B, 16, 32, 32]
        x = self.layer2(x)
        # x: [B, 32, 16, 16]
        x = self.layer3(x)
        # x: [B, 64, 8, 8]

        x = self.avgpool(x)
        # x: [B, 64, 1, 1]
        x = torch.flatten(x, start_dim=1)
        # x: [B, 64]
        logits = self.fc(x)
        # logits: [B, num_classes]
        return logits


def resnet18(**kwargs) -> ImageNetResNet:
    return ImageNetResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ImageNetResNet:
    return ImageNetResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ImageNetResNet:
    return ImageNetResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs) -> ImageNetResNet:
    return ImageNetResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs) -> ImageNetResNet:
    return ImageNetResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


def cifar_resnet(depth: int, **kwargs) -> CifarResNet:
    return CifarResNet(depth=depth, **kwargs)
