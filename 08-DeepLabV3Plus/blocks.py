"""DeepLabv3+ 的基础运算、ASPP 和 decoder；不调用现成分割模型。"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    """普通卷积；3×3 空洞卷积用 padding=rate 保持 stride=1 时的尺寸。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilation=1, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=eps)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class AtrousSeparableConv(nn.Module):
    """Fig.3：空洞 depthwise -> pointwise，明确展开两种卷积。

    每个输入通道独立做 3×3 卷积：groups=in_channels；不混合通道。
    1×1 pointwise 再混合通道。dilation 只作用于 depthwise。
    pre_activation=True 用于 Xception 残差流；False 用于出口尾部/分割头。
    按论文 §3.2，在所有 depthwise 后保留 BN+ReLU；与当前官方 TF 代码
    部分预激活块的 ReLU 位置不同，详见 README 的实现边界。
    """

    def __init__(self, in_channels, out_channels, stride=1, dilation=1,
                 pre_activation=False, eps=1e-5):
        super().__init__()
        self.pre_activation = pre_activation
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=dilation,
            dilation=dilation, groups=in_channels, bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels, eps=eps)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels, eps=eps)

    def forward(self, x):
        # 预激活只影响当前卷积支路；不用 inplace，避免改变 residual/浅层特征。
        if self.pre_activation:
            x = F.relu(x)
        x = self.depthwise(x)             # [B, Cin, H/s, W/s]
        x = F.relu(self.depthwise_bn(x))  # 论文明确增加的 DW 后 BN+ReLU
        x = self.pointwise(x)             # [B, Cout, H/s, W/s]
        x = self.pointwise_bn(x)
        if not self.pre_activation:
            x = F.relu(x)
        return x


class XceptionBlock(nn.Module):
    """Fig.4 的三层可分离卷积；shortcut 为投影、恒等或无连接。

    返回 (块输出, 第二层特征)，便于在第三层下采样前取 decoder 的浅层输入。
    """

    def __init__(self, in_channels, channels, stride=1, dilation=1,
                 skip_type="conv", pre_activation=True):
        super().__init__()
        c1, c2, c3 = channels
        self.sep1 = AtrousSeparableConv(
            in_channels, c1, dilation=dilation,
            pre_activation=pre_activation, eps=1e-3,
        )
        self.sep2 = AtrousSeparableConv(
            c1, c2, dilation=dilation,
            pre_activation=pre_activation, eps=1e-3,
        )
        self.sep3 = AtrousSeparableConv(
            c2, c3, stride=stride, dilation=dilation,
            pre_activation=pre_activation, eps=1e-3,
        )
        self.skip_type = skip_type
        if skip_type == "conv":
            self.shortcut_conv = nn.Conv2d(in_channels, c3, 1,
                                           stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(c3, eps=1e-3)
        elif skip_type == "sum":
            if in_channels != c3 or stride != 1:
                raise ValueError("恒等残差要求通道数相同且 stride=1")
        elif skip_type != "none":
            raise ValueError("skip_type 必须为 conv、sum 或 none")

    def forward(self, x):
        identity = x
        x = self.sep1(x)
        low_level = self.sep2(x)          # 第三层 stride 生效前的特征
        x = self.sep3(low_level)
        if self.skip_type == "conv":
            shortcut = self.shortcut_bn(self.shortcut_conv(identity))
            x = x + shortcut             # 逐元素相加，不是 concat
        elif self.skip_type == "sum":
            x = x + identity
        return x, low_level


class ResNetBottleneck(nn.Module):
    """展开 1×1 -> 3×3 -> 1×1 和 residual，供 ResNet-101 对照骨干使用。"""

    def __init__(self, in_channels, bottleneck_channels, stride=1, dilation=1):
        super().__init__()
        out_channels = bottleneck_channels * 4
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        # 采用 PyTorch 常见的 v1.5 排布：stride 放在中间的 3×3 卷积。
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, 3, stride=stride,
            padding=dilation, dilation=dilation, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.project = in_channels != out_channels or stride != 1
        if self.project:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1,
                                           stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.project:
            identity = self.shortcut_bn(self.shortcut_conv(identity))
        return F.relu(x + identity)


class ASPP(nn.Module):
    """Fig.2：1×1、三路 3×3 空洞卷积、全局平均池化。

    默认每路 256 通道；concat 得 1280 通道，再用 1×1 压回 256。
    image-level 分支可关闭，用于论文 Cityscapes 的消融设定。
    """

    def __init__(self, in_channels=2048, output_stride=16,
                 use_separable_conv=True, image_pooling=True):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride 仅支持 8 或 16")
        rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)
        self.rates = rates
        self.image_pooling = image_pooling
        self.branch1 = ConvBNReLU(in_channels, 256, kernel_size=1)
        Conv3 = AtrousSeparableConv if use_separable_conv else ConvBNReLU
        self.branch2 = Conv3(in_channels, 256, dilation=rates[0])
        self.branch3 = Conv3(in_channels, 256, dilation=rates[1])
        self.branch4 = Conv3(in_channels, 256, dilation=rates[2])
        if image_pooling:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.pool_project = ConvBNReLU(in_channels, 256, kernel_size=1)
        self.project = ConvBNReLU(256 * (5 if image_pooling else 4), 256,
                                   kernel_size=1)
        # 官方工程实现的补充：keep_prob=0.9，即普通逐元素 Dropout(p=0.1)。
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x: [B, 2048, H/OS, W/OS]；四个卷积分支的空间尺寸都不变。
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        branches = [b1, b2, b3, b4]
        if self.image_pooling:
            b5 = self.global_pool(x)      # [B, 2048, 1, 1]
            b5 = self.pool_project(b5)    # [B, 256, 1, 1]
            b5 = F.interpolate(b5, size=x.shape[-2:], mode="bilinear",
                               align_corners=True)
            branches.append(b5)
        x = torch.cat(branches, dim=1)    # 默认 [B, 1280, H/OS, W/OS]
        x = self.project(x)              # [B, 256, H/OS, W/OS]
        return self.dropout(x)


class DeepLabDecoder(nn.Module):
    """§4.1：浅层 256->48，concat 256+48=304，两层 3×3 输出 256。"""

    def __init__(self, low_channels=256, use_separable_conv=True):
        super().__init__()
        self.low_project = ConvBNReLU(low_channels, 48, kernel_size=1)
        Conv3 = AtrousSeparableConv if use_separable_conv else ConvBNReLU
        self.refine1 = Conv3(256 + 48, 256)
        self.refine2 = Conv3(256, 256)

    def forward(self, high_level, low_level):
        low_level = self.low_project(low_level)  # [B, 48, H/4, W/4]
        # OS=16 时约放大4倍；OS=8 时约2倍。按目标尺寸可处理奇数边长。
        high_level = F.interpolate(
            high_level, size=low_level.shape[-2:],
            mode="bilinear", align_corners=True,
        )
        x = torch.cat((high_level, low_level), dim=1)  # [B, 304, H/4, W/4]
        x = self.refine1(x)
        return self.refine2(x)                        # [B, 256, H/4, W/4]
