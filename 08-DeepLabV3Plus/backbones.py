"""论文主干：Fig.4 的完整 Xception-65，以及 Xception-71/ResNet-101 对照。

所有主干均返回 (high_level, low_level)，通道分别为 2048 和 256。
这里不加载预训练参数；ResNet stride 排布的工程差异见 README。
"""

from torch import nn

if __package__:
    from .blocks import ConvBNReLU, XceptionBlock, ResNetBottleneck
else:
    from blocks import ConvBNReLU, XceptionBlock, ResNetBottleneck


class AlignedXception(nn.Module):
    """完整保留 16 个 Middle Flow 残差块，不用缩小网络替代论文结构。"""

    def __init__(self, in_channels=3, output_stride=16, variant="xception65"):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride 仅支持 8 或 16")
        if variant not in ("xception65", "xception71"):
            raise ValueError("variant 必须为 xception65 或 xception71")
        self.stem1 = ConvBNReLU(in_channels, 32, stride=2, eps=1e-3)
        self.stem2 = ConvBNReLU(32, 64, eps=1e-3)

        # (每层通道数, 分类主干原始 stride)。X-71 的新增层来自官方实现。
        if variant == "xception65":
            entry_specs = [(128, 2), (256, 2), (728, 2)]
            self.low_block_index = 1
        else:
            entry_specs = [(128, 2), (256, 1), (256, 2), (728, 1), (728, 2)]
            self.low_block_index = 2

        current_stride, current_rate, channels = 2, 1, 64

        def make_block(depths, original_stride, skip_type, pre_activation=True):
            # 构建时把超出目标 OS 的 stride=2 改为1，并将之后的 rate 翻倍。
            # 注意：本块先使用旧 rate，后续块才使用更新后的 rate。
            nonlocal current_stride, current_rate, channels
            rate = current_rate
            if current_stride == output_stride:
                stride = 1
                current_rate *= original_stride
            else:
                stride = original_stride
                current_stride *= original_stride
            block = XceptionBlock(
                channels, depths, stride=stride, dilation=rate,
                skip_type=skip_type, pre_activation=pre_activation,
            )
            channels = depths[-1]
            return block

        self.entry = nn.ModuleList([
            make_block((c, c, c), stride, "conv") for c, stride in entry_specs
        ])
        self.middle = nn.ModuleList([
            make_block((728, 728, 728), 1, "sum") for _ in range(16)
        ])
        self.exit_block = make_block((728, 1024, 1024), 2, "conv")
        self.exit_tail = make_block((1536, 1536, 2048), 1, "none",
                                    pre_activation=False)

    def forward(self, x):
        # 以下 shape 以 X-65、OS=16、输入 [B,3,256,256] 为例。
        x = self.stem1(x)                 # [B, 32, 128, 128]
        x = self.stem2(x)                 # [B, 64, 128, 128]
        low_level = None
        for index, block in enumerate(self.entry):
            x, before_stride = block(x)
            if index == self.low_block_index:
                low_level = before_stride  # [B, 256, 64, 64]，sep2 后/sep3 前
        # entry 完成后 [B,728,16,16]；OS=8 则为 [B,728,32,32]。
        for block in self.middle:
            x, _ = block(x)               # 16 个块都保持空间大小和728通道
        x, _ = self.exit_block(x)         # [B, 1024, 16, 16]，stride 已取消
        x, _ = self.exit_tail(x)          # [B, 2048, 16, 16]，rate=2
        return x, low_level


class DilatedResNet101(nn.Module):
    """[3,4,23,3] Bottleneck；以 PyTorch v1.5 排布实现论文 ResNet 对照。

    与官方 TF 的块末下采样/中间端点有细节差异；不宣称权重兼容。
    layer1 输出为 OS=4 的 Conv2 特征；取消后续下采样时同步增大 dilation。
    """

    def __init__(self, in_channels=3, output_stride=16):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride 仅支持 8 或 16")
        self.stem = ConvBNReLU(in_channels, 64, kernel_size=7, stride=2)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        channels, rate = 64, 1

        def make_stage(width, count, stride=1, replace_stride=False):
            nonlocal channels, rate
            previous_rate = rate
            if replace_stride:
                rate *= stride
                stride = 1
            layers = [ResNetBottleneck(channels, width, stride, previous_rate)]
            channels = width * 4
            layers.extend(ResNetBottleneck(channels, width, dilation=rate)
                          for _ in range(count - 1))
            return nn.Sequential(*layers)

        self.layer1 = make_stage(64, 3)
        self.layer2 = make_stage(128, 4, stride=2)
        self.layer3 = make_stage(256, 23, stride=2,
                                 replace_stride=output_stride == 8)
        self.layer4 = make_stage(512, 3, stride=2, replace_stride=True)

    def forward(self, x):
        x = self.pool(self.stem(x))       # [B, 64, H/4, W/4]
        low_level = self.layer1(x)        # [B, 256, H/4, W/4]
        x = self.layer2(low_level)        # [B, 512, H/8, W/8]
        x = self.layer3(x)                # [B, 1024, H/OS, W/OS]
        x = self.layer4(x)                # [B, 2048, H/OS, W/OS]
        return x, low_level
