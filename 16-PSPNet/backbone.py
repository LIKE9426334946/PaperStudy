"""输出步长为 8 的膨胀 ResNet；不调用 torchvision 的现成骨干。"""

from torch import nn

from blocks import Bottleneck


class DilatedResNet(nn.Module):
    """论文要求 ResNet + dilation + output stride 8。

    工程补全参考作者 hszhao/semseg：三层 3×3 stem，res4/res5 的
    所有 3×3 卷积分别使用 dilation=2/4。这里默认随机初始化，未下载预训练权重。
    """

    def __init__(self, depth=50):
        super().__init__()
        block_counts = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
        counts = block_counts[depth]

        # Deep stem：3→64→64→128；仅第一个卷积 stride=2。
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.res2 = self._make_stage(128, 64, counts[0], stride=1, dilation=1)
        self.res3 = self._make_stage(256, 128, counts[1], stride=2, dilation=1)
        # 不再做 /16、/32 下采样，改用空洞卷积扩大感受野。
        self.res4 = self._make_stage(512, 256, counts[2], stride=1, dilation=2)
        self.res5 = self._make_stage(1024, 512, counts[3], stride=1, dilation=4)

    @staticmethod
    def _make_stage(in_channels, channels, count, stride, dilation):
        blocks = [Bottleneck(in_channels, channels, stride, dilation)]
        for _ in range(1, count):
            # 每次构造一个新块，参数不共享。
            blocks.append(Bottleneck(channels * 4, channels, dilation=dilation))
        return nn.Sequential(*blocks)

    def forward(self, x):
        # 以下形状以输入 [B, 3, 256, 256] 为例。
        x = self.relu(self.bn1(self.conv1(x)))  # [B, 64, 128, 128]
        x = self.relu(self.bn2(self.conv2(x)))  # [B, 64, 128, 128]
        x = self.relu(self.bn3(self.conv3(x)))  # [B, 128, 128, 128]
        x = self.maxpool(x)                    # [B, 128, 64, 64]
        x = self.res2(x)                       # [B, 256, 64, 64]
        x = self.res3(x)                       # [B, 512, 32, 32]
        aux_features = self.res4(x)            # [B, 1024, 32, 32]
        main_features = self.res5(aux_features)  # [B, 2048, 32, 32]
        return aux_features, main_features
