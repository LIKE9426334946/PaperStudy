"""FCN 的 VGG16 特征提取、分类层卷积化、双线性反卷积初始化和裁剪。

依据上传的 CVPR 2015 论文；padding=100、ceil pooling、crop offset
采用作者 Caffe 模型的空间对齐方式，详见 README 的实现边界。
"""

import torch
from torch import nn
from torch.nn import functional as F


class VGG16Features(nn.Module):
    """展开全部 13 个卷积；不引入 BatchNorm、残差块或全局平均池化。"""

    def __init__(self):
        super().__init__()
        # 第一层的大 padding 为后续 7x7 valid 卷积和裁剪保留边界空间。
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        # 以下尺寸以 x=[B,3,256,256] 为例。
        x = F.relu(self.conv1_1(x))         # [B,64,454,454]
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)                  # [B,64,227,227]

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)                  # [B,128,114,114]

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        pool3 = self.pool3(x)              # [B,256,57,57]，有效步幅 8

        x = F.relu(self.conv4_1(pool3))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        pool4 = self.pool4(x)              # [B,512,29,29]，有效步幅 16

        x = F.relu(self.conv5_1(pool4))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        pool5 = self.pool5(x)              # [B,512,15,15]，有效步幅 32
        return pool3, pool4, pool5


class ConvolutionalClassifier(nn.Module):
    """fc6: Linear(512*7*7,4096) -> Conv2d(512,4096,7)。

    fc7: Linear(4096,4096) -> Conv2d(4096,4096,1)。
    原来的 1000 类 fc8 被丢弃，新的 score_fr 在 FCN 中单独定义。
    fc_channels=4096 是原始宽度；更小的宽度只是调试变体。
    """

    def __init__(self, fc_channels=4096):
        super().__init__()
        self.fc6 = nn.Conv2d(512, fc_channels, kernel_size=7, padding=0)
        self.drop6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Conv2d(fc_channels, fc_channels, kernel_size=1)
        self.drop7 = nn.Dropout(p=0.5)

    def forward(self, pool5):
        x = F.relu(self.fc6(pool5))         # [B,4096,9,9]
        x = self.drop6(x)                  # 原始逐元素 Dropout，不是 Dropout2d
        x = F.relu(self.fc7(x))            # [B,4096,9,9]
        return self.drop7(x)


def bilinear_upsampler(num_classes, stride, trainable):
    """以双线性核初始化 ConvTranspose2d；中间层可学习，末级固定。

    对角通道初始化为同一个二维帐篷核，非对角通道初始化为 0。
    上采样操作仍留在 forward 图内；冻结参数不会阻断输入梯度。
    """
    kernel_size = 2 * stride
    layer = nn.ConvTranspose2d(
        num_classes, num_classes, kernel_size=kernel_size,
        stride=stride, padding=0, bias=False,
    )
    # k=2s 时中心 c=s-0.5，b(u)=max(0,1-|u-c|/s)。
    positions = torch.arange(kernel_size, dtype=layer.weight.dtype,
                             device=layer.weight.device)
    one_dimensional = (1 - (positions - (stride - 0.5)).abs() / stride).clamp(min=0)
    kernel = one_dimensional[:, None] * one_dimensional[None, :]
    with torch.no_grad():
        layer.weight.zero_()
        indices = torch.arange(num_classes, device=layer.weight.device)
        layer.weight[indices, indices] = kernel
    layer.weight.requires_grad_(trainable)
    return layer


def crop_like(source, reference, offset):
    """用固定采样原点 offset 对齐，再按 reference 的 H/W 裁剪。

    不是按尺寸差中心裁剪，也不是插值；ONNX 对应 Slice。
    offsets 依赖本目录的 padding/ceil-pooling/kernel/stride，不能孤立修改。
    """
    return source[:, :, offset:offset + reference.shape[2],
                  offset:offset + reference.shape[3]]
