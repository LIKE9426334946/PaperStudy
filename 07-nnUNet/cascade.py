"""3D low-res -> full-res 的输入衔接，不把两个网络伪装为联合训练。"""

import torch
import torch.nn.functional as F


def make_cascade_input(full_image, lowres_probability):
    """两个输入覆盖同一物理区域并使用相同轴顺序。

    full_image: [B,C,D,H,W]；lowres_probability: [B,K,d,h,w]。
    先对概率恢复分辨率，再 argmax、one-hot，移除背景通道并拼接。
    论文描述拼接分割结果；这里采用前景 one-hot 的常见实现补全。
    训练第二阶段必须用第一阶段的折外预测，不能输入真值或拟合内预测。
    """
    probability = F.interpolate(lowres_probability.detach(),
                                size=full_image.shape[2:], mode="trilinear",
                                align_corners=False)
    labels = probability.argmax(dim=1)
    foreground = F.one_hot(labels, probability.shape[1]).movedim(-1, 1)[:, 1:]
    return torch.cat((full_image, foreground.to(full_image.dtype)), dim=1)
