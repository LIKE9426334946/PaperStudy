"""论文式 (2) 的逐项映射，以及注明补全内容的二分类 BCE + Dice。

logits/target: [B, 1, H, W] 或 [B, 1, D, H, W]。
target 为 0/1，不能把 0/255 的掩码直接送入；函数不做隐式阈值化。
flatten(1) 保留 batch 维：每个样本先计算 Dice，再在 batch 内平均。
"""

import torch
from torch.nn import functional as F


def soft_dice_per_image(logits, target, smooth=1e-6):
    """每幅图/体数据一个 Dice；平滑项是数值实现补全。"""
    p = logits.sigmoid().flatten(start_dim=1)  # [B, M]，M 为像素/体素数
    y = target.to(dtype=logits.dtype).flatten(start_dim=1)
    intersection = (p * y).sum(dim=1)          # [B]，逐像素乘积后求和
    denominator = p.sum(dim=1) + y.sum(dim=1)
    return (2 * intersection + smooth) / (denominator + smooth)


def paper_equation2_loss(logits, target, smooth=1e-6):
    """保留印刷式 (2) 中可见的前景 log 项和负 Dice，供对照阅读。

    印刷式没有展开背景项、像素规约和 epsilon，因此不存在唯一的
    逐字数值实现。这里选择像素均值，并加入 Dice 平滑；它们是补全。
    此函数没有 (1-y)*log(1-p)，并非完整 BCE，默认训练例子不用它。
    """
    y = target.to(dtype=logits.dtype)
    # log(sigmoid(z)) 用 logsigmoid 稳定计算；不能先把预测阈值化。
    foreground_log = (y * F.logsigmoid(logits)).flatten(1).mean(dim=1)
    dice = soft_dice_per_image(logits, y, smooth)
    # 式 (2) 的最外层负号和 1/N -> 负的 batch mean。
    return -(0.5 * foreground_log + dice).mean()


def bce_dice_loss(logits, target, smooth=1e-6):
    """采用完整 BCE 的补全版本：0.5*BCE + (1-Dice)。

    系数 0.5 对应论文式 (2)。完整 BCE 与作者公开实现的意图一致，
    但背景项没有显式写在论文公式里，不能冒称这是公式的逐字翻译。
    相比 0.5*BCE-Dice 加了常数 1，不改变梯度。Dice 按图计算遵循
    论文的样本索引；作者 Keras helper 则把整个 batch flatten。
    """
    y = target.to(dtype=logits.dtype)
    # = -mean[y*log(p) + (1-y)*log(1-p)]，直接从 logits 稳定计算。
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = bce.flatten(start_dim=1).mean(dim=1)  # [B]
    dice = soft_dice_per_image(logits, y, smooth)
    return (0.5 * bce + 1.0 - dice).mean()


def deep_supervision_loss(outputs, target, loss_fn=bce_dice_loss):
    """对各个预测头分别监督，再等权平均损失；不是先平均预测。

    四头训练传入 (z1, z2, z3, z4)；无深监督时可传单个 z4。
    等权平均是本文的实现选择，论文没有规定总损失的权重/规约。
    如需对照印刷公式，可传 loss_fn=paper_equation2_loss。
    """
    if isinstance(outputs, torch.Tensor):
        return loss_fn(outputs, target)
    return torch.stack([loss_fn(logits, target) for logits in outputs]).mean()
