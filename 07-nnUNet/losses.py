"""Dice + CE、多尺度深监督和 poly 学习率。"""

import torch
import torch.nn.functional as F


def dice_ce_loss(logits, target, batch_dice=False, smooth=1e-5):
    """logits: [B,K,*spatial]；target: [B,*spatial]，整数 0..K-1。

    Dice 不计算背景；CE 包含背景。batch_dice、平滑项属于实现补全。
    这里使用 1-Dice；相对 -Dice 只相差常数，不改变梯度。
    """
    ce = F.cross_entropy(logits, target)
    probability = logits.softmax(dim=1)
    one_hot = F.one_hot(target, logits.shape[1]).movedim(-1, 1).to(logits.dtype)
    axes = tuple(range(2, logits.ndim))
    if batch_dice:
        axes = (0,) + axes
    intersection = (probability * one_hot).sum(axes)
    denominator = (probability + one_hot).sum(axes)
    dice = (2 * intersection + smooth) / (denominator + smooth)
    foreground_dice = dice[1:] if batch_dice else dice[:, 1:]
    return ce + 1 - foreground_dice.mean()


def deep_supervision_loss(outputs, target, batch_dice=False):
    """outputs 已经排除不参与监督的粗尺度；权重为归一化的 1,1/2,1/4,...。"""
    weights = [0.5 ** i for i in range(len(outputs))]
    total_weight = sum(weights)
    loss = outputs[0].new_zeros(())
    for weight, logits in zip(weights, outputs):
        # 最近邻缩放的是标签，不是对类别 ID 做线性插值。
        scaled_target = F.interpolate(target[:, None].float(),
                                      size=logits.shape[2:], mode="nearest")
        loss = loss + weight / total_weight * dice_ce_loss(
            logits, scaled_target[:, 0].long(), batch_dice)
    return loss


def poly_learning_rate(epoch, max_epochs=1000, initial_lr=0.01):
    return initial_lr * (1 - epoch / max_epochs) ** 0.9


def foreground_patch_count(batch_size):
    # 论文保证至少一个前景 patch；batch=2 时一个随机、一个前景。
    return max(1, round(batch_size / 3))
