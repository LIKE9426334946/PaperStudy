"""论文采用多类 Sorensen-Dice 与深监督；下面将未指定的细节明确补全。"""

import torch
from torch.nn import functional as F


def soft_dice_loss(logits, target, eps=1e-6):
    """logits: [B,K,*S]；target: [B,*S]，整型类别 ID，无 ignore_index。

    默认包含背景，对 batch 和空间维求和，再对类别平均。
    选择平方分母的 soft Dice；平滑项、聚合方式是参考实现选择。
    K=1 时将 sigmoid 前景概率扩展为 [背景,前景]，仍对两类计算 Dice。
    """
    if logits.shape[1] == 1:
        foreground = torch.sigmoid(logits)
        probabilities = torch.cat((1.0 - foreground, foreground), dim=1)
        classes = 2
    else:
        probabilities = torch.softmax(logits, dim=1)
        classes = logits.shape[1]

    # [B,*S] -> [B,*S,K] -> [B,K,*S]
    one_hot = F.one_hot(target.long(), num_classes=classes).movedim(-1, 1)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    dims = (0,) + tuple(range(2, logits.ndim))
    intersection = (probabilities * one_hot).sum(dim=dims)  # [K]
    denominator = (probabilities.square() + one_hot.square()).sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def deep_supervision_loss(outputs, target, weights=(1.0, 0.5, 0.25)):
    """三个输出已经与 target 同分辨率，不对离散标签做线性插值。

    权重 (1,.5,.25) 及归一化加权平均是教学补全，不是论文给定的超参数。
    关闭深监督时 outputs 为单个 Tensor，只计算主头损失。
    """
    if isinstance(outputs, torch.Tensor):
        return soft_dice_loss(outputs, target)
    if len(outputs) != len(weights) or not weights or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError("每个输出需对应非负权重，且权重和必须大于 0")
    losses = [weight * soft_dice_loss(logits, target)
              for weight, logits in zip(weights, outputs)]
    return sum(losses) / sum(weights)
