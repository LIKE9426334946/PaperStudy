"""半 patch 重叠、Gaussian 概率加权、镜像 TTA，以及 2D 逐层预测。"""

from itertools import product
import math

import torch
import torch.nn.functional as F


def gaussian_importance(patch_size, device):
    # sigma=patch/8 是常见实现补全；论文正文只要求 Gaussian 中心加权。
    coordinates = [torch.arange(n, device=device).float() for n in patch_size]
    grid = torch.meshgrid(*coordinates, indexing="ij")
    distance = sum(((g - (n - 1) / 2) / (n / 8)) ** 2
                   for g, n in zip(grid, patch_size))
    weight = torch.exp(-0.5 * distance)
    return (weight / weight.max()).clamp_min(1e-7)


def window_starts(size, patch):
    count = math.ceil((size - patch) / (patch * 0.5)) + 1
    if count == 1:
        return [0]
    # 末窗精确贴边，实际步长不大于半个 patch，确保无空隙。
    return [round(i * (size - patch) / (count - 1)) for i in range(count)]


def mirrored_probability(model, patch, mirror=True):
    dims = patch.ndim - 2
    probability = None
    combinations = list(product((False, True), repeat=dims)) if mirror else [(False,) * dims]
    for switches in combinations:
        axes = tuple(i + 2 for i, use in enumerate(switches) if use)
        x = torch.flip(patch, axes) if axes else patch
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        prediction = logits.softmax(1)
        prediction = torch.flip(prediction, axes) if axes else prediction
        probability = prediction if probability is None else probability + prediction
    return probability / len(combinations)


@torch.inference_mode()
def sliding_window_predict(model, image, patch_size, mirror=True):
    """model 已设为 eval；image 为单个病例 [C,*spatial]，与模型同 device。

    返回 [K,*spatial] 的概率，未做 argmax/连通域处理。
    当前参考实现将累加器放在同一 device；超大体积可改为 CPU 累加。
    """
    original_shape = image.shape[1:]
    padded_shape = [max(s, p) for s, p in zip(original_shape, patch_size)]
    padding = [v for s, n in reversed(list(zip(original_shape, padded_shape))) for v in (0, n - s)]
    padded = F.pad(image, padding)  # 归一化图像采用零填充，属于实现补全。
    importance = gaussian_importance(patch_size, image.device)
    accumulated = torch.zeros((model.num_classes, *padded_shape), device=image.device)
    weight_sum = torch.zeros(padded_shape, device=image.device)
    starts = [window_starts(s, p) for s, p in zip(padded_shape, patch_size)]
    for start in product(*starts):
        region = tuple(slice(a, a + p) for a, p in zip(start, patch_size))
        patch = padded[(slice(None),) + region].unsqueeze(0)
        probability = mirrored_probability(model, patch, mirror)[0].float()
        accumulated[(slice(None),) + region] += probability * importance
        weight_sum[region] += importance
    probability = accumulated / weight_sum.unsqueeze(0)
    return probability[(slice(None),) + tuple(slice(0, n) for n in original_shape)]


@torch.inference_mode()
def predict_volume_with_2d(model, image, patch_size, slice_axis=0, mirror=True):
    """[C,D,H,W] -> 各切片二维预测 -> [K,D,H,W]，恢复原轴次序。"""
    slices = image.movedim(slice_axis + 1, 1)
    predictions = [sliding_window_predict(model, slices[:, i], patch_size, mirror)
                   for i in range(slices.shape[1])]
    return torch.stack(predictions, dim=1).movedim(1, slice_axis + 1)
