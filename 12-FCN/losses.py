"""对应 FCN 的逐像素 multinomial logistic loss 和 Sec.5 四项指标。"""

import torch
from torch.nn import functional as F


def pixelwise_cross_entropy(logits, target, ignore_index=255, reduction="mean"):
    """logits [B,K,H,W]；target [B,H,W]，torch.long 类别索引。

    reduction='sum' 对应论文空间损失求和公式。
    默认 'mean' 除以有效像素数，便于教学，是明确的工程选择。
    全部标签均忽略时返回可反传的 0，避免 mean 空集合产生 NaN。
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction 仅支持 mean 或 sum")
    # 内部包含 log_softmax 和负对数似然；不要先对 logits 做 softmax。
    per_pixel = F.cross_entropy(logits, target, ignore_index=ignore_index,
                                reduction="none")
    total = per_pixel.sum()
    if reduction == "sum":
        return total
    return total / (target != ignore_index).sum().clamp_min(1)


@torch.no_grad()
def confusion_matrix(logits, target, ignore_index=255):
    """n_ij：真实类 i、预测类 j 的像素数；行是真实类别，列是预测类别。

    数据集指标应先逐批累加混淆矩阵，再调用 segmentation_metrics；
    不能简单平均每个 batch 的 mIoU。
    """
    num_classes = logits.shape[1]
    prediction = logits.argmax(dim=1)
    valid = target != ignore_index
    truth = target[valid].to(torch.int64)
    if torch.any((truth < 0) | (truth >= num_classes)):
        raise ValueError("非忽略标签必须位于 [0,num_classes) 内")
    indices = num_classes * truth + prediction[valid]
    return torch.bincount(indices, minlength=num_classes ** 2).reshape(
        num_classes, num_classes,
    )


def segmentation_metrics(matrix):
    """从累计 n_ij 展开 pixel acc、mean acc、mean IU、frequency-weighted IU。

    mIoU 包含有有效并集的背景类；GT 和预测均不存在的类记为 NaN 并排除。
    GT 不存在但被误预测出的类并集非零，IoU=0，仍计入本实现的 mIoU。
    这是缺失类别的显式约定；特定数据集的评测脚本可能采用另一类集合。
    """
    counts = matrix.to(torch.float64)
    true_count = counts.sum(dim=1)          # t_i = sum_j n_ij
    predicted_count = counts.sum(dim=0)     # sum_j n_ji
    true_positive = counts.diag()          # n_ii
    total = true_count.sum()
    union = true_count + predicted_count - true_positive
    nan = torch.full_like(true_count, float("nan"))
    accuracy = torch.where(true_count > 0, true_positive / true_count.clamp_min(1), nan)
    iou = torch.where(union > 0, true_positive / union.clamp_min(1), nan)
    total_or_nan = torch.where(total > 0, total, total.new_tensor(float("nan")))
    return {
        "pixel_accuracy": true_positive.sum() / total_or_nan,
        "mean_accuracy": accuracy.nanmean(),
        "mean_iou": iou.nanmean(),
        "frequency_weighted_iou": (true_count * iou.nan_to_num()).sum() / total_or_nan,
        "per_class_iou": iou,
    }
