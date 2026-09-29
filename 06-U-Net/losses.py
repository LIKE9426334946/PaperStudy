"""U-Net 论文的像素权重图（式 2）和加权交叉熵（式 1）。"""

from torch import Tensor
import torch
import torch.nn.functional as F


def boundary_weight_map(
    class_weight_map: Tensor,
    instance_distances: Tensor,
    w0: float = 10.0,
    sigma: float = 5.0,
) -> Tensor:
    """式 (2)：w = w_c + w0 * exp(-(d1+d2)^2 / (2*sigma^2))。

    class_weight_map: [B, H, W]，预先按训练集类别频率确定的 w_c。
    instance_distances: [B, N, H, W]，每个像素到每一个细胞边界的
        非负欧氏距离（单位：像素），由带实例 ID 的标注预计算。
        N 维必须对应不同细胞，不能是同一细胞的不同边界点。
        batch 中缺少的实例可用 +inf 距离填充。

    论文未指定 w_c 的具体归一化、形态学结构元素及距离计算程序；
    本函数将这些数据准备结果作为输入，不臆造它们是论文原始代码。
    少于两个实例时不添加分隔边界项，是明示的边界情况处理。
    """
    if instance_distances.shape[1] < 2:
        return class_weight_map
    nearest_two = torch.topk(
        instance_distances, k=2, dim=1, largest=False,
    ).values  # [B, 2, H, W]，沿“不同细胞”这一维选择。
    d1, d2 = nearest_two.unbind(dim=1)  # 各为 [B, H, W]
    separation_weight = w0 * torch.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
    return class_weight_map + separation_weight


def weighted_cross_entropy(
    logits: Tensor,
    target: Tensor,
    weight_map: Tensor | None = None,
    reduction: str = "sum",
) -> Tensor:
    """展开像素 Softmax + 加权负对数似然；默认求和对应论文式 (1)。

    logits: [B, K, H, W]，未经 Softmax 的网络输出。
    target: [B, H, W]，torch.long，类别编号为 0 到 K-1。
    weight_map: [B, H, W]，非负像素权重；None 表示全为 1。
    target 和 weight_map 必须事先裁剪到输出区域，不会在此自动缩放。

    上传论文的式 (1) 写作 sum(w*log(p))，没有负号。
    本函数作为待最小化的交叉熵，明确使用 -sum(w*log(p))。
    """
    # log p_k = a_k - log(sum_j exp(a_j))；log_softmax 避免数值溢出。
    log_probabilities = F.log_softmax(logits, dim=1)  # [B, K, H, W]
    true_class_log_prob = log_probabilities.gather(
        dim=1, index=target.unsqueeze(1),
    ).squeeze(1)  # [B, H, W]，取每个像素真实类别的 log p。
    pixel_loss = -true_class_log_prob
    if weight_map is not None:
        pixel_loss = weight_map * pixel_loss

    if reduction == "sum":
        return pixel_loss.sum()
    if reduction == "mean":
        # 工程可选项：除以 B*H*W，不是除以 weight_map.sum()。
        return pixel_loss.mean()
    if reduction == "none":
        return pixel_loss
    raise ValueError("reduction 必须为 'sum'、'mean' 或 'none'")
