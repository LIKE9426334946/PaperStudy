"""论文图 2 的 overlap-tile：输入块重叠，仅拼接各块的有效输出区域。"""

import torch
from torch import Tensor, nn


def _reflection_indices(start: int, count: int, length: int, device) -> Tensor:
    """将越界位置反复镜像回图像内，不重复边缘像素。

    这是具体的镜像约定，不是论文规定的唯一方式。
    使用索引而非一次性 F.pad，让小于 tile 的图像也能反复镜像。
    """
    positions = torch.arange(start, start + count, device=device)
    if length == 1:
        return torch.zeros_like(positions)
    period = 2 * (length - 1)
    positions = positions.remainder(period)
    return torch.where(positions < length, positions, period - positions)


@torch.no_grad()
def overlap_tile_predict(
    model: nn.Module,
    image: Tensor,
    tile_size: int = 572,
) -> Tensor:
    """用本目录的 UNet 对 [B, C, H, W] 原图分块推理，返回 [B, K, H, W]。

    调用前使用 model.eval()，并使图像与模型的 device、dtype 一致。
    tile_size=572 时，每块输出 388x388，输入上下左右各多取 92 像素。
    输出块按 388 像素步长铺排；最后一块只保留图像范围内的结果。
    这里只拼接 logits；没有概率平均、旋转集成或后处理。
    """
    if model.training:
        raise ValueError("请先调用 model.eval()，关闭推理阶段的 Dropout")
    if tile_size < 188 or tile_size % 16 != 12:
        raise ValueError("tile_size 须 >= 188 且除以 16 的余数为 12")

    batch, _, height, width = image.shape
    border = 92
    output_size = tile_size - 2 * border
    result = image.new_empty(batch, model.num_classes, height, width)

    for top in range(0, height, output_size):
        rows = _reflection_indices(top - border, tile_size, height, image.device)
        for left in range(0, width, output_size):
            columns = _reflection_indices(
                left - border, tile_size, width, image.device,
            )
            tile = image.index_select(-2, rows).index_select(-1, columns)
            tile_logits = model(tile)  # [B, K, output_size, output_size]

            valid_height = min(output_size, height - top)
            valid_width = min(output_size, width - left)
            result[..., top:top + valid_height, left:left + valid_width] = (
                tile_logits[..., :valid_height, :valid_width]
            )
    return result
