"""论文 §3.2：微调到新分辨率时，调整已有位置编码。"""

import torch
import torch.nn.functional as F


def resize_position_embedding(position_embedding, old_grid_size, new_grid_size):
    """保留 class token 的位置向量，只对 patch 的二维位置网格插值。

    输入：[1, 1 + old_h*old_w, D]；输出：[1, 1 + new_h*new_w, D]。
    论文明确采用二维插值，但没有在正文指定插值核与 align_corners。
    此处选 bicubic + align_corners=False，属于工程补充。
    这是微调前的参数迁移，不是为了导出 ONNX 临时替换 forward。
    """
    old_h, old_w = old_grid_size
    new_h, new_w = new_grid_size
    if min(old_h, old_w, new_h, new_w) <= 0:
        raise ValueError("新旧位置网格的高、宽都必须为正。")
    if (position_embedding.ndim != 3 or position_embedding.shape[0] != 1
            or position_embedding.shape[1] != 1 + old_h * old_w):
        raise ValueError("position_embedding 形状与 old_grid_size 不匹配。")
    if (old_h, old_w) == (new_h, new_w):
        return position_embedding

    cls_position = position_embedding[:, :1, :]     # [1,1,D]，不插值
    patch_positions = position_embedding[:, 1:, :]  # [1,old_h*old_w,D]
    embed_dim = patch_positions.shape[-1]
    patch_positions = patch_positions.reshape(1, old_h, old_w, embed_dim)
    patch_positions = patch_positions.permute(0, 3, 1, 2)  # [1,D,old_h,old_w]
    patch_positions = F.interpolate(
        patch_positions, size=(new_h, new_w), mode="bicubic", align_corners=False
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1)
    patch_positions = patch_positions.reshape(1, new_h * new_w, embed_dim)
    return torch.cat((cls_position, patch_positions), dim=1)
