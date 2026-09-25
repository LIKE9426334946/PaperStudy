"""论文第 3.1 节的核心增强：3x3 高斯位移网格 -> 平滑弹性形变。

这不是完整数据增强流水线。平移、旋转、灰度变化只在 README 中说明。
"""

import torch
from torch import Tensor
import torch.nn.functional as F


def elastic_deform_pair(
    image: Tensor,
    target: Tensor,
    displacement_std: float = 10.0,
) -> tuple[Tensor, Tensor]:
    """对原图和标签使用同一采样网格，返回尺寸不变的一对 Tensor。

    image: [B, C, H, W] 浮点图像；target: [B, H, W] 整数标签。
    H、W > 1，二者空间尺寸与 device 相同。
    默认 3x3 网格、位移标准差 10 像素及位移场的 bicubic 插值来自论文。
    反向采样、align_corners=True、反射边界、图像 bilinear 与标签
    nearest 插值均是本实现明确选择的细节，论文没有逐项给出。
    """
    batch, _, height, width = image.shape

    # 两个通道依次存 dx、dy；每张图独立采样。
    coarse_displacement = torch.randn(
        batch, 2, 3, 3, device=image.device, dtype=image.dtype,
    ) * displacement_std
    displacement = F.interpolate(
        coarse_displacement, size=(height, width), mode="bicubic",
        align_corners=True,
    )  # [B, 2, H, W]；单位仍是输入像素。

    # grid_sample 的坐标顺序为 (x, y)，归一化范围为 [-1, 1]。
    ys = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
    xs = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    normalized_displacement = torch.stack(
        [2 * displacement[:, 0] / (width - 1),
         2 * displacement[:, 1] / (height - 1)],
        dim=-1,
    )  # [B, H, W, 2]
    sample_grid = base_grid + normalized_displacement

    # 两者共享 sample_grid，避免图像和标签发生几何错位。
    deformed_image = F.grid_sample(
        image, sample_grid, mode="bilinear", padding_mode="reflection",
        align_corners=True,
    )
    deformed_target = F.grid_sample(
        target.unsqueeze(1).to(image.dtype), sample_grid, mode="nearest",
        padding_mode="reflection", align_corners=True,
    ).squeeze(1).to(target.dtype)  # 最近邻不会产生新的类别编号。
    return deformed_image, deformed_target
