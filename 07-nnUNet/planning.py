"""论文 Methods 中的规则规划，输入 fingerprint，输出网络及 patch 配置。

重要边界：显存用 feature_budget（特征元素数代理）表示，必须按硬件实测标定。
本文件不声称复刻官方 planner 的显存常数或保证某个 GB 数绝不 OOM。
"""

import numpy as np


def target_spacing(spacings, cropped_shapes):
    spacings = np.asarray(spacings, dtype=float)
    shapes = np.median(cropped_shapes, axis=0)
    target = np.median(spacings, axis=0)
    lowres_axis = int(target.argmax())  # spacing 越大，物理分辨率越低。
    other = [i for i in range(len(target)) if i != lowres_axis]
    spacing_anisotropy = target[lowres_axis] > 3 * target[other].max()
    voxel_anisotropy = shapes[lowres_axis] * 3 < shapes[other].min()
    if spacing_anisotropy and voxel_anisotropy:
        target[lowres_axis] = np.percentile(spacings[:, lowres_axis], 10)
    return target


def configure_topology(patch_size, spacing):
    shape = np.asarray(patch_size, dtype=int).copy()
    current_spacing = np.asarray(spacing, dtype=float).copy()
    strides, kernels = [], []
    # 论文的 pseudo-2D 卷积核调整针对 3D；二维模板保持 3x3。
    use_three = np.ones(2, dtype=bool) if len(shape) == 2 else np.zeros(3, dtype=bool)
    while True:
        # 一旦该轴使用 3，就保持 3。各向异性粗轴初期为 1。
        use_three |= current_spacing <= 2 * current_spacing.min()
        kernels.append(tuple(np.where(use_three, 3, 1).tolist()))
        can_pool = (shape >= 8) & (current_spacing <= 2 * current_spacing.min())
        if not can_pool.any():
            break  # 再降采样会小于 4，或仍不满足物理分辨率约束。
        stride = np.where(can_pool, 2, 1)
        strides.append(tuple(stride.tolist()))
        shape //= stride
        current_spacing *= stride
    return kernels, strides


def feature_elements(patch_size, kernels, strides, in_channels, num_classes):
    """单个样本的特征图规模代理，不等于运行时显存字节数。

    累计输入、编码器、转置卷积、concat、解码器、监督头的特征元素。
    不包含优化器状态、卷积 workspace 和框架缓存，所以需要参考实测标定。
    """
    shape = np.array(patch_size, dtype=int)
    limit = 512 if len(shape) == 2 else 320
    channels = [min(32 * 2 ** i, limit) for i in range(len(kernels))]
    shapes = [shape.copy()]
    for stride in strides:
        shape = shape // stride
        shapes.append(shape.copy())
    score = np.prod(shapes[0]) * in_channels
    for i, shape in enumerate(shapes):
        score += 2 * np.prod(shape) * channels[i]
        if i < len(shapes) - 1:
            score += 5 * np.prod(shape) * channels[i]  # up + concat + 两次卷积
        if i < max(1, len(shapes) - 2):
            score += np.prod(shape) * num_classes
    return float(score)


def plan_configuration(median_shape, spacing, total_voxels, feature_budget,
                       in_channels=1, num_classes=2):
    """feature_budget 为整个 batch 的标定预算；优先容纳 batch=2 的大 patch。"""
    median_shape = np.asarray(median_shape, dtype=float)
    patch = np.maximum(np.rint(median_shape).astype(int), 4)
    reduced = False
    while True:
        kernels, strides = configure_topology(patch, spacing)
        divisor = np.prod(strides, axis=0) if strides else np.ones(len(patch), int)
        padded = ((patch + divisor - 1) // divisor) * divisor
        if not np.array_equal(patch, padded):
            patch = padded
            continue
        score = feature_elements(patch, kernels, strides, in_channels, num_classes)
        if 2 * score <= feature_budget:
            break
        candidates = np.where(patch - divisor >= 4)[0]
        if len(candidates) == 0:
            raise ValueError("该预算无法容纳最小 patch 的 batch=2，请提高标定预算")
        axis = candidates[np.argmax(patch[candidates] / median_shape[candidates])]
        patch[axis] -= divisor[axis]
        reduced = True

    memory_batch = 2 if reduced else int(feature_budget // score)
    dataset_batch = int(0.05 * total_voxels / np.prod(patch))
    # 极小数据集出现冲突时优先保留 batch>=2，这是明确的实现补全。
    batch_size = max(2, min(memory_batch, dataset_batch))
    return {"patch_size": tuple(patch.tolist()), "spacing": tuple(spacing),
            "kernel_sizes": kernels, "strides": strides, "batch_size": batch_size,
            "feature_elements_per_sample": score,
            "coverage": float(np.prod(patch) / np.prod(median_shape)),
            "in_channels": in_channels, "num_classes": num_classes}


def plan_experiment(fingerprint, feature_budget_2d, feature_budget_3d):
    """3D 原始病例 -> 2D、3D full-res，以及按需生成的 low-res/cascade 配置。

    返回的 2D 配置注明切片轴；预处理时保留这个轴的原始 spacing。
    """
    spacings = np.asarray(fingerprint["spacings"])
    shapes = np.asarray(fingerprint["cropped_shapes"])
    c, k = len(fingerprint["modalities"]), fingerprint["num_classes"]

    def plan_at(spacing, budget, channels=c):
        resampled = np.maximum(1, np.rint(shapes * spacings / spacing)).astype(int)
        return plan_configuration(np.median(resampled, axis=0), spacing,
                                  np.prod(resampled, axis=1).sum(), budget, channels, k)

    full_spacing = target_spacing(spacings, shapes)
    full = plan_at(full_spacing, feature_budget_3d)
    slice_axis = int(np.median(spacings, axis=0).argmax())
    axes = [a for a in range(3) if a != slice_axis]
    spacing_2d = np.median(spacings, axis=0)[axes]
    inplane_shapes = np.maximum(1, np.rint(shapes[:, axes] * spacings[:, axes] / spacing_2d))
    plan_2d = plan_configuration(np.median(inplane_shapes, axis=0), spacing_2d,
                                (np.prod(inplane_shapes, axis=1) * shapes[:, slice_axis]).sum(),
                                feature_budget_2d, c, k)
    plan_2d.update(slice_axis=slice_axis, inplane_axes=axes)
    result = {"2d": plan_2d, "3d_fullres": full}
    if full["coverage"] < 0.125:
        low_spacing = full_spacing.copy()
        low = full
        while low["coverage"] <= 0.25:
            axes_to_increase = low_spacing < low_spacing.max() / 2
            if not axes_to_increase.any():
                axes_to_increase[:] = True
            low_spacing[axes_to_increase] *= 1.01
            low = plan_at(low_spacing, feature_budget_3d)
        result["3d_lowres"] = low
        # 论文：第二阶段沿用 full-res 几何配置，增加低分辨率分割通道。
        result["3d_cascade_fullres"] = dict(full, in_channels=c + k - 1)
        result["3d_cascade_fullres"]["feature_elements_per_sample"] = feature_elements(
            full["patch_size"], full["kernel_sizes"], full["strides"], c + k - 1, k)
    return result
