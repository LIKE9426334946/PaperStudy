"""训练集 fingerprint、非零裁剪、强度归一化和重采样。

输入数组为 [C,D,H,W] 或 [C,H,W]；标签没有 channel 维。
调用者负责统一医学图像的方向、坐标系和 spacing 的轴顺序。
"""

import numpy as np
from scipy import ndimage


def crop_nonzero(image, label=None):
    mask = ndimage.binary_fill_holes(np.any(image != 0, axis=0))
    coordinates = np.where(mask)
    box = tuple(slice(int(a.min()), int(a.max()) + 1) for a in coordinates)
    cropped_label = None if label is None else label[box]
    return image[(slice(None),) + box], cropped_label, mask[box], box


def dataset_fingerprint(cases, modalities, num_classes):
    """cases: [{image, label, spacing}, ...]，只传训练集，不传测试标签。

    为清楚展示统计公式，本参考实现拼接全部前景值；大数据集应流式统计/抽样。
    """
    before, after, spacings, ratios = [], [], [], []
    foreground_values = [[] for _ in modalities]
    class_counts = np.zeros(num_classes, dtype=np.int64)
    for case in cases:
        image, label, _, _ = crop_nonzero(case["image"], case["label"])
        before.append(case["image"].shape[1:])
        after.append(image.shape[1:])
        spacings.append(case["spacing"])
        ratios.append(label.size / case["label"].size)
        class_counts += np.bincount(label.ravel(), minlength=num_classes)
        for c in range(len(modalities)):
            foreground_values[c].append(image[c][label > 0])
    stats = []
    for values in foreground_values:
        values = np.concatenate(values)
        stats.append({"mean": float(values.mean()), "std": float(values.std()),
                      "lower": float(np.percentile(values, 0.5)),
                      "upper": float(np.percentile(values, 99.5))})
    return {"original_shapes": before, "cropped_shapes": after, "spacings": spacings,
            "modalities": list(modalities), "num_classes": num_classes,
            "num_cases": len(cases), "intensity_stats": stats,
            "class_ratios": (class_counts / class_counts.sum()).tolist(),
            "use_nonzero_mask": bool(np.mean(ratios) <= 0.75)}


def normalize(image, nonzero_mask, fingerprint):
    normalized = image.astype(np.float32).copy()
    for c, modality in enumerate(fingerprint["modalities"]):
        if modality.upper() == "CT":
            stats = fingerprint["intensity_stats"][c]
            clipped = np.clip(normalized[c], stats["lower"], stats["upper"])
            normalized[c] = (clipped - stats["mean"]) / max(stats["std"], 1e-8)
        else:
            mask = nonzero_mask if fingerprint["use_nonzero_mask"] else np.ones_like(nonzero_mask)
            values = normalized[c][mask]
            normalized[c] = (normalized[c] - values.mean()) / max(float(values.std()), 1e-8)
            normalized[c][~mask] = 0
    return normalized


def resize_array(array, target_shape, order):
    # 明确使用体素中心坐标和最近边界扩展；坐标约定属于实现补全。
    coordinates = [(np.arange(n) + 0.5) * old / n - 0.5
                   for old, n in zip(array.shape, target_shape)]
    grid = np.stack(np.meshgrid(*coordinates, indexing="ij"))
    return ndimage.map_coordinates(array, grid, order=order, mode="nearest")


def resample_channels(image, target_shape, order, lowres_axis=None):
    def resize_channel(channel):
        if lowres_axis is None:
            return resize_array(channel, target_shape, order)
        # 先独立重采样每一张高分辨率切片，再对粗轴用最近邻。
        slices = np.moveaxis(channel, lowres_axis, 0)
        inplane_shape = [n for i, n in enumerate(target_shape) if i != lowres_axis]
        inplane = np.stack([resize_array(s, inplane_shape, order) for s in slices])
        inplane = np.moveaxis(inplane, 0, lowres_axis)
        return resize_array(inplane, target_shape, order=0)
    return np.stack([resize_channel(channel) for channel in image]).astype(np.float32)


def preprocess_case(image, spacing, target_spacing, fingerprint, label=None):
    """返回 image、label、恢复到原网格需要的 metadata。

    2D 切片模式下，target_spacing 的切片轴填入该病例原 spacing，避免跨层重采样。
    标签采用论文的 one-hot -> 线性插值 -> argmax，而非类别 ID 的线性插值。
    """
    original_shape = image.shape[1:]
    image, label, mask, box = crop_nonzero(image, label)
    cropped_shape = image.shape[1:]
    image = normalize(image, mask, fingerprint)
    spacing, target_spacing = np.asarray(spacing), np.asarray(target_spacing)
    target_shape = np.maximum(1, np.rint(np.array(cropped_shape) * spacing / target_spacing)).astype(int)
    lowres_axis = None
    for candidate in (spacing, target_spacing):
        # 多个同样粗的轴时不选唯一 out-of-plane 轴，这是实现补全。
        if candidate.max() / candidate.min() > 3 and np.sum(candidate == candidate.max()) == 1:
            lowres_axis = int(candidate.argmax())
            break
    image = resample_channels(image, target_shape, order=3, lowres_axis=lowres_axis)
    if label is not None:
        # moveaxis 保持原始空间轴次序，不能通过任意 transpose 重排。
        one_hot = np.moveaxis(np.eye(fingerprint["num_classes"], dtype=np.float32)[label], -1, 0)
        label = resample_channels(one_hot, target_shape, order=1,
                                  lowres_axis=lowres_axis).argmax(0)
    metadata = {"original_shape": original_shape, "cropped_shape": cropped_shape,
                "box": box, "lowres_axis": lowres_axis}
    return image, label, metadata


def restore_prediction(probability, metadata):
    """恢复概率到裁剪前网格，最后 argmax；裁剪区域外填背景 0。"""
    probability = resample_channels(probability, metadata["cropped_shape"], order=1,
                                    lowres_axis=metadata["lowres_axis"])
    segmentation = np.zeros(metadata["original_shape"], dtype=np.int64)
    segmentation[metadata["box"]] = probability.argmax(0)
    return segmentation
