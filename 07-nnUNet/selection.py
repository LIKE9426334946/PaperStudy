"""用折外验证结果选单模型/两模型集成，再决定是否移除小连通域。"""

from itertools import combinations

import numpy as np
from scipy import ndimage


def mean_dice_per_class(predictions, targets, num_classes):
    scores = []
    for prediction, target in zip(predictions, targets):
        case = []
        for label in range(1, num_classes):
            p, t = prediction == label, target == label
            denominator = p.sum() + t.sum()
            # 二者均空时不参与平均，是明确的评估实现约定。
            case.append(2 * np.logical_and(p, t).sum() / denominator
                        if denominator else np.nan)
        scores.append(case)
    return np.nanmean(scores, axis=0)


def select_ensemble(oof_probabilities, targets, num_classes):
    """dict[name] = 每个病例的 [K,*spatial] 概率，病例次序及物理网格一致。

    输入必须是交叉验证的折外预测；禁止用最终测试集选择配置。
    """
    names = list(oof_probabilities)
    candidates = [(name,) for name in names] + list(combinations(names, 2))
    scores = {}
    for members in candidates:
        predictions = [np.mean([oof_probabilities[m][i] for m in members], axis=0).argmax(0)
                       for i in range(len(targets))]
        scores[members] = float(np.nanmean(mean_dice_per_class(predictions, targets, num_classes)))
    selected = max(scores, key=scores.get)
    return selected, scores


def keep_largest(segmentation, classes):
    # 按面邻接定义连通性（2D 4 邻接/3D 6 邻接），属于实现补全。
    components, count = ndimage.label(np.isin(segmentation, classes))
    result = segmentation.copy()
    if count == 0:
        return result
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    result[(components != 0) & (components != sizes.argmax())] = 0
    return result


def select_postprocessing(predictions, targets, num_classes):
    """先将全部前景视为一类评估，再逐类评估，接受有益且不伤害任何类的规则。"""
    current = [p.copy() for p in predictions]
    rules = []
    baseline = mean_dice_per_class(current, targets, num_classes)
    groups = [tuple(range(1, num_classes))] + [(c,) for c in range(1, num_classes)]
    for classes in groups:
        candidate = [keep_largest(p, classes) for p in current]
        score = mean_dice_per_class(candidate, targets, num_classes)
        comparable = np.isfinite(baseline) & np.isfinite(score)
        if np.nanmean(score) > np.nanmean(baseline) and np.all(score[comparable] >= baseline[comparable]):
            current, baseline = candidate, score
            rules.append(classes)
    return rules, current


def apply_postprocessing(segmentation, rules):
    for classes in rules:
        segmentation = keep_largest(segmentation, classes)
    return segmentation
