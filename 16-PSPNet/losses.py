"""论文 §4/§5.1：主分支交叉熵 + 0.4 × 辅助分支交叉熵。"""

from torch.nn import functional as F


def pspnet_loss(main_logits, aux_logits, target, aux_weight=0.4, ignore_index=255):
    """logits: [B, K, H, W]；target: [B, H, W]，dtype=torch.long。

    target 值为 0..K-1；ignore_index=255 是本实现采用的标签约定。
    cross_entropy 内部含 log_softmax，因此输入不能先做 softmax。
    """
    main_loss = F.cross_entropy(main_logits, target, ignore_index=ignore_index)
    aux_loss = F.cross_entropy(aux_logits, target, ignore_index=ignore_index)
    total_loss = main_loss + aux_weight * aux_loss
    return total_loss, main_loss, aux_loss
