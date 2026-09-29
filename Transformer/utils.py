"""与论文计算直接相关的掩码、标签平滑和学习率公式。"""

import torch
from torch.nn import functional as F


def make_padding_mask(tokens: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """[B, L] -> [B, 1, 1, L]，True 表示这个 key 是 PAD。"""
    return tokens.eq(pad_id).unsqueeze(1).unsqueeze(2)


def make_target_mask(tokens: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """目标 key padding mask 与 causal mask 的并集；输入应为右填充序列。"""
    length = tokens.shape[1]
    # 行 i 是 query，列 j 是 key；j > i 时禁止关注，主对角线允许。
    causal_mask = torch.ones(
        length, length, dtype=torch.bool, device=tokens.device
    ).triu(diagonal=1)
    return make_padding_mask(tokens, pad_id) | causal_mask[None, None, :, :]


def label_smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int = 0,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """logits: [B,T,V]；targets: [B,T]，忽略 PAD 标签位置。

    采用 (1-eps)*one_hot + eps/V 的均匀混合约定，与 PyTorch 的
    CrossEntropyLoss(label_smoothing=eps) 对齐。论文只给出 eps=0.1，
    没有明确 PAD 类如何分配平滑质量；这里是明确标注的实现补全。
    PAD 标签位置不参与平均，但平滑分布的 V 个类别包含 PAD 类。
    """
    log_probabilities = F.log_softmax(logits, dim=-1)
    nll = -log_probabilities.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)  # [B, T]，真实类别的负对数概率
    uniform_loss = -log_probabilities.mean(dim=-1)  # [B, T]
    token_loss = (1.0 - smoothing) * nll + smoothing * uniform_loss
    return token_loss[targets.ne(pad_id)].mean()


def transformer_learning_rate(
    step_num: int, d_model: int = 512, warmup_steps: int = 4000
) -> torch.Tensor:
    """论文公式 (3)，step_num 从 1 开始；返回零维 Tensor。"""
    step = torch.as_tensor(step_num, dtype=torch.float32)
    return d_model ** (-0.5) * torch.minimum(
        step.pow(-0.5), step * warmup_steps ** (-1.5)
    )
