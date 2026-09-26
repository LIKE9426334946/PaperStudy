"""论文 §3.2 的准确模式和快速模式；推理前由调用者执行 model.eval()。"""

import torch


@torch.no_grad()
def predict_accurate(model, images):
    """四个 sigmoid 概率图取平均，返回概率，不在这里二值化。"""
    if not model.deep_supervision:
        raise ValueError("准确模式需要经过深监督训练的四个预测头")
    logits = model(images)
    probabilities = torch.stack([z.sigmoid() for z in logits], dim=0)
    # [4, B, 1, *S] -> [B, 1, *S]
    # mean(sigmoid(z_j)) 一般不等于 sigmoid(mean(z_j))。
    return probabilities.mean(dim=0)


@torch.no_grad()
def predict_fast(model, images, branch=3):
    """选择 Lj：forward 提前返回，只执行该分支所需的依赖节点。

    这是执行图上的剪枝：可减少运算/中间激活，但模型对象依然保留
    完整参数，state_dict 文件不会自动缩小。分支选择须通过验证集。
    """
    return model(images, branch=branch).sigmoid()
