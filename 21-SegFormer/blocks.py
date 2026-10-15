"""SegFormer 核心模块：重叠嵌入、空间缩减注意力、Mix-FFN。

按论文公式 (1)-(3) 展开真实运算，不调用封装好的 MultiheadAttention。
Pre-LN、缩减后的 LayerNorm、bias 和 DropPath 属于实现补全，详见 README。
"""

import torch
from torch import nn


class OverlapPatchEmbedding(nn.Module):
    """用重叠卷积窗口把二维图像/特征图转换成 token 序列。"""

    def __init__(self, in_channels, embed_dim, kernel_size, stride, padding):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, feature_map):
        # Stage 1: kernel=7, stride=4, padding=3；后续为 3, 2, 1。
        feature_map = self.projection(feature_map)  # [B, C, H_i, W_i]
        height, width = feature_map.shape[-2:]
        tokens = feature_map.flatten(2).transpose(1, 2)  # [B, H_i*W_i, C]
        tokens = self.norm(tokens)  # 在最后的通道维 C 上做 LayerNorm。
        return tokens, height, width


class SpatialReduction(nn.Module):
    """公式 (2)：把每个 r×r 区域映射为一个 C 维向量，供 K、V 使用。

    论文正文的序列缩减率 R=r²；代码的 sr_ratio=r 是单个空间轴的缩减率。
    普通卷积等价于对局部 r×r×C 元素做共享线性投影；groups=1。
    这里采用二维局部分组，不能把任意连续 r² 个展平 token 当作同一区域。
    """

    def __init__(self, dim, sr_ratio):
        super().__init__()
        self.dim = dim
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.reduce = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio, padding=0
            )
            self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, tokens, height, width):
        if self.sr_ratio == 1:
            return tokens  # 第四阶段不缩减，不额外添加 LayerNorm。

        batch = tokens.shape[0]
        feature_map = tokens.transpose(1, 2).reshape(
            batch, self.dim, height, width
        )  # [B, C, H_i, W_i]
        feature_map = self.reduce(feature_map)  # [B, C, floor(H_i/r), floor(W_i/r)]
        reduced_tokens = feature_map.flatten(2).transpose(1, 2)
        return self.norm(reduced_tokens)  # [B, N_reduced, C]


class EfficientSelfAttention(nn.Module):
    """保留全部 Q，只缩短 K/V 序列；仍然让每个 Q 关注全部缩减后的 K。"""

    def __init__(self, dim, num_heads, sr_ratio, dropout=0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim 必须能被 num_heads 整除")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.reduction = SpatialReduction(dim, sr_ratio)
        # 为便于逐行调试，把官方合并的 KV 投影拆成独立的 K、V 线性层。
        self.q_projection = nn.Linear(dim, dim)
        self.k_projection = nn.Linear(dim, dim)
        self.v_projection = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens, height, width):
        batch, num_tokens, _ = tokens.shape  # [B, N, C], N=H_i*W_i

        # 1. Q 使用全部输入 token；K、V 使用同一份缩减后的输入。
        reduced_tokens = self.reduction(tokens, height, width)
        num_reduced = reduced_tokens.shape[1]
        query = self.q_projection(tokens)         # [B, N, C]
        key = self.k_projection(reduced_tokens)    # [B, N_reduced, C]
        value = self.v_projection(reduced_tokens)  # [B, N_reduced, C]

        # 2. 拆分多头：C = num_heads * head_dim。
        query = query.reshape(
            batch, num_tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B, heads, N, d]
        key = key.reshape(
            batch, num_reduced, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B, heads, N_reduced, d]
        value = value.reshape(
            batch, num_reduced, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B, heads, N_reduced, d]

        # 3. 公式 (1)：QK^T / sqrt(d)，在所有 Key 的位置上归一化。
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        # scores / attention_weights: [B, heads, N, N_reduced]

        # 4. Attention Weight @ V；输出仍然保留 N 个 Query 位置。
        context = torch.matmul(attention_weights, value)  # [B, heads, N, d]

        # 5. 合并 heads，再进行输出线性投影。
        context = context.transpose(1, 2).reshape(batch, num_tokens, self.dim)
        output = self.output_projection(context)  # [B, N, C]
        return self.output_dropout(output)


class MixFFN(nn.Module):
    """公式 (3) 的非残差部分；残差加法统一放在 TransformerBlock 中。"""

    def __init__(self, dim, expansion_ratio, dropout=0.0):
        super().__init__()
        self.hidden_dim = int(dim * expansion_ratio)
        self.expand = nn.Linear(dim, self.hidden_dim)
        self.depthwise_conv = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.hidden_dim,  # 每个通道独立卷积，不是空间缩减卷积。
        )
        self.activation = nn.GELU()
        self.project = nn.Linear(self.hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, height, width):
        batch = tokens.shape[0]
        # 第一个 MLP/Linear：只扩展每个 token 的通道维。
        hidden = self.expand(tokens)  # [B, N, E*C]

        # 恢复二维布局，3×3 depth-wise 卷积混合相邻空间位置。
        hidden = hidden.transpose(1, 2).reshape(
            batch, self.hidden_dim, height, width
        )  # [B, E*C, H_i, W_i]
        hidden = self.depthwise_conv(hidden)
        hidden = hidden.flatten(2).transpose(1, 2)  # [B, N, E*C]

        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        output = self.project(hidden)  # [B, N, C]
        return self.dropout(output)


class DropPath(nn.Module):
    """训练时按样本丢弃残差分支；eval 时是恒等映射。工程补全。"""

    def __init__(self, probability=0.0):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("DropPath probability 必须位于 [0, 1)")
        self.probability = probability

    def forward(self, branch):
        if not self.training or self.probability == 0:
            return branch
        keep_probability = 1 - self.probability
        mask_shape = (branch.shape[0],) + (1,) * (branch.ndim - 1)
        keep_mask = branch.new_empty(mask_shape).bernoulli_(keep_probability)
        return branch * keep_mask / keep_probability


class TransformerBlock(nn.Module):
    """Pre-LN + Efficient Attention + Residual + Pre-LN + Mix-FFN + Residual。"""

    def __init__(self, dim, num_heads, sr_ratio, expansion_ratio, drop_path=0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim, eps=1e-6)
        self.attention = EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFFN(dim, expansion_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, tokens, height, width):
        # 第一次残差：注意力不改变 [B, H_i*W_i, C]。
        attention_input = self.attention_norm(tokens)
        attention_output = self.attention(attention_input, height, width)
        tokens = tokens + self.drop_path(attention_output)

        # 第二次残差：对应公式 (3) 最后的 + x_in；这里只加一次。
        ffn_input = self.ffn_norm(tokens)
        ffn_output = self.ffn(ffn_input, height, width)
        tokens = tokens + self.drop_path(ffn_output)
        return tokens
