"""Attention Is All You Need：展开核心计算的 PyTorch 模块。

约定：所有特征均为 batch-first；布尔 mask 中 True 表示禁止关注。
默认 d_k = d_v = d_model // num_heads，对应论文的 base / big 配置。
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """论文 §3.5：正弦位置编码，逐元素加到 embedding 上。

    d_model 使用偶数。频率是 buffer，不是可学习参数；按当前长度生成 PE，
    不额外设置 max_len。动态生成方式属于实现选择，公式与论文一致。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # [D/2]；下标 0, 2, 4, ... 对应论文中的 2i。
        inverse_frequency = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("inverse_frequency", inverse_frequency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]；位置从 0 开始。
        length = x.shape[1]
        position = torch.arange(
            length, device=x.device, dtype=self.inverse_frequency.dtype
        ).unsqueeze(1)  # [L, 1]
        angles = position * self.inverse_frequency.unsqueeze(0)  # [L, D/2]

        pe = torch.zeros(
            length, self.d_model, device=x.device, dtype=angles.dtype
        )
        pe[:, 0::2] = torch.sin(angles)  # PE(pos, 2i)
        pe[:, 1::2] = torch.cos(angles)  # PE(pos, 2i+1)
        return x + pe.to(dtype=x.dtype).unsqueeze(0)  # [B, L, D]


class MultiHeadAttention(nn.Module):
    """论文 §3.2：投影、分头、缩放点积、掩码、加权、拼头、输出投影。

    没有调用 nn.MultiheadAttention 或 fused attention。
    一次 D -> D 的投影包含所有头各自的参数，不是把一个头复制 h 份。
    默认投影无 bias，与论文中只含 W 的注意力公式对应。
    attention_dropout 是可选工程补充，默认 0.0，保持公式 (1) 的计算。
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # PyTorch Linear 的 weight 为 [out_features, in_features]。
        self.q_projection = nn.Linear(d_model, d_model, bias=False)
        self.k_projection = nn.Linear(d_model, d_model, bias=False)
        self.v_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        blocked_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # query: [B, Lq, D]；key / value: [B, Lk, D]。
        # 交叉注意力允许 Lq != Lk；key 与 value 的序列长度相同。
        batch_size, query_length, _ = query.shape
        key_length = key.shape[1]

        # 1. Q / K / V Projection，各自使用独立的可学习权重。
        q = self.q_projection(query)  # [B, Lq, D]
        k = self.k_projection(key)    # [B, Lk, D]
        v = self.v_projection(value)  # [B, Lk, D]

        # 2. Split Heads：[B, L, h*d_k] -> [B, h, L, d_k]。
        q = q.reshape(batch_size, query_length, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, key_length, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, key_length, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 3. 公式 (1)：Q K^T / sqrt(d_k)，转置每个头最后两个维度。
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)  # [B, h, Lq, Lk]

        # 4. 在 softmax 之前把非法连接设为 -inf。
        # mask 可广播到 [B, h, Lq, Lk]；每个 query 至少保留一个 key。
        if blocked_mask is not None:
            scores = scores.masked_fill(blocked_mask, float("-inf"))

        # 5. 沿 key 维归一化：每个 query 决定从哪些位置读取信息。
        attention_weights = F.softmax(scores, dim=-1)  # [B, h, Lq, Lk]

        # 6. 对 V 加权求和。返回的 weights 是 dropout 之前的概率。
        weights_used = self.attention_dropout(attention_weights)
        context = torch.matmul(weights_used, v)  # [B, h, Lq, d_v]

        # 7. Head Concatenate；transpose 后用 contiguous 保证 view 合法。
        context = context.transpose(1, 2).contiguous()  # [B, Lq, h, d_v]
        context = context.view(batch_size, query_length, self.d_model)

        # 8. Concat(head_1, ..., head_h) W^O。
        output = self.output_projection(context)  # [B, Lq, D]
        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """论文公式 (2)：FFN(x) = max(0, xW1+b1)W2+b2。"""

    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 同一层中的所有 token 共享这两个线性层；不混合不同位置。
        x = self.linear1(x)  # [B, L, D] -> [B, L, d_ff]
        x = F.relu(x)
        x = self.linear2(x)  # [B, L, d_ff] -> [B, L, D]
        return x


class EncoderLayer(nn.Module):
    """原论文的 Post-LN Encoder：Self-Attention -> FFN，各有 Add & Norm。"""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, num_heads, attention_dropout
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.attention_output_dropout = nn.Dropout(dropout)
        self.ffn_output_dropout = nn.Dropout(dropout)
        # eps 数值是实现补全，不是论文给定的超参数。
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(
        self, x: torch.Tensor, source_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, S, D]；Q、K、V 都由当前源序列 x 投影得到。
        attention_output, _ = self.self_attention(x, x, x, source_mask)
        x = self.norm1(x + self.attention_output_dropout(attention_output))

        # Residual Connection -> LayerNorm；保持 [B, S, D]。
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.ffn_output_dropout(ffn_output))
        return x


class DecoderLayer(nn.Module):
    """原论文的 Post-LN Decoder：Masked Self-Attention -> Cross-Attention -> FFN。"""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, num_heads, attention_dropout
        )
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, attention_dropout
        )
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.self_output_dropout = nn.Dropout(dropout)
        self.cross_output_dropout = nn.Dropout(dropout)
        self.ffn_output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, T, D]；memory: [B, S, D]，是最后一层 Encoder 的输出。

        # 1. Masked Multi-Head Self-Attention：禁止读取未来的目标 token。
        attention_output, _ = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.self_output_dropout(attention_output))

        # 2. Cross-Attention：Q 来自上一步 Decoder 状态；K、V 来自 memory。
        # scores 为 [B, h, T, S]；这里只屏蔽源 PAD，不加目标因果掩码。
        attention_output, _ = self.cross_attention(
            x, memory, memory, source_mask
        )
        x = self.norm2(x + self.cross_output_dropout(attention_output))

        # 3. Position-wise FFN + Residual Connection + LayerNorm。
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.ffn_output_dropout(ffn_output))
        return x  # [B, T, D]
