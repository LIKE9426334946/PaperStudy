"""对应论文公式 (1)-(3)：特征 token 化、位置编码和 Transformer 编码器。"""

import math

import torch
from torch import nn


class HybridPatchEmbedding(nn.Module):
    """对 CNN 的 /16 特征取 1x1 patch；不是对该特征再切 16x16 patch。"""

    def __init__(self, grid_size, embed_dim=768, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_tokens = grid_size[0] * grid_size[1]
        # 1x1 卷积等价于每个空间位置独立使用同一个 Linear(1024, D)。
        # 带偏置是采用官方实现的补充细节；论文公式 (1) 只写了矩阵 E。
        self.projection = nn.Conv2d(1024, embed_dim, kernel_size=1, stride=1)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        # 阅读演示采用随机位置向量；不是论文要求的初始化方案。
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, features):
        # features: [B,1024,14,14]
        x = self.projection(features)       # [B,D,14,14]
        x = x.flatten(start_dim=2)          # [B,D,196]；按行展开空间位置
        x = x.transpose(1, 2)              # [B,196,D]
        # 公式 (1)：z0 = [x_p^1 E; ...; x_p^N E] + E_pos。
        # 无 class token，因此 N=196，不是分类 ViT 的 197。
        x = x + self.position_embedding    # [1,N,D] 沿 batch 维广播
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """完整展开 Q/K/V、分头、注意力权重、加权求和、拼接及输出投影。"""

    def __init__(self, embed_dim=768, num_heads=12, attention_dropout=0.0):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须为正，且能被正整数 num_heads 整除。")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape   # [B,N,D]

        # 同一个 x 经三个独立可学习矩阵，得到不同的 Q、K、V。
        q = self.query(x)                            # [B,N,D]
        k = self.key(x)
        v = self.value(x)

        # D = h * d_head：Base 配置为 768 = 12 * 64。
        q = q.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)                    # [B,h,N,d_head]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # 标准 MSA 展开：softmax(QK^T / sqrt(d_head)) V。
        # 图像编码器允许所有 token 两两交互，没有 causal mask。
        scores = torch.matmul(q, k.transpose(-2, -1))  # [B,h,N,N]
        scores = scores / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)  # 对 key 维归一化
        attention_weights = self.attention_dropout(attention_weights)
        context = torch.matmul(attention_weights, v)   # [B,h,N,d_head]

        # 拼接 head，不是把各 head 逐元素相加。
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, num_tokens, embed_dim)  # [B,N,D]
        output = self.output_projection(context)
        return self.output_dropout(output)


class FeedForwardNetwork(nn.Module):
    """逐 token 的 MLP：D -> M -> D；Base 为 768 -> 3072 -> 768。"""

    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)       # [B,N,768] -> [B,N,3072]
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)       # [B,N,3072] -> [B,N,768]
        return self.dropout2(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-LN：归一化在子层前，两个子层分别有残差连接。"""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, attention_dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, attention_dropout)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = FeedForwardNetwork(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        # 公式 (2)：z'_l = MSA(LN(z_{l-1})) + z_{l-1}。
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = residual + x

        # 公式 (3)：z_l = MLP(LN(z'_l)) + z'_l。
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x
        return x                 # [B,N,D]；所有层保持 N 和 D 不变


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072,
                 dropout=0.1, attention_dropout=0.0):
        super().__init__()
        if depth < 1 or mlp_dim < 1:
            raise ValueError("depth 和 mlp_dim 必须为正整数。")
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout, attention_dropout)
            for _ in range(depth)
        ])
        # 论文公式未单列末端 LN；按 ViT 和官方实现补充。
        self.final_norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, tokens):
        for block in self.blocks:
            tokens = block(tokens)
        return self.final_norm(tokens)
