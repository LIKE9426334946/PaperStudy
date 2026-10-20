"""ViT 的核心模块：对应论文 §3.1、公式 (1)-(8) 与附录 A。

为方便阅读，显式展开 patch 划分、Q/K/V 投影、多头注意力和残差连接。
符号：B=batch，N=patch 数量，S=N+1，D=embedding 维度，h=head 数量。
"""

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """公式 (1)：非重叠 patch -> 展平 -> 所有 patch 共享的线性投影。"""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or min(image_size) <= 0 or patch_size <= 0:
            raise ValueError("image_size 必须是正整数或两个正整数，patch_size 必须为正。")
        if image_size[0] % patch_size or image_size[1] % patch_size:
            raise ValueError("图像高、宽必须能被 patch_size 整除；本实现不自动裁剪或填充。")
        if in_channels <= 0 or embed_dim <= 0:
            raise ValueError("in_channels 和 embed_dim 必须为正。")
        self.image_size = tuple(image_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_dim = patch_size * patch_size * in_channels
        # 论文 E: [P²C, D]；nn.Linear.weight 存储为 [D, P²C]。
        # bias 是工程补充，公式为了简洁省略了偏置。
        self.projection = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, images):
        # images: [B, C, H, W]。每个模型实例固定空间尺寸，B 可变化。
        if images.ndim != 4 or images.shape[1:] != (self.in_channels, *self.image_size):
            raise ValueError(f"期望 [B, {self.in_channels}, {self.image_size[0]}, "
                             f"{self.image_size[1]}]，实际为 {tuple(images.shape)}。")
        batch = images.shape[0]
        grid_h, grid_w = self.grid_size
        p = self.patch_size

        # [B,C,H,W] -> [B,C,Gh,P,Gw,P]：把高、宽各拆成网格与 patch 内坐标。
        patches = images.reshape(batch, self.in_channels, grid_h, p, grid_w, p)
        # -> [B,Gh,Gw,P,P,C]：patch 按从左到右、从上到下排列。
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        # -> [B,N,P²C]：每个 patch 内采用 (行, 列, 通道) 的展平顺序。
        patches = patches.reshape(batch, self.num_patches, self.patch_dim)
        return self.projection(patches)  # [B,N,P²C] -> [B,N,D]


class MultiHeadSelfAttention(nn.Module):
    """附录 A 的公式 (5)-(8)，不使用 nn.MultiheadAttention 等封装。"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, qkv_bias=True):
        super().__init__()
        if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim 必须是 num_heads 的正整数倍。")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # 三个独立的线性投影等价于公式 (5) 中合并的 U_qkv。
        # 每个投影先产生所有 head 的特征，再沿最后一维拆成 h 份。
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, embed_dim = x.shape  # [B,S,D]

        # 公式 (5)：Q、K、V 来自同一个 x，但使用不同的可学习参数。
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)  # 各为 [B,S,D]

        # [B,S,D] -> [B,S,h,d] -> [B,h,S,d]，其中 d=D/h。
        query = query.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 公式 (6)：每个 query 与所有 key 点积，除以 sqrt(d)。
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [B,h,S,S]
        attention = torch.softmax(scores, dim=-1)  # 对 key 所在的最后一维归一化
        # 图像分类使用双向 self-attention；没有 causal mask。
        # 按论文公式保留完整注意力，默认不额外对 attention 权重加 dropout。

        # 公式 (7)：每个 token 对各个 value 做加权求和。
        context = torch.matmul(attention, value)  # [B,h,S,d]
        # 公式 (8)：拼接各 head，不是把 head 相加。
        context = context.transpose(1, 2).reshape(batch, seq_len, embed_dim)  # [B,S,D]
        output = self.out_proj(context)  # U_msa：混合所有 head 的信息
        return self.proj_dropout(output)


class MLP(nn.Module):
    """逐 token 的两层前馈网络：D -> D_mlp -> D；非线性是 GELU。"""

    def __init__(self, embed_dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)        # [B,S,D] -> [B,S,D_mlp]
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)        # [B,S,D_mlp] -> [B,S,D]
        return self.dropout2(x)


class EncoderBlock(nn.Module):
    """公式 (2)、(3)：两个 Pre-LN 子层，每个子层后接残差相加。"""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0,
                 qkv_bias=True, norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout, qkv_bias)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x):
        # z'_l = MSA(LN(z_{l-1})) + z_{l-1}
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = residual + x

        # z_l = MLP(LN(z'_l)) + z'_l
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x  # 所有 block 都保持 [B,S,D]


class ClassificationHead(nn.Module):
    """§3.1 / 附录 D.3：预训练为 Linear -> Tanh -> Linear，微调为 Linear。"""

    def __init__(self, embed_dim, num_classes, representation_size=None):
        super().__init__()
        if representation_size is None:
            self.pre_logits = nn.Identity()
            self.activation = nn.Identity()
            head_dim = embed_dim
        else:
            if representation_size <= 0:
                raise ValueError("representation_size 必须为正整数或 None。")
            self.pre_logits = nn.Linear(embed_dim, representation_size)
            self.activation = nn.Tanh()
            head_dim = representation_size
        self.classifier = nn.Linear(head_dim, num_classes)

    def forward(self, cls_feature):
        x = self.pre_logits(cls_feature)  # [B,D] -> [B,R]，或保持 [B,D]
        x = self.activation(x)
        return self.classifier(x)  # [B,K]，原始 logits，不在模型内部做 softmax
