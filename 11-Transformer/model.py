"""Attention Is All You Need 的完整 Encoder-Decoder 参考实现。

默认是 Transformer-base：N=6, D=512, h=8, d_ff=2048。
共用源/目标词表，源 embedding、目标 embedding 和输出投影共享参数。
不包含 tokenizer、训练循环或 beam search；forward 返回未 softmax 的 logits。
"""

import math

import torch
from torch import nn

from .blocks import DecoderLayer, EncoderLayer, SinusoidalPositionalEncoding
from .utils import make_padding_mask, make_target_mask


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # §3.4：两侧都查询同一张表 E；每次查询结果乘 sqrt(D)。
        # 不设置 padding_idx：PAD 通过注意力 mask 和 loss mask 排除。
        # 输出投影还会使用同一行权重，因此不把 PAD 行理解为冻结参数。
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = SinusoidalPositionalEncoding(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # 每次构造一个新层：结构相同，参数各自独立，不使用 [layer] * N。
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, attention_dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, attention_dropout)
            for _ in range(num_layers)
        ])

        # PyTorch 的 Linear 计算 h @ weight.T；weight 与 E 都是 [V, D]。
        # 赋值共享的是同一个 Parameter，而不是 copy 一份初始数值。
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """初始化是合理补全，论文未给出这里所用的完整初始化方案。"""
        # 共享 embedding 只初始化一次，避免按输出 Linear 再次覆盖。
        nn.init.normal_(self.token_embedding.weight, std=self.d_model ** (-0.5))
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.output_projection:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # LayerNorm 保留 PyTorch 的 gamma=1、beta=0 默认初始化。

    def encode(
        self, source_tokens: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        # source_tokens: [B, S]，整数 token ID。
        x = self.token_embedding(source_tokens) * math.sqrt(self.d_model)
        x = self.position_encoding(x)   # [B, S, D]，embedding + PE
        x = self.embedding_dropout(x)   # §5.4：在二者相加后 dropout

        # 双向 Self-Attention + FFN；每层都有两次 Post-LN。
        for layer in self.encoder_layers:
            x = layer(x, source_mask)
        return x  # memory: [B, S, D]，没有额外的最终 LayerNorm

    def decode(
        self,
        target_input: torch.Tensor,
        memory: torch.Tensor,
        target_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        # target_input: [B, T]，外部已右移，例如 [BOS, y1, y2]。
        x = self.token_embedding(target_input) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        x = self.embedding_dropout(x)  # [B, T, D]

        # 每个 Decoder 层都有自己的 Q/K/V 投影，但读取相同的最终 memory。
        for layer in self.decoder_layers:
            x = layer(x, memory, target_mask, source_mask)
        return x  # [B, T, D]

    def forward(
        self, source_tokens: torch.Tensor, target_input: torch.Tensor
    ) -> torch.Tensor:
        # source_tokens: [B, S]；target_input: [B, T]，二者长度可以不同。
        # 1. 构造禁止关注的位置；所有 mask 都是 True = blocked。
        source_mask = make_padding_mask(source_tokens, self.pad_id)
        target_mask = make_target_mask(target_input, self.pad_id)

        # 2. Encoder：完整读取源序列，生成 memory。
        memory = self.encode(source_tokens, source_mask)

        # 3. Decoder：读取已知目标前缀，并通过 cross-attention 查询 memory。
        hidden = self.decode(target_input, memory, target_mask, source_mask)

        # 4. 共享词表投影：[B, T, D] -> [B, T, vocab_size]。
        # 训练时直接送入交叉熵；需要概率时，在外部 softmax(dim=-1)。
        logits = self.output_projection(hidden)
        return logits
