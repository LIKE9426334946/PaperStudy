"""四阶段 Mix Transformer 编码器；配置优先采用上传论文的附录表 6。"""

from torch import nn

if __package__:
    from .blocks import OverlapPatchEmbedding, TransformerBlock
else:
    from blocks import OverlapPatchEmbedding, TransformerBlock


# 注意：这不是官方代码配置的直接复制。
# 论文表 6：B0-B4 的 expansion_ratios=(8,8,4,4)，B2/B3 的第二阶段深度为 3。
# 官方代码：所有版本的 expansion_ratios=(4,4,4,4)，B2/B3 的第二阶段深度为 4。
# 根据 request.md 的“论文优先”，这里保留表 6 的数值，差异见 README。
MIT_CONFIGS = {
    "b0": dict(channels=(32, 64, 160, 256), depths=(2, 2, 2, 2),
               expansion_ratios=(8, 8, 4, 4), decoder_dim=256),
    "b1": dict(channels=(64, 128, 320, 512), depths=(2, 2, 2, 2),
               expansion_ratios=(8, 8, 4, 4), decoder_dim=256),
    "b2": dict(channels=(64, 128, 320, 512), depths=(3, 3, 6, 3),
               expansion_ratios=(8, 8, 4, 4), decoder_dim=768),
    "b3": dict(channels=(64, 128, 320, 512), depths=(3, 3, 18, 3),
               expansion_ratios=(8, 8, 4, 4), decoder_dim=768),
    "b4": dict(channels=(64, 128, 320, 512), depths=(3, 8, 27, 3),
               expansion_ratios=(8, 8, 4, 4), decoder_dim=768),
    "b5": dict(channels=(64, 128, 320, 512), depths=(3, 6, 40, 3),
               expansion_ratios=(4, 4, 4, 4), decoder_dim=768),
}


class EncoderStage(nn.Module):
    """一个阶段：重叠 Patch Embedding -> 若干 Transformer Block -> 特征图。"""

    def __init__(self, in_channels, dim, kernel_size, stride, padding,
                 num_heads, sr_ratio, expansion_ratio, drop_path_rates):
        super().__init__()
        self.dim = dim
        self.patch_embedding = OverlapPatchEmbedding(
            in_channels, dim, kernel_size, stride, padding
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, sr_ratio, expansion_ratio, rate)
            for rate in drop_path_rates
        ])
        self.output_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, feature_map):
        tokens, height, width = self.patch_embedding(feature_map)
        # 每个 Block 内都有显式的 Q/K/V、注意力和 Mix-FFN 运算。
        for block in self.blocks:
            tokens = block(tokens, height, width)
        tokens = self.output_norm(tokens)  # [B, H_i*W_i, C_i]
        return tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.dim, height, width
        ).contiguous()  # [B, C_i, H_i, W_i]


class MixTransformerEncoder(nn.Module):
    """MiT-B0 至 MiT-B5。没有 CLS token、绝对位置编码或窗口注意力。"""

    def __init__(self, variant="b0", in_channels=3, drop_path_rate=0.1):
        super().__init__()
        variant = variant.lower()
        if variant not in MIT_CONFIGS:
            raise ValueError(f"未知 variant={variant!r}，可选 {tuple(MIT_CONFIGS)}")
        if in_channels < 1 or not 0 <= drop_path_rate < 1:
            raise ValueError("in_channels 必须为正数，drop_path_rate 必须位于 [0, 1)")
        config = MIT_CONFIGS[variant]
        self.variant = variant
        self.channels = config["channels"]
        self.decoder_dim = config["decoder_dim"]
        heads = (1, 2, 5, 8)
        spatial_reductions = (8, 4, 2, 1)  # 对应正文 R=(64,16,4,1)。
        depths = config["depths"]

        # 参考官方的 stochastic depth 线性调度；不是论文公式的核心部分。
        total_blocks = sum(depths)
        rates = [drop_path_rate * i / (total_blocks - 1)
                 for i in range(total_blocks)]
        stages = []
        offset = 0
        previous_channels = in_channels
        for i in range(4):
            kernel, stride, padding = (7, 4, 3) if i == 0 else (3, 2, 1)
            stages.append(EncoderStage(
                in_channels=previous_channels,
                dim=self.channels[i],
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                num_heads=heads[i],
                sr_ratio=spatial_reductions[i],
                expansion_ratio=config["expansion_ratios"][i],
                drop_path_rates=rates[offset:offset + depths[i]],
            ))
            previous_channels = self.channels[i]
            offset += depths[i]
        self.stages = nn.ModuleList(stages)

    def forward(self, image):
        # 以下 H/4 等写法假设 H、W 是 32 的倍数；一般尺寸按卷积公式向上取整。
        feature1 = self.stages[0](image)     # [B, C1, H/4,  W/4]
        feature2 = self.stages[1](feature1)  # [B, C2, H/8,  W/8]
        feature3 = self.stages[2](feature2)  # [B, C3, H/16, W/16]
        feature4 = self.stages[3](feature3)  # [B, C4, H/32, W/32]
        return feature1, feature2, feature3, feature4
