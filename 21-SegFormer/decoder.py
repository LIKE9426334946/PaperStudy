"""按论文公式 (4) 展开的 All-MLP 解码器，输出 1/4 分辨率 logits。

这里采用公式中的逐位置 Linear，不添加官方训练头中的 SyncBN/ReLU/Dropout。
这是有意的论文公式参考实现，不能视为官方 checkpoint 的原样复刻。
"""

import torch
from torch import nn
from torch.nn import functional as F


class ChannelProjection(nn.Module):
    """在每一个空间位置独立执行 Linear(C_i, C)，不混合相邻像素。"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, feature_map):
        batch, _, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)  # [B, H_i*W_i, C_i]
        tokens = self.linear(tokens)  # [B, H_i*W_i, C]
        return tokens.transpose(1, 2).reshape(
            batch, self.out_channels, height, width
        )


class AllMLPDecoder(nn.Module):
    """四路通道对齐 -> 空间对齐 -> 通道拼接 -> 线性融合 -> 逐像素分类。"""

    def __init__(self, encoder_channels, embedding_dim, num_classes):
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.projection1 = ChannelProjection(c1, embedding_dim)
        self.projection2 = ChannelProjection(c2, embedding_dim)
        self.projection3 = ChannelProjection(c3, embedding_dim)
        self.projection4 = ChannelProjection(c4, embedding_dim)
        self.fusion = ChannelProjection(4 * embedding_dim, embedding_dim)
        self.classifier = ChannelProjection(embedding_dim, num_classes)

    def forward(self, features):
        feature1, feature2, feature3, feature4 = features
        target_size = feature1.shape[-2:]  # 用实际尺寸，兼容非正方形/奇数尺寸。

        # 公式 (4) 第一行：每一路 C_i -> C。
        projected1 = self.projection1(feature1)
        projected2 = self.projection2(feature2)
        projected3 = self.projection3(feature3)
        projected4 = self.projection4(feature4)

        # 公式 (4) 第二行：都对齐到第 1 阶段；第 1 路已经是目标分辨率。
        projected2 = F.interpolate(projected2, size=target_size,
                                   mode="bilinear", align_corners=False)
        projected3 = F.interpolate(projected3, size=target_size,
                                   mode="bilinear", align_corners=False)
        projected4 = F.interpolate(projected4, size=target_size,
                                   mode="bilinear", align_corners=False)

        # 公式 (4) 第三行：沿通道维拼接，不是逐元素相加。
        combined = torch.cat(
            [projected1, projected2, projected3, projected4], dim=1
        )  # [B, 4*C, H/4, W/4]
        fused = self.fusion(combined)  # [B, C, H/4, W/4]

        # 公式 (4) 第四行：输出 logits，不在模型内部做 softmax/argmax。
        logits = self.classifier(fused)  # [B, num_classes, H/4, W/4]
        return logits
