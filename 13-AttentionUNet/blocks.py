"""Attention U-Net 的卷积块与加性注意力门；公式编号对应论文第 2 节。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def spatial_layers(spatial_dims):
    """只选择 2D/3D 算子，核心计算仍在各模块 forward 中展开。"""
    if spatial_dims == 2:
        return nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d, "bilinear"
    if spatial_dims == 3:
        return nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d, "trilinear"
    raise ValueError("spatial_dims 必须为 2 或 3")


class DoubleConv(nn.Module):
    """两次 Conv -> BatchNorm -> ReLU；3D 是论文实验设定，2D 是教学改写。"""

    def __init__(self, in_channels, out_channels, spatial_dims=3):
        super().__init__()
        Conv, BatchNorm, _, _ = spatial_layers(spatial_dims)
        # 合理补全：padding=1 保持空间尺寸；BN 前的卷积省略 bias。
        self.conv1 = Conv(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = BatchNorm(out_channels)
        self.conv2 = Conv(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = BatchNorm(out_channels)

    def forward(self, x):
        # x: [B, Cin, *S]，*S 为 (H,W) 或 (D,H,W)。
        x = self.conv1(x)  # [B, Cout, *S]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)  # [B, Cout, *S]
        x = self.bn2(x)
        x = F.relu(x)
        return x


class AdditiveAttentionGate(nn.Module):
    """论文式 (1)(2)：x 是 encoder skip，g 是粗尺度 decoder 上下文。

    在 g 的网格上计算单通道 alpha，再上采样并广播乘回原始 x。
    alpha 是空间权重，不是 Transformer 中的 token-token 注意力矩阵。
    """

    def __init__(self, x_channels, g_channels, inter_channels, spatial_dims=3):
        super().__init__()
        Conv, _, _, self.mode = spatial_layers(spatial_dims)
        # 对应 W_x^T x；不设置 bias，避免重复加入公式中的 b_g。
        self.W_x = Conv(x_channels, inter_channels, kernel_size=1, bias=False)
        # 对应 W_g^T g + b_g。
        self.W_g = Conv(g_channels, inter_channels, kernel_size=1, bias=True)
        # 对应 psi^T h + b_psi。这里不插入 BN，以保持式 (1) 的计算顺序。
        self.psi = Conv(inter_channels, 1, kernel_size=1, bias=True)

        # 论文要求初始时各位置近似放行，但没有指定初始化数值。
        # 合理补全：小的非零 psi 权重与 sigmoid^{-1}(0.99) 偏置。
        # 非零权重让 W_x、W_g 在第一次反向传播时也能接收到梯度。
        nn.init.normal_(self.psi.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.psi.bias, math.log(0.99 / 0.01))

    def forward(self, x, g):
        # x: [B, Fx, *Sx]；g: [B, Fg, *Sg]，Sg 比 Sx 粗一层。
        # 论文明确：先把 x 对齐到 g 的粗网格，并使用 1x1(x1) 线性投影。
        # 下采样方式未规定；这里用双/三线性插值，align_corners=False。
        x_coarse = F.interpolate(x, size=g.shape[2:], mode=self.mode,
                                 align_corners=False)
        theta_x = self.W_x(x_coarse)  # [B, Fint, *Sg]
        phi_g = self.W_g(g)  # [B, Fint, *Sg]，已经包含 b_g

        # 式 (1)：q_att = psi^T ReLU(W_x^T x + W_g^T g + b_g) + b_psi。
        h = F.relu(theta_x + phi_g)
        q_att = self.psi(h)  # [B, 1, *Sg]

        # 式 (2)：逐位置 sigmoid，而不是对全部位置做 softmax。
        alpha_coarse = torch.sigmoid(q_att)  # [B, 1, *Sg]
        alpha = F.interpolate(alpha_coarse, size=x.shape[2:], mode=self.mode,
                              align_corners=False)  # [B, 1, *Sx]

        # x_hat[i,c] = x[i,c] * alpha[i]；所有通道共享该位置的权重。
        # 必须乘原始分辨率的 x；不能用 x_coarse 替代高分辨率 skip。
        # 不 detach，autograd 自动包含论文式 (3) 的两项乘积求导。
        x_hat = x * alpha  # [B, Fx, *Sx]
        return x_hat, alpha


class MultiAttentionGate(nn.Module):
    """可选多子门扩展：不同子门学习互补区域，不预先绑定器官类别。

    num_heads=1 严格保留标量门 x*alpha；大于 1 时拼接后用 1x1 卷积融合。
    论文提出多维/多子门，但未给出唯一融合公式；该融合方式是合理补全。
    """

    def __init__(self, x_channels, g_channels, inter_channels,
                 spatial_dims=3, num_heads=1):
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads 必须至少为 1")
        Conv, _, _, _ = spatial_layers(spatial_dims)
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            AdditiveAttentionGate(x_channels, g_channels, inter_channels, spatial_dims)
            for _ in range(num_heads)
        ])
        self.fuse = (Conv(num_heads * x_channels, x_channels, 1)
                     if num_heads > 1 else nn.Identity())

    def forward(self, x, g):
        gated_features, attention_maps = [], []
        for gate in self.heads:
            x_hat, alpha = gate(x, g)
            gated_features.append(x_hat)
            attention_maps.append(alpha)
        if self.num_heads == 1:
            return gated_features[0], attention_maps[0]
        merged = torch.cat(gated_features, dim=1)  # [B, heads*Fx, *Sx]
        output = self.fuse(merged)  # [B, Fx, *Sx]
        maps = torch.cat(attention_maps, dim=1)  # [B, heads, *Sx]
        return output, maps


class UpConcatBlock(nn.Module):
    """decoder 上采样 -> 与已经门控的 skip 拼接 -> 两次卷积。"""

    def __init__(self, decoder_channels, skip_channels, out_channels, spatial_dims=3):
        super().__init__()
        _, _, _, self.mode = spatial_layers(spatial_dims)
        self.convs = DoubleConv(decoder_channels + skip_channels,
                               out_channels, spatial_dims)

    def forward(self, decoder, skip):
        # 合理补全：使用插值上采样，不另加转置卷积。
        upsampled = F.interpolate(decoder, size=skip.shape[2:], mode=self.mode,
                                  align_corners=False)
        merged = torch.cat((skip, upsampled), dim=1)  # 按通道拼接，不是相加
        output = self.convs(merged)
        return output
