"""2018 UNet++ 的阅读用 PyTorch 参考实现。

对应论文图 1 与式 (1)，完整展开 15 个节点和 10 条上采样边。
默认通道数 C_i = 32 * 2**i。输入空间维度应为 16 的正整数倍。
节点双卷积、same padding 和 2 倍转置卷积参考作者 Keras 实现补全。
forward 返回 logits；论文的 sigmoid 在损失/推理函数中显式执行。
"""

import torch
from torch import nn

from .blocks import ConvBlock


class UNetPlusPlus(nn.Module):
    """嵌套 U-Net；spatial_dims=2/3 分别处理图像/体数据。

    deep_supervision=True：forward(x) 返回 (z1, z2, z3, z4)。
    deep_supervision=False：forward(x) 仅返回 z4，不创建辅助预测头。
    forward(x, branch=j)：返回单个 zj，并跳过后续不需要的计算。
    分支 1/2/3 只有经过深监督训练后才有独立分割意义。
    输出形状均为 [B, out_channels, *input_spatial_shape]。
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        deep_supervision=True,
        spatial_dims=2,
    ):
        super().__init__()
        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims 必须是 2 或 3")
        self.deep_supervision = deep_supervision
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        Pool = nn.MaxPool2d if spatial_dims == 2 else nn.MaxPool3d
        Up = nn.ConvTranspose2d if spatial_dims == 2 else nn.ConvTranspose3d
        c0, c1, c2, c3, c4 = [base_channels * 2**i for i in range(5)]

        # j=0：共享编码器，四次 MaxPool 将各空间维度缩小到 1/16。
        self.pool = Pool(kernel_size=2, stride=2)
        self.conv0_0 = ConvBlock(in_channels, c0, spatial_dims)
        self.conv1_0 = ConvBlock(c0, c1, spatial_dims)
        self.conv2_0 = ConvBlock(c1, c2, spatial_dims)
        self.conv3_0 = ConvBlock(c2, c3, spatial_dims)
        self.conv4_0 = ConvBlock(c3, c4, spatial_dims)

        # U: C_(i+1) -> C_i，各空间维度放大 2 倍。
        # 每条边有独立可学习参数，没有在多个节点之间共享同一个 Up。
        self.up0_1 = Up(c1, c0, kernel_size=2, stride=2)
        self.up1_1 = Up(c2, c1, kernel_size=2, stride=2)
        self.up2_1 = Up(c3, c2, kernel_size=2, stride=2)
        self.up3_1 = Up(c4, c3, kernel_size=2, stride=2)
        self.up0_2 = Up(c1, c0, kernel_size=2, stride=2)
        self.up1_2 = Up(c2, c1, kernel_size=2, stride=2)
        self.up2_2 = Up(c3, c2, kernel_size=2, stride=2)
        self.up0_3 = Up(c1, c0, kernel_size=2, stride=2)
        self.up1_3 = Up(c2, c1, kernel_size=2, stride=2)
        self.up0_4 = Up(c1, c0, kernel_size=2, stride=2)

        # j>0：cat 接收同层 j 个旧节点和下层 1 个上采样节点。
        # 本实现 U 已把通道数变为 C_i，故 cat 后为 (j+1)*C_i。
        self.conv0_1 = ConvBlock(2 * c0, c0, spatial_dims)
        self.conv1_1 = ConvBlock(2 * c1, c1, spatial_dims)
        self.conv2_1 = ConvBlock(2 * c2, c2, spatial_dims)
        self.conv3_1 = ConvBlock(2 * c3, c3, spatial_dims)
        self.conv0_2 = ConvBlock(3 * c0, c0, spatial_dims)
        self.conv1_2 = ConvBlock(3 * c1, c1, spatial_dims)
        self.conv2_2 = ConvBlock(3 * c2, c2, spatial_dims)
        self.conv0_3 = ConvBlock(4 * c0, c0, spatial_dims)
        self.conv1_3 = ConvBlock(4 * c1, c1, spatial_dims)
        self.conv0_4 = ConvBlock(5 * c0, c0, spatial_dims)

        # 论文 §4：顶层四个节点各接一个 1x1（3D 为 1x1x1）卷积。
        # 深监督关闭时，内部密集路径仍保留，只移除三个辅助预测头。
        if deep_supervision:
            self.head1 = Conv(c0, out_channels, kernel_size=1)
            self.head2 = Conv(c0, out_channels, kernel_size=1)
            self.head3 = Conv(c0, out_channels, kernel_size=1)
        self.head4 = Conv(c0, out_channels, kernel_size=1)

    def forward(self, x, branch=None):
        if branch is not None and branch not in (1, 2, 3, 4):
            raise ValueError("branch 必须是 None、1、2、3 或 4")
        if not self.deep_supervision and branch in (1, 2, 3):
            raise ValueError("浅层分支需要 deep_supervision=True 并接受对应训练")

        # 按所需网络深度递增展开，便于在 L1/L2/L3 真正提前返回。
        # 以下注释以 2D、base_channels=32 为例；3D 多一个 D 维度。
        # 输入 x: [B, C_in, H, W]。

        # L1：x0_1 = H(cat(x0_0, U(x1_0)))。
        x0_0 = self.conv0_0(x)                # [B, 32, H, W]
        x1_0 = self.conv1_0(self.pool(x0_0))   # [B, 64, H/2, W/2]
        u0_1 = self.up0_1(x1_0)               # [B, 32, H, W]
        x0_1 = self.conv0_1(torch.cat([x0_0, u0_1], dim=1))
        if branch == 1:
            return self.head1(x0_1)

        # L2：先生成 x1_1，再汇集顶层所有旧节点生成 x0_2。
        x2_0 = self.conv2_0(self.pool(x1_0))   # [B, 128, H/4, W/4]
        u1_1 = self.up1_1(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, u1_1], dim=1))
        u0_2 = self.up0_2(x1_1)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, u0_2], dim=1))
        # x0_2 的 cat: [B, 96, H, W] -> H -> [B, 32, H, W]。
        if branch == 2:
            return self.head2(x0_2)

        # L3：x3_0 -> x2_1 -> x1_2 -> x0_3。
        x3_0 = self.conv3_0(self.pool(x2_0))   # [B, 256, H/8, W/8]
        u2_1 = self.up2_1(x3_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, u2_1], dim=1))
        u1_2 = self.up1_2(x2_1)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, u1_2], dim=1))
        u0_3 = self.up0_3(x1_2)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, u0_3], dim=1))
        if branch == 3:
            return self.head3(x0_3)

        # L4：完整网络，x4_0 是分辨率最低的 bottleneck。
        x4_0 = self.conv4_0(self.pool(x3_0))   # [B, 512, H/16, W/16]
        u3_1 = self.up3_1(x4_0)
        x3_1 = self.conv3_1(torch.cat([x3_0, u3_1], dim=1))
        u2_2 = self.up2_2(x3_1)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, u2_2], dim=1))
        u1_3 = self.up1_3(x2_2)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, u1_3], dim=1))
        u0_4 = self.up0_4(x1_3)
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, u0_4], dim=1)
        )  # cat: [B, 160, H, W] -> [B, 32, H, W]

        z4 = self.head4(x0_4)                 # [B, K, H, W]，尚未 sigmoid
        if branch == 4 or not self.deep_supervision:
            return z4
        return self.head1(x0_1), self.head2(x0_2), self.head3(x0_3), z4
