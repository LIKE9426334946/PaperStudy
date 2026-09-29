"""UNet++ 的节点运算：展开 H 内部的卷积与激活。

论文式 (1) 用 H 表示卷积/激活，但未逐层列出节点内部配置。
这里参考作者 Keras standard_unit，用两个 3x3 卷积补全 H；
padding=1、ReLU、双卷积属于实现选择，不能全视作式 (1) 的明文规定。
3D 时相同运算改为两个 3x3x3 卷积。为突出主干，省略作者代码中的
Dropout 和 L2 正则化，不加入论文未要求的 BatchNorm。
"""

from torch import nn


class ConvBlock(nn.Module):
    """H: [B, C_in, *S] -> [B, C_out, *S]，空间尺寸不变。"""

    def __init__(self, in_channels, out_channels, spatial_dims=2):
        super().__init__()
        Conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一次卷积：融合输入通道，包括 cat 汇集的不同来源特征。
        x = self.conv1(x)  # [B, C_out, *S]
        x = self.relu(x)
        # 第二次卷积：继续提取融合后的局部特征。
        x = self.conv2(x)  # [B, C_out, *S]
        x = self.relu(x)
        return x
