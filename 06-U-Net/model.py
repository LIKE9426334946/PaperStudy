"""2015 U-Net 的论文参考实现，默认对应图 1 的 1 通道输入、2 类输出。

仅包含网络及初始化，不引入数据集、优化器或训练工程。
论文未明确的 Dropout 配置与初始化细节见 README 的一致性说明。
"""

import math
import torch

from torch import Tensor, nn

if __package__:
    from .blocks import DoubleConv, UpBlock
else:
    from blocks import DoubleConv, UpBlock


class UNet(nn.Module):
    """输入 [B, C, H, W]，输出未归一化 logits [B, K, H-184, W-184]。

    H 和 W 分别满足：>= 188，且除以 16 的余数为 12。
    这使四次池化前的空间尺寸均为偶数，例如 572、588、604。
    构造后所有权重都是随机初始化的；没有加载预训练编码器。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Contracting path：每次下采样后，下一组卷积将通道数翻倍。
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)

        # 合理补全：论文只说收缩路径末端使用 Dropout，未明确位置和 p。
        # 此处在 enc4 和 bottleneck 输出后使用逐元素 Dropout，默认 p=0.5。
        self.drop4 = nn.Dropout(p=dropout_p)
        self.drop5 = nn.Dropout(p=dropout_p)

        # Expansive path：内部完整展开 up-conv、crop、cat 和两次卷积。
        self.dec4 = UpBlock(1024, 512)
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)

        # Pixel-wise classifier：不接 ReLU、Softmax 或 Sigmoid。
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """论文第 3 节：零均值高斯分布，标准差 sqrt(2/N)。"""
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                # 3x3 卷积时，一个输出神经元有 9*C_in 个输入连接。
                fan_in = layer.in_channels * math.prod(layer.kernel_size)
            elif isinstance(layer, nn.ConvTranspose2d):
                # 合理补全：此处 kernel=stride=2，没有空间重叠。
                # 每个输出位置由一个输入位置的 C_in 个通道贡献，N=C_in。
                # 不直接根据转置卷积 weight 的第 1 维推断输入通道。
                fan_in = layer.in_channels
            else:
                continue
            # 对没有 ReLU 的 1x1 head 也使用此规则，是本实现的补全。
            nn.init.normal_(layer.weight, mean=0.0, std=math.sqrt(2.0 / fan_in))
            nn.init.zeros_(layer.bias)  # 偏置初始化为 0，也是实现选择。

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        if any(size < 188 or size % 16 != 12 for size in (height, width)):
            raise ValueError("H、W 须分别 >= 188 且 H % 16 == W % 16 == 12")

        # 以下 Shape 均以 x: [B, 1, 572, 572] 为例。
        e1 = self.enc1(x)  # [B,   64, 568, 568]
        p1 = self.pool(e1)  # [B,   64, 284, 284]
        e2 = self.enc2(p1)  # [B,  128, 280, 280]
        p2 = self.pool(e2)  # [B,  128, 140, 140]
        e3 = self.enc3(p2)  # [B,  256, 136, 136]
        p3 = self.pool(e3)  # [B,  256,  68,  68]
        e4 = self.enc4(p3)  # [B,  512,  64,  64]
        e4 = self.drop4(e4)
        p4 = self.pool(e4)  # [B,  512,  32,  32]

        z = self.bottleneck(p4)  # [B, 1024,  28,  28]
        z = self.drop5(z)

        # 同尺度 encoder 特征经中心裁剪后，沿通道维拼接。
        d4 = self.dec4(z, e4)  # [B,  512,  52,  52]
        d3 = self.dec3(d4, e3)  # [B,  256, 100, 100]
        d2 = self.dec2(d3, e2)  # [B,  128, 196, 196]
        d1 = self.dec1(d2, e1)  # [B,   64, 388, 388]
        logits = self.head(d1)  # [B,    K, 388, 388]
        return logits


if __name__ == "__main__":
    model = UNet(dropout_p=0.0).eval()
    x = torch.randn(1, 1, 188, 188)
    with torch.no_grad():
        model(x)
    torch.onnx.export(
        model,
        x,
        "unet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
    )
