"""Attention U-Net 论文参考实现，支持 3D 原理演示和 2D 教学改写。

直接运行本文件：同一个 model、同一个 x、同一个 eval forward 先演示，再导出
ONNX，并用 ONNX Runtime 逐输出核对。未加载训练权重，输出不代表分割效果。
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .blocks import DoubleConv, MultiAttentionGate, UpConcatBlock, spatial_layers
else:
    from blocks import DoubleConv, MultiAttentionGate, UpConcatBlock, spatial_layers


class AttentionUNet(nn.Module):
    """四尺度 encoder-decoder，通道数为 C、2C、4C、8C。

    spatial_dims=3: [B, Cin, D, H, W]，遵循论文 3D 设定。
    spatial_dims=2: [B, Cin, H, W]，将相同核心计算改写为 2D。
    deep_supervision=True: 总是返回 (logits, aux2_logits, aux3_logits)。
    deep_supervision=False: 总是返回 logits。
    返回规则不依赖 self.training，演示和导出不会偷偷切换计算分支。

    论文正文称最浅 skip 不门控，故 gate_first_skip=False；Fig.1 的示意图
    仍在该位置画了门，可用 gate_first_skip=True 显式研究该变体。
    空间边长须 >= 8；训练 BN 时每个通道需有多个批次/空间元素。
    """

    def __init__(self, in_channels=1, num_classes=2, base_channels=32,
                 spatial_dims=3, deep_supervision=True, attention_heads=1,
                 gate_first_skip=False):
        super().__init__()
        if min(in_channels, num_classes, base_channels, attention_heads) < 1:
            raise ValueError("通道数、类别输出数和注意力子门数量必须为正数")
        Conv, _, MaxPool, self.mode = spatial_layers(spatial_dims)
        self.deep_supervision = deep_supervision
        c = base_channels

        # Fig.1 给出四尺度示意；具体宽度及每级统一双卷积是明确记录的补全。
        self.encoder1 = DoubleConv(in_channels, c, spatial_dims)
        self.encoder2 = DoubleConv(c, 2 * c, spatial_dims)
        self.encoder3 = DoubleConv(2 * c, 4 * c, spatial_dims)
        self.bottleneck = DoubleConv(4 * c, 8 * c, spatial_dims)
        self.pool = MaxPool(kernel_size=2, stride=2)

        # Fint=Fx/2 为本实现选择。g 来自下一粗尺度的 decoder 特征。
        self.gate3 = MultiAttentionGate(4 * c, 8 * c, 2 * c,
                                        spatial_dims, attention_heads)
        self.decoder3 = UpConcatBlock(8 * c, 4 * c, 4 * c, spatial_dims)
        self.gate2 = MultiAttentionGate(2 * c, 4 * c, c,
                                        spatial_dims, attention_heads)
        self.decoder2 = UpConcatBlock(4 * c, 2 * c, 2 * c, spatial_dims)
        self.gate1 = (MultiAttentionGate(c, 2 * c, max(1, c // 2),
                                         spatial_dims, attention_heads)
                      if gate_first_skip else None)
        self.decoder1 = UpConcatBlock(2 * c, c, c, spatial_dims)

        self.head = Conv(c, num_classes, kernel_size=1)
        if deep_supervision:
            # 论文使用深监督，但未指定头位置/融合细节；这里在两个中间解码层
            # 分别预测，并将 logits 插值至输入尺寸后独立计算辅助损失。
            self.aux2_head = Conv(2 * c, num_classes, kernel_size=1)
            self.aux3_head = Conv(4 * c, num_classes, kernel_size=1)

    def forward(self, x):
        # 以 2D、C=16、x=[1,3,128,128] 为例；3D 在空间轴增加 D。
        x1 = self.encoder1(x)                # [1, 16, 128, 128]
        x2 = self.encoder2(self.pool(x1))    # [1, 32,  64,  64]
        x3 = self.encoder3(self.pool(x2))    # [1, 64,  32,  32]
        x4 = self.bottleneck(self.pool(x3))  # [1,128,  16,  16]

        # 粗尺度 x4 先引导 x3 的门控，再上采样并拼接。
        skip3, alpha3 = self.gate3(x3, x4)  # alpha3: [1,heads,32,32]
        d3 = self.decoder3(x4, skip3)        # [1, 64, 32, 32]

        # 下一个 g 使用已经融合上下文的 d3，不始终复用瓶颈 x4。
        skip2, alpha2 = self.gate2(x2, d3)  # alpha2: [1,heads,64,64]
        d2 = self.decoder2(d3, skip2)        # [1, 32, 64, 64]

        skip1 = x1
        if self.gate1 is not None:
            skip1, alpha1 = self.gate1(x1, d2)
        d1 = self.decoder1(d2, skip1)        # [1, 16,128,128]
        logits = self.head(d1)              # [1, K,128,128]，未经 softmax

        if not self.deep_supervision:
            return logits

        # train/eval 都计算并返回相同的三个头；ONNX 也导出三个头。
        aux2_logits = self.aux2_head(d2)     # [1, K,64,64]
        aux2_logits = F.interpolate(aux2_logits, size=x.shape[2:],
                                    mode=self.mode, align_corners=False)
        aux3_logits = self.aux3_head(d3)     # [1, K,32,32]
        aux3_logits = F.interpolate(aux3_logits, size=x.shape[2:],
                                    mode=self.mode, align_corners=False)
        return logits, aux2_logits, aux3_logits


if __name__ == "__main__":
    import numpy as np
    import onnx
    import onnxruntime as ort

    torch.manual_seed(13)

    # 只修改此处即可选择 2D 或 3D；演示和 ONNX 导出会同步使用该设置。
    spatial_dims = 2
    input_shape = (1, 3, 128, 128) if spatial_dims == 2 else (1, 1, 16, 32, 32)
    model = AttentionUNet(
        in_channels=input_shape[1], num_classes=2, base_channels=16,
        spatial_dims=spatial_dims, deep_supervision=True,
        attention_heads=1, gate_first_skip=False,
    ).cpu().eval()
    x = torch.randn(*input_shape, dtype=torch.float32)
    output_names = ["logits", "aux2_logits", "aux3_logits"]
    onnx_path = Path(__file__).with_name(f"attention_unet_{spatial_dims}d.onnx")

    with torch.no_grad():
        # 1. 演示：直接调用此模型的 forward；没有额外的后处理或替代模块。
        outputs = model(x)
        print("input:", tuple(x.shape))
        for name, output in zip(output_names, outputs):
            print(name, tuple(output.shape), "前6个数:", output.flatten()[:6])

        # 2. 导出：原封不动使用刚才的 model 和 x，不重建、不重新初始化、
        # 不切换 train/eval、不使用 wrapper，也不跳过任何一个输出头。
        # 显式选择兼容导出器；本次环境的 dynamo 导出器在 Conv3d(bias=False)
        # 上生成错误的 bias 形状。这里只切换导出器，不改变模型计算。
        torch.onnx.export(
            model, (x,), str(onnx_path),
            input_names=["image"], output_names=output_names,
            opset_version=17, dynamo=True,
        )

    # 3. 用刚才同一份 x 核对全部输出；容许浮点运算顺序带来的微小误差。
    onnx.checker.check_model(str(onnx_path), full_check=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(output_names, {"image": x.numpy()})
    for name, expected, actual in zip(output_names, outputs, onnx_outputs):
        expected_array = expected.numpy()
        np.testing.assert_allclose(actual, expected_array, rtol=1e-4, atol=1e-5)
        print(f"{name}: 最大绝对误差 {np.max(np.abs(actual - expected_array)):.8g}")
    print(f"ONNX 导出及全部输出一致性校验通过: {onnx_path}")
