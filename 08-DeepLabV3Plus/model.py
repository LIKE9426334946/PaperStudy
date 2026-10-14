"""DeepLabv3+ 论文参考实现；运行本文件可演示并导出同一模型。

输出始终为原图大小的 logits，不在 forward 内做 softmax/argmax。
演示与导出使用同一个 model、同一个 x、同一种 eval 模式和同一个 forward。
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .backbones import AlignedXception, DilatedResNet101
    from .blocks import ASPP, DeepLabDecoder
else:
    from backbones import AlignedXception, DilatedResNet101
    from blocks import ASPP, DeepLabDecoder


class DeepLabV3Plus(nn.Module):
    """Backbone -> ASPP -> 低层特征融合 -> 两次3×3 -> 分类 -> 双线性上采样。

    backbone: xception65（论文 Fig.4 默认）、xception71 或 resnet101。
    use_separable_conv 只切换 ASPP/decoder 的3×3卷积；Xception 本身始终
    使用可分离卷积。False 对应论文加入头部可分离卷积之前的对照设定。
    image_pooling=False 可考察论文 Cityscapes 的无 image-level 分支变体。
    """

    def __init__(self, in_channels=3, num_classes=21, output_stride=16,
                 backbone="xception65", use_separable_conv=True,
                 image_pooling=True):
        super().__init__()
        if in_channels < 1 or num_classes < 1:
            raise ValueError("in_channels 和 num_classes 必须为正整数")
        if backbone in ("xception65", "xception71"):
            self.backbone = AlignedXception(in_channels, output_stride, backbone)
        elif backbone == "resnet101":
            self.backbone = DilatedResNet101(in_channels, output_stride)
        else:
            raise ValueError("backbone 必须为 xception65、xception71 或 resnet101")
        self.aspp = ASPP(2048, output_stride, use_separable_conv, image_pooling)
        self.decoder = DeepLabDecoder(256, use_separable_conv)
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # 以 B=1、256×256、OS=16、K=21 为例。
        input_size = x.shape[-2:]
        high_level, low_level = self.backbone(x)  # [1,2048,16,16] / [1,256,64,64]
        high_level = self.aspp(high_level)        # [1,256,16,16]
        x = self.decoder(high_level, low_level)  # [1,256,64,64]
        logits = self.classifier(x)              # [1,21,64,64]
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=True,
        )
        return logits                            # [1,21,256,256]


if __name__ == "__main__":
    # 导出依赖仅在演示入口导入；单纯 import 模型只需 torch。
    import numpy as np
    import onnx
    import onnxruntime as ort

    torch.manual_seed(0)
    model = DeepLabV3Plus(
        in_channels=3, num_classes=21, output_stride=16,
        backbone="xception65", use_separable_conv=True, image_pooling=True,
    ).cpu().eval()
    x = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    onnx_path = Path(__file__).with_name("deeplabv3plus.onnx")

    with torch.no_grad():
        # 1. 直接运行完整模型。随机参数仅用于看数据流，不代表分割能力。
        output = model(x)
        print("input:", tuple(x.shape))
        print("logits:", tuple(output.shape), "前6个数:", output.flatten()[:6])

        # 2. 同一个 model 和 x，不重建网络、不替换算子、不切换训练状态。
        # optimize=False 尽量保留结构，便于在 Netron 对照 forward 阅读。
        # 固定形状导出；单个 ONNX 内嵌权重，避免伴随 .data 文件。
        torch.onnx.export(
            model, (x,), str(onnx_path),
            input_names=["image"], output_names=["logits"],
            opset_version=18, dynamo=True, optimize=True, external_data=False,
        )

    # 3. 把同一份 x 送入 ONNX Runtime，对照刚才保存的 PyTorch logits。
    onnx.checker.check_model(str(onnx_path), full_check=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"image": x.numpy()})[0]
    expected = output.numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    print(f"最大绝对误差: {np.max(np.abs(actual - expected)):.8g}")
    print(f"ONNX 导出及数值一致性校验通过: {onnx_path}")
