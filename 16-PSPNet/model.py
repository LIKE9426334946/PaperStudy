"""PSPNet 论文阅读用参考实现；仅包含网络与简短前向示例。"""

import torch
from torch import nn
from torch.nn import functional as F

from backbone import DilatedResNet
from blocks import PyramidPoolingModule, SegmentationHead


class PSPNet(nn.Module):
    """训练返回 (main_logits, aux_logits)，eval 返回 main_logits。

    num_classes=2 可用于背景/水域两类，搭配 CrossEntropyLoss。
    depth 可选 50/101/152。初始化没有加载 ImageNet 预训练权重。
    """

    def __init__(self, num_classes=21, depth=50):
        super().__init__()
        self.backbone = DilatedResNet(depth)
        self.ppm = PyramidPoolingModule(in_channels=2048)
        self.main_head = SegmentationHead(self.ppm.out_channels, 512, num_classes)
        # res4 输出处增加辅助监督；ResNet101 对应论文 Fig.4 的 res4b22 后。
        self.aux_head = SegmentationHead(1024, 256, num_classes)

    def forward(self, x):
        input_size = x.shape[-2:]  # (H, W)，也支持非正方形输入
        aux_features, main_features = self.backbone(x)
        context = self.ppm(main_features)  # [B, 4096, h, w]
        main_logits = self.main_head(context)  # [B, K, h, w]
        main_logits = F.interpolate(
            main_logits,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )  # [B, K, H, W]

        if self.training:
            # 不 detach，辅助损失可以反向传播到 res4 及之前的所有层。
            aux_logits = self.aux_head(aux_features)
            aux_logits = F.interpolate(
                aux_logits,
                size=input_size,
                mode="bilinear",
                align_corners=True,
            )  # [B, K, H, W]
            return main_logits, aux_logits
        return main_logits  # 推理不执行辅助分类头


if __name__ == "__main__":
    model = PSPNet(num_classes=21, depth=50).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = model(x)
        prediction = logits.argmax(dim=1)
    print("logits:", logits.shape)  # [1, 21, 256, 256]
    print("prediction:", prediction.shape)  # [1, 256, 256]

    torch.onnx.export(
        model,
        x,
        "pspnet.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamo=True,
    )
