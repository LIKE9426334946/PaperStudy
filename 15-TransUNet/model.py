"""TransUNet (2021 arXiv v1) 的 PyTorch 论文阅读参考实现。

运行 python model.py：完整 Base 配置前向传播 -> ONNX 导出 -> 数值核验。
采用随机权重，不包含训练循环或预训练权重加载，不代表论文实验精度。
"""

from pathlib import Path

import torch
from torch import nn

from backbone import HybridCNNEncoder
from transformer import HybridPatchEmbedding, TransformerEncoder
from decoder import CascadedUpsampler, SegmentationHead


class TransUNet(nn.Module):
    def __init__(self, image_size=224, in_channels=3, num_classes=9,
                 embed_dim=768, depth=12, num_heads=12, mlp_dim=3072,
                 dropout=0.1, attention_dropout=0.0, skip_scales=(8, 4, 2)):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)
        if len(self.image_size) != 2 or any(
            not isinstance(side, int) or side < 32 or side % 16 != 0
            for side in self.image_size
        ):
            raise ValueError("image_size 必须是 >=32 且能被 16 整除的高和宽。")
        if in_channels not in (1, 3) or num_classes < 1:
            raise ValueError("in_channels 必须为 1 或 3；num_classes 必须为正。")
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.grid_size = (self.image_size[0] // 16, self.image_size[1] // 16)

        self.cnn = HybridCNNEncoder()
        self.patch_embedding = HybridPatchEmbedding(self.grid_size, embed_dim, dropout)
        self.transformer = TransformerEncoder(
            embed_dim, depth, num_heads, mlp_dim, dropout, attention_dropout
        )
        self.decoder = CascadedUpsampler(embed_dim, skip_scales)
        self.head = SegmentationHead(num_classes)

    def forward(self, images):
        # 固定输入 C/H/W；构造时可以选不同分辨率，不能只改运行时输入尺寸。
        # ONNX trace 会将尺寸检查固化为示例输入的结果，不声明动态 C/H/W。
        if images.ndim != 4:
            raise ValueError("输入应为 [B,C,H,W] 四维 tensor。")
        if images.shape[1] != self.in_channels or tuple(images.shape[2:]) != self.image_size:
            raise ValueError("输入通道数及空间尺寸必须与构造参数一致。")

        # 工程补全：灰度图沿通道维复制三次，沿用官方对单通道图像的处理。
        # 这个条件取决于构造参数，演示和导出均执行相同路径。
        if self.in_channels == 1:
            images = images.repeat(1, 3, 1, 1)

        # 1. CNN 局部特征和跳跃连接。不是对原图直接执行 Patch Embedding。
        deep_features, skips = self.cnn(images)          # [B,1024,H/16,W/16]

        # 2. 公式 (1)：每个 CNN 空间位置作为 token，加可学习位置编码。
        tokens = self.patch_embedding(deep_features)    # [B,N,D]，N=H*W/256

        # 3. 公式 (2)、(3)：L 层全局自注意力 + 逐 token MLP。
        tokens = self.transformer(tokens)               # [B,N,D]

        # 4. 按 flatten 时的相同空间顺序还原，不使用 sqrt(N) 猜测尺寸。
        feature_map = tokens.transpose(1, 2).contiguous()
        feature_map = feature_map.reshape(
            tokens.shape[0], self.embed_dim, self.grid_size[0], self.grid_size[1]
        )                                               # [B,D,H/16,W/16]

        # 5. CUP 四次 2x 上采样，融合 /8、/4、/2 的 CNN 特征。
        features = self.decoder(feature_map, skips)     # [B,16,H,W]

        # 6. 分割头。演示与导出都返回完整 logits，而不是隐藏特征或 argmax。
        logits = self.head(features)                    # [B,K,H,W]
        return logits


if __name__ == "__main__":
    import numpy as np
    import onnx
    import onnxruntime as ort

    torch.manual_seed(0)
    # 完整 R50-ViT Base：D=768、12 层、12 个 head、MLP=3072，三条 skip。
    model = TransUNet(image_size=224, in_channels=3, num_classes=9).cpu().eval()
    images = torch.randn(1, 3, 224, 224)
    onnx_path = Path(__file__).resolve().parent / "transunet.onnx"

    with torch.no_grad():
        # 1. 演示数据经过模型：直接调用完整 model.forward()。
        logits = model(images)
        print("Input shape:", tuple(images.shape))
        print("Output logits shape:", tuple(logits.shape))
        print("First-pixel logits:", logits[0, :, 0, 0].tolist())
        print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))

        # 2. 相同 model 对象、相同 images 对象、相同 eval 状态、相同 forward。
        # 不重建模型、不更换权重、不去掉 skip，也没有导出专用计算分支。
        torch.onnx.export(
            model,
            images,
            str(onnx_path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
            export_params=True,
        )

    # 3. 校验文件和实际推理数值，不仅检查能否导出。
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"images": images.numpy()})[0]
    pytorch_logits = logits.numpy()
    assert np.isfinite(pytorch_logits).all() and np.isfinite(onnx_logits).all()
    np.testing.assert_allclose(onnx_logits, pytorch_logits, rtol=1e-3, atol=1e-4)
    print("ONNX max absolute error:", float(np.abs(onnx_logits - pytorch_logits).max()))
    print("ONNX allclose passed (rtol=1e-3, atol=1e-4).")
    print("ONNX saved to:", onnx_path)
