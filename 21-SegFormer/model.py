"""SegFormer 论文参考模型。

直接运行本文件：同一个 model、同一个 image、同一个 forward，先演示再导出 ONNX。
编码器配置遵循上传论文表 6；解码器遵循公式 (4)，差异说明见 README。
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .encoder import MixTransformerEncoder
    from .decoder import AllMLPDecoder
else:
    from encoder import MixTransformerEncoder
    from decoder import AllMLPDecoder


class SegFormer(nn.Module):
    """输入 [B, in_channels, H, W]，输出 [B, num_classes, H, W] logits。

    H、W 建议至少为 32；支持非正方形和不能被 32 整除的尺寸。
    解码器本身输出第 1 阶段分辨率，最后插值回原图是常用的分割接口补全。
    num_classes=1 用于单通道二分类；num_classes>=2 用于互斥类别分类。
    """

    def __init__(self, num_classes=2, variant="b0", in_channels=3,
                 drop_path_rate=0.1):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes 必须为正数")
        self.encoder = MixTransformerEncoder(variant, in_channels, drop_path_rate)
        self.decoder = AllMLPDecoder(
            self.encoder.channels, self.encoder.decoder_dim, num_classes
        )

    def forward(self, image):
        original_size = image.shape[-2:]  # [H, W]

        # 1. 四阶段层次化 Transformer：输出四种分辨率。
        features = self.encoder(image)

        # 2. 公式 (4)：多尺度特征投影、插值、通道拼接、融合和分类。
        low_resolution_logits = self.decoder(features)

        # 3. 通用分割接口补全：将 logits 恢复到输入尺寸。
        # 该操作始终在 forward 中；演示和 ONNX 都会执行。
        logits = F.interpolate(
            low_resolution_logits,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        return logits  # 不做 softmax、sigmoid 或 argmax。


if __name__ == "__main__":
    import onnx

    # 只创建一次模型和输入。随机权重仅用于观察数据流，不代表训练后的分割结果。
    torch.manual_seed(42)
    model = SegFormer(num_classes=2, variant="b0").cpu()
    model.eval()  # 两个过程都保持 eval；DropPath 在两者中都关闭。
    image = torch.randn(1, 3, 256, 256)
    onnx_path = Path(__file__).with_name("segformer_b0.onnx")

    with torch.no_grad():
        # 过程一：演示数据经过完整模型。
        logits = model(image)
        print("Input shape :", tuple(image.shape))
        print("Output shape:", tuple(logits.shape))
        print("Pixel logits:", logits[0, :, 0, 0].tolist())

        # 过程二：直接导出上面的 model 和 image，仍走完全相同的 forward。
        # 不重建模型、不替换模块、不增加 export 专用计算分支。
        torch.onnx.export(
            model,
            image,
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,  # 显式选择传统导出器，便于观察展开的计算节点。
            do_constant_folding=False,
            training=torch.onnx.TrainingMode.EVAL,
        )

    onnx.checker.check_model(str(onnx_path))
    print("ONNX exported:", onnx_path)

    # 可选的数值核对：同一份 image，比较 ONNX 与上面已计算好的 logits。
    # 没装 onnxruntime 时，仅跳过此核对；ONNX 导出已完成。
    try:
        import onnxruntime as ort
    except ImportError:
        print("未安装 onnxruntime；可安装后重跑，以核对两种推理结果。")
    else:
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        onnx_logits = session.run(["logits"], {"image": image.numpy()})[0]
        onnx_logits = torch.from_numpy(onnx_logits)
        torch.testing.assert_close(onnx_logits, logits, rtol=1e-4, atol=1e-5)
        maximum_error = (onnx_logits - logits).abs().max().item()
        print(f"ONNX/PyTorch 一致性检查通过，最大绝对误差: {maximum_error:.3e}")
