"""CVPR 2015 FCN-VGG16：FCN-32s / FCN-16s / FCN-8s。

直接执行：同一 model、同一 x、同一 eval forward 先输出 logits，再导出
ONNX，并由 ONNX Runtime 核对数值。没有导出专用的替代 forward。
"""

from pathlib import Path

import torch
from torch import nn

if __package__:
    from .blocks import (
        ConvolutionalClassifier, VGG16Features, bilinear_upsampler, crop_like,
    )
else:
    from blocks import (
        ConvolutionalClassifier, VGG16Features, bilinear_upsampler, crop_like,
    )


class FCN(nn.Module):
    """output_stride=32、16、8 分别选择论文中的三种 FCN。

    输出始终为 [B,num_classes,H,W] 的 logits。8s 的 8 表示最后一次
    上采样前的有效步幅，不表示最终输出比输入小 8 倍。
    默认没有下载/加载预训练权重；zero_init_scores=True 遵循论文零初始化。
    fc_channels=4096 保留论文宽度；减小它会改变模型容量。
    """

    def __init__(self, num_classes=21, output_stride=8, fc_channels=4096,
                 zero_init_scores=True):
        super().__init__()
        if output_stride not in (32, 16, 8):
            raise ValueError("output_stride 只能取 32、16 或 8")
        if num_classes < 1 or fc_channels < 1:
            raise ValueError("num_classes 和 fc_channels 必须为正整数")
        self.output_stride = output_stride
        self.features = VGG16Features()
        self.classifier = ConvolutionalClassifier(fc_channels)
        self.score_fr = nn.Conv2d(fc_channels, num_classes, kernel_size=1)

        if output_stride <= 16:
            self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
            self.upscore2 = bilinear_upsampler(num_classes, stride=2, trainable=True)
        if output_stride == 8:
            self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
            self.upscore_pool4 = bilinear_upsampler(num_classes, stride=2, trainable=True)

        # 上传的 CVPR 2015 论文 Sec.4.3：末级固定双线性，中间层可学习。
        self.upscore_final = bilinear_upsampler(
            num_classes, stride=output_stride, trainable=False,
        )
        self.final_crop_offset = {32: 19, 16: 27, 8: 31}[output_stride]

        # 骨干未加载预训练时采用 PyTorch 默认初始化，属于工程补全。
        # 论文中新加的类别评分层采用零初始化。
        for name in ("score_fr", "score_pool4", "score_pool3"):
            layer = getattr(self, name, None)
            if layer is not None:
                if zero_init_scores:
                    nn.init.zeros_(layer.weight)
                else:
                    # 仅供未训练演示/非零数值校验；不是论文训练配置。
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        # 以 x=[B,3,256,256]、K=num_classes 为例。
        pool3, pool4, pool5 = self.features(x)
        conv7 = self.classifier(pool5)      # [B,4096,9,9]
        score = self.score_fr(conv7)        # [B,K,9,9]，步幅 32 的类别分数

        if self.output_stride <= 16:
            # FCN-16s / FCN-8s：先投影到 K 类，再融合类别分数。
            up2 = self.upscore2(score)      # [B,K,20,20]
            skip4 = self.score_pool4(pool4) # [B,K,29,29]
            skip4 = crop_like(skip4, up2, offset=5)
            score = up2 + skip4             # [B,K,20,20]，逐元素相加，不是 cat

        if self.output_stride == 8:
            up_pool4 = self.upscore_pool4(score)  # [B,K,42,42]
            skip3 = self.score_pool3(pool3)       # [B,K,57,57]
            skip3 = crop_like(skip3, up_pool4, offset=9)
            score = up_pool4 + skip3        # [B,K,42,42]，步幅 8 的融合分数

        dense = self.upscore_final(score)   # 32s:320² / 16s:336² / 8s:344²
        logits = crop_like(dense, x, offset=self.final_crop_offset)
        return logits                      # [B,K,256,256]；不在此处 softmax/argmax

    @torch.no_grad()
    def load_vgg16_weights(self, state_dict):
        """从 torchvision 标准 VGG16（非 BN 版）state_dict 迁移预训练参数。

        不自动联网下载。fc6/fc7 用 reshape 将 Linear 权重变为卷积核，
        不复制原 1000 类 fc8；类别头和上采样核保持各自初始化。
        torchvision RGB 预处理与作者 Caffe BGR 预处理不同，见 README。
        """
        if self.classifier.fc6.out_channels != 4096:
            raise ValueError("迁移原始 VGG16 权重要求 fc_channels=4096")
        names = (
            "conv1_1", "conv1_2", "conv2_1", "conv2_2",
            "conv3_1", "conv3_2", "conv3_3", "conv4_1", "conv4_2", "conv4_3",
            "conv5_1", "conv5_2", "conv5_3",
        )
        indices = (0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28)
        for name, index in zip(names, indices):
            layer = getattr(self.features, name)
            layer.weight.copy_(state_dict[f"features.{index}.weight"])
            layer.bias.copy_(state_dict[f"features.{index}.bias"])
        for name, index in (("fc6", 0), ("fc7", 3)):
            layer = getattr(self.classifier, name)
            layer.weight.copy_(state_dict[f"classifier.{index}.weight"].reshape_as(layer.weight))
            layer.bias.copy_(state_dict[f"classifier.{index}.bias"])


if __name__ == "__main__":
    import numpy as np
    import onnx
    import onnxruntime as ort

    torch.manual_seed(12)
    # 限制 CPU 线程只影响执行效率，不改变网络；演示/导出/核验均用 CPU float32。
    torch.set_num_threads(min(4, torch.get_num_threads()))

    # 保留完整 4096 通道，仅评分层改为随机初始化以得到非零演示输出。
    # 学习论文初始化时用 zero_init_scores=True；未训练结果不代表分割效果。
    model = FCN(num_classes=21, output_stride=8,
                fc_channels=4096, zero_init_scores=False).eval()
    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        # 1. 直接执行模型。此后不改 model、权重、模式或输入 x。
        output = model(x)
        print("输入形状:", tuple(x.shape))
        print("输出 logits 形状:", tuple(output.shape))
        print("第一个像素的前 5 类 logits:", output[0, :5, 0, 0])

        # 2. 导出完全相同的 model(x)，没有包装器、替代算子或导出专用分支。
        onnx_path = Path(__file__).resolve().with_name("fcn8s.onnx")
        torch.onnx.export(
            model,
            (x,),
            str(onnx_path),
            input_names=["image"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            optimize=False,        # 便于观察导出的数据流；不是逐 Python 行的展示
            external_data=True,    # 大型 VGG16 权重保存到同目录的 .onnx.data
        )

    # 3. 用同一个 x 验证 ONNX 与上面的 PyTorch output 数值一致（浮点容差内）。
    # 导出示例固定 [1,3,256,256]；PyTorch forward 本身支持其它 H/W。
    onnx.checker.check_model(str(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    onnx_output = session.run(["logits"], {"image": x.numpy()})[0]
    np.testing.assert_allclose(onnx_output, output.numpy(), rtol=1e-4, atol=1e-5)
    print("ONNX 最大绝对误差:", np.abs(onnx_output - output.numpy()).max())
    print("ONNX 检查及数值对比通过:", onnx_path)
