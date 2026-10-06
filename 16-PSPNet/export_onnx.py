"""独立 ONNX 导出脚本：固定空间尺寸，精确保留自适应平均池化语义。

默认执行：python export_onnx.py
输出到脚本所在目录 pspnet_resnet50.onnx，可直接用 Netron 打开。
仅导出 eval 主分支；随机权重用于看结构，不能直接用于有效预测。
"""

from copy import deepcopy
from pathlib import Path

import onnx
import torch
from torch import nn

from model import PSPNet


class FixedAdaptiveAvgPool2d(nn.Module):
    """用 Slice + ReduceMean 展开固定 H/W 的 AdaptiveAvgPool2d。

    不能把 32→6 简化为固定 kernel/stride 的普通池化。
    每个 bin 的起止位置遵循 PyTorch 的 floor/ceil 定义，边界可重叠。
    这是导出适配，属于工程补全，不是论文的新模块。
    """

    def __init__(self, input_size, bins):
        super().__init__()
        h, w = input_size
        self.rows = [(i * h // bins, ((i + 1) * h + bins - 1) // bins)
                     for i in range(bins)]
        self.cols = [(j * w // bins, ((j + 1) * w + bins - 1) // bins)
                     for j in range(bins)]

    def forward(self, x):
        rows = []
        for top, bottom in self.rows:
            cells = []
            for left, right in self.cols:
                region = x[:, :, top:bottom, left:right]
                cells.append(region.mean(dim=(-2, -1), keepdim=True))
            rows.append(torch.cat(cells, dim=-1))
        return torch.cat(rows, dim=-2)


def export_onnx(model, example, output_path):
    """接受本目录的 PSPNet；输出固定 B/C/H/W 的 ONNX。

    对模型副本执行替换，不改变调用者的模型或训练模式。
    model 与 example 应使用相同设备和 dtype。
    """
    export_model = deepcopy(model).eval()
    with torch.no_grad():
        reference = export_model(example)
        _, features = export_model.backbone(example)
        feature_size = tuple(features.shape[-2:])

    for bins, branch in zip(export_model.ppm.bin_sizes, export_model.ppm.branches):
        branch.pool = FixedAdaptiveAvgPool2d(feature_size, bins)

    with torch.no_grad():
        torch.testing.assert_close(export_model(example), reference, rtol=1e-4, atol=1e-5)

    # 明确选用 TorchScript 导出器，避免新版默认 dynamo 的切换影响此示例。
    # 所有空间尺寸固定；修改 example 后重新导出即可，不能给 H/W 随意标动态。
    torch.onnx.export(
        export_model,
        example,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    onnx.checker.check_model(str(output_path))
    print(f"ONNX 已导出并通过结构检查：{output_path}")
    print(f"输入：{tuple(example.shape)}；输出：{tuple(reference.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    depth = 50                       # 可改为 101 或 152
    num_classes = 21                 # 水域二分类可改为 2
    model = PSPNet(num_classes=num_classes, depth=depth)
    # 有本参考实现训练出的权重时，在此使用 model.load_state_dict(...)。
    example = torch.randn(1, 3, 256, 256)
    output = Path(__file__).resolve().parent / f"pspnet_resnet{depth}.onnx"
    export_onnx(model, example, output)
