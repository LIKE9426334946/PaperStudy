"""独立导出脚本：运行 python export_onnx.py，再用 Netron 打开生成文件。

这是便于查看结构的固定示例配置，不是针对某个真实数据集自动生成的配置。
默认导出 2D；把 DIM 改为 3 可查看各向异性 3D 结构。
随机权重仅供结构学习；真实推理需载入与网络配置匹配的已训练权重。
"""

from pathlib import Path

import onnx
import torch

from model import NNUNet


DIM = 2
IN_CHANNELS = 1
NUM_CLASSES = 3  # 包括背景；二分类应设为 2。


def export_model(dim=DIM, output_path=None):
    if dim == 2:
        patch_size = (128, 128)
        kernel_sizes = [(3, 3)] * 4
        strides = [(2, 2)] * 3
    else:
        patch_size = (16, 64, 64)
        kernel_sizes = [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
        strides = [(1, 2, 2), (2, 2, 2), (2, 2, 2)]

    torch.manual_seed(0)
    model = NNUNet(IN_CHANNELS, NUM_CLASSES, kernel_sizes, strides,
                   deep_supervision=False).eval()
    x = torch.randn(1, IN_CHANNELS, *patch_size)
    path = Path(output_path) if output_path is not None else (
        Path(__file__).resolve().parent / f"nnunet_{dim}d.onnx")
    # 显式指定导出后端和静态输入尺寸，避免不同版本的默认行为差异。
    with torch.no_grad():
        torch.onnx.export(model, (x,), str(path), input_names=["image"],
                          output_names=["logits"], opset_version=18,
                          dynamo=True, external_data=False)
    onnx.checker.check_model(str(path))
    print(f"已导出：{path}")
    print(f"输入：{tuple(x.shape)}；输出：[1, {NUM_CLASSES}, {', '.join(map(str, patch_size))}]")
    return model, x, path


if __name__ == "__main__":
    export_model()
