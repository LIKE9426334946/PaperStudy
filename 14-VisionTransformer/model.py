"""Vision Transformer (ICLR 2021) 的论文阅读参考实现。

运行 python model.py：完整 ViT-B/16 前向传播、导出 ONNX、验证数值一致性。
没有训练循环；默认没有加载预训练权重，随机 logits 不能用于实际分类。
"""

from pathlib import Path

import torch
from torch import nn

from blocks import PatchEmbedding, EncoderBlock, ClassificationHead


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_dim=3072, dropout=0.0, qkv_bias=True, norm_eps=1e-6,
                 representation_size=None, zero_init_head=True):
        """representation_size=None 对应单层微调头；整数对应预训练 MLP 头。

        微调时，新分类器按照 §3.2 默认零初始化。
        随机权重演示显式设 zero_init_head=False，以免全零输出掩盖导出错误。
        初始化分布、bias、LayerNorm epsilon 等补充选择见 README。
        """
        super().__init__()
        if depth <= 0 or mlp_dim <= 0 or num_classes <= 0:
            raise ValueError("depth、mlp_dim、num_classes 必须为正。")
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        # 可学习的 class token 与一维位置编码；不是固定正余弦位置编码。
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)
        # 每一层独立创建，层与层不共享权重。
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_dim, dropout, qkv_bias, norm_eps)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.head = ClassificationHead(embed_dim, num_classes, representation_size)

        # 工程补充：显式初始化，便于随机输入演示；并非论文完整训练配方。
        self.apply(self._init_weights)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.cls_token)
        if zero_init_head:
            nn.init.zeros_(self.head.classifier.weight)
            nn.init.zeros_(self.head.classifier.bias)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(self, images):
        # 以 ViT-B/16、224x224 为例：N=14*14=196，D=768。
        x = self.patch_embed(images)  # [B,3,224,224] -> [B,196,768]

        # 公式 (1)：把一个可学习的 class token 放到序列开头。
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B,1,768]
        x = torch.cat((cls_tokens, x), dim=1)                  # [B,197,768]
        x = x + self.position_embedding                     # 广播 batch 维
        x = self.embedding_dropout(x)

        # 公式 (2)、(3)：12 层编码器，始终保持 197 个 token、每个 768 维。
        for block in self.blocks:
            x = block(x)

        # 公式 (4)：取最后一层 class token，再对其 D 维做 LayerNorm。
        # LayerNorm 不混合 token，所以这与先 norm 全序列再取第 0 项等价。
        cls_feature = x[:, 0, :]        # [B,768]
        return self.norm(cls_feature)  # [B,768]

    def forward(self, images):
        # 演示与 ONNX 都直接执行这个 forward，没有导出专用分支或包装模型。
        cls_feature = self.forward_features(images)
        logits = self.head(cls_feature)  # [B,768] -> [B,K]
        return logits


def vit_base_patch16(**kwargs):
    """ViT-B/16：Table 1 的 Base 配置。"""
    return VisionTransformer(patch_size=16, embed_dim=768, depth=12,
                             num_heads=12, mlp_dim=3072, **kwargs)


def vit_base_patch32(**kwargs):
    return VisionTransformer(patch_size=32, embed_dim=768, depth=12,
                             num_heads=12, mlp_dim=3072, **kwargs)


def vit_large_patch16(**kwargs):
    return VisionTransformer(patch_size=16, embed_dim=1024, depth=24,
                             num_heads=16, mlp_dim=4096, **kwargs)


def vit_large_patch32(**kwargs):
    return VisionTransformer(patch_size=32, embed_dim=1024, depth=24,
                             num_heads=16, mlp_dim=4096, **kwargs)


def vit_huge_patch14(**kwargs):
    return VisionTransformer(patch_size=14, embed_dim=1280, depth=32,
                             num_heads=16, mlp_dim=5120, **kwargs)


if __name__ == "__main__":
    import numpy as np
    import onnx
    import onnxruntime as ort

    torch.manual_seed(0)
    # 完整的 ViT-B/16；不是减小层数或隐藏维度的教学替身。
    # 使用非零随机分类头，让下面的数值检查真正覆盖整个 backbone。
    model = vit_base_patch16(
        image_size=224, num_classes=1000, dropout=0.0,
        representation_size=None, zero_init_head=False,
    ).cpu().eval()
    images = torch.randn(1, 3, 224, 224)
    onnx_path = Path(__file__).resolve().parent / "vit_b16.onnx"

    # 1. 演示前向传播。
    # 2. 在同一 no_grad 环境导出同一个 model、同一个 images。
    # 两次执行之间不改变权重、分类头、模式、输入或 forward。
    with torch.no_grad():
        logits = model(images)
        print("Input shape:", tuple(images.shape))
        print("Output shape:", tuple(logits.shape))
        print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))
        print("First 5 logits (random weights):", logits[0, :5].tolist())

        torch.onnx.export(
            model,
            images,
            str(onnx_path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=17,
            dynamo=False,
            export_params=True,
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )

    # 完整检查 ONNX 文件，再用刚才那一份输入比较刚才保存的 PyTorch 输出。
    # ONNX 的基础算子名称/分解方式不必与 Python 类逐字对应，数值允许浮点误差。
    onnx.checker.check_model(str(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["logits"], {"images": images.numpy()})[0]
    reference = logits.numpy()
    np.testing.assert_allclose(onnx_logits, reference, rtol=1e-4, atol=1e-5)
    print("ONNX max absolute error:", float(np.abs(onnx_logits - reference).max()))
    print("ONNX validation passed:", onnx_path)
