"""FPN 论文的 PyTorch 参考实现：骨干网络、特征金字塔和共享检测头。"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import roi_pool


class ResNet50BottomUp(nn.Module):
    """取 ResNet-50 每个残差 stage 的最后输出 C2、C3、C4、C5。"""

    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()
        net = resnet50(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stage2 = net.layer1
        self.stage3 = net.layer2
        self.stage4 = net.layer3
        self.stage5 = net.layer4

    def forward(self, images):
        # images: [B, 3, H, W]
        x = self.stem(images)   # [B, 64, H/4, W/4]
        c2 = self.stage2(x)     # [B, 256, H/4, W/4]
        c3 = self.stage3(c2)    # [B, 512, H/8, W/8]
        c4 = self.stage4(c3)    # [B, 1024, H/16, W/16]
        c5 = self.stage5(c4)    # [B, 2048, H/32, W/32]
        return c2, c3, c4, c5


class FeaturePyramid(nn.Module):
    """论文 §3：1×1 侧向连接 + 最近邻上采样相加 + 3×3 输出卷积。"""

    def __init__(self, channels=256):
        super().__init__()
        self.lateral2 = nn.Conv2d(256, channels, kernel_size=1)
        self.lateral3 = nn.Conv2d(512, channels, kernel_size=1)
        self.lateral4 = nn.Conv2d(1024, channels, kernel_size=1)
        self.lateral5 = nn.Conv2d(2048, channels, kernel_size=1)
        self.output2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.output3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.output4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.output5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, c2, c3, c4, c5):
        # 顶层从 C5 开始，不做额外的激活或归一化。
        m5 = self.lateral5(c5)  # [B, 256, H/32, W/32]

        # 上采样后与相应的浅层特征逐元素相加；指定 size 可对齐奇数尺寸。
        m4 = self.lateral4(c4) + F.interpolate(
            m5, size=c4.shape[-2:], mode="nearest"
        )  # [B, 256, H/16, W/16]
        m3 = self.lateral3(c3) + F.interpolate(
            m4, size=c3.shape[-2:], mode="nearest"
        )  # [B, 256, H/8, W/8]
        m2 = self.lateral2(c2) + F.interpolate(
            m3, size=c2.shape[-2:], mode="nearest"
        )  # [B, 256, H/4, W/4]

        # 3×3 卷积减弱上采样的混叠；P2～P5 的通道数统一为 256。
        p2 = self.output2(m2)
        p3 = self.output3(m3)
        p4 = self.output4(m4)
        p5 = self.output5(m5)

        # 论文脚注：P6 只用于 RPN，由 P5 以步长 2 下采样得到。
        p6 = p5[:, :, ::2, ::2]
        return {"P2": p2, "P3": p3, "P4": p4, "P5": p5, "P6": p6}


class SharedRPNHead(nn.Module):
    """§4.1：所有金字塔层复用同一组 3×3 / 1×1 / 1×1 参数。"""

    def __init__(self, channels=256, anchors_per_location=3):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.objectness = nn.Conv2d(channels, anchors_per_location, kernel_size=1)
        self.box_regression = nn.Conv2d(
            channels, 4 * anchors_per_location, kernel_size=1
        )

    def forward(self, pyramid):
        objectness_logits, box_deltas = {}, {}
        for level in ("P2", "P3", "P4", "P5", "P6"):
            # [B, 256, Hk, Wk] -> [B, 3, Hk, Wk] 和 [B, 12, Hk, Wk]
            hidden = F.relu(self.conv3x3(pyramid[level]))
            objectness_logits[level] = self.objectness(hidden)
            box_deltas[level] = self.box_regression(hidden)
        return objectness_logits, box_deltas


def assign_rois_to_levels(rois):
    """论文式 (1)。rois: [R, 5] = (batch_index, x1, y1, x2, y2)。"""
    width = rois[:, 3] - rois[:, 1]
    height = rois[:, 4] - rois[:, 2]
    # k = floor(4 + log2(sqrt(w*h) / 224))；Fast R-CNN 只使用 P2～P5。
    level = torch.floor(4 + torch.log2(torch.sqrt(width * height) / 224.0))
    return level.to(torch.int64).clamp(2, 5)


class PyramidRoIPool(nn.Module):
    """§4.2：按 RoI 尺度路由到 P2～P5，再各自做 7×7 RoI pooling。"""

    def forward(self, pyramid, rois):
        # 输入坐标是原图坐标，列 0 是 batch index。
        levels = assign_rois_to_levels(rois)
        pooled = pyramid["P2"].new_empty((rois.shape[0], 256, 7, 7))
        for level in (2, 3, 4, 5):
            indices = torch.nonzero(levels == level).flatten()
            if indices.numel() == 0:
                continue
            selected_rois = rois.index_select(0, indices)
            level_features = roi_pool(
                pyramid[f"P{level}"],
                selected_rois,
                output_size=(7, 7),
                spatial_scale=1.0 / (2**level),
            )  # [Rk, 256, 7, 7]
            pooled = pooled.index_copy(0, indices, level_features)
        return pooled, levels


class SharedFastRCNNHead(nn.Module):
    """§4.2：两个 1024 维全连接层和共享的分类、回归预测器。"""

    def __init__(self, num_classes, channels=256):
        super().__init__()
        self.fc1 = nn.Linear(channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.classifier = nn.Linear(1024, num_classes)
        self.box_regression = nn.Linear(1024, num_classes * 4)

    def forward(self, pooled):
        x = pooled.flatten(start_dim=1)  # [R, 256*7*7]
        x = F.relu(self.fc1(x))            # [R, 1024]
        x = F.relu(self.fc2(x))            # [R, 1024]
        return self.classifier(x), self.box_regression(x)


class FPNDetectorReference(nn.Module):
    """串联论文中的特征构建与两个检测头；RoI 由外部候选框算法提供。"""

    def __init__(self, num_classes, weights=ResNet50_Weights.IMAGENET1K_V1):
        super().__init__()
        self.backbone = ResNet50BottomUp(weights)
        self.fpn = FeaturePyramid()
        self.rpn_head = SharedRPNHead()
        self.roi_pool = PyramidRoIPool()
        self.fast_rcnn_head = SharedFastRCNNHead(num_classes)

    def forward(self, images, rois=None):
        # 自底向上：一次图像前向得到 C2、C3、C4、C5。
        c2, c3, c4, c5 = self.backbone(images)
        # 自顶向下 + 侧向融合：得到 P2～P6。
        pyramid = self.fpn(c2, c3, c4, c5)
        # 每一层都做 RPN 预测；P2～P6 共享同一个 RPN head。
        objectness, rpn_box_deltas = self.rpn_head(pyramid)
        outputs = {
            "pyramid": pyramid,
            "rpn_objectness": objectness,
            "rpn_box_deltas": rpn_box_deltas,
        }

        if rois is not None:
            # 外部生成的候选框 -> 按论文式 (1) 分配层级 -> 7×7 RoI 池化。
            pooled, levels = self.roi_pool(pyramid, rois)
            # 所有层复用同一个 Fast R-CNN head。
            class_logits, box_deltas = self.fast_rcnn_head(pooled)
            outputs.update({
                "roi_levels": levels,
                "class_logits": class_logits,
                "box_deltas": box_deltas,
            })
        return outputs
