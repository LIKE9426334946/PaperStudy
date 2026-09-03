# Deep Residual Learning for Image Recognition（ResNet）

## 论文信息

- **标题**：Deep Residual Learning for Image Recognition
- **作者**：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **会议**：IEEE Conference on Computer Vision and Pattern Recognition（CVPR）
- **年份**：2016
- **论文链接**：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
- **作者公开实现**：[KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)

## 论文简介

网络加深后，训练误差本应至少不高于浅层网络：理论上，新增加的层只要学成恒等映射，深层网络就可以复用浅层网络的解。然而实验中，普通网络随着深度增加反而出现更高的**训练误差**。论文将其称为 degradation problem（退化问题）。它不同于过拟合，也不只是传统的梯度消失或梯度爆炸问题。

ResNet 不再让若干层直接拟合目标映射 \(\mathcal{H}(x)\)，而是让它们学习相对于输入的残差：

\[
\mathcal{F}(x) := \mathcal{H}(x) - x,
\qquad
\mathcal{H}(x) = \mathcal{F}(x) + x.
\]

当理想映射接近恒等映射时，让非线性层把 \(\mathcal{F}(x)\) 推向 0，通常比让它们直接拟合恒等函数更容易。基于这一重参数化，论文成功训练了最高 152 层的 ImageNet 网络，并在 CIFAR-10 上分析了最高 1202 层的网络。

## 核心思想

### 1. 残差学习

论文式（1）定义了一个残差块：

\[
y = \mathcal{F}(x, \{W_i\}) + x.
\]

对包含两层的基础残差块，论文写为：

\[
\mathcal{F}(x) = W_2\,\sigma(W_1x),
\]

其中 \(\sigma\) 表示 ReLU。卷积网络的真实数据流还包括 Batch Normalization：

\[
\mathcal{F}(x)
= \operatorname{BN}_2\!\left(
  \operatorname{Conv}_2\!\left(
    \operatorname{ReLU}\!\left(
      \operatorname{BN}_1(\operatorname{Conv}_1(x))
    \right)
  \right)
\right).
\]

最后执行逐元素加法，再进行一次 ReLU：

\[
y = \operatorname{ReLU}(\mathcal{F}(x) + x).
\]

对应 `blocks.py` 中的核心 PyTorch 运算：

```python
identity = self.shortcut(x)

residual = self.conv1(x)
residual = self.bn1(residual)
residual = F.relu(residual, inplace=True)
residual = self.conv2(residual)
residual = self.bn2(residual)

out = residual + identity
out = F.relu(out, inplace=True)
```

这正是论文原始的 **ResNet v1 / post-activation** 结构。后来的 pre-activation ResNet 不属于本篇论文的结构。

### 2. 恒等 shortcut 与投影 shortcut

残差分支和 shortcut 分支必须具有相同形状，才能逐元素相加。当输入、输出维度不同时，论文式（2）使用线性投影：

\[
y = \mathcal{F}(x, \{W_i\}) + W_sx.
\]

论文比较了三种方案：

| 方案 | 维度不变时 | 维度增加时 | 是否引入额外参数 |
|---|---|---|---|
| A | 恒等映射 | stride 下采样 + 新通道补 0 | 否 |
| B | 恒等映射 | \(1\times1\) 卷积投影，必要时 stride=2 | 是，仅维度变化处 |
| C | \(1\times1\) 卷积投影 | \(1\times1\) 卷积投影 | 是，每个残差块都有 |

论文认为恒等 shortcut 是解决退化问题的关键，投影并非必需。ResNet-50/101/152 使用方案 B，以兼顾精度、参数量和计算量。

### 3. 为什么 shortcut 有效

- 恒等 shortcut 不增加参数，逐元素加法的计算成本也几乎可以忽略。
- 残差分支只需学习输入的修正量，而不是从零学习完整目标映射。
- 信息和梯度可以沿 shortcut 直接传播，极深网络因而更容易优化。
- 论文实验表明，更深的普通网络具有更高训练误差，而更深的 ResNet 能获得更低训练误差和验证误差。

## 核心模块

### BasicBlock

用于 ResNet-18 和 ResNet-34：

```text
x -> 3x3 Conv -> BN -> ReLU -> 3x3 Conv -> BN --+
|                                                    + -> ReLU -> y
+---------------- identity / projection ------------+
```

当阶段发生切换时，第一个 \(3\times3\) 卷积使用 stride=2，使空间尺寸减半；输出通道数同时翻倍。

### Bottleneck

用于 ResNet-50、ResNet-101 和 ResNet-152：

```text
x -> 1x1 Conv -> BN -> ReLU
  -> 3x3 Conv -> BN -> ReLU
  -> 1x1 Conv -> BN -------------------------------+
|                                                   + -> ReLU -> y
+--------------- identity / projection ------------+
```

第一个 \(1\times1\) 卷积压缩通道，中间 \(3\times3\) 卷积处理空间信息，最后一个 \(1\times1\) 卷积将通道恢复为基础宽度的 4 倍。例如，Figure 5 的残差分支为 \(256\rightarrow64\rightarrow64\rightarrow256\)。Bottleneck 的主要目的不是改变残差学习原理，而是降低深层网络的计算成本。

原作者公开的 Caffe 模型在切换 stage 时，把 stride=2 放在 Bottleneck 的第一个 \(1\times1\) 卷积上；本实现保持这一原始位置。`torchvision` 中常见的 ResNet-v1.5 则把 stride=2 移到中间的 \(3\times3\) 卷积上，两者输出 shape 相同，但具体计算与精度可能略有差异。

### OptionAShortcut

`OptionAShortcut` 对空间维度执行 stride 采样，并用 0 补齐新增通道。它完整展示了论文方案 A 的无参数维度匹配过程，没有用可学习卷积代替。

### ImageNetResNet

负责组合 stem、四个 residual stages、全局平均池化和分类器。网络深度不统计 BN、ReLU 和池化层，只统计卷积层与最后的全连接层。

### CifarResNet

实现论文第 4.2 节的 CIFAR 结构。网络以一个 \(3\times3\) 卷积开始，在 \(32\times32\)、\(16\times16\)、\(8\times8\) 三个尺度上各使用 \(n\) 个 BasicBlock，因此总深度为：

\[
1 + 3\times(2n) + 1 = 6n + 2.
\]

## ImageNet 模型整体结构

下表对应论文 Table 1。方括号内是一个 block 的卷积配置，乘数表示 block 重复次数。

| Stage | 输出尺寸 | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | ResNet-152 |
|---|---:|---:|---:|---:|---:|---:|
| conv1 | 112×112 | 7×7, 64, s=2 | 同左 | 同左 | 同左 | 同左 |
| max pool | 56×56 | 3×3, s=2 | 同左 | 同左 | 同左 | 同左 |
| conv2_x | 56×56 | [3×3, 64; 3×3, 64] ×2 | ×3 | [1×1, 64; 3×3, 64; 1×1, 256] ×3 | ×3 | ×3 |
| conv3_x | 28×28 | [3×3, 128; 3×3, 128] ×2 | ×4 | [1×1, 128; 3×3, 128; 1×1, 512] ×4 | ×4 | ×8 |
| conv4_x | 14×14 | [3×3, 256; 3×3, 256] ×2 | ×6 | [1×1, 256; 3×3, 256; 1×1, 1024] ×6 | ×23 | ×36 |
| conv5_x | 7×7 | [3×3, 512; 3×3, 512] ×2 | ×3 | [1×1, 512; 3×3, 512; 1×1, 2048] ×3 | ×3 | ×3 |
| head | 1×1 | global average pool, 1000-d FC | 同左 | 同左 | 同左 | 同左 |
| FLOPs（论文） | - | 1.8×10⁹ | 3.6×10⁹ | 3.8×10⁹ | 7.6×10⁹ | 11.3×10⁹ |

每个新 stage 的第一个 block 使用 stride=2 下采样。论文 Table 1 将其记为 `conv3_1`、`conv4_1` 和 `conv5_1`。

## Forward 数据流与 Tensor Shape

以 `resnet50()` 和输入 `x: [B, 3, 224, 224]` 为例：

| 步骤 | 主要运算 | Tensor Shape |
|---|---|---|
| 输入 | ImageNet 图像 batch | `[B, 3, 224, 224]` |
| stem conv | 7×7 Conv, 64, stride=2 | `[B, 64, 112, 112]` |
| stem pool | 3×3 MaxPool, stride=2 | `[B, 64, 56, 56]` |
| conv2_x | 3 个 Bottleneck | `[B, 256, 56, 56]` |
| conv3_x | 4 个 Bottleneck，首块 stride=2 | `[B, 512, 28, 28]` |
| conv4_x | 6 个 Bottleneck，首块 stride=2 | `[B, 1024, 14, 14]` |
| conv5_x | 3 个 Bottleneck，首块 stride=2 | `[B, 2048, 7, 7]` |
| avg pool | 全局平均池化 | `[B, 2048, 1, 1]` |
| flatten | `torch.flatten(x, 1)` | `[B, 2048]` |
| fc | 1000 类线性分类器 | `[B, 1000]` |

BasicBlock 的 `expansion=1`，因此 ResNet-18/34 四个 stage 的通道数为 64、128、256、512。Bottleneck 的 `expansion=4`，因此 ResNet-50/101/152 的输出通道数为 256、512、1024、2048。

## 公式与 PyTorch 运算对应

| 论文概念/公式 | PyTorch 参考实现 |
|---|---|
| \(\mathcal{F}(x,\{W_i\})\) | 连续的 `Conv2d -> BatchNorm2d -> ReLU` 运算 |
| 恒等映射 \(x\) | `nn.Identity()` |
| 投影 \(W_sx\) | `Conv2d(kernel_size=1, stride=...)` + `BatchNorm2d` |
| \(\mathcal{F}(x)+x\) | `residual + identity` |
| 加法后的 \(\sigma(y)\) | `F.relu(out, inplace=True)` |
| 全局平均池化 | `nn.AdaptiveAvgPool2d((1, 1))` |
| 向量化 | `torch.flatten(x, start_dim=1)` |
| 分类映射 | `nn.Linear(features, num_classes)` |
| 分类概率 | `logits.softmax(dim=1)`（需要概率时） |

## 与经典普通网络的区别

| 普通深层网络 | ResNet |
|---|---|
| 直接学习 \(\mathcal{H}(x)\) | 学习残差 \(\mathcal{F}(x)=\mathcal{H}(x)-x\) |
| 层之间只有顺序连接 | 每隔两层或三层增加 shortcut |
| 加深后可能出现更高训练误差 | 极深网络仍较容易优化 |
| 必须由堆叠层自行逼近恒等函数 | 将恒等映射显式加入模型 |
| VGG 风格网络大量使用 3×3 卷积但没有残差加法 | 保留规则的卷积 stage，并在每个 block 执行逐元素残差加法 |

论文中的 ResNet-34 与 plain-34 具有相同的卷积深度、宽度和近似计算量；主要差异就是 shortcut 与逐元素加法，因此实验能更直接地说明残差重参数化的作用。

## 论文明确内容与合理补全

### 论文明确描述

- 原始 post-activation 残差块：卷积后接 BN，残差相加后再接 ReLU。
- BasicBlock 使用两个 \(3\times3\) 卷积。
- Bottleneck 使用 \(1\times1\)、\(3\times3\)、\(1\times1\) 卷积。
- 作者公开实现把 Bottleneck 的 stage-transition stride 放在第一个 \(1\times1\) 卷积，而非后来 ResNet-v1.5 的 \(3\times3\) 卷积。
- 特征图尺寸减半时，通道数翻倍；下采样由 stride=2 卷积完成。
- ImageNet 各深度的 block 数量及 A/B/C 三种 shortcut 方案。
- CIFAR 模型采用 16、32、64 个通道，总深度为 \(6n+2\)，全部使用方案 A。
- 使用 rectifier-aware 初始化、BN，不使用 dropout。

### 参考实现中的合理补全/工程惯例

- `AdaptiveAvgPool2d((1, 1))`：对 224×224 输入等价于论文的 7×7 全局平均池化，同时允许参考代码处理其他输入尺寸。
- `MaxPool2d(..., padding=1)`：论文给出 3×3、stride=2 及目标 shape；padding=1 是得到 Table 1 尺寸的主流实现方式。
- 投影 shortcut 后的 BN：遵循“每个卷积后接 BN”的论文描述和主流 ResNet v1 实现。
- forward 返回 logits 而不是显式 softmax：这是 PyTorch 与 `nn.CrossEntropyLoss` 配合时的标准做法。论文分类概率所需的 softmax 可在推理时通过 `logits.softmax(dim=1)` 计算。
- 全连接层初始化为 `normal_(std=0.01)`：论文没有明确给出分类头的具体初始化参数；此处仅作为常见实现补全，不影响残差结构。

本实现没有加入现代实现中常见的最后一个 BN 零初始化、ResNet-D stem、anti-aliasing、SE 模块或 stochastic depth，因为这些都不是原论文结构。

## 使用示例

```python
import torch

from model import cifar_resnet, resnet34, resnet50


# ImageNet ResNet-50：论文用于 50/101/152 层网络的 option B。
model = resnet50(num_classes=1000, shortcut_option="B")
x = torch.randn(2, 3, 224, 224)
logits = model(x)                 # [2, 1000]
probabilities = logits.softmax(1)

# 论文 ResNet-34 的无参数 shortcut 对照实验（option A）。
model34_a = resnet34(shortcut_option="A")

# CIFAR-10 ResNet-110：110 = 6 * 18 + 2，论文默认 option A。
cifar_model = cifar_resnet(depth=110, num_classes=10)
cifar_x = torch.randn(4, 3, 32, 32)
cifar_logits = cifar_model(cifar_x)  # [4, 10]
```

## 阅读心得

ResNet 的贡献并不只是“多加一条支路”，而是改变了深层网络要优化的函数形式。它没有依赖复杂门控，也几乎不增加参数，却让网络能够从 18/34 层稳定扩展到 50/101/152 层。论文最有说服力的地方，是把 plain network 与 residual network 的深度、宽度和计算量控制得基本一致，从而把性能变化集中到 residual learning 本身。

## 文件结构

```text
ResNet/
├── README.md   # 论文解读、公式、结构、数据流和实现边界
├── blocks.py   # BasicBlock、Bottleneck 及 shortcut A 的具体计算
└── model.py    # shortcut A/B/C、ImageNet 与 CIFAR 完整模型组合
```

代码是面向论文阅读的参考实现：保留真实 PyTorch API 和核心计算过程，但有意省略数据集、DataLoader、训练循环、优化器、日志、checkpoint、AMP 和分布式训练等工程代码。
