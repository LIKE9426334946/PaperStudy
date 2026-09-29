# U-Net：论文学习笔记与 PyTorch 参考实现

## 1. 论文信息

| 项目 | 内容 |
| --- | --- |
| 标题 | U-Net: Convolutional Networks for Biomedical Image Segmentation |
| 作者 | Olaf Ronneberger、Philipp Fischer、Thomas Brox |
| 会议 | Medical Image Computing and Computer-Assisted Intervention（MICCAI） |
| 出版信息 | Springer，LNCS 9351，234–241 页 |
| 年份 | 2015 |
| 论文 | [arXiv:1505.04597](https://arxiv.org/abs/1505.04597) |
| 出版链接 | [DOI: 10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28) |
| 作者主页与原始发布 | [University of Freiburg U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) |

本目录依据提供的 `UNet.pdf`，即 arXiv v1（2015-05-18），以论文文字、公式和图 1 为优先依据。作者最初发布的是基于 Caffe 的实现；这里是便于逐行阅读的 PyTorch 参考实现，不是官方权重移植，也没有声称复现论文实验指标。

## 2. 论文简介与核心思想

图像分类通常为一张图输出一个类别，语义分割则需要为每个像素输出类别。医学图像中往往只有少量标注，目标边界又需要精确定位。U-Net 将提取上下文的收缩路径与恢复空间分辨率的扩张路径组合起来，并在对应尺度之间传递高分辨率特征。

核心思想包括：

1. **收缩路径提取上下文**：两次卷积后进行池化，逐层减小空间尺寸、增加特征通道。
2. **扩张路径恢复定位能力**：可学习上采样后，拼接同尺度编码特征，再用卷积融合。
3. **较宽的解码路径**：上采样过程中保留较多特征通道，让语义信息参与精细定位。
4. **少样本训练策略**：利用弹性形变等增强，让有限标注提供更多有效训练变化。
5. **相邻细胞分离**：为细胞之间的狭窄背景分隔区赋予较大的损失权重。
6. **大图分块预测**：每个输入块只输出拥有完整上下文的中心区域，用 overlap-tile 覆盖原图，图像外部采用镜像补充。

U-Net 的输出仍然是语义类别，不会直接输出不同细胞的实例 ID。训练权重图需要实例边界信息，与网络输出语义标签是两个不同层面。

## 3. 文件结构与阅读顺序

| 文件 | 内容 |
| --- | --- |
| `README.md` | 论文信息、结构、公式、Tensor Shape、使用示例与实现边界 |
| `blocks.py` | 两次 valid convolution、中心裁剪、上采样与 skip 拼接 |
| `model.py` | 完整 U-Net、逐级 forward 和权重初始化 |
| `losses.py` | 不同细胞的最近边界距离选择、权重图和加权交叉熵 |
| `augmentation.py` | 3×3 随机位移网格、bicubic 位移场、图像与标签同步变形 |
| `inference.py` | 镜像取块与 overlap-tile 输出拼接 |
| `__init__.py` | 支持从仓库根目录导入模型和基础函数 |

建议先阅读 `blocks.py`，再跟踪 `model.py` 的 forward，最后阅读损失、增强和分块推理。所有计算使用真实 PyTorch API，没有依赖第三方分割模型库。数据集、DataLoader、训练循环、优化器封装及日志等工程代码按学习用途省略。

## 4. 模型整体结构与每个模块的作用

模型有四次下采样、一个最深层卷积块、四次上采样和一个逐像素分类头。默认输入通道为 1，输出类别数为 2，与图 1 一致。

| 模块 | 计算 | 作用 |
| --- | --- | --- |
| `DoubleConv` | `Conv2d(3×3, padding=0)` → ReLU → 同样的第二次卷积 → ReLU | 学习特征；高、宽各减少 4 |
| `pool` | `MaxPool2d(2, stride=2)` | 降低空间分辨率，不改变通道数 |
| `bottleneck` | 512 → 1024 → 1024 的两个卷积 | 在最低分辨率提取上下文 |
| `UpBlock.up` | `ConvTranspose2d(2, stride=2)` | 空间扩大 2 倍，通道减半 |
| `center_crop` | 对 encoder 特征的最后两个维度切片 | 与对应上采样特征对齐 |
| `torch.cat` | 沿 `dim=1` 拼接 | 保留两路特征，供后续卷积学习融合 |
| `UpBlock.convs` | 拼接后的两个 valid convolution | 融合细节和上下文 |
| `head` | `Conv2d(64, K, kernel_size=1)` | 将每个位置的 64 维特征映射为 K 个类别 logits |

共有 9 个 `DoubleConv`，即 18 个 3×3 卷积，再加上 4 个 up-convolution 和 1 个 1×1 分类卷积，合计 **23 个卷积层**。池化、裁剪、Dropout 和 ReLU 不计入这个数量。

本实现没有 BatchNorm、残差相加、Attention、全连接层或预训练编码器。构造模型时全部权重随机初始化。

## 5. Forward 数据流与 Tensor Shape

以下统一采用 `[B, C, H, W]` 格式。表中省略 batch 维，输入为 `[B, 1, 572, 572]`。

### 5.1 收缩路径

| 阶段 | 输入 C×H×W | 第一个卷积 + ReLU | 第二个卷积 + ReLU | 池化结果 |
| --- | --- | --- | --- | --- |
| `enc1` | 1×572×572 | 64×570×570 | 64×568×568 | 64×284×284 |
| `enc2` | 64×284×284 | 128×282×282 | 128×280×280 | 128×140×140 |
| `enc3` | 128×140×140 | 256×138×138 | 256×136×136 | 256×68×68 |
| `enc4` | 256×68×68 | 512×66×66 | 512×64×64 | 512×32×32 |
| `bottleneck` | 512×32×32 | 1024×30×30 | 1024×28×28 | 无 |

`e1` 到 `e4` 保存各级卷积结果，供对应解码级使用。`enc4` 和 `bottleneck` 后的 Dropout 不改变尺寸，其位置与概率属于第 9 节标明的补全。

### 5.2 扩张路径

| 阶段 | up-conv 之后 | skip 原尺寸 → 裁剪尺寸 | 拼接结果 | 两次卷积后的空间尺寸 | 最终输出 |
| --- | --- | --- | --- | --- | --- |
| `dec4` | 512×56×56 | 512×64×64 → 512×56×56 | 1024×56×56 | 56 → 54 → 52 | 512×52×52 |
| `dec3` | 256×104×104 | 256×136×136 → 256×104×104 | 512×104×104 | 104 → 102 → 100 | 256×100×100 |
| `dec2` | 128×200×200 | 128×280×280 → 128×200×200 | 256×200×200 | 200 → 198 → 196 | 128×196×196 |
| `dec1` | 64×392×392 | 64×568×568 → 64×392×392 | 128×392×392 | 392 → 390 → 388 | 64×388×388 |
| `head` | 无 | 无 | 无 | 388 → 388 | K×388×388 |

四级 skip 分别在上、下、左、右各裁掉 4、16、40、88 个像素。这里裁剪的是对应层的**特征图**；最终监督标签则从输入大小裁到输出大小，每侧裁掉 92 像素。

### 5.3 为什么不是输入、输出同尺寸？

无填充的 3×3、stride=1 卷积满足：

$$
H_{out}=H_{in}-2,\qquad W_{out}=W_{in}-2.
$$

两次卷积使每个空间维减少 4。上采样会放大空间尺寸，但随后的 valid convolution 仍会削去边缘。因此论文图 1 的结果是：

$$
[B,1,572,572]\longrightarrow[B,2,388,388].
$$

### 5.4 输入尺寸为什么有限制？

论文第 2 节要求每次 2×2 池化之前的高、宽均为偶数。对任意一个空间维 n，四次“减 4 再除 2”后得到：

$$
p_4=\frac{n-60}{16}.
$$

故合规尺寸必须满足 `n % 16 == 12`。还要保证解码后的卷积有效，最小为 188。令 `n = 188 + 16m`，其中 m 为非负整数，则最终输出尺寸为：

$$
n_{out}=n-184=4+16m.
$$

572、588、604 都可使用；高、宽也可以不同。模型主动检查这个条件。某些不合规输入可能被默认池化向下取整后继续计算，但本实现不采用这种偏离论文尺寸约束的行为。原始数据为 256×256 或 512×512 时，不能直接套用这里的同尺寸训练习惯；可准备符合要求的输入块，整图预测则使用第 8 节的分块函数。

## 6. 公式与 PyTorch 运算对应

### 6.1 卷积、激活和 skip 拼接

用索引表达一个 valid convolution：

$$
a_{o,h,w}=b_o+\sum_c\sum_{u=0}^{2}\sum_{v=0}^{2}
W_{o,c,u,v}x_{c,h+u,w+v},\qquad y=\max(0,a).
$$

分别对应 `nn.Conv2d(..., kernel_size=3, padding=0)` 和 `F.relu`。这段索引解释是为理解代码展开的，不是论文中的编号公式。

解码级的数据融合为：

$$
u=\operatorname{UpConv}(z),\quad
s=\operatorname{CenterCrop}(e,\operatorname{size}(u)),\quad
f=\operatorname{Concat}_{channel}(s,u).
$$

对应 `self.up(x)`、Tensor 切片和 `torch.cat([skip, x], dim=1)`。拼接会使通道数相加；它不是 ResNet 中要求维度匹配的逐元素残差相加。

### 6.2 像素级 Softmax

论文第 3 节：

$$
p_k(x)=\frac{\exp(a_k(x))}{\sum_{j=1}^{K}\exp(a_j(x))}.
$$

对于 `[B, K, H, W]` 的 logits，类别维是 `dim=1`：

```python
probabilities = torch.softmax(logits, dim=1)
prediction = logits.argmax(dim=1)  # [B, H, W]
```

Softmax 不改变最大值所在类别，故预测标签可以直接对 logits 做 `argmax`。训练函数接收 logits，在损失内部执行 `F.log_softmax`，不要先传入已经 Softmax 的概率。

### 6.3 式（1）：加权交叉熵与负号说明

提供的 PDF 中式（1）写成：

$$
E=\sum_{x\in\Omega}w(x)\log p_{\ell(x)}(x).
$$

这个表达式**没有负号**。若直接最小化它，会降低真实类别概率，与交叉熵训练目标相反。因此本实现明确采用待最小化的负对数似然：

$$
L=-\sum_{b,x}w_b(x)\log p_{b,\ell_b(x)}(x).
$$

`losses.py` 中逐步完成：

1. `F.log_softmax(logits, dim=1)`：稳定计算所有类别的 log 概率。
2. `gather(dim=1, index=target.unsqueeze(1))`：取每个像素真实类别的 log 概率。
3. 加负号：形成非负的单像素交叉熵。
4. 乘 `weight_map`：增加重要像素的贡献。
5. `sum()`：默认按论文的求和形式聚合；batch 维也一并求和。

`reduction="mean"` 是可选工程方式，除数为 `B*H*W`；不等同于除以 `weight_map.sum()`。`reduction="none"` 返回像素损失图。`weight_map=None` 表示所有像素权重为 1。

标签必须是 `torch.long`，范围 `0..K-1`，空间尺寸须与输出一致。这里只使用有效类别编号；部分未标注像素可填任意合法类别并把相应像素权重设为 0，不可直接将 255 传入 `gather`。本函数没有隐含的 ignore_index 处理。

### 6.4 式（2）：细胞分隔边界权重

$$
w(x)=w_c(x)+w_0\exp\left(-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2}\right).
$$

| 符号 | 含义 | 代码 |
| --- | --- | --- |
| `w_c(x)` | 补偿训练集中类别像素频率差异的权重图 | `class_weight_map` |
| `d_1(x)` | 到最近细胞边界的距离 | `nearest_two[:, 0]` |
| `d_2(x)` | 到第二近、且属于另一个细胞的边界距离 | `nearest_two[:, 1]` |
| `w_0` | 分隔项的强度，论文取 10 | `w0=10.0` |
| `sigma` | 距离衰减尺度，论文约为 5 像素 | `sigma=5.0` |

处于两个细胞之间的像素同时接近两侧边界，`d1+d2` 较小，得到较大权重。例如 `w_c=1, d1=1, d2=2` 时，权重为 `1 + 10*exp(-9/50)`，约 9.35。

`instance_distances` 的形状是 `[B, N, H, W]`：先为每个细胞分别得到一张“到该细胞边界的距离图”，再用 `topk(k=2, largest=False, dim=1)` 沿实例维选最近两个距离。不能把所有细胞边界合并后，仅找最近的两个边界像素，因为这两个点可能属于同一个细胞。

预计算距离图需要保留实例 ID 的标注；单纯的水体/背景语义二值图无法自动提供论文所需的细胞实例关系。论文提到用形态学操作构造相邻细胞间的背景分隔，但没有交代具体结构元素；也没有完整指定类别平衡权重的计算与归一化。本实现把这部分数据准备结果作为输入，展开权重公式，不将某一种预处理算法冒充为论文原文。可以自行选用训练集逆频率权重，再通过 `class_weights[target]` 得到 `w_c`，但那属于补全。

这里严格照式（2）叠加指数项，没有额外乘“仅背景”的掩码；少于两个实例时返回 `w_c`，batch 中不存在的实例可以用正无穷距离填充。这些边界情况处理不是论文明确规定。

### 6.5 权重初始化

论文第 3 节说明，对交替卷积与 ReLU 的网络，可使用：

$$
W\sim\mathcal N\left(0,\frac{2}{N}\right),\qquad
\operatorname{std}(W)=\sqrt{\frac{2}{N}}.
$$

这里 N 是一个输出神经元的输入连接数。普通 3×3 卷积中 `N=9*C_in`；例如输入通道为 64 时，`N=576`。对应 `nn.init.normal_(weight, mean=0, std=sqrt(2/N))`。

代码中的 2×2、stride=2 转置卷积没有空间重叠，每个输出位置接收一个输入位置的所有输入通道，所以本实现取 `N=C_in`。这是对论文原则的具体映射；论文没有给出该层的初始化代码。转置卷积权重按 `[C_in, C_out, 2, 2]` 存储，不能误把第二维当成输入通道数。分类头也沿用该高斯规则，所有偏置置零；这两点同样作为补全标注。

## 7. 弹性形变与训练策略

论文第 3 节及第 3.1 节明确写到：

- 使用随机梯度下降，batch size 为 1，momentum 为 0.99；倾向于使用较大的输入块。
- 加强平移、旋转、形变和灰度变化的数据增强。
- 弹性形变从 **3×3 粗网格**上的随机位移生成，位移服从标准差为 **10 像素**的高斯分布。
- 使用 **bicubic 插值**生成逐像素平滑位移。
- 在收缩路径末端使用 Dropout。

`augmentation.py` 对弹性形变展开了完整的核心计算：

1. `torch.randn(B, 2, 3, 3) * 10`，两个通道分别存 dx 和 dy。
2. `F.interpolate(..., mode="bicubic")` 得到 `[B, 2, H, W]` 的位移场。
3. 通过 `meshgrid` 建立基准坐标，再将像素位移归一化为 `grid_sample` 的坐标偏移。
4. 将位移加到基准网格，建立输出位置到输入采样位置的映射。
5. 原图使用 bilinear，标签使用 nearest，同一网格同步采样。

这里 **bicubic 用于位移场**，不代表语义标签也应采用 bicubic。标签的最近邻采样可避免插值产生无意义的类别编号。

对完整输入图及其同尺寸标签先做同步增强，再裁出输出监督区域。若使用细胞距离权重，应在同步变形的实例标注上重新生成距离图和权重图，保持几何关系一致。论文的其他增强、训练循环及旋转预测集成没有扩展成工程流水线。

## 8. Overlap-tile 推理

论文图 2 的思想是：预测原图的一块区域时，额外读取周围上下文。默认每次输入 572×572，得到中心 388×388 的结果，相当于四周各多读取 92 像素。

`overlap_tile_predict` 按输出块尺寸铺排原图，从镜像延拓的输入中抽取对应 tile，再把 logits 写回原图坐标。相邻输入块重叠；输出块直接相邻。最右边、最下边的块只保留属于原图范围的部分，所以最终输出为 `[B, K, H_original, W_original]`。

函数支持原图尺寸不符合网络直接输入的约束，也支持小于 tile 的图像。镜像采用不重复边缘像素、必要时反复反射的索引方式；这是明确选定的边界约定。调用者须先执行 `model.eval()` 关闭 Dropout。

这里实现的是上下文取块和完整覆盖，不声称不同 tile 设置与一次整图 forward 逐位相等。池化的采样起点和上下文范围可能影响接缝附近的数值；本函数没有额外进行融合、概率平均或去缝优化。网络激活按块计算，但示例仍在模型所在设备保存原图与整幅输出，没有实现超大图的磁盘流式 I/O。

## 9. 论文明确内容与实现选择

| 项目 | 论文依据 | 本实现 |
| --- | --- | --- |
| 3×3 valid convolution + ReLU | 第 2 节、图 1 明确给出 | `padding=0`，每级两次 |
| 4 次池化及通道配置 | 图 1 明确给出 | 64、128、256、512、1024 |
| 2×2 up-convolution | 图 1、第 2 节给出操作及尺寸 | 用 `ConvTranspose2d(kernel_size=2, stride=2)` 映射 |
| copy and crop + concat | 第 2 节明确给出 | 用中心切片实现 crop，沿通道拼接 |
| up-conv 后激活 | 图 1 单独标 up-conv，不标 ReLU | 该位置不额外插入 ReLU |
| Dropout | 第 3.1 节只说明位于收缩路径末端 | 选择 enc4、bottleneck 输出后逐元素 Dropout，默认 p=0.5；位置、类型与概率均是补全 |
| 损失负号 | 上传版本式（1）未写负号 | 为最小化交叉熵明确加负号 |
| `w_c` 与实例距离预处理 | 给出含义，没有完整算法 | 由调用者预计算，本目录实现距离选择及权重公式 |
| 高斯初始化 | 给出 std=sqrt(2/N) | 普通卷积直接按输入连接数；up-conv、head、偏置规则明确说明 |
| 输入与输出通道 | 图 1 示例为 1 → 2；正文允许所需类别数 | 默认 1 → 2，允许配置 RGB 输入及多类别 |
| 弹性形变核心 | 3×3、std=10、bicubic 明确给出 | 按这些参数生成位移场；采样方向、边界和插值设置是补全 |
| 镜像 overlap-tile | 图 2、第 1 节明确描述 | 简洁的镜像索引、分块取样和有效输出拼接 |

`dropout_p=0.0` 可关闭本实现补全的 Dropout，方便只研究图 1 的计算图，但这会省去正文提到的正则化。上述选择都没有被称为官方 Caffe 的精确逐项配置。

## 10. 与经典方法及常见 U 形网络的区别

| 方法/实现 | 主要特点 | 与本目录原版 U-Net 的区别 |
| --- | --- | --- |
| 滑窗 CNN 像素分类 | 每个局部 patch 为中心像素预测类别 | U-Net 一次预测整个有效区域，减少逐 patch 重复计算，并融合不同尺度信息 |
| 早期 FCN 思路 | 将分类网络扩展为全卷积预测，利用上采样恢复分辨率 | U-Net 在此思路上构建较对称、较宽的扩张路径，并逐级融合高分辨率编码特征 |
| ResNet 残差块 | 同形状特征逐元素相加 | U-Net 的跨路径连接使用裁剪后通道拼接，通道数会增加 |
| 常见 same-padding U 形实现 | 常用 `padding=1` 保持 3×3 卷积的空间尺寸 | 本实现忠实保留 valid convolution，因此需要 crop，输出小于输入 |
| 使用预训练 backbone 的分割模型 | 编码器可能替换为预训练分类网络 | 本实现使用论文原生卷积编码器，并从随机权重开始 |
| 单通道二分类输出 | 1 个 logit，常搭配 Sigmoid 和 BCEWithLogitsLoss | 图 1 及默认实现使用 2 个 logits，配合 Softmax 交叉熵 |

“U-Net”这个名称不意味着所有库都使用相同的 padding、通道数、编码器或训练策略。进行自己的水体分割实验时，应把这些设定分别对齐后再比较结果。

## 11. 最小使用与逐步调试

在 `PaperStudy` 仓库根目录启动 Python；只需要已安装 PyTorch 的环境。

### 11.1 对照图 1 查看输出

```python
import torch
from UNet import UNet

model = UNet(in_channels=1, num_classes=2).eval()
x = torch.randn(1, 1, 572, 572)

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    prediction = logits.argmax(dim=1)

print(logits.shape)         # torch.Size([1, 2, 388, 388])
print(probabilities.shape)  # torch.Size([1, 2, 388, 388])
print(prediction.shape)     # torch.Size([1, 388, 388])
```

可以在 `forward` 中给 `e1`、`p1`、`e2` 等语句设置断点，查看中间 Tensor。若直接进入 `UNet/` 目录调试，也可以使用 `from model import UNet`。

RGB 水体二分类使用 `UNet(in_channels=3, num_classes=2)`；四类别灰度分割使用 `UNet(in_channels=1, num_classes=4)`。这些只是输入、输出通道的适配，空间裁剪规则保持不变。当前加权 Softmax 损失需要至少两个类别；不要设置 `num_classes=1` 后继续使用它，单类别 Softmax 恒为 1，损失也会退化。

### 11.2 用小 Tensor 验证标签裁剪与反向传播

188 是合规的最小尺寸，只用于快速检查运算，不是推荐训练分辨率。

```python
import torch
from UNet import UNet, center_crop, weighted_cross_entropy

model = UNet(dropout_p=0.0)
x = torch.randn(1, 1, 188, 188)
full_target = torch.randint(0, 2, (1, 188, 188), dtype=torch.long)
full_weight = torch.ones(1, 188, 188)

logits = model(x)  # [1, 2, 4, 4]
target = center_crop(full_target, logits.shape[-2:])  # [1, 4, 4]
weight = center_crop(full_weight, logits.shape[-2:])  # [1, 4, 4]
loss = weighted_cross_entropy(logits, target, weight, reduction="mean")
loss.backward()
```

这里的随机标签和全 1 权重只用于检查计算，不表示真实细胞训练数据。使用论文的权重图时，在裁剪之前根据标注准备 `full_weight`。

### 11.3 单独检查权重公式

```python
import torch
from UNet import boundary_weight_map

wc = torch.ones(1, 1, 2)  # 两个像素的类别平衡权重
distances = torch.tensor([[[[1.0, 2.0]],
                           [[2.0, 4.0]],
                           [[9.0, 8.0]]]])  # [B=1, N=3, H=1, W=2]
weights = boundary_weight_map(wc, distances)
print(weights)  # 约 tensor([[[9.3527, 5.8675]]])
```

### 11.4 同步弹性形变与整图分块预测

```python
import torch
from UNet import UNet
from UNet.augmentation import elastic_deform_pair
from UNet.inference import overlap_tile_predict

image = torch.randn(1, 1, 572, 572)
mask = torch.randint(0, 2, (1, 572, 572))
augmented_image, augmented_mask = elastic_deform_pair(image, mask)

model = UNet().eval()
original_image = torch.randn(1, 1, 512, 512)
full_logits = overlap_tile_predict(model, original_image)
print(full_logits.shape)  # torch.Size([1, 2, 512, 512])
```

## 12. 阅读心得

U-Net 的关键是让上下文提取和像素定位互相配合。池化有助于扩大上下文，但细节会丢失；对应尺度的高分辨率特征为解码器补充定位线索。解码器的卷积学习如何组合这些信息，skip connection 本身不会自动完成分割。

另一点是，网络结构只是论文的一部分。少量标注下的数据增强、相邻细胞分隔权重、有效输出区域和分块推理共同构成了方法。用于遥感水体时，可以学习这种编码解码与多尺度融合思路，但细胞实例边界权重是否合适，需要结合自己的标注与任务验证。

本目录优先服务于结构和公式学习；随机输入上的计算验证不能证明复现了论文在真实数据集上的精度。

## 13. 验证记录

本次在 Python 3.12、PyTorch 2.5.1+cpu 环境进行了真实数值验证：

- 1×1×572×572 输入完成前向计算，各 encoder、bottleneck、decoder 和 head 的实际形状与图 1 一致。
- 统计得到 19 个普通卷积、4 个转置卷积；默认模型有 31,030,658 个可学习参数（含偏置）。
- 最小合规输入 188×188 输出 4×4；损失可反向传播，编码器、上采样层和输出头的梯度均为有限值。
- batch=2 的 RGB 矩形输入 188×204、4 类输出得到 `[2, 4, 4, 20]`。
- 加权交叉熵的像素值、sum/mean 聚合及梯度与 PyTorch 内置交叉熵对照一致；极端 logits 下仍保持数值稳定。
- 边界权重符合手工指数计算，并覆盖实例顺序打乱、少于两个实例及无效实例使用正无穷距离的情况。
- 弹性形变通过零位移恒等检查，以及坐标编码图像和标签的同步几何检查。
- overlap-tile 通过已知输出的覆盖检查，包含非整块边缘、小图和重复镜像，并完成真实 U-Net 的分块前向验证。

以上是结构与运算验证，没有执行真实数据集训练，也没有复现论文的分割指标。

## 14. 参考资料

- 提供的 `UNet.pdf`：图 1 与第 2 节对应模型；第 3 节对应损失与初始化；第 3.1 节对应增强；图 2 对应分块推理。
- [作者官方论文与发布页面](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)：核对作者、会议及原始 Caffe 发布信息。
- [PyTorch ConvTranspose2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)：核对上采样尺寸及参数布局。
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：核对 logits、类别索引和交叉熵定义。
- [PyTorch grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)：核对采样坐标顺序及插值设置。
