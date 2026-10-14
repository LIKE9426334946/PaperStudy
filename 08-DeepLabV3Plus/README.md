# DeepLabv3+ 论文学习与 PyTorch 参考实现

## 1. 论文信息

| 项目 | 内容 |
| --- | --- |
| 标题 | Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation |
| 作者 | Liang-Chieh Chen、Yukun Zhu、George Papandreou、Florian Schroff、Hartwig Adam |
| 会议 / 年份 | ECCV 2018 |
| 论文 | <https://arxiv.org/abs/1802.02611> |
| 官方 TensorFlow 实现 | <https://github.com/tensorflow/models/tree/master/research/deeplab> |
| 阅读版本 | 上传论文，arXiv:1802.02611v3，2018-08-22 |

本目录是用于理解论文的参考实现：展开 backbone、空洞卷积、深度可分离卷积、ASPP、浅层特征融合和 decoder。默认使用 Fig.4 的完整 Xception-65，支持 encoder output stride（OS）为 16 或 8；另提供 Xception-71 和 ResNet-101 对照。未加载预训练权重、未训练数据集，不宣称复现论文精度。

## 2. 论文简介与核心思想

DeepLabv3 的 ASPP 能提取多尺度上下文，但直接把低分辨率预测放大，难以充分恢复物体边界。DeepLabv3+ 在它后面加入一个简单 decoder：先融合较高分辨率的浅层特征，再做卷积细化，最后恢复原图大小。

三个重点：

1. **空洞卷积控制分辨率和采样范围。** 取消某些下采样，同时增加后续卷积的 dilation，使高层特征更密集。
2. **ASPP 与 decoder 分工。** ASPP 聚合多尺度语义；decoder 融合浅层空间细节，改善边界。
3. **深度可分离卷积降低开销。** 在改进 Xception、ASPP 和 decoder 中，将空间卷积与通道混合分开。

论文 §4.1 通过消融选择：浅层特征用 `1×1` 卷积压缩到 **48 通道**，拼接后采用 **两层、每层256通道的3×3卷积**。不是把所有 encoder 层都接到 decoder。

## 3. 文件结构与阅读顺序

| 文件 | 内容 |
| --- | --- |
| `model.py` | 完整 `DeepLabV3Plus`；`__main__` 中的前向演示、ONNX 导出和数值比较 |
| `blocks.py` | 普通卷积、空洞可分离卷积、Xception 块、ResNet Bottleneck、ASPP、decoder |
| `backbones.py` | 完整 Xception-65 / Xception-71、空洞 ResNet-101 |
| `README.md` | 论文笔记、公式映射、形状分析、实现差异和运行说明 |
| `.gitignore` | 忽略 Python 缓存和运行时生成的 ONNX 文件 |

建议先看 `model.py::forward`，再看 `ASPP` 和 `DeepLabDecoder`，最后展开 backbone。目录互不依赖，没有引用仓库其他论文目录中的代码。

## 4. 整体结构、模块作用与 Forward 数据流

| 步骤 | 模块 | 计算与作用 |
| --- | --- | --- |
| 1 | `backbone` | 编码图像，同时保留 OS=4 的浅层特征和 OS=16/8 的高层特征 |
| 2 | `aspp.branch1` | `1×1 Conv`，在当前位置混合通道 |
| 3 | `aspp.branch2/3/4` | 三个不同 dilation 的 `3×3` 分支，采集不同范围的上下文 |
| 4 | `aspp.global_pool` | 全局平均池化，再投影并插值回高层特征尺寸 |
| 5 | `aspp.project` | 拼接各分支后，用 `1×1` 将1280通道压回256 |
| 6 | `decoder.low_project` | 将浅层256通道压成48，避免其在融合中占比过大 |
| 7 | `decoder` | 高层特征插值到浅层尺寸，沿通道拼接，再经过两层3×3细化 |
| 8 | `classifier` | `1×1 Conv` 输出每个类别的 logits |
| 9 | `F.interpolate` | 将 logits 恢复到输入图像的准确尺寸 |

最终输出为 `[B,K,H,W]`，每个像素有 K 个未归一化得分。多分类预测可在模型外使用 `logits.argmax(dim=1)`；训练时直接将 logits 交给 `CrossEntropyLoss`。`forward` 不包含 Softmax、Argmax 或阈值处理。

## 5. 公式解析与真实 PyTorch 运算

### 5.1 论文公式 (1)：空洞卷积

$$
y[i] = \sum_k x[i+r\cdot k]w[k].
$$

`r` 是卷积核采样点之间的间隔；二维时 `i`、`k` 都表示两个空间坐标。`r=1` 就是普通卷积。它不增加可学习的卷积核元素，也不等于把输出下采样。

```python
nn.Conv2d(Cin, Cout, kernel_size=3, stride=1,
          padding=r, dilation=r, bias=False)
```

3×3 核的有效覆盖边长为 `3 + 2*(r-1) = 2*r+1`：ASPP 的 rate 6/12/18 分别覆盖13/25/37的空间范围，但每个输入通道仍只有9个采样位置。边界之外使用零填充；特征图很小时，大 rate 会采到较多填充值。

### 5.2 Fig.3：空洞深度可分离卷积

以下是对 Fig.3 的数学展开，不是论文新增的编号公式：

$$
z_c[i]=\sum_k x_c[i+r\cdot k]d_c[k],\qquad
y_o[i]=\sum_c p_{o,c}z_c[i].
$$

| 运算 | PyTorch 映射 | 通道关系 |
| --- | --- | --- |
| Depthwise | `Conv2d(Cin,Cin,3,groups=Cin,dilation=r,padding=r)` | 每个通道独立做空间卷积 |
| Pointwise | `Conv2d(Cin,Cout,1)` | 在每个位置混合所有输入通道 |
| 归一化与非线性 | `BatchNorm2d`、`F.relu` | 见 `AtrousSeparableConv.forward` |

忽略 bias、BN 和激活，标准卷积参数量为 `9*Cin*Cout`，可分离卷积为 `9*Cin + Cin*Cout`。例如 `Cin=Cout=256`，分别为589824和67840。**可分离卷积改变了运算的参数化方式，不是任意标准卷积的完全等价分解。**

### 5.3 ASPP 的融合

ASPP 是并行计算，不是按 rate 6、12、18 依次串行卷积。默认结构可写为：

$$
A=\operatorname{Project}_{1\times1}\bigl[
f_{1\times1}(X),\ f_{3\times3}^{6}(X),\ f_{3\times3}^{12}(X),\
f_{3\times3}^{18}(X),\ \operatorname{Resize}(f_{1\times1}(\operatorname{GAP}(X)))
\bigr].
$$

方括号表示 `torch.cat(..., dim=1)`，不是相加。`GAP` 对应 `AdaptiveAvgPool2d(1)`；恢复尺寸对应双线性 `F.interpolate`。公式省略了 BN、ReLU 与补充的 Dropout，实际代码保留这些运算。

### 5.4 Decoder 的融合

$$
L'=\operatorname{Project}_{1\times1}^{48}(L),\quad
U=\operatorname{Resize}(A,\operatorname{size}(L')),\quad
D=f_{3\times3}^{256}(f_{3\times3}^{256}([U,L'])).
$$

```python
low_level = self.low_project(low_level)
high_level = F.interpolate(high_level, size=low_level.shape[-2:],
                           mode="bilinear", align_corners=True)
x = torch.cat((high_level, low_level), dim=1)
x = self.refine1(x)
x = self.refine2(x)
```

`48+256=304` 是拼接后的通道数。低层特征与高层特征空间大小相同即可拼接，通道数不必相同。最后分类与插值为 `Resize(Conv1x1(D), input_size)`。

## 6. Backbone 与 Output Stride

### 6.1 改进的 Aligned Xception-65

Fig.4 的完整层数被保留。入口使用两个普通卷积，后续用可分离卷积替代最大池化；Middle Flow 完整重复16次。

| 阶段 | 通道序列 | 残差连接 |
| --- | --- | --- |
| Stem | 3 → 32 → 64 | 无 |
| Entry block 1 | 64 → 128 → 128 → 128 | 1×1 投影 |
| Entry block 2 | 128 → 256 → 256 → 256 | 1×1 投影 |
| Entry block 3 | 256 → 728 → 728 → 728 | 1×1 投影 |
| Middle，每块三层，共16块 | 728 → 728 → 728 → 728 | 恒等残差 |
| Exit block | 728 → 728 → 1024 → 1024 | 1×1 投影 |
| Exit tail | 1024 → 1536 → 1536 → 2048 | 无 |

默认浅层特征取 **Entry block 2 的第二个可分离卷积输出**，即该块最后一次下采样之前的 `[B,256,H/4,W/4]`。这个具体端点由官方 `feature_extractor.py` 补全，不能误取已经下采样到 OS=8 的块输出。

### 6.2 取消 stride 时如何改变 dilation

下表是 **Xception-65** 的设定，每个块内三层使用相同 rate；stride 只放在第三层和投影 shortcut。

| 阶段 | OS=16 的 stride / rate | OS=8 的 stride / rate |
| --- | --- | --- |
| Stem 第1层 | 2 / 1 | 2 / 1 |
| Entry block 1 | 2 / 1 | 2 / 1 |
| Entry block 2 | 2 / 1 | 2 / 1 |
| Entry block 3 | 2 / 1 | 1 / 1 |
| Middle flow | 1 / 1 | 1 / 2 |
| Exit block | 1 / 1 | 1 / 2 |
| Exit tail | 1 / 2 | 1 / 4 |
| ASPP 三个3×3分支 | 1 / (6,12,18) | 1 / (12,24,36) |

取消一次 stride=2 后，后续层才把 rate 乘2。`make_block` 明确区分本块使用的 rate 和随后更新的 rate。没有额外使用 multi-grid；论文 §4.3 指出它对该 Xception 设定没有改善。

OS 表示主干的名义下采样倍数。对于不是 OS 整数倍的边长，输出约为 `ceil(H/OS)`，并不要求输入高除以输出高恰好等于 OS。decoder 分类前的网格为 OS=4；本实现最终返回已恢复到原图尺寸的 logits。

### 6.3 对照骨干与开关

| 配置 | 含义 |
| --- | --- |
| `backbone="xception65"` | 默认 Fig.4 主干 |
| `backbone="xception71"` | §4.5 的更深入口变体；入口为128/256/256/728/728，Middle仍16块 |
| `backbone="resnet101"` | 展开 `[3,4,23,3]` 个 Bottleneck 的 ResNet-101 对照 |
| `use_separable_conv=False` | ASPP/decoder 的3×3改为普通卷积；不改变 Xception 内部卷积类型 |
| `image_pooling=False` | 移除全局池化分支，ASPP 拼接通道由1280变为1024 |

Xception-71 的浅层特征改取 Entry block 3 的第二层输出；具体入口配置来自官方源码。论文在 Cityscapes 上发现移除 image-level 分支有益，这不是在所有数据集上都必须移除的结论。

## 7. Tensor Shape 分析

默认 Xception-65、OS=16、`x=[1,3,256,256]`、`K=21`：

| 位置 | Shape |
| --- | --- |
| 输入 | `[1,3,256,256]` |
| Stem 完成 | `[1,64,128,128]` |
| Entry block 1 完成 | `[1,128,64,64]` |
| 浅层特征，Entry block 2 内部 | `[1,256,64,64]` |
| Entry block 2 完成 | `[1,256,32,32]` |
| Entry block 3 / Middle 完成 | `[1,728,16,16]` |
| Exit block 完成 | `[1,1024,16,16]` |
| Backbone 高层输出 | `[1,2048,16,16]` |
| ASPP 各分支，池化分支已插值 | 每路 `[1,256,16,16]` |
| ASPP 拼接 / 投影后 | `[1,1280,16,16]` / `[1,256,16,16]` |
| 浅层压缩 / 高层插值后 | `[1,48,64,64]` / `[1,256,64,64]` |
| Decoder 拼接 / 两层细化后 | `[1,304,64,64]` / `[1,256,64,64]` |
| 1×1 分类后 | `[1,21,64,64]` |
| 最终 logits | `[1,21,256,256]` |

改为 OS=8 时，高层与 ASPP 的空间大小为32×32，decoder 仍在64×64融合，最终输出仍256×256。奇数输入如513×513，低层为129×129，高层在 OS=16 时为33×33；代码使用 `size=目标张量尺寸`，避免盲目乘4造成拼接或输出尺寸不一致。

## 8. 与经典方法的区别

| 方法 | 多尺度上下文 | 恢复空间细节的主要方式 |
| --- | --- | --- |
| FCN | 分类主干及不同层特征 | 粗预测上采样；FCN-16s/8s融合较浅层预测 |
| U-Net | 编码器层级特征 | 多级 decoder 与多个高分辨率 skip 拼接 |
| PSPNet | 对高层特征做不同网格大小的池化 | 汇聚金字塔池化特征后预测并上采样 |
| DeepLabv3 | 多个 rate 的 ASPP 与 image-level pooling | 主要将粗尺度 logits 直接插值 |
| DeepLabv3+ | 保留 ASPP | 新增简单 decoder，融合一个 OS=4 浅层端点 |

ASPP 与 PSPNet 的差异在于：前者主要改变卷积采样间隔，后者改变池化网格。DeepLabv3+ 的 decoder 比 U-Net 简单；重点是把多尺度上下文与边界细化组合起来。

## 9. 论文明确内容、官方补充与实现边界

| 内容 | 依据与处理 |
| --- | --- |
| 空洞卷积、depthwise+pointwise | 公式(1)、Fig.3，完整展开 |
| X-65通道及16次中间流 | Fig.4，完整保留 |
| 低层48通道、两层256通道细化 | §4.1、Table 1/2，按论文实现 |
| ASPP分支及rate 6/12/18 | Fig.2；OS=8时按更密的网格倍增rate |
| 浅层具体端点、X-71入口排布 | 从官方 `feature_extractor.py`、`xception.py` 补全 |
| Depthwise 后 ReLU | 按论文 §3.2/Fig.4 的文字保留；当前官方TF预激活残差块设 `activation_fn_in_separable_conv=False`，其内部DW后没有独立ReLU。本实现优先论文，明确存在这一差异 |
| 其余激活位置 | Xception残差流采用卷积前ReLU、pointwise后BN、相加后不额外激活；出口尾部/ASPP/decoder采用pointwise后BN+ReLU。这是具体排布的补全 |
| padding与插值 | 对称padding；双线性 `align_corners=True` 参照官方工程实现，论文未规定该参数 |
| BN与初始化 | Xception BN eps=1e-3，头部/ResNet eps=1e-5；保留PyTorch默认momentum=0.1、默认随机初始化。没有照搬TF移动平均衰减或初始化分布 |
| ASPP Dropout | 使用官方实现的普通 `Dropout(p=0.1)`，属于工程补充；eval时不随机丢弃元素 |
| ResNet-101排布 | 使用PyTorch常见v1.5排布，在新stage首块的3×3放stride；浅层取layer1最终输出。官方TF采用不同的下采样位置及内部端点，因此此项是论文骨干的教学改写，并非TF逐节点移植 |
| 默认21类 | 便于对应VOC的20类前景加背景；可设置为其他类别数 |

因此，本实现适合逐层学习和调试，不直接兼容官方 TensorFlow checkpoint。论文的 ImageNet/COCO/JFT 预训练、完整训练协议、多尺度/翻转测试和 Cityscapes coarse 数据微调没有加入这个参考实现。

论文 Table 5 中，Xception、OS=16、无COCO/JFT、无多尺度/翻转时，加入decoder的普通卷积版本为79.93% mIoU，头部采用可分离卷积为79.79%，Multiply-Adds由89.76B降为54.17B。说明这一替换主要降低计算量，单个设定并不保证提高精度。论文报告的VOC测试集89.0%还使用了JFT预训练等设置，不能理解为运行本目录随机模型就有该表现。

## 10. 前向演示与 ONNX 导出

在本目录运行：

```bash
python -m pip install "torch>=2.9" onnx onnxscript onnxruntime numpy
python model.py
```

`model.py` 的 `if __name__ == "__main__":` 中包含完整流程：

1. 创建一次模型并调用 `.cpu().eval()`，创建一次 `x=torch.randn(1,3,256,256)`。
2. 计算 `output=model(x)` 并打印输入、输出形状及部分数值。
3. 将**同一个模型对象和同一个输入张量**传入 `torch.onnx.export`，导出 `deeplabv3plus.onnx`。
4. 用ONNX checker检查图，再将同一份x送入ONNX Runtime，与步骤2保存的全部logits逐元素比较。

两次执行都在eval模式，走同一个 `forward`，包含同一套backbone、ASPP、decoder、分类和最终插值。没有导出专用分支、简化模型、包装器或替代算子。配置只修改模型构造处，演示和导出就会一起改变。

导出使用 `dynamo=True`、opset18、`optimize=False`，并设置 `external_data=False`，便于用一个ONNX文件在Netron中阅读。ONNX图可能包含框架转换产生的Shape、Constant、Resize等节点；BN、Dropout在eval下的表现也与训练图不同。**执行语义和数值一致，不表示Python语句与ONNX节点一一对应。** 浮点运算允许 `rtol=1e-4, atol=1e-5` 的误差。

本例ONNX固定输入形状为 `[1,3,256,256]`，没有声明动态轴；修改示例输入后重新导出即可。PyTorch模型本身按输入尺寸插值。生成的ONNX包含随机权重，体积较大，由目录内 `.gitignore` 忽略；仓库提交的是生成和验证它的代码。

只想在自己的代码中调用模型，可以在该目录中导入：

```python
import torch
from model import DeepLabV3Plus

model = DeepLabV3Plus(num_classes=2).eval()
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    logits = model(x)          # [1,2,256,256]
    mask = logits.argmax(1)   # [1,256,256]，数值为0或1
```

两通道二分类可以配合 `CrossEntropyLoss`，标签为 `[B,H,W]` 的long张量。若设置 `num_classes=1`，则使用 `BCEWithLogitsLoss`，标签为 `[B,1,H,W]` 的浮点0/1张量，预测时用 `logits.sigmoid()>0.5`；单通道不能用argmax获得前景。

**训练时的BN要求：** 默认image-level分支池化后是 `[B,256,1,1]`，普通训练模式BatchNorm需要 `B>=2`。单张前向/导出使用 `model.eval()`。如果后续用batch size=1训练，需要明确采用冻结BN等策略；本实现不会为了绕过这个问题而静默改成另一种归一化或另一条forward。

## 11. 本次实际验证

- 环境：PyTorch 2.9.1+cpu、ONNX 1.23.2、ONNX Runtime 1.31.0、ONNX Script 0.7.2。
- 直接执行 `python model.py`：默认完整Xception-65输出 `[1,21,256,256]`，ONNX checker通过，PyTorch与ONNX Runtime最大绝对误差为 `2.9802322e-08`。
- ONNX中保留147个Conv节点，包含depthwise分组和rate 6/12/18的ASPP分支；有3个Resize节点，对应池化分支恢复、decoder融合和最终输出恢复。
- Xception-65、Xception-71、ResNet-101各自的OS=8/16配置，均通过65×79奇数输入的高层/浅层/输出形状及有限值检查。
- 普通卷积头、关闭image-level分支、单通道输入和单通道二分类输出通过前向检查。
- 默认模型在训练模式下，以batch size=2计算多分类交叉熵并反向传播：所有可学习参数均获得有限梯度，主干、ASPP、池化分支、浅层投影和分类头的关键权重均有非零梯度。

这些检查验证计算路径与导出可用性，不是数据集训练或精度复现；仅默认配置执行了ONNX数值对照，其他配置验证的是PyTorch前向。

## 12. 阅读心得

DeepLabv3+ 的关键不在于堆叠复杂的解码器，而在于各部分职责明确：高层ASPP获取多尺度上下文，浅层特征提供定位细节，用少量卷积完成融合。学习时应分别理解 `stride`、`dilation` 和上采样：它们改变的分别是输出采样密度、卷积采样间隔和特征网格大小。可分离卷积进一步减少计算，但不会自动解决边界问题，也不保证在任何数据集上都比普通卷积更准。

## 13. 参考资料

- [论文原文](https://arxiv.org/abs/1802.02611)
- [官方 Xception 定义](https://github.com/tensorflow/models/blob/master/research/deeplab/core/xception.py)
- [官方 decoder 特征端点](https://github.com/tensorflow/models/blob/master/research/deeplab/core/feature_extractor.py)
- [官方 ASPP、decoder 与插值实现](https://github.com/tensorflow/models/blob/master/research/deeplab/model.py)
- [PyTorch ONNX 导出文档](https://docs.pytorch.org/docs/stable/onnx.html)
