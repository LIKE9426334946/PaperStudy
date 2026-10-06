# PSPNet：Pyramid Scene Parsing Network

本目录按论文组织 PyTorch 参考实现，重点是读懂膨胀 ResNet、金字塔池化和辅助监督。代码不包含数据集、训练工程或自动下载权重；**结构与梯度验证不等于复现论文精度**。

## 1. 论文信息

| 项目 | 内容 |
| --- | --- |
| 标题 | Pyramid Scene Parsing Network |
| 作者 | Hengshuang Zhao、Jianping Shi、Xiaojuan Qi、Xiaogang Wang、Jiaya Jia |
| 会议 | IEEE Conference on Computer Vision and Pattern Recognition（CVPR） |
| 年份 | 2017 |
| 本次阅读版本 | 用户提供的 `16-PSPNet.pdf`，arXiv:1612.01105v2，2017-04-27 |
| 论文 | [arXiv](https://arxiv.org/abs/1612.01105) / [CVPR 页面](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html) |
| 原始官方代码 | [hszhao/PSPNet（Caffe）](https://github.com/hszhao/PSPNet) |
| 工程细节参考 | 作者后续 [semseg/model/pspnet.py](https://github.com/hszhao/semseg/blob/master/model/pspnet.py) 与 [resnet.py](https://github.com/hszhao/semseg/blob/master/model/resnet.py) |

后续 PyTorch 代码用于补全正文未逐层给出的配置，不把它与 2017 年 Caffe 实现视为逐项相同。

## 2. 论文简介与核心思想

语义分割给每个像素分配类别。仅凭局部外观，一艘船可能被认成汽车，一整栋楼可能被割裂为 building 与 skyscraper，外观接近床单的枕头也容易被忽略。论文 §3.1 将这些问题归纳为上下文关系不匹配、类别混淆与不显眼类别。

PSPNet 在深层特征之上构造多尺度区域上下文。全局池化提供整个场景线索，更细的网格保留粗略空间关系，再将它们上采样，与原始深层特征共同预测每个像素。

主要设计：

1. **膨胀 ResNet**：保持较高特征分辨率，最终输出步长为 8。
2. **Pyramid Pooling Module（PPM）**：并行产生 `1×1、2×2、3×3、6×6` 四种网格的区域表示。
3. **辅助监督**：在 res4 后连接额外分类头，用 `主损失 + 0.4 × 辅助损失` 优化共享骨干；推理时只走主分类头。

这里的“全局先验”是从输入图像学习出的上下文表示，不是手工编写“船必须在水上”的规则。“open vocabulary”是原文描述复杂类别场景的措辞；本参考实现仍为预定义 K 类分类，不具备现代文本驱动开放词汇分割能力。

## 3. 论文要求与实现补全的边界

| 项目 | 依据与本实现选择 |
| --- | --- |
| ResNet、膨胀卷积、输出步长 8 | 正文 §3.3 明确描述 |
| 四级池化、1×1 降维、双线性上采样、拼接原特征 | 正文 §3.2 明确描述 |
| 平均池化 | §5.2 / Table 1 消融结果优于最大池化，采用最佳配置 |
| 每分支 2048→512 通道 | 四个 level，每分支降至输入通道的 1/4；不是按 1、2、3、6 分别除通道数 |
| res4 后辅助分类器、权重 0.4、推理去除辅助分支 | §4、§5.1；ResNet101 对应 Fig.4 的 res4b22 后 |
| 三层 3×3 stem，通道 64/64/128 | 参考作者后续 PyTorch 实现；正文未逐层指定 |
| res4/res5 所有 3×3 卷积 dilation=2/4 | 达成输出步长 8 的具体配置，参考作者后续实现 |
| 下采样 stride 放在 Bottleneck 的 3×3 上 | 作者后续 PyTorch 实现的选择；与原始 ResNet 某些版本不同 |
| PPM 分支卷积后的 BN/ReLU | 作者后续实现，正文未完整指定此顺序 |
| 主头 4096→512→K、辅助头 1024→256→K | 作者后续实现的具体通道/卷积配置 |
| 两个分类头 Dropout2d=0.1，插值 align_corners=True | 作者后续实现的工程配置 |
| 单设备 BatchNorm2d | 阅读用简化；论文采用跨 GPU 汇总统计的 BN，训练 batch size 为 16 |
| 随机初始化 | 本代码使用 PyTorch 默认初始化，没有加载论文使用的 ImageNet 预训练权重 |
| 输出直接插值到原始 H/W | 本实现的便利接口，允许 256×256 和非正方形尺寸；不沿用后续代码的 `(H-1)%8==0` 输入限制 |
| ignore_index=255 | 本实现的标签约定，不是 PSPNet 模型的必要条件 |
| ONNX 的固定尺寸池化展开 | 导出适配，数学等价于当前输入尺寸下的 PyTorch 自适应平均池化 |

支持论文实验中使用的 ResNet-50/101/152；未加入 ResNet-269。也未实现论文全部增强、多尺度测试、模型集成或数据集训练流程。上述默认随机初始化模型不能与论文指标直接比较；deep stem 的形状、键名也与普通 torchvision ResNet 不完全相同，不应直接无检查地套用其权重。

## 4. 整体结构与模块职责

| 模块 | 作用 | 后续去向 |
| --- | --- | --- |
| 三层卷积 stem + 最大池化 | 提取浅层特征，降至约 1/4 尺寸 | res2 |
| res2、res3 | 残差特征提取；res3 降至约 1/8 | res4 |
| res4，dilation=2 | 保持分辨率、扩大空间覆盖 | res5，以及训练用辅助分类头 |
| res5，dilation=4 | 得到 2048 通道深层特征 | PPM |
| PPM | 汇集全局与不同网格区域的上下文 | 4096 通道拼接结果 |
| 主分类头 | 融合 4096 通道，并产生每像素 K 类 logits | 双线性插值回输入大小 |
| 辅助分类头 | 对 res4 特征施加额外监督 | 训练时计算辅助交叉熵 |

`backbone.py` 的四个残差阶段分别对应常见 torchvision 命名中的 `layer1/2/3/4`。因此本目录的 **res4 等于 layer3**，不要将它误认成最后的 layer4。

| 骨干深度 | res2 块数 | res3 块数 | res4 块数 | res5 块数 |
| --- | ---: | ---: | ---: | ---: |
| 50 | 3 | 4 | 6 | 3 |
| 101 | 3 | 4 | 23 | 3 |
| 152 | 3 | 8 | 36 | 3 |

深度名称沿用 ResNet 骨干家族命名；使用三层 stem、额外 PPM 与分类头后，不能把整套 PSPNet 的卷积数直接当成名字中的 50/101/152。

## 5. Forward 数据流与 Tensor Shape

以 `x=[B,3,256,256]`、类别数 K 为例：

| 运算 | 输出形状 |
| --- | --- |
| stem conv1，3×3/s2 | `[B,64,128,128]` |
| stem conv2，3×3/s1 | `[B,64,128,128]` |
| stem conv3，3×3/s1 | `[B,128,128,128]` |
| MaxPool，3×3/s2 | `[B,128,64,64]` |
| res2 | `[B,256,64,64]` |
| res3 | `[B,512,32,32]` |
| res4，辅助分支输入 | `[B,1024,32,32]` |
| res5，PPM 输入 | `[B,2048,32,32]` |
| PPM 四个池化输出 | `[B,2048,1,1]`、`[B,2048,2,2]`、`[B,2048,3,3]`、`[B,2048,6,6]` |
| 各自 1×1 卷积 + BN + ReLU | `[B,512,s,s]`，s∈{1,2,3,6} |
| 各自插值到 32×32 | 四个 `[B,512,32,32]` |
| 原特征 + 四分支，按通道 cat | `[B,4096,32,32]`，因为 2048+4×512=4096 |
| 主头 3×3 融合 | `[B,512,32,32]` |
| 主头 1×1 分类 | `[B,K,32,32]` |
| 主 logits 插值到输入大小 | `[B,K,256,256]` |
| 辅助头（仅训练） | `[B,1024,32,32]` → `[B,256,32,32]` → `[B,K,32,32]` → `[B,K,256,256]` |

对任意输入 H/W，本实现深层空间大小为 `h=ceil(H/8)`、`w=ceil(W/8)`：473×473 得到 60×60；255×321 得到 32×41。所谓“1/8”是输出步长概念，奇数尺寸需要考虑 padding 和取整。

四个 PPM 分支读取**同一个 res5 输出**，没有从 1×1 逐层传到 6×6。1×1 分支上采样后，每个空间位置接收相同的全局描述；2/3/6 分支仍有粗粒度位置差异。拼接发生在通道维 `dim=1`，不是逐元素相加。

## 6. 公式与 PyTorch 运算对应

以下除 poly 学习率表达式外，主要是对正文算法的数学展开，便于对照代码，并非声称论文为每一项都列出了编号公式。

### 6.1 残差块与膨胀卷积

残差块为：

$$y=\operatorname{ReLU}(F(x)+S(x)).$$

F 是三次卷积与归一化组成的残差分支，S 是恒等映射或 `1×1 Conv + BN` 投影。对应 `Bottleneck.forward()` 中的 `x + identity`。

省略 batch 和 bias，一个 stride=1、3×3、padding=d 的膨胀卷积可以写为：

$$y_{o,i,j}=\sum_c\sum_{u=-1}^{1}\sum_{v=-1}^{1}
W_{o,c,u+1,v+1}\,x_{c,i+du,j+dv}.$$

越界输入按零填充。其有效核跨度为：

$$k_{\mathrm{eff}}=k+(k-1)(d-1).$$

3×3 核在 d=2/4 时分别覆盖 5×5/9×9 跨度，但只使用原有 9 个采样位置，参数量不变。对应 `nn.Conv2d(..., kernel_size=3, dilation=d, padding=d)`。

### 6.2 自适应平均池化

对输入特征 $X\in\mathbb{R}^{B\times C\times h\times w}$，输出网格为 s×s。第 i 行 bin 的范围是：

$$a_i=\lfloor ih/s\rfloor,\qquad b_i=\lceil(i+1)h/s\rceil.$$

第 j 列同理得到 $a'_j,b'_j$：

$$P_s(X)_{b,c,i,j}=
\frac{\sum_{u=a_i}^{b_i-1}\sum_{v=a'_j}^{b'_j-1}X_{b,c,u,v}}
{(b_i-a_i)(b'_j-a'_j)}.$$

对应 `nn.AdaptiveAvgPool2d((s,s))`。注意 s 是**输出网格大小**，不是固定卷积核大小。不可整除时，各区域的边界可能重叠；例如 32→3 的行区间为 `[0,11)、[10,22)、[21,32)`。

这种 floor/ceil 划分是本参考实现采用的 PyTorch 语义；不宣称非整除尺寸下与早期 Caffe 池化边界逐项相同。

### 6.3 金字塔上下文融合

每个分支：

$$Z_s=\operatorname{Resize}_{h,w}
\left(\operatorname{ReLU}(\operatorname{BN}(\operatorname{Conv}_{1\times1}(P_s(X))))\right).$$

拼接原始局部特征与上下文：

$$Z=\operatorname{Concat}_{\mathrm{channel}}[X,Z_1,Z_2,Z_3,Z_6].$$

对应 `branch(x)`、`F.interpolate(..., mode="bilinear")` 与 `torch.cat(features, dim=1)`。

分支权重分别学习；原特征和各尺度表示通过后续 3×3 融合卷积共同影响像素分类，没有预设某个尺度必须负责某种物体。

### 6.4 像素分类与辅助损失

对每个有效像素 p，类别 logits 为 $z_{p,k}$：

$$q_{p,k}=\frac{e^{z_{p,k}}}{\sum_j e^{z_{p,j}}},\qquad
\mathcal{L}_{CE}=-\frac{1}{|\Omega|}\sum_{p\in\Omega}\log q_{p,y_p}.$$

$$\mathcal{L}=\mathcal{L}_{main}+0.4\mathcal{L}_{aux}.$$

`F.cross_entropy(logits, target)` 内部完成 log-softmax 和负对数似然，排除 `ignore_index` 对应像素。`losses.py` 返回总损失和两个分支损失，便于观察。

两条损失都能更新 stem/res2/res3/res4；只有主损失通过 res5、PPM 和主分类头。辅助损失更新辅助分类头，但不会倒着流入并未参与辅助预测的 res5。代码不使用 `detach()` 阻断共享部分的梯度，也不把两个头的输出相加作为预测。

### 6.5 Poly 学习率（论文 §5.1）

$$\eta_t=\eta_0\left(1-\frac{t}{T}\right)^{0.9}.$$

对应的真实 Python/PyTorch 用法如下；其中 `iteration` 按优化器更新次数计数，取值 0..max_iterations：

```python
lr = base_lr * (1 - iteration / max_iterations) ** 0.9
for group in optimizer.param_groups:
    group["lr"] = lr
```

论文设置 base_lr=0.01、momentum=0.9、weight_decay=0.0001；ADE20K/PASCAL VOC/Cityscapes 的迭代数分别为 150K/30K/90K。这里记录其含义，不附带完整训练循环。

## 7. 与经典方法的区别

| 方法 | 主要融合方式 | 与 PSPNet 的区别 |
| --- | --- | --- |
| FCN | 全卷积逐像素预测，可融合不同层级特征 | PSPNet 显式增加深层特征上的多尺度区域上下文 |
| U-Net | 编码器特征通过跳连送入逐级上采样解码器 | PSPNet 没有对称 U 形解码器，也没有逐层融合浅层特征的跳连 |
| FPN | 跨骨干层级的自顶向下融合与横向连接 | PSPNet 的 PPM 在同一个深层特征图上并行池化；“金字塔”来源不同 |
| SPP 分类网络 | 将不同网格的池化结果展平，形成定长分类表示 | PSPNet 将池化结果恢复空间分辨率，再进行像素级预测 |
| 仅全局平均池化 | 每幅图仅一个全局向量 | PSPNet 另外保留 2×2、3×3、6×6 网格上下文 |
| 多尺度膨胀卷积分支 | 用不同 dilation 采样空间邻域 | PPM 用区域池化显式汇总上下文；骨干本身也使用膨胀卷积 |

PSPNet 的 residual add 属于 ResNet 块内部连接，不等于 U-Net 中连接编码器与解码器的跨层拼接。

## 8. 最小使用示例

进入本目录后运行以下代码；文件采用同目录导入，无需从带数字前缀的目录名进行普通 Python import。

```bash
cd 16-PSPNet
pip install torch onnx
python model.py
```

推理和类别图：

```python
import torch
from model import PSPNet

model = PSPNet(num_classes=2, depth=50).eval()
x = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    logits = model(x)                  # [1, 2, 256, 256]
    probabilities = logits.softmax(1)
    prediction = logits.argmax(1)     # [1, 256, 256]，0=背景，1=水域
```

这里只演示形状，随机输入与随机模型权重没有分割语义。类别预测直接对 logits 做 argmax 与先 softmax 再 argmax 等价。

辅助损失与反向传播：

```python
import torch
from model import PSPNet
from losses import pspnet_loss

model = PSPNet(num_classes=2, depth=50).train()
x = torch.randn(2, 3, 256, 256)
target = torch.randint(0, 2, (2, 256, 256), dtype=torch.long)
main_logits, aux_logits = model(x)
loss, main_loss, aux_loss = pspnet_loss(main_logits, aux_logits, target)
loss.backward()
```

训练示例使用 B=2：PPM 的 1×1 分支经过 BN 时，单设备 B=1 只有一个统计样本，会报错。`eval()` 下使用已有 running statistics，B=1 可以前向；`torch.no_grad()` 本身不会把 BN 切换到 eval。

本目录二分类示例使用 **2 通道 logits + CrossEntropyLoss**。如果改为单通道 BCE，需要同时调整标签形状与损失，不能只将 `num_classes` 改为 1 而继续使用这里的交叉熵。

## 9. 独立 ONNX 导出与 Netron

```bash
python export_onnx.py
```

生成脚本所在目录的 `pspnet_resnet50.onnx`，默认输入 `[1,3,256,256]`、输出 `[1,21,256,256]`。用 [Netron](https://netron.app/) 打开即可。可在脚本底部修改 depth、num_classes 和 example；修改后重新导出。默认随机权重用于结构阅读，需先加载本实现训练出的匹配权重才能用于实际预测。

导出注意点：

- `export_onnx()` 对模型副本调用 `eval()`，只导出主分支，原模型保持不变；辅助头不会出现在最终计算图中。
- B/C/H/W 全部固定，不宣称导出结果支持动态尺寸。
- 256 输入对应 32×32 深层特征，32 不能被 3、6 整除。TorchScript ONNX 导出器对这类自适应池化有限制，不能简单写一次 export 就假设成功。
- `FixedAdaptiveAvgPool2d` 按同一 floor/ceil 边界，用切片与均值精确展开；会出现 `Slice / ReduceMean / Concat`，而不一定是一个名为“PPM”或“AdaptiveAvgPool”的节点。这不代表结构错误。
- 替换后先对完整模型输出做 `torch.testing.assert_close`，再导出并运行 `onnx.checker.check_model`。
- 明确使用 `dynamo=False`、opset 17，保证此脚本的固定尺寸展开路径；这是有意选择 TorchScript 导出接口，并非声称它是 PyTorch 当前默认/推荐路径。接口说明见 [PyTorch ONNX 文档](https://docs.pytorch.org/docs/stable/onnx.html)。
- 优化可能把 BN 融合进卷积，Dropout 在 eval 下不参与随机丢弃；Netron 中节点数不必与 Python 模块数相等。

仓库提交导出脚本，不提交带随机权重的大型 ONNX 二进制文件。

## 10. 论文结果与阅读心得

论文 Table 1 的 ADE20K 单尺度验证实验中，ResNet50 baseline 为 37.23% mIoU，四级平均池化加降维为 41.68%，提高 **4.45 个百分点**。这些是原文结果，不是本目录运行得到的指标，也不是只改网络结构就保证能达到的结果。

阅读时值得抓住三个问题：

1. 理论感受野很大，不代表模型能充分利用远处信息；PPM 提供了显式聚合区域信息的路径。
2. 全局池化能提供场景线索，却丢失空间差异；不同网格在上下文范围与空间信息之间取得不同折中。
3. 辅助监督改变训练信号到达中间层的方式，在推理时不需要保留辅助分类头。

用于水域分割时，可以把岸线附近的局部纹理和整个场景的水域上下文联系起来理解这个设计；是否优于 U-Net，需要在相同数据划分、输入尺寸、预训练与训练配置下实际比较。

## 11. 文件结构与阅读顺序

| 文件 | 内容 |
| --- | --- |
| `README.md` | 论文信息、结构、公式、形状、实现边界与使用方法 |
| `blocks.py` | Bottleneck、单个池化分支、PPM、分割分类头 |
| `backbone.py` | 完整展开 stem 和各个阶段的膨胀 ResNet |
| `model.py` | 主分支、辅助分支的完整 forward 及简单输入示例 |
| `losses.py` | 主/辅助交叉熵和加权总损失 |
| `export_onnx.py` | 固定尺寸池化展开、独立 ONNX 导出与结构检查 |

建议先看 `model.py` 的整体流向，再看 `blocks.py` 的 PPM，然后理解 `backbone.py` 如何保持输出步长为 8，最后看损失与导出细节。

## 12. 本次验证记录

环境：Python 3.12、PyTorch 2.5.1+cpu、ONNX 1.17.0、ONNX Runtime 1.20.1。均使用随机输入和随机初始化权重。

- 对 32×32、32×41、60×60 特征，四个尺度的导出池化与 `adaptive_avg_pool2d` 数值一致。
- ResNet50 在 256×256 和非正方形 255×321 输入上，输出空间尺寸正确；ResNet101/152 也完成实际前向检查。
- 两类分割的主/辅助联合损失完成反向传播，stem、res5、PPM 和辅助分类头均得到有限非零梯度。
- 仅反传辅助损失时，共享骨干得到梯度，res5 不得到梯度，符合分支结构。
- ResNet50、21 类、输入 `[1,3,256,256]` 实际导出成功并通过 ONNX checker；ONNX Runtime 与 PyTorch 输出最大绝对误差约 `3.73e-8`（本次样本）。

未进行真实数据集训练、精度评估或 GPU 性能评估；ONNX 数值对齐验证针对上述固定输入规格。
