# TransUNet 论文学习与 PyTorch 参考实现

## 1. 论文信息

| 项目 | 内容 |
| --- | --- |
| 标题 | **TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation** |
| 作者 | Jieneng Chen, Yongyi Lu, Qihang Yu, Xiangde Luo, Ehsan Adeli, Yan Wang, Le Lu, Alan L. Yuille, Yuyin Zhou |
| 年份 | 2021 |
| 发表会议 / 期刊 | 本次上传的是 **arXiv:2102.04306v1，2021-02-08 预印本**；该版本未标注正式会议或期刊，不把后续扩展版信息混用到本版本。 |
| 论文链接 | [arXiv v1](https://arxiv.org/abs/2102.04306v1) · [PDF](https://arxiv.org/pdf/2102.04306v1) |
| 官方代码 | [Beckschen/TransUNet](https://github.com/Beckschen/TransUNet) |
| 任务 | 医学图像语义分割；本文实现对二维切片预测逐像素类别。 |

本目录是用于阅读论文的参考实现。核心计算完整展开，没有数据集、训练循环、优化器或 checkpoint 下载。默认模型采用完整 Base 配置，但权重随机初始化；运行成功和导出一致性检查不等于复现论文的分割精度。

## 2. 论文简介与核心思想

U-Net 使用 CNN 编码器提取特征，再逐步上采样并融合跳跃连接。卷积能够逐层扩大感受野，但单层卷积只在局部邻域计算，缺少显式的全局位置交互。Transformer 自注意力可以直接建立任意两个 token 之间的联系，不过只对低分辨率 token 做简单上采样，容易丢失精细的边界和定位信息。

TransUNet 将两类能力结合起来：

1. **CNN 提取局部特征**，并保留多个分辨率的空间细节。
2. **Transformer 处理 CNN 最深层的特征序列**，建模全局上下文。
3. **CUP 解码器逐级上采样**，在 `/8`、`/4`、`/2` 尺度拼接对应的 CNN 特征。
4. **分割头输出每个像素的类别分数**。

Transformer 位于编码端，解码器仍然使用插值和卷积。三条跨尺度 skip 来自 CNN，不是来自三个不同的 Transformer 层。

## 3. 文件结构与阅读顺序

| 文件 | 作用 |
| --- | --- |
| `backbone.py` | 权重标准化卷积、展开的残差瓶颈块、R50-ViT 混合 CNN 骨干。 |
| `transformer.py` | 特征 patch embedding、可学习位置编码、Q/K/V、多头注意力、MLP、Pre-LN 编码器。 |
| `decoder.py` | CUP 上采样、通道拼接、卷积融合和分割头。 |
| `model.py` | 组合完整 TransUNet；`if __name__ == "__main__":` 中包含演示、ONNX 导出及数值核验。 |
| `README.md` | 论文笔记、公式到代码的对应、形状分析、实现取舍和运行说明。 |

建议先看 `model.py` 的 `forward()`，然后按上述三个模块文件追踪细节。

## 4. 论文、官方代码与本实现的边界

遵循 `request.md`：论文描述优先；论文没有写明的细节明确标注来源。

| 细节 | 本实现及依据 |
| --- | --- |
| 总体架构 | 按论文 §3.2 / Figure 1：混合编码器、四次 2 倍上采样、三条 CNN skip。 |
| Transformer 大小 | Base：`D=768, L=12, heads=12, MLP=3072`。§4.4 的 Model Scaling 段落中，hidden size 与层数的文字排列疑似颠倒；按 Figure 1、ViT Base 和官方配置解释。 |
| CNN 层配置 | 论文称 R50-ViT，未展开细节；采用官方混合骨干的 `(3,4,9)` 个瓶颈块、通道 `256/512/1024`、输出步长 16。不是直接调用 torchvision ResNet-50 的四个 stage。 |
| CNN 归一化 | 采用官方的 Weight Standardization + GroupNorm，并展开计算；不声称这些是本论文的新贡献。 |
| CNN 残差块顺序 | 按官方实际执行顺序使用 `Conv -> GN -> ReLU`，第三个卷积后先 GN，再残差相加和 ReLU；没有因为官方类名含 PreAct 就改写成另一种顺序。 |
| CNN `/4` 尺寸对齐 | 采用官方 `MaxPool2d(3,2,padding=0)`。224 输入对应 `55x55` 的 stage1 特征；只给 skip 的右侧、下侧各补一列/行零到 `56x56`。主路仍使用原始 `55x55` 特征。 |
| Patch 与位置编码 | 论文明确在 CNN 特征图上取 `1x1` patch，添加可学习位置编码；无需 class token。投影使用带 bias 的 `1x1 Conv`，bias 为参考官方的实现补充。 |
| Dropout 与末端 LN | embedding/MLP dropout 为 0.1，attention dropout 为 0，LayerNorm epsilon 为 `1e-6`，编码器末端再加 LN，参考官方实现补全。 |
| CUP 每级卷积次数 | **按论文 §3.2，每级一次 `2x upsample -> 3x3 Conv -> ReLU`**，有 skip 的层在上采样后、卷积前拼接。官方代码每级是两次 Conv-BN-ReLU；本实现未采用该扩展，也未加 decoder BatchNorm。 |
| 解码器通道 | 按 Figure 1：`D -> 512 -> 256 -> 128 -> 64 -> 16`。 |
| 插值参数 | 双线性插值，`align_corners=False`，属于合理补全；官方 `UpsamplingBilinear2d` 使用的角点对齐方式不同。 |
| 分割头 | 主模型 head 的核大小未写明，选择 `1x1 Conv(16,K)` 作为补全；官方用 `3x3`。论文明确提及的 `1x1` 分类层在 naive baseline 段落，不能据此声称主模型 head 也被明确规定为 `1x1`。 |
| 单通道输入 | 构造时设 `in_channels=1`，在 forward 中复制成三个通道，参考官方灰度图处理；三个通道不是三张相邻切片。 |
| 初始化与预训练 | 位置向量用 `Normal(0,0.02)`，其余层用 PyTorch 默认初始化，均为演示选择。论文使用 ImageNet 预训练；这里未加载预训练参数，也不直接兼容官方完整 checkpoint。 |
| 输入尺寸 | 为保持代码简明，构造时固定高宽，两者均需为 16 的倍数且至少为 32；支持构造固定的矩形输入，是实现补充。不实现运行时位置编码插值。 |

以上差异意味着本代码是**以论文文字为主的参考实现**，不是官方模型逐层、逐权重完全一致的复制。演示和 ONNX 导出则严格使用本实现的同一个模型。

## 5. 模型整体结构与 Forward 数据流

默认设置：输入 `[B,3,224,224]`，`D=768`，`L=12`，`K=9`。这里的 9 类是 Synapse 场景的 8 个器官加背景，不是网络结构固定只能分 9 类。

| 步骤 | 模块 / 操作 | Tensor shape |
| --- | --- | --- |
| 1 | 输入图像 | `[B,3,224,224]` |
| 2 | CNN stem：7x7 Conv，stride=2，GN，ReLU | `[B,64,112,112]`，保存 `/2` skip |
| 3 | 3x3 MaxPool，stride=2，padding=0 | `[B,64,55,55]` |
| 4 | CNN stage1，3 个瓶颈块 | `[B,256,55,55]`，skip 补零为 `[B,256,56,56]` |
| 5 | CNN stage2，4 个瓶颈块 | `[B,512,28,28]`，保存 `/8` skip |
| 6 | CNN stage3，9 个瓶颈块 | `[B,1024,14,14]` |
| 7 | 1x1 Conv 投影到 D 维 | `[B,768,14,14]` |
| 8 | flatten 空间维、transpose | `[B,196,768]` |
| 9 | 加位置编码 `[1,196,768]` | `[B,196,768]` |
| 10 | 12 个 Transformer Block + 末端 LN | `[B,196,768]` |
| 11 | transpose、reshape 回二维网格 | `[B,768,14,14]` |
| 12 | 3x3 Conv + ReLU | `[B,512,14,14]` |
| 13 | 上采样、拼接 `/8` skip、3x3 Conv + ReLU | `[B,256,28,28]` |
| 14 | 上采样、拼接 `/4` skip、3x3 Conv + ReLU | `[B,128,56,56]` |
| 15 | 上采样、拼接 `/2` skip、3x3 Conv + ReLU | `[B,64,112,112]` |
| 16 | 上采样、3x3 Conv + ReLU，无 skip | `[B,16,224,224]` |
| 17 | 1x1 分割头 | `[B,9,224,224]` logits |

跨尺度融合的通道数可以单独检查：

| 解码尺度 | 上采样后的通道数 | CNN skip 通道数 | 拼接后的通道数 | 卷积后的通道数 |
| --- | ---: | ---: | ---: | ---: |
| `/8` | 512 | 512 | 1024 | 256 |
| `/4` | 256 | 256 | 512 | 128 |
| `/2` | 128 | 64 | 192 | 64 |
| 原尺寸 | 64 | 0 | 64 | 16 |

`torch.cat(..., dim=1)` 增加通道数；Transformer 和 CNN 内部的残差 `x + residual` 不增加通道数。这里不能把拼接与相加混为一谈。

## 6. 公式解析与代码对应

### 6.1 公式 (1)：Patch embedding 与位置编码

论文先用纯 ViT 说明图像序列化：原图 patch 边长为 P 时，

$$N=\frac{HW}{P^2},\qquad x_p^i\in\mathbb{R}^{P^2C}.$$

$$z_0=[x_p^1E;x_p^2E;\ldots;x_p^NE]+E_{pos}.$$

**在混合编码器中，取的是 CNN 特征图上的 1x1 patch。** 默认 CNN 输出 `[B,1024,14,14]`，因此每个 patch 是一个 1024 维向量，投影到 768 维。

```python
x = self.projection(features)   # [B,1024,14,14] -> [B,768,14,14]
x = x.flatten(2).transpose(1, 2)
x = x + self.position_embedding
```

默认 `N=14*14=196`。论文所说的 `P=16` 对应相对输入图像的 token 网格步长；混合骨干中一个位置的真实感受野可以覆盖更大范围，不等于互不重叠的原图 16x16 小块。

### 6.2 公式 (2)：Pre-LN 自注意力与残差

$$z'_\ell=\operatorname{MSA}(\operatorname{LN}(z_{\ell-1}))+z_{\ell-1}.$$

对应 `TransformerEncoderBlock.forward()` 前半段。以下是标准 MSA 的展开，不是声称论文另列了这些编号公式：

$$Q=XW_Q+b_Q,\quad K=XW_K+b_K,\quad V=XW_V+b_V,$$

$$A=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_{head}}}\right),\quad O=AV,$$

$$\operatorname{MSA}(X)=\operatorname{Concat}(O_1,\ldots,O_h)W_O+b_O.$$

这里 `X` 是 LN 后的 token 特征，Q/K/V 的投影矩阵相互独立。

| 计算 | Base 配置 shape | PyTorch 操作 |
| --- | --- | --- |
| Q/K/V 投影 | `[B,196,768]` | 三个独立 `nn.Linear` |
| 分成 12 个 head | `[B,12,196,64]` | `reshape`、`permute` |
| `Q @ K.transpose(-2,-1)` | `[B,12,196,196]` | `torch.matmul` |
| 除以 `sqrt(64)`、对 key 维 softmax | `[B,12,196,196]` | 除法、`torch.softmax(dim=-1)` |
| 注意力权重乘 V | `[B,12,196,64]` | `torch.matmul` |
| 拼接 head、输出投影 | `[B,196,768]` | `permute`、`reshape`、`nn.Linear` |

这是图像编码器的全局注意力，不使用文本自回归预测的 causal mask。空间位置由位置编码提供，Q/K/V 本身不是二维坐标。

### 6.3 公式 (3)：MLP 与第二条残差

$$z_\ell=\operatorname{MLP}(\operatorname{LN}(z'_\ell))+z'_\ell.$$

MLP 采用 `Linear(768,3072) -> GELU -> Dropout -> Linear(3072,768) -> Dropout`。两次线性变换沿最后一维执行；MLP 不在不同 token 之间做信息交换，token 间交互由前面的注意力完成。

LN 对每个 token 的 768 维做归一化。12 个 Block 均保持序列长度与隐藏维度不变，各层参数独立。

### 6.4 CNN 瓶颈块中的计算

`HybridBottleneck.forward()` 完整展开：

1. 1x1 卷积将通道缩小到中间维度，GN、ReLU。
2. 3x3 卷积提取空间信息，必要时 `stride=2`，GN、ReLU。
3. 1x1 卷积恢复输出通道，GN。
4. 若形状不同，残差支路通过 1x1 卷积和 GN 对齐。
5. 两支路逐元素相加，再 ReLU。

卷积权重标准化对每个输出卷积核单独计算：

$$\widehat W_o=\frac{W_o-\operatorname{mean}(W_o)}{\sqrt{\operatorname{var}(W_o)+10^{-5}}}.$$

`torch.var_mean(..., dim=(1,2,3), unbiased=False)` 对应上述均值和总体方差。随后仍用 `F.conv2d` 做普通卷积，`groups=1`；它不是逐通道卷积。

### 6.5 CUP 与分割输出

序列先以同一展平顺序还原为 `[B,D,H/16,W/16]`。模型保存构造时的二维网格，不依赖 `sqrt(N)`，所以固定矩形输入也能保持位置对应关系。

设本级上采样后的特征为 U，对应 CNN skip 为 S，则有 skip 的解码级可写成：

$$F=\operatorname{ReLU}\left(\operatorname{Conv}_{3\times3}(\operatorname{Concat}(U,S))\right).$$

此式是对论文流程的整理。最后一个解码级没有 skip；分割头把 16 个通道映射成 K 类 logits。

```python
logits = model(images)                  # [B,K,H,W]
probabilities = logits.softmax(dim=1)   # 多类别概率，可选后处理
labels = logits.argmax(dim=1)           # [B,H,W] 类别 ID，可选后处理
```

`softmax` 和 `argmax` 不在模型 `forward()` 内。若设置单通道二分类输出 `num_classes=1`，后处理应使用 `sigmoid` 和阈值，不能沿长度为 1 的类别维 argmax。

## 7. 与经典方法的区别

| 方法 | 全局建模与特征提取 | 解码与细节恢复 | 输出 |
| --- | --- | --- | --- |
| U-Net | CNN 编码器；逐层扩大感受野 | 上采样 + CNN skip 拼接 | 逐像素类别 |
| 分类 ViT | 通常对原图切 patch，经 Transformer 编码 | 使用 class token 分类，不需要恢复像素网格 | 整张图的类别 |
| 论文 ViT-CUP | 纯 ViT 编码器 | CUP，上采样但无混合 CNN 的多尺度 skip | 逐像素类别 |
| 论文 R50-ViT-CUP | CNN + Transformer | CUP，无 CNN skip 融合 | 逐像素类别 |
| TransUNet | CNN 提取局部特征，低分辨率 Transformer 显式建模全局关系 | CUP + 三条 CNN skip | 逐像素类别 |
| Attention U-Net | CNN 为主体，使用门控筛选 skip 特征 | 解码器融合经过门控的 skip | 逐像素类别 |

TransUNet 的 Q/K/V 自注意力与 Attention U-Net 的加性注意力门不是同一种模块。它也不是把 U-Net 的每一层都换成 Transformer。

## 8. 论文实验与消融阅读

以下是上传的 **2021 v1 论文报告值**，不是本目录代码重新训练得到的结果。

| Synapse 方法 | 平均 DSC (%) | 平均 HD (mm) |
| --- | ---: | ---: |
| ViT + 直接上采样 | 61.50 | 39.61 |
| ViT-CUP | 67.86 | 36.11 |
| R50-ViT-CUP | 71.29 | 32.87 |
| TransUNet | 77.48 | 31.69 |

对应论文 Table 1。CUP 改善解码，混合 CNN 增强编码，而跨尺度 skip 进一步恢复细节；不能只把最后的提升归因于加入 Transformer。Table 5 的 ACDC 平均 DSC 为 89.71%。

§4.4 比较了无 skip、单条 skip 和三条 skip。**单条 skip 使用的是 `/4` 尺度**，不是默认从最深的 `/8` 取一条。本实现用明确的尺度参数表达：

```python
model = TransUNet(skip_scales=())          # 无 skip 的结构变体
model = TransUNet(skip_scales=(4,))        # 论文 1-skip 的位置
model = TransUNet(skip_scales=(8, 4, 2))   # 默认 3-skip
```

这些只方便观察结构变化，未重新训练或声称重现消融指标。论文还研究了不同分辨率、patch 大小、模型规模和 skip 内的轻量 Transformer；本目录聚焦主模型，不实现所有扩展实验。

输入从 224 增大到 512 时，默认 token 数从 196 增至 1024。注意力矩阵大小与 `N²` 成正比，约增至原来的 27.3 倍，因此提升分辨率同时会明显增加注意力计算和存储开销。

## 9. 运行与 ONNX 导出

在本目录运行。验证环境使用 Python 3.12、PyTorch `2.9.1+cpu`、ONNX `1.19.1`、ONNX Runtime `1.23.2`、NumPy `2.3.5`。如果已经安装适合自己 CUDA 环境的 PyTorch，可保留现有版本；下面给出本次 CPU 验证所用的安装方式：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install onnx==1.19.1 onnxruntime==1.23.2 numpy==2.3.5
python model.py
```

`model.py` 的主程序会：

1. 创建完整 Base 模型，设置 `.cpu().eval()`，构造随机输入。
2. 用 `logits = model(images)` 演示，打印输入/输出形状、一个像素的 logits 和参数量。
3. 直接执行 `torch.onnx.export(model, images, ...)`。
4. 用 `onnx.checker` 检查文件，再用 ONNX Runtime 对同一份输入推理。
5. 比较全部 logits，并打印最大绝对误差；采用 `rtol=1e-3, atol=1e-4`。

输出文件为本目录下的 `transunet.onnx`，包含当前模型的随机权重。仓库提交源码和笔记，ONNX 大型二进制文件由主程序在本地生成。

**演示与导出的一致性约定：**

- 相同的 `model` 对象、相同参数、相同 `images` 对象。
- 相同的 `eval()` 状态，Dropout 在两处均关闭。
- 两处都调用完整 `TransUNet.forward()` 并输出相同的 logits。
- 没有导出专用模型、导出专用分支、临时删层或更换输入尺寸。
- 固定导出输入形状为 `[1,3,224,224]`，不声明动态 batch 或动态空间尺寸。

`dynamo=False` 明确选择在验证环境可用的传统 TorchScript ONNX 导出路径；它不是另一套模型结构。PyTorch 2.9 默认推荐新导出器，参见 [官方文档](https://docs.pytorch.org/docs/2.9/onnx.html)。这里显式指定该参数，便于对照展开的算子并保持已验证的导出设置；不代表 `dynamo=True` 在原理上不正确。

导出器可能把常量计算折叠、把一个 PyTorch 模块拆成多个 ONNX 算子。Netron 显示的节点数量和 Python 类的数量因此不会一一相等；同一前向路径也不保证跨运行时逐 bit 相同，所以还要检查浮点数值误差。输入尺寸检查在 trace 时会成为常量，不应据此把导出的模型当作可接受任意形状。

本次实际验证结果：

| 检查 | 结果 |
| --- | --- |
| 完整默认 Base 配置 | 104,497,577 个参数；输入 `[1,3,224,224]`，输出 `[1,9,224,224]`。 |
| ONNX checker 与 CPU Runtime | 通过；与 PyTorch 比较所有 logits，最大绝对误差为 `3.2782554626464844e-7`。 |
| 灰度矩形输入与 skip 尺度 | `[2,1,32,48]`，分别检查三条、仅 `/4`、无 skip，所有中间形状与输出形状通过。 |
| 反向传播 | 交叉熵 backward 后所有参数梯度存在且有限；CNN、位置编码、Q 投影、解码器和分割头的代表参数梯度非零。 |
| 无效配置 | 非法分辨率、输入通道、skip 尺度、重复 skip、无法均分的 head 数均被拒绝。 |

灰度/矩形与梯度检查使用完整 CNN、`D=96,L=2,heads=3,MLP=192` 的较小 Transformer 以节省检查成本；**上面的 ONNX 数值检查使用的是完整默认 Base 模型**。这些检查验证实现和导出，没有测试训练收敛或分割精度。

只想在 Python 中使用模型时：

```python
import torch
from model import TransUNet

model = TransUNet(image_size=224, in_channels=1, num_classes=4).eval()
images = torch.randn(2, 1, 224, 224)
with torch.no_grad():
    logits = model(images)  # [2,4,224,224]，灰度图在 forward 内复制成三通道
```

改变空间尺寸时，需要重新构造匹配的模型，例如 `image_size=(224,320)`；位置编码长度随之改变，不能把已训练权重直接当作不受影响。

## 10. 阅读心得

TransUNet 的关键是让全局上下文和细节恢复分工明确。Transformer 建立低分辨率位置之间的联系，CNN skip 保留高分辨率信息，解码器将两者融合。仅在 bottleneck 处加入 Transformer，和带完整三条 skip 的 TransUNet，在信息流上并不相同。

阅读时最容易混淆的是三个“patch/尺寸”：原图中的 16 像素网格步长、CNN 特征图上的 1x1 patch，以及 Transformer 中的 196 个 token。把它们与每一步 Tensor shape 对齐，结构就容易理解。

## 11. 参考来源

- 上传论文：2021 arXiv v1，重点为 §3.1、§3.2、§4.2、§4.4 和 Figure 1。
- 官方实现核对版本：[`02ef0010b36eb8328b5e689eadaf613602edf9b8`](https://github.com/Beckschen/TransUNet/tree/02ef0010b36eb8328b5e689eadaf613602edf9b8)。
- [官方 CNN 骨干](https://github.com/Beckschen/TransUNet/blob/02ef0010b36eb8328b5e689eadaf613602edf9b8/networks/vit_seg_modeling_resnet_skip.py)。
- [官方配置](https://github.com/Beckschen/TransUNet/blob/02ef0010b36eb8328b5e689eadaf613602edf9b8/networks/vit_seg_configs.py)。
- [官方 Transformer 与解码器](https://github.com/Beckschen/TransUNet/blob/02ef0010b36eb8328b5e689eadaf613602edf9b8/networks/vit_seg_modeling.py)。
