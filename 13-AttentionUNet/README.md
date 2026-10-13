# Attention U-Net: Learning Where to Look for the Pancreas

## 1. 论文信息

| 项目 | 内容 |
| --- | --- |
| 标题 | Attention U-Net: Learning Where to Look for the Pancreas |
| 作者 | Ozan Oktay, Jo Schlemper, Loic Le Folgoc, Matthew Lee, Mattias Heinrich, Kazunari Misawa, Kensaku Mori, Steven McDonagh, Nils Y Hammerla, Bernhard Kainz, Ben Glocker, Daniel Rueckert |
| 会议 | Medical Imaging with Deep Learning（MIDL 2018），Amsterdam；non-archival conference track |
| 年份 | 2018 |
| 阅读版本 | 用户提供的 arXiv:1804.03999v3，2018-05-20 |
| 论文链接 | [arXiv](https://arxiv.org/abs/1804.03999)、[v3 PDF](https://arxiv.org/pdf/1804.03999v3) |
| 作者代码 | [ozan-oktay/Attention-Gated-Networks](https://github.com/ozan-oktay/Attention-Gated-Networks) |
| 作者会议记录 | [Bernhard Kainz 的论文页面](https://wp.doc.ic.ac.uk/bkainz/publication/attention-u-net-learning-where-to-look-for-the-pancreas-3/) |

这是用于阅读论文、单步调试的 PyTorch **参考实现**。展开注意力门、跳跃连接、粗细尺度对齐和深监督，提供 3D 模型及相同计算规则的 2D 改写。没有数据集、训练循环或预训练权重，未复现论文实验指标。

论文没有完整给出每一层的通道数、深监督权重和初始化数值。本目录逐项标注合理补全，不将其称为原始实验网络的逐层复刻。

## 2. 论文简介与核心思想

胰腺在 CT 中体积较小、形状变化大，周围组织容易干扰预测。经典 U-Net 把 encoder 特征直接送入 decoder，这些跳跃连接既携带边缘和位置信息，也携带大量背景响应。

本文在拼接前加入 Attention Gate（AG）：利用粗尺度特征的语义上下文，决定高分辨率 skip 中哪些位置值得保留。整张图仍然经过网络，不需要先用另一个网络定位器官、裁剪 ROI，再进行分割。

一个门有两个输入：`x` 是 encoder 的高分辨率细节特征；`g` 是下一粗尺度的 decoder 上下文。先对齐到粗网格，分别线性投影，相加、ReLU、单通道投影、Sigmoid，得到注意力；再上采样注意力并乘回原始 `x`。

默认每个空间位置只有一个标量权重，全部通道共享它。接近 0 时抑制，接近 1 时接近直接传递。权重不是类别概率，不要求所有位置的权重加和为 1。

论文虽使用 attention / self-attention 的术语，但这里没有 Transformer 的 Q、K、V 或两两位置的点积矩阵。这是由粗尺度上下文引导的**加性空间注意力门**。

论文在 CT-150 和 CT-82 上评估。Table 1 的 CT-150、120/30 划分中，胰腺 Dice 均值从 0.814 提升至 0.840。这是论文原始结果，不是本目录代码的测试结果，也不保证在其他数据集上同样提高。

## 3. 整体结构、模块与 Forward 数据流

参考 Fig.1 的四尺度示意，使用三次池化：

| 阶段 | 模块 | 数据流及作用 |
| --- | --- | --- |
| 编码 1 | `encoder1` | 双卷积提取浅层 `x1` |
| 编码 2 | `pool`、`encoder2` | 空间减半，得到 `x2` |
| 编码 3 | `pool`、`encoder3` | 再减半，得到 `x3` |
| 瓶颈 | `pool`、`bottleneck` | 得到最粗尺度 `x4` |
| 解码 3 | `gate3`、`decoder3` | `gate3(x3,x4)`；上采样 `x4`，与门控 skip 拼接，卷积得到 `d3` |
| 解码 2 | `gate2`、`decoder2` | `gate2(x2,d3)`；上采样 `d3` 并拼接，得到 `d2` |
| 解码 1 | `decoder1` | 上采样 `d2`，与浅层 `x1` 拼接，得到 `d1` |
| 主预测头 | `head` | `1×1(×1)` 卷积将 `d1` 映射为 `K` 个 logits |
| 辅助预测头 | `aux2_head`、`aux3_head` | 从 `d2`、`d3` 预测，插值至输入分辨率，用于深监督 |

`DoubleConv` 展开两次 `Conv -> BatchNorm -> ReLU`；`UpConcatBlock` 展开插值、按通道拼接和双卷积。瓶颈表示编码与解码之间的最粗空间尺度，不表示这里使用了 ResNet 的 Bottleneck 残差块。

第二个门的 `g` 来自融合过上下文的 `d3`，不是始终复用瓶颈 `x4`。默认最浅 skip 直接传递，论文正文与结构图对此存在差异，见第 7 节。

## 4. 公式与真实 PyTorch 运算

### 4.1 普通卷积特征

论文第 2 节以卷积及激活描述特征提取：

$$
x_c^l=\operatorname{ReLU}\left(\sum_{c'}x_{c'}^{l-1}*k_{c',c}\right).
$$

对应 `DoubleConv` 的 `self.conv1(x)` 和 `F.relu(x)`。实验还采用 BatchNorm，因此实现将 BN 放在卷积与 ReLU 之间。

### 4.2 式 (1)：加性注意力得分

$$
q_{\mathrm{att},i}^{l}
=\psi^T\operatorname{ReLU}\left(W_x^T x_i^l+W_g^T g_i+b_g\right)+b_\psi.
$$

公式默认两者已经位于相同网格。代码先下采样 `x`，再投影：

```python
x_coarse = F.interpolate(x, size=g.shape[2:], mode=self.mode,
                         align_corners=False)
theta_x = self.W_x(x_coarse)
phi_g = self.W_g(g)
h = F.relu(theta_x + phi_g)
q_att = self.psi(h)
```

| 数学项 | 代码 | 通道变化 |
| --- | --- | --- |
| $W_x^T x$ | `W_x`，无 bias 的点卷积 | `Fx -> Fint` |
| $W_g^T g+b_g$ | `W_g`，有 bias 的点卷积 | `Fg -> Fint` |
| $\sigma_1$ | `F.relu` | 不变 |
| $\psi^T h+b_\psi$ | `psi`，有 bias 的点卷积 | `Fint -> 1` |

点卷积为 2D 的 `1×1` 或 3D 的 `1×1×1`。它在每个位置对通道向量做可学习线性变换。PyTorch 卷积权重的存储方向已经对应公式中的转置，不需要对输入特征图手动调用 `.T`。

`W_g.bias` 承担 `b_g`，`psi.bias` 承担 `b_psi`。没有重复加入偏置，也没有在门内部额外插入 BN 改变公式的计算顺序。

### 4.3 式 (2)：Sigmoid、重采样与门控

$$
\alpha_i^l=\operatorname{sigmoid}(q_{\mathrm{att},i}^l),\qquad
\hat{x}_{i,c}^l=x_{i,c}^l\alpha_i^l.
$$

```python
alpha_coarse = torch.sigmoid(q_att)
alpha = F.interpolate(alpha_coarse, size=x.shape[2:], mode=self.mode,
                      align_corners=False)
x_hat = x * alpha
```

Fig.2 指定用三线性插值恢复 3D 注意力分辨率；2D 改写使用双线性插值。例：`x=[1,64,32,32]`、`g=[1,128,16,16]`、`Fint=32`：

| 张量 | Shape |
| --- | --- |
| `x_coarse` | `[1,64,16,16]` |
| `theta_x`、`phi_g`、`h` | `[1,32,16,16]` |
| `q_att`、`alpha_coarse` | `[1,1,16,16]` |
| `alpha` | `[1,1,32,32]` |
| `x_hat` | `[1,64,32,32]` |

乘法沿通道广播，没有改变 skip 尺寸。最终相乘的是原始高分辨率 `x`，不能用 `x_coarse` 替代，否则会丢失 skip 的细节。

论文选择 Sigmoid，是为了避免连续使用 Softmax 导致过稀疏的激活。多个区域可以同时获得接近 1 的权重。

### 4.4 式 (3)：门控如何影响梯度

令 `x=f(...;Phi)`，则乘积求导为：

$$
\frac{\partial(\alpha x)}{\partial\Phi}
=\alpha\frac{\partial x}{\partial\Phi}
+x\frac{\partial\alpha}{\partial\Phi}.
$$

第一项表示直接经过 skip 的特征梯度被 `alpha` 缩放；第二项来自注意力本身依赖 `x` 的路径。因此不能把完整梯度简单理解成“原梯度乘 alpha”，也不能保证 encoder 参数总梯度一定变小，因为它还参与其他路径。

`x_hat = x * alpha` 不做 `detach()`，`loss.backward()` 会自动包含两项。注意力通过分割损失学习，不需要额外的注意力标注。

### 4.5 多子门扩展

`attention_heads=1` 是公式的标量形式，直接返回 `x*alpha`。大于 1 时，每个子门独立生成一张注意力图，将加权特征按通道拼接，再以点卷积融合回 `Fx` 通道。

论文描述了互补的多维/多子门，但没有唯一的融合公式。这里的拼接融合是明确补全的实现选择。子门数量不必等于类别数，不能预先把某张注意力图解释成固定器官。

## 5. 完整 Tensor Shape 分析

### 5.1 默认脚本：2D、C=16、K=2

输入 `[1,3,128,128]`，`attention_heads=1`。

| 张量/操作 | Shape |
| --- | --- |
| `x1` | `[1,16,128,128]` |
| `x2` | `[1,32,64,64]` |
| `x3` | `[1,64,32,32]` |
| `x4` | `[1,128,16,16]` |
| `alpha3`、`skip3` | `[1,1,32,32]`、`[1,64,32,32]` |
| `decoder3` 拼接输入 | `[1,192,32,32]`，即 `64+128` 通道 |
| `d3` | `[1,64,32,32]` |
| `alpha2`、`skip2` | `[1,1,64,64]`、`[1,32,64,64]` |
| `decoder2` 拼接输入 | `[1,96,64,64]`，即 `32+64` 通道 |
| `d2` | `[1,32,64,64]` |
| `decoder1` 拼接输入 | `[1,48,128,128]`，即 `16+32` 通道 |
| `d1` | `[1,16,128,128]` |
| 主输出 `logits` | `[1,2,128,128]` |
| `aux2_logits` 插值前/后 | `[1,2,64,64]` / `[1,2,128,128]` |
| `aux3_logits` 插值前/后 | `[1,2,32,32]` / `[1,2,128,128]` |

### 5.2 3D、C=16、K=2

将脚本中的 `spatial_dims` 改为 `3`，同一段代码会选择 `[1,1,16,32,32]` 输入，并使用 Conv3d、BatchNorm3d、MaxPool3d 和三线性插值。

| 张量 | Shape |
| --- | --- |
| `x1` | `[1,16,16,32,32]` |
| `x2` | `[1,32,8,16,16]` |
| `x3` | `[1,64,4,8,8]` |
| `x4` | `[1,128,2,4,4]` |
| `d3` | `[1,64,4,8,8]` |
| `d2` | `[1,32,8,16,16]` |
| `d1` | `[1,16,16,32,32]` |
| 三个输出 | 均为 `[1,2,16,32,32]` |

类的默认值是 `spatial_dims=3, base_channels=32`；主程序显式选择 2D、16 通道，便于调试。不要混淆类默认值与演示参数。

PyTorch 前向支持非整除尺寸，通过 `size=skip.shape[2:]` 对齐。每条空间边长须至少为 8；训练 BN 时还需保证每通道有多个批次/空间元素，不能用 `B=1`、瓶颈只有单个体素的输入训练。

## 6. 深监督与 Dice 损失

论文第 2 节末尾和第 3 节指出使用 deep supervision，让中间特征具有语义信息；训练使用覆盖语义类别的 Sorensen-Dice。

本实现默认返回 `(logits, aux2_logits, aux3_logits)`，三个头都参与损失。辅助输出不是注意力图；只需要主预测时取 `outputs[0]`。训练和 eval 的返回规则保持相同。

`losses.py` 选择平方分母的 soft Dice：

$$
\operatorname{Dice}_c=
\frac{2\sum_i p_{i,c}y_{i,c}+\epsilon}
{\sum_i p_{i,c}^2+\sum_i y_{i,c}^2+\epsilon},\qquad
\mathcal{L}_{Dice}=1-\frac{1}{K}\sum_c\operatorname{Dice}_c.
$$

`i` 遍历 batch 和空间位置，包含背景，类别均匀平均。多通道 logits 经 Softmax 转为概率；单通道二分类经 Sigmoid 后补成 `[背景,前景]` 两通道。平方分母、聚合方式、平滑项是补全，论文未明确给出这些细节。

深监督权重也属于教学选择：

$$
\mathcal{L}=\frac{\mathcal{L}_{main}+0.5\mathcal{L}_{aux2}+0.25\mathcal{L}_{aux3}}{1+0.5+0.25}.
$$

在本目录下执行的最小反向示例：

```python
import torch
from model import AttentionUNet
from losses import deep_supervision_loss

model = AttentionUNet(in_channels=3, num_classes=2, base_channels=16,
                      spatial_dims=2, deep_supervision=True)
x = torch.randn(2, 3, 64, 64)
target = torch.randint(0, 2, (2, 64, 64))  # 类别 ID，无通道维
outputs = model(x)
loss = deep_supervision_loss(outputs, target)
loss.backward()
print(loss.item())
print(model.gate2.heads[0].W_x.weight.grad.shape)
```

也可设置 `deep_supervision=False` 返回单个 Tensor。`num_classes=1` 代表二分类单输出通道；`num_classes>=2` 代表多个互斥类别的 logits。预测时在模型外做 Sigmoid / Softmax / argmax；不要先 Softmax 再交给 `CrossEntropyLoss`。当前 Dice 示例不实现 `ignore_index`。

## 7. 论文规定、合理补全与可选变体

| 事项 | 依据及本实现决定 |
| --- | --- |
| 3D / 2D | 论文实验明确采用 3D。2D 是便于普通图像调试的改写 |
| 四尺度和宽度 | 参考 Fig.1；`C,2C,4C,8C` 及统一双卷积是具体化选择，不保证匹配论文实验参数量 |
| 卷积与 BN | Fig.1 给出卷积与 ReLU，实验采用 BN；`padding=1`、BN 位置、卷积 bias 设置为补全 |
| 点映射 | 按公式及正文用 `1×1(×1)`，没有以 `2×2(×2)` 卷积替代点映射 |
| 粗网格对齐 | 正文要求下采样至 `g` 网格；具体下采样方法未规定，这里选择插值 |
| 注意力上采样 | Fig.2 指定三线性插值；2D 使用双线性，`align_corners=False` 为补全 |
| 中间通道 | 选择 `Fint=max(1,Fx//2)`，不是论文唯一指定比例 |
| 最浅 skip | 第 5 页正文称最初的低层 skip 不用于门控，但 Fig.1 仍画有门。默认按正文不门控；`gate_first_skip=True` 可显式启用该变体 |
| decoder 上采样 | 选择插值、拼接、双卷积；不另加转置卷积 |
| 多子门融合 | 拼接加点卷积是对论文文字描述的补全 |
| 初始化 | 论文要求初始近似放行；具体选 `psi.weight~N(0,0.001²)`、`psi.bias=logit(0.99)`。这些数值是补全，Sigmoid 不精确等于 1 |
| 深监督 | 论文明确使用；头位置、logits 插值、权重 `(1,.5,.25)` 和 eval 返回辅助头是补全 |
| Dice 细节 | 平方分母、类别平均、含背景和 batch 聚合是明确记录的选择 |

本实现优先依据论文正文和公式，不声称与作者仓库的每个工程细节相同。

## 8. 同一个 Forward 演示与 ONNX 导出

安装依赖后，从仓库根目录运行：

```bash
python -m pip install "torch>=2.7" onnx onnxscript onnxruntime numpy
python 13-AttentionUNet/model.py
```

`model.py` 自带 `if __name__ == "__main__":`：

1. 只创建一次模型和输入 `x`，设置 CPU、float32、`eval()`。
2. 直接执行 `outputs = model(x)`，打印三个输出的 Shape 和部分数值。
3. 将原来的 `model`、原来的 `x` 交给 `torch.onnx.export`。
4. ONNX checker 检查后，用 ONNX Runtime 对同一输入逐输出比较，打印最大绝对误差。

导出显式使用 `dynamo=False, opset_version=17`，权重放在单个 ONNX 文件内。输出名为 `logits`、`aux2_logits`、`aux3_logits`。没有重建模型、重新初始化、修改参数、切换 train/eval、额外 wrapper、替换池化或插值、关闭深监督或省略辅助头，也没有 `is_in_onnx_export()` 特殊计算分支。

选择兼容导出器的原因：本次 PyTorch 2.14.1、ONNX Script 0.7.2 环境中，`dynamo=True` 为无 bias 的 Conv3d 生成了不正确的 bias 形状，ONNX Runtime 无法执行。为让 2D/3D 使用同一导出方式，统一采用 `dynamo=False`。这是 PyTorch 已标记弃用的传统 TorchScript 导出路径，运行时可能出现弃用提示；它仍可用于本次经过数值验证的静态前向网络。将来改用新版导出器时，应重新做全部输出的数值核对。

默认在本目录生成 `attention_unet_2d.onnx`。把 `spatial_dims` 改成 `3` 后，演示和导出同步选择 3D 配置，生成 `attention_unet_3d.onnx`。

“一样”指**同一配置、参数、输入、forward 和全部输出的数值一致性**，比较容差为 `rtol=1e-4, atol=1e-5`。不同后端的浮点结果不承诺逐 bit 一致；ONNX 的算子分解、形状运算和常量折叠也可能导致 Netron 节点不能与 Python 语句一一对应，例如 eval 中的卷积与 BatchNorm 可能融合。不能把图中节点数量变化理解成模型换了一套 forward。

文件采用演示输入的固定 Shape，不声称支持任意动态尺寸。更改输入大小应重新导出。导出的网络只包含前向计算，不包含损失和反向传播。

如果自行将主程序改为 `deep_supervision=False`，同时将 `output_names` 改为单个名称，并把 Tensor 输出包装为单元素序列用于打印和逐输出比较。默认主程序已经将三个输出完整对齐。

## 9. 与经典方法的区别

| 方法 | 特征传递方式 | 特点 |
| --- | --- | --- |
| 经典 U-Net | encoder 特征直接拼接至 decoder | 保留细节，也传递背景响应 |
| Attention U-Net | 拼接前用粗尺度上下文生成空间权重，乘回 skip | 通过分割损失学习筛选位置，无需额外 ROI 标注 |
| 定位后分割的级联 | 先预测 ROI，再裁剪并交给后续网络 | 有额外定位流程，本文希望用门控减少此需求 |

与 Transformer 相比，本门不构造 `N×N` 位置相关矩阵。与通道注意力相比，默认权重随空间位置变化，并广播至全部通道。

## 10. 阅读心得

本文的变化集中在 skip 如何被使用：encoder 提供细节，粗尺度 decoder 提供上下文，门学习两者结合后的保留程度。

随机初始化时注意力近似均匀放行，是有意的初始化策略，不是已经学会定位器官。多子门也不保证自动分工为指定类别，需要结合验证集观察。

建议先单步跟踪 `gate3` 的 `x` 与 `g`，确认网格对齐，然后查看 `q_att -> alpha_coarse -> alpha -> x_hat`，最后回到 `torch.cat`，区分乘法门控与通道拼接的作用。

## 11. 文件结构与参考定位

| 文件 | 用途 |
| --- | --- |
| `README.md` | 论文笔记、公式对应、Shape 分析和实现差异 |
| `blocks.py` | 双卷积、加性门、多子门融合、上采样拼接 |
| `model.py` | 完整 2D/3D 数据流、主程序演示、ONNX 导出及数值核对 |
| `losses.py` | soft Dice 与深监督损失 |
| `.gitignore` | 排除生成的 ONNX 与 Python 缓存 |

ONNX 是脚本可重建的未训练演示网络，不将随机权重二进制文件提交到仓库。

参考定位：论文第 3 页 Fig.1，第 4 页 Fig.2 / 式 (1)(2)，第 5 页式 (3) 及门控、深监督说明，第 6 页实验细节；[PyTorch ONNX API](https://docs.pytorch.org/docs/stable/onnx.html) 说明导出参数。

## 12. 验证记录

以下为代码与数值验证，不包含真实数据集的训练或精度复现。

环境：Python 3.12、PyTorch 2.14.1+cpu、ONNX 1.23.2、ONNX Runtime 1.31.0。

| 检查 | 结果 |
| --- | --- |
| 主程序 2D：`[1,3,128,128]` | 三个输出全部匹配，最大绝对误差 `6.71e-8` |
| 主程序切换 3D：`[1,1,16,32,32]` | 三个输出全部匹配，最大绝对误差 `5.96e-8` |
| 2D 非整除尺寸、两个子门、启用浅层门 | 前向/反向通过；两个不同输入的全部 ONNX 输出匹配，最大绝对误差 `8.94e-8` |
| 3D 非整除尺寸、默认门配置 | 前向/反向通过；两个不同输入的全部 ONNX 输出匹配，最大绝对误差 `5.96e-8` |
| 导出前后的参数与 BN 状态 | 完全相同，模型保持 eval |
| 注意力梯度 | 有限差分 gradcheck 通过；反向时各门参数梯度存在且有限 |
| 二分类单通道、关闭深监督 | 2D/3D 的 Shape 与损失均通过检查 |
| Dice 语义检查 | 高置信正确预测接近零损失，反向错误预测接近一 |

默认脚本包含可重复执行的 ONNX checker 和逐输出数值比较；完整模型尚未在真实影像数据上训练。
