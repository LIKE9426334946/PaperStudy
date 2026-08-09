# LeNet-5 (LeCun et al., 1998) PyTorch 复现

本目录复现论文 **Gradient-Based Learning Applied to Document Recognition** 中的 LeNet-5/MNIST 核心实验。实现保留论文版 LeNet-5 的 60,000 个可训练参数、C3 稀疏连接、可训练下采样、缩放 tanh、84 维固定 RBF 输出及随机对角 Levenberg-Marquardt 训练模式。

## 论文信息

- 作者：Yann LeCun、Léon Bottou、Yoshua Bengio、Patrick Haffner
- 期刊：Proceedings of the IEEE, 86(11), 1998
- 论文：[Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
- 完整方法分析：[paper_analysis.md](paper_analysis.md)

## 目录结构

~~~text
LeNet5_1998/
├── configs/        # 严格论文配置与快速验证配置
├── datasets/       # MNIST、论文归一化、虚拟形变集与固定采样顺序
├── models/         # LeNet-5、C3、S2/S4、RBF 与损失
├── optim/          # 随机对角 Levenberg-Marquardt
├── utils/          # 配置、指标、复现与检查点工具
├── tests/          # 结构、公式、数据和优化器测试
├── checkpoints/    # 训练权重（不提交）
├── results/        # JSONL 指标和测试结果（不提交）
├── train.py
└── test.py
~~~

## 环境配置

建议使用 Python 3.10-3.12：

~~~bash
cd LeNet5_1998
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

验证安装：

~~~bash
pytest
python train.py --smoke-test
~~~

## 数据集

第一次训练时 torchvision 会自动下载 MNIST 到 data/。若服务器不能联网，可先在其他机器准备 MNIST，再把文件复制到 data/MNIST/raw/。

论文配置会将 28x28 图像补成 32x32，并把背景映射为 -0.1、前景映射为 1.175。

## 训练

### 论文设置

~~~bash
python train.py --config configs/paper.yaml
~~~

该设置使用 batch size 1、20 轮、600,000 虚拟增强样本、每轮 60,000 次样本呈现、每轮 500 样本精确对角 Gauss-Newton 估计。纯 PyTorch Jacobian 实现重视可核验性而不是速度，完整运行可能很慢。

中断后继续：

~~~bash
python train.py \
  --config configs/paper.yaml \
  --resume checkpoints/paper/latest.pth
~~~

### 快速现代验证

~~~bash
python train.py --config configs/fast.yaml
~~~

fast.yaml 使用 mini-batch SGD，只用于较快地确认数据、模型、保存和评估流程；它的结果不能标注为论文严格复现结果。

## 测试

~~~bash
python test.py --checkpoint checkpoints/paper/best.pth
~~~

输出：

- results/evaluation/test_metrics.json
- results/evaluation/confusion_matrix.csv

## 原论文结果与复现记录

| 设置 | 原论文测试错误率 | 本仓库完整运行 |
| --- | ---: | --- |
| LeNet-5，无随机形变 | 0.95% | 待在目标训练硬件上执行 |
| LeNet-5，随机形变 | 约 0.8%（82/10,000） | 待在目标训练硬件上执行 |

代码提交前执行的是不依赖下载数据的结构测试、公式测试、前后向与优化器烟雾测试。完整 20 轮训练耗时取决于硬件，运行后的逐轮结果会写入 results/paper/metrics.jsonl，不会在没有实际运行时预填数字。

## 关键复现差异

1. 论文未给出仿射形变的具体范围，本工程把采用的范围完整放入 YAML。
2. 原始 84 维 RBF 字符码只以论文图片提供；本工程以明确的 12x7 模板转录其设计，便于检查和替换。
3. PyTorch 中精确曲率通过 F6 Jacobian 计算，数学上对应式 (8) 的对角 Gauss-Newton，但比论文的专用二阶反向传播实现慢。
4. fast.yaml 是工程验证配置，不是论文结果配置。

更详细的公式、代码映射、训练流程和限制说明见 [paper_analysis.md](paper_analysis.md)。

## 常见问题

### 为什么预测使用 argmin？

输出不是十类 logits，而是 F6 到十个 RBF 原型的平方欧氏距离。距离越小，类别越匹配。

### 为什么不用 MaxPool 和 CrossEntropyLoss？

那是常见的现代教学版 LeNet，不是论文中的 LeNet-5。论文的 S2/S4 有可训练 scale/bias，输出是固定 RBF 惩罚，并使用式 (8) 或式 (9) 的损失。

### 为什么严格配置很慢？

论文逐样本更新，并在每轮前估计各参数的二阶尺度。原文在当时的专用实现和硬件上也需要约 2-3 个 CPU 日。此实现优先保证公式与代码可核验。
