# Additive GP 高维预测实验

此实验直接运行本项目的 C++ GPR，不以 Python 模型替代待验证实现。
目标是区分两个问题：

1. 在相同高维数据上，已知变量组的加性核是否优于全维核？
2. 固定训练样本量时，维度增加后，加性模型的绝对预测精度是否仍能保持？

高维并不会因为可加就自动比低维更容易。变量组数量增加后，未知分量和待估计超参数也增加。

## 复现

```bash
cmake --preset release -DCMAKE_CXX_STANDARD=14 -DSUPER_GPR_BUILD_EXPERIMENTS=ON
cmake --build --preset release --target additive_experiment
./build/release/bin/additive_experiment > experiments/additive_results.csv
python3 experiments/summarize.py --output experiments/RESULTS.md
```

快速检查：

```bash
./build/release/bin/additive_experiment --quick > build/additive-quick.csv
python3 experiments/summarize.py build/additive-quick.csv
```

正式实验的原始记录在 [additive_results.csv](additive_results.csv)，
统计表在 [RESULTS.md](RESULTS.md)。汇总脚本检查模型配对、超参数数量、参数有限性、
最终 LML 不低于初始 LML，以及 NRMSE 与 R² 的一致性。

## 数据与模型

输入在 `[-1, 1]^D` 独立均匀采样；D 取 4、20、80，每组 4 个变量。
加性数据的组内函数为：

```text
g_j(a,b,c,d) = w_j * [sin(2.5a + 1.2b) + 0.6 cos(2c - 0.7d) + 0.5ac]
w_j = 0.6 + 0.2 * (j mod 5)
f(x) = sum_j g_j(x[S_j]) / sqrt(number_of_groups)
```

每组包括组内非线性及交互，不只是一维独立效应。除以组数的平方根使不同 D 的信号方差
保持相近量级，使用 NRMSE 和 R² 比较，而不是通过目标幅度改变制造优势。
4 维时只有一组，加性模型与普通各向同性 RBF GP 应基本一致，这是一个对照检查。

另设非加性对照：64 个稠密随机 Fourier 特征之和，频率向量的每个维度服从
`Normal(0, 2/sqrt(D))`，相位服从 `Uniform(0, 2*pi)`。每个特征同时依赖全部变量，
包含跨组交互，不能被上述纯加性核完整表示。训练集、测试集使用同一组频率和相位。
该对照在 20 和 80 维运行。

所有模型在同一个数据划分上训练、预测：

| 模型 | 信号 Kernel | 优化参数数量 |
| --- | --- | --- |
| Additive GP | 各 4 维组的 `ConstantKernel * RBF` 之和 | `2*(D/4)+1` |
| Full RBF GP | 全维 `ConstantKernel * RBF` | 3 |
| Full ARD GP | 全维 `ConstantKernel * RBF(ARD)` | `D+2` |

每个模型额外包含一个可训练的 `WhiteKernel`。Additive GP 的各组幅度和长度尺度独立，
通过完整输出的同一个 LML 联合优化，不向任何模型提供分量标签。
分组是预先给定的真实结构，未进行自动分组学习；这是 Additive GP 的结构先验优势。

## 固定实验设置

- 加性数据：训练样本数 128 和 256；测试样本数 1024。
- 非加性对照：训练样本数 256；测试样本数 1024。
- 三个随机种子：11、29、47。每个种子的所有模型共享相同输入、标签和噪声。
- 训练目标加入标准差为 0.03 的独立高斯噪声；测试标签为无噪声真实函数值。
- 各模型都启用 `normalize_y`，训练协方差的 `alpha=1e-10`。
- 所有 LML 优化使用同一个 L-BFGS-B 实现，最多 500 次迭代、梯度容差 `1e-5`、
  函数容差 `1e-8`，单个起点，无额外重启。
- 初始总信号方差均为 1。加性核分配初始每组方差 `1/M`，每组长度尺度 0.7；
  全维核长度尺度初值为 `0.7*sqrt(M)`，避免高维初始相关性直接退化为零。
- 信号方差边界 `[1e-5, 100]`；组内长度尺度 `[0.1, 20]`；全维长度尺度 `[0.1, 100]`。
- 噪声方差初值 0.001，边界 `[1e-8, 0.5]`。测试标签不参与初值、边界或超参数选择。
- CSV 保留每次运行的 RMSE、NRMSE、R²、时间、迭代数、函数评估次数、收敛标志、
  初始和最终 LML，以及全部优化后的物理超参数。

`physical_parameters` 以分号分隔：加性模型为每组 `[variance, length_scale]` 后接噪声方差；
各向同性模型为 `[variance, length_scale, noise]`；ARD 模型为
`[variance, length_scale_0, ..., length_scale_(D-1), noise]`。
幅度参数表示先验方差，不是每个样本上分量贡献的固定权重。

## 本次运行的观察

正式结果共 72 次拟合，每个设置使用 3 个种子，全部达到优化器收敛条件。
下表是 256 个训练样本时的加性函数 R² 均值：

| 总维度 | Additive GP | Full RBF GP | Full ARD GP |
| --- | --- | --- | --- |
| 4 | 0.995 | 0.995 | 0.997 |
| 20 | 0.951 | 0.511 | 0.597 |
| 80 | 0.444 | 0.381 | 0.117 |

20 维时，加性结构带来了较大的预测改善。80 维时，加性模型的均值改善较小，
重复间波动较大（Additive GP 的 R² 样本标准差约 0.108），不能据此断言普遍优势。
80 维仅有 128 个训练样本时，加性模型平均 R² 为 0.183，低于普通 RBF 的 0.263。
因此，相同数据量下高维仍更难，独立组超参数也需要足够数据支持。

在非加性对照中，20 维 Additive GP / Full RBF GP 的平均 R² 为 0.425 / 0.644，
80 维为 0.181 / 0.352。纯加性假设遗漏跨组交互时会损失精度。

拟合成本也增加：本次机器上，80 维、256 样本的加性模型平均拟合约 7.1 秒，
普通 RBF 约 0.05 秒。加性核使用 41 个优化参数，普通 RBF 为 3 个；
这组实验验证的是预测效果及适用条件，没有显示训练速度优势。

## 如何解释结果

应首先比较相同 D、相同训练样本数和种子上的模型，再看各模型跨维度的精度变化。
三次重复只是小规模合成实验，不能证明对任意高维黑盒函数都有效。

加性先验正确且数据足够时，可以通过低维子核减少需要学习的交互结构；
数据过少时，更多组的独立幅度与长度尺度仍可能难以估计。若存在明显跨组交互，
纯加性模型存在结构偏差，可通过 `additive + global_kernel + WhiteKernel` 扩展。

保留未收敛的运行并报告状态。LML 非凸，单一起点不保证最佳局部解；
训练 LML 改善也不保证留出集误差改善。时间仅用于记录当前机器上的成本，不是跨平台性能保证。

实验设计参考 [Kandasamy et al., ICML 2015](https://proceedings.mlr.press/v37/kandasamy15.html)
的已知低维加性分组模型；此处测量预测误差，不进行贝叶斯优化或声称验证其 regret 理论。
