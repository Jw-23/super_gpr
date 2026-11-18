# Additive GP 高维预测实验

此实验直接运行本项目的 C++ GPR，不以 Python 模型替代待验证实现。
目标是区分两个问题：

1. 在相同高维数据上，已知变量组的加性核是否优于全维核？
2. 固定训练样本量时，维度增加后，加性模型的绝对预测精度是否仍能保持？

高维并不会因为可加就自动比低维更容易。变量组数量增加后，未知分量和待估计超参数也增加。

针对高维预测不准的后续诊断、共享 ARD 和收缩先验改进，见
[ADDITIVE_DIAGNOSIS.md](ADDITIVE_DIAGNOSIS.md)。该实验使用新种子，并在训练集内选择模型，
不替换本文件中原实验的数据和结果。
增加到 512、1024 个训练样本的固定测试集学习曲线见
[SAMPLE_LEARNING_CURVE.md](SAMPLE_LEARNING_CURVE.md)。
在完全相同数据上与普通全维 Matérn 2.5 GP 的对照见
[MATERN_BASELINE_COMPARISON.md](MATERN_BASELINE_COMPARISON.md)。
七类新函数（含周期、尖点、多尺度与跨组交互）的配对对照见
[FUNCTION_SUITE.md](FUNCTION_SUITE.md)。
旋转、病态尺度、狭窄谷地和多峰结构的 12 类 BBOB 黑盒函数实验见
[BBOB_PATHOLOGICAL.md](BBOB_PATHOLOGICAL.md)，同时检查全域与最优点附近的预测误差。

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

## Matérn 与 RBF 的配对对照

可通过 `--kernel=rbf|matern15|matern25` 选择核族；不传参数时仍运行 RBF。
Matérn 的 nu 固定为 1.5 或 2.5，幅度、长度尺度和噪声通过相同的 LML 优化。
三个核族使用相同的数据生成函数、输入、噪声、随机种子、初始化规则、参数边界和优化预算。
各组仍通过核相加构成同一个联合 GP。

```bash
./build/release/bin/additive_experiment --kernel=matern15 > experiments/matern15_results.csv
./build/release/bin/additive_experiment --kernel=matern25 > experiments/matern25_results.csv
python3 experiments/summarize.py experiments/additive_results.csv \
  experiments/matern15_results.csv experiments/matern25_results.csv \
  --output experiments/MATERN_RESULTS.md
```

新增 144 次 Matérn 拟合，与原有 72 次 RBF 记录合并，共 216 次拟合，全部达到优化器收敛条件。
完整结果见 [MATERN_RESULTS.md](MATERN_RESULTS.md)，原始记录见
[matern15_results.csv](matern15_results.csv) 和 [matern25_results.csv](matern25_results.csv)。
汇总脚本兼容旧 RBF CSV：缺少 `kernel` 列时按 RBF 读取，旧模型名 `full_rbf` 对应
新输出中的 `full_iso`。跨核比较时检查三个核族的数据集键完全一致。

4 维、使用 ARD 时的 RMSE 均值如下，测试集每次包含 1024 个无噪声真值，重复 3 个种子：

| 训练样本数 | RBF | Matérn 1.5 | Matérn 2.5 |
| --- | --- | --- | --- |
| 128 | 0.0395 | 0.0464 | 0.0419 |
| 256 | 0.0245 | 0.0306 | 0.0275 |

256 个样本时，三个核的平均 R² 分别为 0.9975、0.9960、0.9968。
在这组低维光滑函数上，三个核都能接近真值，RBF 略好，Matérn 2.5 更接近 RBF 的表现。

对于加性结构，R² 均值如下：

| 维度 | 训练样本数 | RBF | Matérn 1.5 | Matérn 2.5 |
| --- | --- | --- | --- | --- |
| 20 | 256 | 0.9505 | 0.8980 | 0.9249 |
| 80 | 128 | 0.1825 | 0.2553 | 0.2271 |
| 80 | 256 | 0.4439 | 0.4569 | 0.4505 |

Matérn 1.5 改善了本次 80 维模型的平均表现，但 256 样本时改善较小，
仍没有达到低维模型的预测精度。20 维加性函数则以 RBF 更好。
非加性对照的全部结果也保留在报告中。

本实验的函数由正弦、余弦、多项式或 Fourier 特征组成，均很光滑，
没有覆盖粗糙函数或不连续函数，不能据此断言 RBF 在所有任务中优于 Matérn。
Matérn 的 nu 控制光滑性，较小的 nu 允许较粗糙的函数；
1.5 和 2.5 的定义及固定 nu 的约定见
[scikit-learn Matérn 文档](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html)。
三个种子的重复不足以支持普遍优劣结论，且收敛不保证全局最优。
不同核族的拟合时间采集于不同时段，仅作运行成本记录。

库内使用方式：

```cpp
auto kernel = ConstantKernel({1.0, 1e-5, 100.0})
            * Matern({0.7, 0.1, 20.0}, 2.5)
            + WhiteKernel({0.001, 1e-8, 0.5});
GaussianProcessRegressor<decltype(kernel)> gp(kernel, options);
```

需要 ARD 时，为 `Matern` 传入每维一个 `Parameter` 的向量；需要加性模型时，
将 `ConstantKernel * Matern` 作为 `make_additive_kernel` 的子核原型。

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
