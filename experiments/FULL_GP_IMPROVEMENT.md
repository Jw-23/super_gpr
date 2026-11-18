# 全维 GP 的改进：新 BBOB 实例与训练内选择

## 正式测试前固定的方案

目标是在相同训练观测预算下，检查普通全维 GP 能否提高预测精度。
继续使用 [BBOB 病态函数实验](BBOB_PATHOLOGICAL.md) 的 12 个函数和 IOH 0.3.22，
正式测试改用未使用过的实例 **4、5、6**，数据种子分别为 **31013、32003、33013**。
三个重复同时改变函数实例和输入采样，不能当作对任意函数的统计保证。

每个实例使用 512 个无噪声训练点、2048 个独立均匀全域测试点、512 个最优点附近测试点。
原坐标范围 `[-5,5]^80`，模型输入除以 5；局部测试为 `clip(xopt+N(0,0.1²I),-5,5)`。
模型只接收输入和总输出，最优点仅用于评估器构造局部测试集。
模型不使用分组、加性分解、旋转矩阵、函数特定公式或测试集调参。

五个候选方案：

| 名称 | 改动 | 超参数数目 |
| --- | --- | --- |
| baseline | 原始 `C * Matérn(ν=2.5) + White`，单起点 | 3 |
| multistart | 相同核，三个确定性起点，按训练 LML 选择最优起点 | 3 |
| quadratic | `C * Matérn(ν=2.5) + Cq * (1 + 3 x·x'/D)² + White`，三个起点 | 4 |
| asinh_matern | 训练目标做 asinh 变换，使用 multistart 核 | 3 |
| asinh_quadratic | 训练目标做 asinh 变换，使用 quadratic 核 | 4 |

二次核通过现有 C++14 模板组合 `ConstantKernel`、`DotProduct` 和乘法实现，
包含所有坐标间的二阶交互，不要求固定四维分组。
二次多项式核的性质可参考 [GPML 第四章](https://gaussianprocess.org/gpml/chapters/RW4.pdf)。
重复启动优化用于降低落入较差局部最优的风险；
[scikit-learn 文档](https://scikit-learn.org/stable/modules/gaussian_process.html) 也支持通过重启优化 LML。
这里的起点是预先指定的，不是 scikit-learn 的随机对数均匀重启。

所有模型目标标准化、`alpha=1e-10`，L-BFGS-B 最多 500 次迭代、梯度容差 `1e-5`、
函数容差 `1e-8`。幅度初值 1、边界 `[1e-5,100]`，长度尺度边界 `[0.1,100]`，
噪声方差边界 `[1e-8,0.5]`。二次项幅度初值 1、边界 `[1e-5,100]`。
三个起点的 `(长度尺度,噪声方差)` 固定为 `(1,0.001)`、`(sqrt(D/3),0.05)`、
`(2sqrt(D/3),0.2)`；其余参数不变。原始起点保留，因此同一核、同一输出变换下，
选定的训练 LML 不应低于原始起点。

### 输出变换与预测

每次拟合只用当前训练折的标签计算中心 `c`（排序后下标 `N/2` 的值）和尺度
`s=Q75-Q25`（排序下标分别为 `3N/4` 和 `N/4`）。尺度为零时使用训练标准差，
若仍为零则使用 1。变换为 `z=asinh((y-c)/s)`，适用于含负值的标签，压缩大幅值尾部。
最终全量训练时重新用全部训练标签计算变换参数。

输出预测使用反变换的条件期望，若变换空间预测为 `Z~N(mu,v)`，则
`E[y]=c+s*exp(v/2)*sinh(mu)`。
这里 `v` 使用现有 GP 的预测方差（包括 WhiteKernel），不把 `sinh(mu)` 误当成原尺度均值。
所有 CV 和测试误差都在原始输出单位上计算。
没有跨变换比较 LML；每个候选只在自己的三个起点内比较 LML。

### 训练内选择与留出评估

对 512 个训练样本按固定种子打乱，做三折交叉验证。
每个候选的目标变换、标准化和超参数优化均只使用该折的训练部分。
按三个验证折合并后的原尺度平方误差选择候选（完全相同时按表中顺序）。
选择在访问全域和局部测试预测前锁定，然后全部五个候选在 512 个训练样本上重新拟合，
同时保留每个候选和训练内选择方案的测试结果。测试集不参与模型选择。

共 `12 函数 × 3 实例 × 5 候选 × (3 CV 折 + 1 全量拟合) = 720` 次拟合，
含 1872 次 L-BFGS-B 优化启动。记录全部状态，包括达到迭代上限或线搜索失败的启动；
统计最优 LML 对应启动是否收敛，而不是将整个模型选择过程概括为一个优化器状态。
训练观测数相同，但改进方案包含更多计算，记录 CV 与最终拟合/预测耗时供参考。

正式模型、边界、变换、选择规则和新实例在测试前提交。
快速开发检查使用旧开发数据（20 维、64 样本、实例 1、种子 101），不混入正式结果。
全域与局部指标沿用上一轮定义；局部归一化误差的分母是全域真值标准差。
这轮仍是从全域训练数据向局部区域泛化，局部误差未改善时不能声称解决了最优区域拟合问题。

## 复现

```bash
python3 experiments/generate_bbob_data.py --fresh --output build/bbob-fresh
cmake --preset release -DCMAKE_CXX_STANDARD=14 -DSUPER_GPR_BUILD_EXPERIMENTS=ON
cmake --build --preset release --target full_gp_improvement
./build/release/bin/full_gp_improvement build/bbob-fresh/run_4 > build/full-improvement-4.csv
./build/release/bin/full_gp_improvement build/bbob-fresh/run_5 > build/full-improvement-5.csv
./build/release/bin/full_gp_improvement build/bbob-fresh/run_6 > build/full-improvement-6.csv
python3 experiments/summarize_full_improvement.py build/full-improvement-4.csv \
  build/full-improvement-5.csv build/full-improvement-6.csv \
  --output experiments/FULL_GP_IMPROVEMENT_RESULTS.md --plot experiments/FULL_GP_IMPROVEMENT.png
```

依赖版本同上一轮：IOH 0.3.22、NumPy 2.2.6、Matplotlib 3.11.2。
三个实例进程可并行，每个进程固定一个 Eigen 线程；耗时不是严格单进程性能基准。
