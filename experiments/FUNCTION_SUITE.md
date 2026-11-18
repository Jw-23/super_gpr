# 多函数上的 80 维 GP 配对对照

## 在正式运行前固定的方案

延续上一轮三个模型的设置：原独立参数加性 RBF（41 个参数）、共享 ARD 加性 RBF
（6 个参数）和普通全维各向同性 Matérn 2.5（3 个参数）。核、边界、初值、目标标准化、
LML 优化设置不变，具体见 [MATERN_BASELINE_COMPARISON.md](MATERN_BASELINE_COMPARISON.md)。
所有模型都仅使用整体标量标签。加性模型继续固定使用连续四维分组，不重新学习分组。
这里同时比较结构、核族与共享方案，不单独隔离加性这一因素。

- 输入独立均匀分布于 `[-1,1]^80`，种子固定为 11003、12007、13001。
- 每个种子使用一个 512 行训练输入池、一个固定的 2048 点测试集，训练、测试、噪声 RNG 独立。
- 三档训练规模为 128、256、512，取同一训练池前 N 行，噪声标签嵌套一致。
- 不同函数复用相同输入和噪声随机数；每个函数的三个模型使用完全相同的标签。
- 训练噪声标准差 0.03，测试标签无噪声。函数之间输出尺度不同，因此不跨函数平均原始 RMSE。
- 每档冷启动，不增加重启、不使用先验、不按测试结果修改模型、函数或选择最好的种子。
- 七个函数 × 三个种子 × 三档样本数 × 三个模型，共 189 次正式拟合。
- 每个进程使用一个 Eigen 线程，三个种子进程并行；保留失败/未收敛状态及所有误差。
- 校验每个函数/种子的训练池和测试集在所有模型、样本规模下保持一致。
- 汇总 R²、RMSE、NRMSE、MAE、绝对误差 95 分位、训练误差、噪声估计、耗时、LML、优化状态。

## 函数定义

这些是明确定义的合成函数，用于覆盖不同结构，不冒充标准公开基准。
令 `M=20`，第 j 个连续四维组为 `(a,b,c,d)`；除另行说明外，
目标是各组子函数之和除以 `sqrt(M)`。

| 名称 | 子函数或整体函数 | 检查的问题 |
| --- | --- | --- |
| polynomial | `0.8a + 0.6(b²-1/3) + 0.5cd + 0.4d³` | 光滑低阶多项式，组内存在交互 |
| periodic | `sin(2πa) + 0.6cos(π(b+c)) + 0.3sin(πd)` | 较快周期变化，当前 RBF/Matérn 都没有显式周期核 |
| nonsmooth | `abs(a+0.5b) + 0.6max(c,0) + 0.4abs(d)` | 带尖点、不可处处微分的加性目标 |
| multiscale | `w[sin(s(2.5a+1.2b)) + 0.6cos((3.6-s)(2c-0.7d)) + 0.5ac]`，`s=0.6+0.6(j mod 5)`，`w=0.6+0.2(j mod 5)` | 各组尺度不同，挑战共享超参数假设 |
| cross_pair | `3/sqrt(40) * sum(i=0..39) x[i]x[i+40]`，这是整体函数 | 所有乘积都跨越固定分组，加性模型无法精确表示 |
| linear | `sum(i=0..79) w[i]x[i] / sqrt(sum(w²)/3)`，`w[i]=0.5+0.2(i mod 7)`，这是整体函数 | 简单线性控制函数 |
| dense_fourier | 64 个全维随机 Fourier 特征之和，乘 `sqrt(2/64)` | 稠密跨组交互 |

Fourier 函数固定使用 `interaction_target(x,424242)`，因此不同数据种子采样同一个函数。
其频率独立服从 `N(0,(2/sqrt(80))²)`，相位均匀分布于 `[0,2π]`。
具体实现见 [problems.hpp](problems.hpp)。函数族、参数和测试方案均在正式测试前提交。

当前加性分组只对前四类及线性函数成立。后两类交互函数的结果用于显示结构失配，
不能将已知正确分组的结果推广为对未知分组、真实高维任务的普遍保证。
比较预测精度时也要同时看绝对 R²：相对更好不意味着已经准确。

## 复现

```bash
cmake --preset release -DCMAKE_CXX_STANDARD=14 -DSUPER_GPR_BUILD_EXPERIMENTS=ON
cmake --build --preset release --target sample_learning_curve
./build/release/bin/sample_learning_curve --suite > experiments/function_suite_results.csv
python3 experiments/summarize_function_suite.py experiments/function_suite_results.csv \
  --output experiments/FUNCTION_SUITE_RESULTS.md --plot experiments/FUNCTION_SUITE.png
```

正式运行分别传入 `--seed=11003`、`--seed=12007`、`--seed=13001` 并行执行，之后合并 CSV。
开发检查使用 `--suite --quick`（20 维、32/64 个训练点、128 个测试点、种子 101），
开发种子不混入正式结果。绘图需要 Matplotlib，其余数据校验与汇总只使用 Python 标准库。
旧的单函数 CSV 保持兼容；新记录增加 `function` 列。
