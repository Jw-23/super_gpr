# 增加训练样本数的学习曲线

本实验保持 80 维、20 个四维变量组、相同加性函数不变，
将训练样本数设为 128、256、512、1024，比较原独立参数核和共享 ARD 核。

## 在测试前固定的方案

- 使用新种子 5003、6007、7001。每个种子单独生成一个 1024 样本训练池和一个 2048 点测试集。
- 每个训练规模取同一个训练池的前 N 行，相同输入对应的噪声标签也保持不变。
- 训练输入、测试输入和训练噪声采用独立 RNG 流；测试集不随训练规模变化。
- 输入仍在 `[-1,1]^80` 独立均匀采样，标签为原 `additive_target`，训练噪声标准差 0.03。
- 测试标签为无噪声真值。Kernel、变量组、参数边界和初始化规则沿用上一轮对照。
- 原独立参数模型有 41 个超参数，共享 ARD 有 6 个；都使用 RBF 和 WhiteKernel。
- 模型结构预先固定，不根据测试集选择模型或修改设置。
- 每档都从相同初始超参数重新拟合，不使用前一档拟合结果做 warm start。
- `normalize_y=true`，`alpha=1e-10`，L-BFGS-B 最多 500 次迭代，梯度容差 `1e-5`、函数容差 `1e-8`。
- 每次拟合限定一个 Eigen 线程。保留全部结果及收敛状态，不预设准确率必须单调提升。
- 记录 RMSE、NRMSE、R²、MAE、绝对误差 95 分位与最大值、训练误差、噪声估计、拟合和预测时间、全部超参数。
- 同时记录训练池和测试集校验值，检查不同 N 和模型使用的是同一训练池与测试集。

这里测试的是增加数据后对该加性函数的预测效果，不是对所有高维函数的样本复杂度保证。
共享 ARD 仍依赖各组内部坐标位置具有可比含义。

## 复现

```bash
cmake --preset release -DCMAKE_CXX_STANDARD=14 -DSUPER_GPR_BUILD_EXPERIMENTS=ON
cmake --build --preset release --target sample_learning_curve
./build/release/bin/sample_learning_curve > experiments/sample_learning_curve_results.csv
python3 experiments/summarize_learning_curve.py experiments/sample_learning_curve_results.csv \
  --output experiments/SAMPLE_LEARNING_CURVE_RESULTS.md \
  --plot experiments/SAMPLE_LEARNING_CURVE.png
```

生成图像时需要 Matplotlib；仅校验和汇总 CSV 不需要额外 Python 依赖。
快速检查可使用 `sample_learning_curve --quick`，仅生成开发种子 101 的小规模数据。
也可以使用 `--seed=5003` 单独运行一个种子，然后将三个 CSV 路径一并传给汇总脚本。

本次正式运行将三个种子分别运行，每个进程单线程，三个进程并行。
相应的拟合时间是并发环境中的实际耗时，只用于描述成本，不作严格的跨模型性能基准。

## 实测结果

执行模型代码版本为 `1760d6a`，使用 Apple M4、Apple Clang 21、C++14 Release、
Eigen 3.4.0 和项目内置 LBFGS++。完整原始记录见
[sample_learning_curve_results.csv](sample_learning_curve_results.csv)，逐种子统计见
[SAMPLE_LEARNING_CURVE_RESULTS.md](SAMPLE_LEARNING_CURVE_RESULTS.md)。

以下为三个种子的平均结果；不同训练规模使用固定的测试集，且训练样本和噪声标签嵌套一致：

| 训练样本数 | 原独立参数模型 R² | 共享 ARD R² | 原模型 RMSE | 共享 ARD RMSE |
| --- | --- | --- | --- | --- |
| 128 | 0.2328 | 0.5140 | 0.7288 | 0.5796 |
| 256 | 0.4346 | 0.6823 | 0.6252 | 0.4680 |
| 512 | 0.7458 | 0.9279 | 0.4195 | 0.2233 |
| 1024 | 0.9482 | 0.9864 | 0.1894 | 0.0969 |

![Nested training sample learning curves](SAMPLE_LEARNING_CURVE.png)

这组加性函数中，增加样本明显提高了留出集预测准确性。
两个模型、三个种子的 18 次相邻训练规模比较中，RMSE 全部下降；24 次拟合全部达到收敛条件。
共享 ARD 在 1024 个样本时的 R² 为 `0.9864 ± 0.0004`，RMSE 为 `0.0969 ± 0.0023`，
误差较 128 样本时降低约 83%。这仍然是对整体输出的预测，不使用子函数标签。

1024 样本共享 ARD 的绝对误差 95 分位数，在各个种子上的平均值约为 0.1892；
因此高 R² 也不意味着每个点都与真值完全一致。

增加数据的代价也很明确：1024 样本时，原模型平均拟合约 110.2 秒，共享 ARD 约 16.9 秒；
512 样本时分别约为 21.7 秒和 6.5 秒。预测时间另外保留在原始记录中。
这些时间是在三个进程并行的条件下记录的，不等于严格的单进程性能基准。

从本实验看，数据覆盖不足确实是之前高维预测不准的重要因素，
合适的共享核结构还能在相同样本数下进一步提高预测精度。
这不构成对任意高维函数的保证，特别是实际函数可能不满足当前加性分组假设。
优化过程中也没有强制准确率单调改善；这里的单调性来自实际测试结果。

图像通过 Matplotlib 3.11.2 生成；如使用现有参考测试环境，可用
`build/reference-env/bin/python` 执行带 `--plot` 的汇总命令。
