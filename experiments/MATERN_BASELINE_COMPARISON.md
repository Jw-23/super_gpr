# 80 维加性 GP 与普通 Matérn 2.5 GP

## 在运行基线前固定的设置

复用上一轮学习曲线的训练池、固定测试集和噪声标签，见
[SAMPLE_LEARNING_CURVE.md](SAMPLE_LEARNING_CURVE.md)。
三个种子为 5003、6007、7001；训练样本数为 128、256、512、1024，
每个种子有 2048 个无噪声测试点，训练噪声标准差为 0.03。
所有数据校验值必须与已有加性模型记录一致。

普通模型使用整个 80 维输入上的各向同性核：

```cpp
ConstantKernel({1.0, 1e-5, 100.0})
    * Matern({1.0, 0.1, 100.0}, 2.5)
    + WhiteKernel({0.001, 1e-8, 0.5})
```

它只有幅度、一个长度尺度和噪声方差，共三个可训练参数；
不使用变量分组、加性结构、ARD、参数共享、先验或重启。
ν 固定为 2.5。长度尺度初值 1.0，各训练规模均从相同初值重新训练。
普通模型与加性模型都做目标标准化，并通过 LML 优化超参数。
沿用 `normalize_y=true`、`alpha=1e-10`、L-BFGS-B 最多 500 次迭代、
梯度容差 `1e-5` 和函数容差 `1e-8`。
不根据测试标签调整模型、初值或边界，保留所有拟合结果和收敛状态。

加性模型复用已提交的两组结果：原独立参数 RBF（41 个参数）和改进后的
共享 ARD RBF（6 个参数）。这是一组实际模型方案的对比，
同时包含加性结构与核族差异，并不单独隔离某一个因素。
测试函数本身是按已知四维组相加的函数；结论适用于这个任务，
不能推断有强跨组交互的任意高维函数也有相同排序。

## 复现

```bash
cmake --preset release -DCMAKE_CXX_STANDARD=14 -DSUPER_GPR_BUILD_EXPERIMENTS=ON
cmake --build --preset release --target sample_learning_curve
./build/release/bin/sample_learning_curve --plain-matern25 > experiments/plain_matern25_learning_curve_results.csv
python3 experiments/summarize_learning_curve.py \
  experiments/sample_learning_curve_results.csv \
  experiments/plain_matern25_learning_curve_results.csv \
  --output experiments/MATERN_BASELINE_COMPARISON_RESULTS.md \
  --plot experiments/MATERN_BASELINE_COMPARISON.png
```

生成图像需要 Matplotlib；仅校验和汇总 CSV 不需要额外 Python 依赖。
也可通过 `--seed=5003` 等参数分别运行三个种子，每个进程单线程。
不同批次的耗时受机器负载影响，本次主要比较预测精度。
