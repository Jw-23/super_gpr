# 80 维病态黑盒函数拟合实验

## 正式运行前固定的方案

使用 IOHexperimenter **0.3.22** 提供的 BBOB 实现，完整保留函数实例的平移、旋转、
非线性变换与多峰结构，不手工重写简化公式。
BBOB 的函数分类和定义见 [COCO 官方说明](https://coco-platform.org/testsuites/bbob/overview.html)，
IOH 接口见 [get_problem 文档](https://iohprofiler.github.io/IOHexperimenter/api/ioh.get_problem.html)。
这里使用标准 BBOB 函数的 80 维实例；不是采用块旋转的 `bbob-largescale` 函数族。
这是基于这些函数的回归实验，结果不是官方 BBOB 优化器竞赛分数。

| ID | 函数 | 主要困难 |
| --- | --- | --- |
| f7 | Step-Ellipsoid | 旋转、台阶和平台、非连续变化 |
| f9 | Rosenbrock rotated | 旋转的狭窄弯曲谷地、变量耦合 |
| f10 | Ellipsoid rotated | 约百万量级条件数、旋转与非线性不规则变换 |
| f11 | Discus | 一个高度敏感方向，尺度悬殊 |
| f12 | Bent Cigar | 高条件数、窄谷与非线性不对称变换 |
| f13 | Sharp Ridge | 尖锐且非光滑的脊，方向耦合 |
| f15 | Rastrigin rotated | 旋转、非线性变换与大量局部极小值 |
| f16 | Weierstrass | 多尺度细密振荡、多峰 |
| f18 | Schaffers F7, condition 1000 | 病态尺度、振荡和变量耦合 |
| f21 | Gallagher 101 peaks | 不规则分布与不同形状的峰 |
| f23 | Katsuura | 非光滑的细尺度结构、大量局部极小值 |
| f24 | Lunacek bi-Rastrigin | 双漏斗与多峰结构 |

上述描述来自官方函数定义；条件数描述针对函数构造，不是 GP 协方差矩阵的条件数。

- 所有正式实例为 **80 维**。使用实例 1/2/3，分别配对数据种子 21001/22003/23003。
  因此三个重复同时包含实例变换和输入采样的差异，不是对同一个固定函数只更换数据种子。
- 每个实例生成 512 个训练点，三档规模取前 128/256/512 行；固定 2048 个全域测试点。
- 训练与全域测试输入在 BBOB 原坐标 `[-5,5]^80` 独立均匀采样，使用独立 RNG。
- GP 接收的输入统一除以 5，映射到 `[-1,1]^80`；输出保持原始函数值，只使用模型已有的训练目标标准化。
- 使用无噪声函数原始输出，不叠加人工噪声；三个模型仍统一保留 WhiteKernel 并优化其参数。
- 每个实例另生成固定 512 个局部测试点：`clip(xopt + Normal(0,0.1² I),-5,5)`。
  `xopt` 只供评估器构造局部测试集、核对函数实例使用，不给模型、不加入训练集。
  这衡量从全域数据预测最优点附近的能力，是分布变化下的诊断，不是局部训练后的插值测试。
- 模型与上一轮一致：独立参数加性 RBF（41 参数）、共享 ARD 加性 RBF（6 参数）、
  普通全维各向同性 Matérn 2.5（3 参数），仍使用连续四维组。
  不向模型提供旋转矩阵、敏感方向、函数 ID、实例信息、最优点或子函数标签。
- 三个方案继续使用相同初始化和参数边界，`normalize_y=true`、`alpha=1e-10`；
  L-BFGS-B 最多 500 次迭代，梯度容差 `1e-5`、函数容差 `1e-8`，单次冷启动，无先验、无重启。
  具体核设置见 [上一轮基线协议](MATERN_BASELINE_COMPARISON.md)。不根据本次测试集修改这些设置。
- 12 函数 × 3 个实例/数据配对 × 3 档样本数 × 3 个模型，共 **324 次正式拟合**。
- 每个进程一个 Eigen 线程，三个实例进程并行；保留全部拟合及未收敛状态，耗时不作为严格速度基准。
- 原始数据留在构建目录，提交生成脚本、包版本、输入及观测文件哈希、原始结果和统计报告。
  生成器检查每个实例维度、有限输出、非零方差及 `f(xopt)=fopt`；
  汇总器检查训练池、全域/局部测试校验值、完整配对和误差指标一致性。

## 误差含义

全域报告 R²、RMSE、NRMSE、MAE、绝对误差 95 分位等；不同函数值的量级差异很大，
不跨函数平均原始 RMSE。`NRMSE = RMSE / 全域测试真值标准差`，R² 越高越好。

局部同时报告原始 RMSE、局部 R²，以及 `局部 RMSE / 全域测试真值标准差`。
最后一个指标便于比较局部误差相对全域变化量有多大，不等于按局部变化量归一化。
局部 R² 的分母使用局部真值方差，可能出现非常大的负数；完整保留，不裁剪。
另记录只预测训练目标均值的局部误差作为参考。

固定加性分组通常不匹配旋转后的结构。本实验测试现有三个方案的适用性，
不是保证加性模型占优，也不单独分离核族、参数共享和加性结构各自的因果作用。
全域预测较准确与能重建最优点附近的谷地是两个独立问题。

## 复现

```bash
python3 -m pip install numpy==2.2.6 ioh==0.3.22 matplotlib==3.11.2
python3 experiments/generate_bbob_data.py --output build/bbob-data
cmake --preset release -DCMAKE_CXX_STANDARD=14 -DSUPER_GPR_BUILD_EXPERIMENTS=ON
cmake --build --preset release --target sample_learning_curve
./build/release/bin/sample_learning_curve --bbob-data=build/bbob-data/run_1 > build/bbob-run1.csv
./build/release/bin/sample_learning_curve --bbob-data=build/bbob-data/run_2 > build/bbob-run2.csv
./build/release/bin/sample_learning_curve --bbob-data=build/bbob-data/run_3 > build/bbob-run3.csv
python3 experiments/summarize_function_suite.py --bbob \
  build/bbob-run1.csv build/bbob-run2.csv build/bbob-run3.csv \
  --output experiments/BBOB_PATHOLOGICAL_RESULTS.md \
  --plot experiments/BBOB_GLOBAL.png --local-plot experiments/BBOB_LOCAL.png
```

快速检查：生成器加 `--quick`，C++ 程序同时加 `--quick`，使用 20 维开发实例、种子 101，
32/64 个训练点、128 个全域测试点、64 个局部测试点，共 72 次开发拟合。
开发数据不混入正式结果。IOH 仅为观测生成器的依赖，C++ 库与拟合程序无需链接它。
