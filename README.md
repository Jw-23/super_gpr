# super_gpr

C++14 高斯过程回归库。Kernel 使用模板静态组合，超参数通过解析 LML 梯度和
L-BFGS-B 优化。提供预测均值、标准差、完整协方差、多输出回归及贝叶斯优化采集函数。

## 构建与验证

需要 CMake 3.24+、Ninja、Git 和支持 C++14 的编译器。
Eigen 3.4 与 LBFGS++ 为必要依赖。Eigen 优先使用已安装版本，否则由 CPM 获取固定版本；
LBFGS++ 的头文件和许可证已内置于 `third_party/lbfgspp`，构建时无需下载。
GoogleTest 和 Google Benchmark 默认启用，可分别通过
`PROJECT_ENABLE_TESTING`、`PROJECT_ENABLE_BENCHMARKS` 关闭。

```bash
cmake --preset debug -DCMAKE_CXX_STANDARD=14
cmake --build --preset debug
ctest --preset debug
./build/debug/bin/super_gpr
```

作为子项目使用：

```cmake
add_subdirectory(super_gpr)
target_link_libraries(your_application PRIVATE super_gpr::super_gpr)
```

启用 clang-tidy 目标：

```bash
cmake --preset debug -DPROJECT_CLANG_TIDY_EXECUTABLE=/path/to/clang-tidy
cmake --build --preset debug --target check-clang-tidy
```

检查覆盖启用的库、应用、测试、基准和公共头文件。保留缺陷分析和性能检查；关闭相邻参数
同类型、强制静态成员函数、数学公式冗余括号等风格规则。Eigen 的轻量视图允许按值传递。

## 示例

```cpp
#include <super_gpr/acquisition.hpp>

using namespace super_gpr;

Matrix x(5, 1); // Rows are samples, columns are features.
x << -2.0, -1.0, 0.0, 1.0, 2.0;
Vector y = (x.col(0).array() - 0.3).square();

auto kernel = ConstantKernel({1.0, 1e-3, 1e3}) * RBF({0.8, 0.1, 10.0})
            + WhiteKernel({0.01, 1e-6, 1.0});
GPROptions options;
options.normalize_y = true;
options.optimizer_restarts = 2;
GaussianProcessRegressor<decltype(kernel)> gp(kernel, options);
gp.fit(x, y);

auto prediction = gp.predict(x, true);
// prediction.mean and standard_deviation have shape [samples, targets].
// prediction.covariance contains one [samples, samples] matrix per target.

Matrix starts(17, 1);
starts.col(0) = Vector::LinSpaced(17, -2.0, 2.0);
Bounds bounds{Vector::Constant(1, -2.0), Vector::Constant(1, 2.0)};
auto next = optimize_acquisition(gp, EI(y.minCoeff(), 0.01), starts, bounds);
// next.x is the suggested point; next.value is the positive EI value.
```

## Kernel 与模板组合

| Kernel | 数学特性 | 适合的函数 |
| --- | --- | --- |
| `RBF` | 平稳、无限可微，支持 ARD | 很平滑的函数 |
| `Matern` | 平稳、可控制光滑性，支持 ARD | 一般非线性函数、BO |
| `RationalQuadratic` | 不同长度尺度的 RBF 混合 | 多尺度变化 |
| `ExpSineSquared` | 平稳、周期性 | 周期函数 |
| `DotProduct` | 非平稳、线性 | 线性趋势 |
| `ConstantKernel` | 常数协方差 | 信号幅度与常数偏置 |
| `WhiteKernel` | 独立观测噪声 | 观测噪声建模 |

`Matern` 提供常用的 `nu = 0.5, 1.5, 2.5, infinity`，默认 `1.5`；`nu` 固定，不参与优化。
其他 `nu` 暂不支持。`RationalQuadratic`、`ExpSineSquared` 使用各向同性长度尺度。
`ExpSineSquared` 的周期性基于欧氏距离，与 scikit-learn 一致，通常用于一维周期输入。
Matérn 1.5、2.5 与 RBF 的同数据预测对照见
[核函数实验](experiments/README.md#matérn-与-rbf-的配对对照)，包含普通、ARD 和加性模型。

```cpp
auto sum = RBF{} + WhiteKernel{};       // AddKernel<RBF, WhiteKernel>
auto product = ConstantKernel{} * sum; // Multiple<ConstantKernel, decltype(sum)>
auto scaled = 2.5 * product;           // Multiple<ConstantKernel, decltype(product)>
```

`Add<K1, K2>` 是 `AddKernel<K1, K2>` 的别名，`Multiply<K1, K2>` 是 `Multiple<K1, K2>`
的别名。运算保存完整嵌套类型，组合子应用和法则与乘积法则传播解析导数。
仅支持加法与乘法；浮点系数转为固定的 `ConstantKernel`，不会增加待优化参数。
协方差的缩放通常应使用非负系数。

只有显式构造 `BoxedKernel` 才发生类型擦除。它使用虚函数转发，复制时深拷贝：

```cpp
BoxedKernel boxed(RBF{} + WhiteKernel{});
GaussianProcessRegressor<BoxedKernel> gp(boxed);
```

### 超参数

`Parameter{value, lower, upper}` 使用原始物理量，默认 `{1.0, 1e-5, 1e5}`。
`Parameter::fixed(value)` 排除优化，也可用相同的上下界表示固定参数。

```cpp
RBF isotropic({0.7, 0.1, 10.0});
RBF ard(std::vector<Parameter>{{0.7, 0.1, 10.0}, Parameter::fixed(1.2)});
Matern matern({0.8, 0.1, 10.0}, 2.5);
RationalQuadratic rq({0.8}, {1.4}); // Length scale, alpha.
ExpSineSquared periodic({0.8}, {1.7}); // Length scale, period.
```

`theta()`、`bounds()`、`set_theta()` 和 `gradient()` 使用**对数超参数坐标**。
固定参数不占 theta 或梯度位置。组合按左子树、右子树拼接。
`RationalQuadratic` 的 theta 顺序为 alpha、length scale，与 scikit-learn 一致，
其构造参数顺序仍为 length scale、alpha。

`covariance(k, X)` 构造训练/先验协方差；`covariance(k, X, Y)` 构造交叉协方差。
`WhiteKernel` 仅在前者的对角线上产生噪声，即使 X 与 Y 的坐标完全相同，后者仍为零。

### 自定义解析 Kernel

C++14 的 `IsKernel<K>` 检测接口，组合子与 GPR 通过编译期断言约束类型。
可以自定义值类型并实现下列接口，也可以使用 `make_analytic_kernel` 提供公式：

```cpp
auto linear = make_analytic_kernel(
    Vector::Zero(1),
    Bounds{Vector::Constant(1, -5.0), Vector::Constant(1, 5.0)},
    [](Point x, Point y, bool, Point theta) {
        return std::exp(theta[0]) * x.dot(y);
    },
    [](Point x, Point y, bool, Point theta, Gradient out) {
        out[0] = std::exp(theta[0]) * x.dot(y);
    },
    [](Point, Point y, Point theta, Gradient out) {
        out = std::exp(theta[0]) * y;
    },
    [](Point x, Point theta, Gradient out) {
        out = 2.0 * std::exp(theta[0]) * x;
    });
auto combined = linear * RBF{};
```

接口契约：

- `parameter_count()`、`theta()`、`bounds()`、`set_theta(Point)`：参数数量、对数参数与边界。
- `value(Point x, Point y, bool same_observation)`：核值。`same_observation` 表示同一个观测索引，非坐标相等。
- `gradient(x, y, same_observation, Gradient out)`：写入 `dk/dlog(parameter)`。
- `input_gradient(x, y, Gradient out)`：写入交叉协方差对第一个输入 x 的偏导。
- `diagonal_gradient(x, Gradient out)`：写入 `d[k(x,x)]/dx`，包含两个输入同时变化的贡献。

自定义核须对称并适合作为协方差；所有梯度回调必须完整写入输出。`Point` 和 `Gradient`
是 Eigen 视图，不应保存到调用范围之外。`Matern(nu=0.5)` 在重合点不可微，输入导数取零。

## Additive GP：变量组上的核相加

`super_gpr/additive.hpp` 提供 `SubspaceKernel<K>` 和 `AdditiveKernel<K>`。
实现的是一个联合 GP 的加性核：

```text
k(x, x') = sum_j k_j(x[S_j], x'[S_j])
```

每个子核有自己的超参数，但训练标签只有完整函数的一个标量；所有参数通过同一个
LML 联合优化。不会为各组分别拟合一份完整 y。变量分组由调用方给定，不自动学习分组。

```cpp
#include <super_gpr/additive.hpp>

using namespace super_gpr;
std::vector<FeatureGroup> groups{{0, 1}, {2, 3}};
auto additive = make_additive_kernel(
    groups, ConstantKernel({0.5, 1e-5, 100.0}) * RBF({0.7, 0.1, 20.0}));
GPROptions options;
options.normalize_y = true;
auto gp = make_additive_gp(additive, {0.001, 1e-8, 0.5}, options);
gp.fit(x, y); // x has four columns; y contains only the total scalar target.
auto total = gp.predict(query, true);

// Use fitted component parameters for conditional component predictions.
const auto &components = gp.kernel().left().components();
auto first = gp.predict_component(components[0], query, true);
auto second = gp.predict_component(components[1], query, true);
auto cross = gp.component_cross_covariance(components[0], components[1], query);
// total.mean = first.mean + second.mean + gp.mean_offset() per row.
```

`AdditiveKernel<K>` 保留静态子核类型，支持运行时指定组数；`SubspaceKernel<K>` 支持
非连续、重排及组间重叠的坐标，但同一个组内不能重复索引。连续组通过 Eigen 视图访问，
非连续组收集到局部向量。输入导数散射回原始维度，组间重叠时累加导数。

原有组合子仍然适用，例如：

```cpp
auto explicit_sum = SubspaceKernel<RBF>(RBF{}, {0, 1})
                  + SubspaceKernel<Matern>(Matern({}, 2.5), {2, 3});
auto hybrid = additive + ConstantKernel({0.01, 1e-6, 1.0}) * RBF({2.0, 0.1, 100.0})
             + WhiteKernel({0.001, 1e-8, 0.5});
GaussianProcessRegressor<decltype(hybrid)> hybrid_gp(hybrid, options);
```

上例的全局核用于建模跨组交互，其幅度也参与 LML 优化。`make_additive_gp` 是普通
`GaussianProcessRegressor<AddKernel<AdditiveKernel<K>, WhiteKernel>>` 的构造便捷函数，
没有额外的训练算法。需要每组不同初值时，直接构造
`AdditiveKernel<K>(std::vector<SubspaceKernel<K>>{...})`。

各分量的后验均值使用共享系数 `alpha = K_total^-1 y_normalized`。
`predict_component` 返回不带全局均值偏置的分量贡献，恢复原始目标的尺度；
`mean_offset()` 返回全局均值。该通用接口要求传入已拟合总核中的实际分量。
`component_cross_covariance(first, second, X)` 仅用于先验独立的两个不同分量：

```text
Cov(g_i(X*), g_j(X*) | y) = -k_i(X*, X) K_total^-1 k_j(X, X*)
```

完整预测方差不能直接相加各分量方差，还必须加入交叉协方差及 WhiteKernel 的预测噪声。
分量贡献取决于核先验；没有中心化约束时，不保证分量的物理可辨识性。
加性结构降低的是函数建模的有效维度，精确联合拟合仍为 O(n^3)。

理论背景：[Kandasamy et al., ICML 2015](https://proceedings.mlr.press/v37/kandasamy15.html)。
高维预测实验与原始数据见 [experiments/README.md](experiments/README.md)。

### 共享超参数与收缩先验

数据较少时，`make_shared_additive_kernel` 可让所有组共享同一份核超参数。
协方差仍然是各组子核之和，解析超参数梯度按组相加。
共享参数不会强制各组潜在函数的取值相同，各组仍对应不同的潜在 GP。
例如两个二维组共享 ARD 形状：

```cpp
std::vector<FeatureGroup> groups{{0, 1}, {2, 3}};
auto prototype = ConstantKernel({0.5, 1e-5, 100.0})
               * RBF(std::vector<Parameter>{{0.7, 0.1, 20.0}, {0.7, 0.1, 20.0}});
auto shared = make_shared_additive_kernel(groups, prototype);
auto gp = make_additive_gp(shared, {0.001, 1e-8, 0.5}, options);
```

这里优化 1 个共同幅度、2 个组内位置长度尺度和 1 个噪声，共 4 个超参数。
普通加性核会为每组分别保留参数。共享核的 `components()` 返回实际带幅度的子核，
因此原来的分量预测接口仍然成立。每组幅度的初值通常取总幅度除以组数。
共享 ARD 要求组内坐标位置具有可比较的含义；组间差异明显时，应通过训练集内验证判断是否合适。
原有 `make_additive_kernel` 的独立参数行为不变。

需要保留组间差异时，可为对数超参数配置可选 Gaussian 收缩先验：

```cpp
auto kernel = make_additive_kernel(groups, prototype) + WhiteKernel({0.001, 1e-8, 0.5});
GPROptions regularized;
regularized.normalize_y = true;
regularized.prior.mean = kernel.theta();
regularized.prior.standard_deviation = Vector::Constant(kernel.parameter_count(), 0.5);
regularized.prior.standard_deviation.tail(1).setConstant(1.5);
GaussianProcessRegressor<decltype(kernel)> regularized_gp(kernel, regularized);
```

此时最大化以下 MAP 目标，而非只最大化 LML：

```text
LML(theta) - 0.5 * sum_i ((theta_i - prior.mean_i) / prior.standard_deviation_i)^2
```

`theta` 始终是对数坐标，固定参数不占位置；先验是这些坐标上的 Gaussian，
不是物理参数密度的 MAP。两个先验向量为空时关闭该功能；标准差为正无穷可关闭某个坐标的惩罚。
先验中心和宽度应基于先验知识或训练集内交叉验证选择。
`log_marginal_likelihood()` 始终报告未加惩罚的 LML，`log_posterior()` 及其带 theta 的重载
报告上述目标和解析梯度，省略与 theta 无关的常数。
有先验时，`optimization_results()` 的目标值是负的 MAP 目标。
训练集内选择与新种子消融验证见
[Additive GP 诊断报告](experiments/ADDITIVE_DIAGNOSIS.md)。

## GPR 的数值约定

使用 Cholesky 分解计算后验和 LML。每次 LML 梯度求值将核导数直接与
`(alpha * alpha^T - targets * K^-1)` 收缩，不存储 `n × n × parameters` 导数张量。
Cholesky 使用列主序，输入矩阵使用行主序，方便逐样本计算核函数。
精确 GPR 的拟合时间为 `O(n^3)`、存储为 `O(n^2)`，适用于中小规模密集问题。

`GPROptions`：

| 选项 | 默认值 | 语义 |
| --- | --- | --- |
| `alpha` | `1e-10` | 训练协方差对角线增加的方差 |
| `sample_noise` | 空 | 非空时替代 alpha，逐观测提供方差 |
| `normalize_y` | `false` | 各输出去均值、按总体标准差归一化 |
| `optimize` | `true` | 使用 L-BFGS-B 最大化 LML |
| `optimizer_restarts` | `0` | 初值之外的额外重启次数 |
| `random_seed` | `0` | 本库内重启可复现 |
| `optimizer` | `LBFGSBOptions{}` | 优化器配置 |
| `prior` | 空 | 对数超参数的可选 Gaussian 收缩先验 |

噪声方差与 scikit-learn 一样作用于归一化后的训练目标。预测时恢复目标的均值和尺度。
`alpha` 不加入预测协方差；若 Kernel 包含 `WhiteKernel`，其噪声包含在预测方差中。
不自动添加隐式 jitter；非正定训练协方差抛出异常，试探超参数的非正定 LML 为负无穷。

`fit(X, Matrix Y)` 支持共享 Kernel 的多个独立输出，不学习输出之间的协方差。
未 fit 时 `predict(X)` 返回零均值先验。fit 后 `log_marginal_likelihood()` 返回缓存值，
`log_marginal_likelihood(theta)` 返回指定参数下的 LML 及解析梯度，不修改已拟合模型。

`predict_with_gradients(point)` 提供单输出模型的均值、标准差及其解析输入导数。
`optimization_results()` 暴露各次超参数优化的状态；达到迭代上限或线搜索失败不算收敛。
随机重启在对数边界内均匀采样，需要有限边界。随机数序列不要求与 Python 完全相同。

## 独立 L-BFGS-B

只包含 `super_gpr/optimizer.hpp` 即可使用，不依赖 Kernel 或 GPR。
底层是内置的 [LBFGS++](https://lbfgspp.statr.me/)，版本来源见
[third_party/lbfgspp/README.md](third_party/lbfgspp/README.md)，包含广义 Cauchy 点、子空间最小化和
More–Thuente 线搜索。与 SciPy 的实现可能采用不同迭代路径。

```cpp
#include <super_gpr/optimizer.hpp>

using namespace super_gpr;
Objective objective = [](const Vector& x, Vector& gradient) {
    gradient = 2.0 * x;
    return x.squaredNorm();
};
Bounds bounds{Vector::Constant(2, -2.0), Vector::Constant(2, 2.0)};
auto result = minimize_lbfgsb(objective, Vector::Ones(2), bounds);
```

支持无穷边界、上下界相等的固定坐标、初值投影和零维问题。回调须返回目标值并写入梯度。
`OptimizationResult` 包含解、目标值、梯度、迭代数、评估次数和状态。
线搜索失败时返回已评估的最佳有限点并标记失败；用户回调抛出的异常继续传播。
`memory=10`、`max_iterations=1000`、`max_line_search=40`、`gradient_tolerance=1e-5`、
`function_tolerance=2.220446049250313e-9`。函数容差设为零可关闭函数变化量停止条件。

## 采集函数

以下均以**最小化目标函数**为约定，令 `I = best - mean - xi`，`z = I / sigma`：

| 函数 | 目标 | 用途 |
| --- | --- | --- |
| `PredictedMean` | min mean | 只追求预测最优 |
| `LCB(kappa)` | min (mean - kappa sigma) | 探索潜在最优点 |
| `EI(best, xi)` | max (I Phi(z) + sigma phi(z)) | 最大化预期改善 |
| `PI(best, xi)` | max Phi(z) | 最大化改善概率 |
| `ConservativeUCB(kappa)` | min (mean + kappa sigma) | 偏好较好预测值和较低不确定性 |

`AcquisitionFunction` 提供虚函数 `evaluate(mean, sigma)`，返回值及其两个偏导；
自定义最大化准则需重写 `maximize()`。`optimize_acquisition` 通过链式法则调用 GPR
输入导数，自动处理最大化的符号，并返回原始采集值及梯度。`best` 与预测使用相同的原始目标尺度。

零方差时 EI 取 `max(I, 0)`；PI 在严格改善时为 1，否则为 0；不可微边界的偏导取零。
L-BFGS-B 是局部优化器，EI/PI 在远离改善区域时可能几乎为零，建议提供覆盖搜索域的多个初值。
默认 `kappa=2`，`xi=0`。采集优化目前使用单输出、已拟合的模型。

## scikit-learn 对照与基准

Python 仅用于可选对照测试，不是库的运行依赖：

```bash
python3 -m venv build/reference-env
build/reference-env/bin/python -m pip install scikit-learn==1.7.2 numpy==2.2.6 scipy==1.16.3
cmake --preset debug -DSUPER_GPR_PYTHON_REFERENCE_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/build/reference-env/bin/python"
cmake --build --preset debug
ctest --preset debug
```

对照覆盖全部内置核、ARD、固定参数、组合核、核解析梯度、归一化、逐样本噪声、
多输出预测、完整协方差、LML 及边界超参数优化。优化器迭代路径不同，因此使用数值容差比较。
其余 C++ 测试独立验证输入导数、组合类型、优化器状态、零方差采集值和采集优化结果。

```bash
cmake --preset release -DCMAKE_CXX_STANDARD=14
cmake --build --preset release
ctest --preset release
./build/release/bin/super_gpr_benchmarks --benchmark_min_time=0.1s
```

基准包含固定超参数拟合、模板/类型擦除 Kernel 矩阵计算以及 LML 梯度求值。

参考实现：[scikit-learn Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html)、
[GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)。
