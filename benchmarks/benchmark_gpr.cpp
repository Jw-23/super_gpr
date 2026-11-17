#include <benchmark/benchmark.h>
#include <super_gpr/gpr.hpp>

namespace {
using namespace super_gpr;

Matrix samples(Eigen::Index count) {
    Matrix x(count, 2);
    for (Eigen::Index row = 0; row < count; ++row) {
        x(row, 0) = static_cast<double>(row) / static_cast<double>(count);
        x(row, 1) = std::cos(x(row, 0));
    }
    return x;
}

void fit_fixed_kernel(benchmark::State &state) {
    const auto count = static_cast<Eigen::Index>(state.range(0));
    const Matrix x = samples(count);
    const Vector y = x.col(0).array().sin();
    GPROptions options;
    options.optimize = false;
    GaussianProcessRegressor<RBF> model(RBF{}, options);
    for (auto iteration : state) {
        (void)iteration;
        model.fit(x, y);
        benchmark::DoNotOptimize(model.log_marginal_likelihood());
    }
    state.SetComplexityN(count);
}
// Google Benchmark owns the registration.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory)
BENCHMARK(fit_fixed_kernel)->RangeMultiplier(2)->Range(64, 512)->Complexity();

using Composite = AddKernel<Multiple<ConstantKernel, RBF>, WhiteKernel>;

template <class K> void kernel_matrix(benchmark::State &state) {
    const auto count = static_cast<Eigen::Index>(state.range(0));
    const Matrix x = samples(count);
    const K kernel(ConstantKernel{} * RBF{} + WhiteKernel{});
    for (auto iteration : state) {
        (void)iteration;
        const Matrix result = covariance(kernel, x);
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }
    state.SetComplexityN(count);
}
// Google Benchmark owns the registration.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(kernel_matrix, Composite)->RangeMultiplier(2)->Range(64, 512)->Complexity();
// Google Benchmark owns the registration.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(kernel_matrix, BoxedKernel)->RangeMultiplier(2)->Range(64, 512)->Complexity();

void likelihood_gradient(benchmark::State &state) {
    const auto count = static_cast<Eigen::Index>(state.range(0));
    const Matrix x = samples(count);
    const Vector y = x.col(0).array().sin();
    GPROptions options;
    options.optimize = false;
    auto kernel = ConstantKernel{} * RBF{} + WhiteKernel({0.01});
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    model.fit(x, y);
    const Vector theta = kernel.theta();
    for (auto iteration : state) {
        (void)iteration;
        auto result = model.log_marginal_likelihood(theta);
        benchmark::DoNotOptimize(result.value);
        benchmark::DoNotOptimize(result.gradient.data());
    }
    state.SetComplexityN(count);
}
// Google Benchmark owns the registration.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory)
BENCHMARK(likelihood_gradient)->RangeMultiplier(2)->Range(64, 512)->Complexity();
} // namespace
