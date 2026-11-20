#include <benchmark/benchmark.h>
#include <random>
#include <super_gpr/additive.hpp>

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
        model.fit(x, y).value();
        benchmark::DoNotOptimize(model.log_marginal_likelihood().value());
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
    model.fit(x, y).value();
    const Vector theta = kernel.theta();
    for (auto iteration : state) {
        (void)iteration;
        auto result = model.log_marginal_likelihood(theta).value();
        benchmark::DoNotOptimize(result.value);
        benchmark::DoNotOptimize(result.gradient.data());
    }
    state.SetComplexityN(count);
}
// Google Benchmark owns the registration.
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory)
BENCHMARK(likelihood_gradient)->RangeMultiplier(2)->Range(64, 512)->Complexity();

template <bool Strided> void shared_additive_likelihood_gradient(benchmark::State &state) {
    constexpr Eigen::Index groups_count = 20;
    constexpr Eigen::Index group_size = 4;
    const auto count = static_cast<Eigen::Index>(state.range(0));
    std::vector<FeatureGroup> groups;
    for (Eigen::Index group = 0; group < groups_count; ++group) {
        FeatureGroup features;
        for (Eigen::Index axis = 0; axis < group_size; ++axis) {
            features.push_back(Strided ? axis * groups_count + group : group * group_size + axis);
        }
        groups.push_back(std::move(features));
    }
    Matrix x(count, group_size * groups_count);
    std::mt19937 random(5003);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (Eigen::Index row = 0; row < count; ++row) {
        for (const auto &group : groups) {
            for (const auto col : group) {
                x(row, col) = uniform(random);
            }
        }
    }
    const Vector y = x.col(0).array().sin();
    const auto component = ConstantKernel(Parameter{0.05}) *
                           RBF(std::vector<Parameter>(group_size, Parameter{0.7}));
    GPROptions options;
    options.optimize = false;
    auto model = make_additive_gp(make_shared_additive_kernel(groups, component), {0.001, 1e-8, 0.5}, options);
    model.fit(x, y).value();
    const Vector theta = model.kernel().theta();
    for (auto iteration : state) {
        (void)iteration;
        auto result = model.log_marginal_likelihood(theta).value();
        benchmark::DoNotOptimize(result.value);
        benchmark::DoNotOptimize(result.gradient.data());
    }
}
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(shared_additive_likelihood_gradient, false)->Arg(512)->Arg(1024);
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory)
BENCHMARK_TEMPLATE(shared_additive_likelihood_gradient, true)->Arg(512)->Arg(1024);
} // namespace
