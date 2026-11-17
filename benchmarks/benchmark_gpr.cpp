#include <super_gpr/gpr.hpp>
#include <benchmark/benchmark.h>

namespace {
void fit_fixed_kernel(benchmark::State& state) {
    using namespace super_gpr;
    const auto count = static_cast<Eigen::Index>(state.range(0));
    Matrix x(count, 2);
    for (Eigen::Index row = 0; row < count; ++row) {
        x(row, 0) = static_cast<double>(row) / static_cast<double>(count);
        x(row, 1) = std::cos(x(row, 0));
    }
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
// NOLINTNEXTLINE(bugprone-throwing-static-initialization,cppcoreguidelines-owning-memory) -- Registration is owned by Google Benchmark.
BENCHMARK(fit_fixed_kernel)->RangeMultiplier(2)->Range(64, 512)->Complexity();
} // namespace
