#include <iostream>
#include <super_gpr/acquisition.hpp>

namespace {
super_gpr::GPRResult<void> run_example() {
    using namespace super_gpr;
    Matrix x(6, 1);
    x << 0.0, 0.8, 1.6, 2.4, 3.2, 4.0;
    const Vector y = x.col(0).array().sin();
    BOOST_OUTCOME_TRY(auto amplitude, ConstantKernel::create({1.0, 1e-3, 1e3}));
    BOOST_OUTCOME_TRY(auto radial, RBF::create({1.0, 0.1, 10.0}));
    BOOST_OUTCOME_TRY(auto noise, WhiteKernel::create({1e-4, 1e-8, 0.1}));
    auto kernel = amplitude * radial + noise;
    GPROptions options;
    options.normalize_y = true;
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    BOOST_OUTCOME_TRY(model.fit(x, y));
    Matrix candidates(3, 1);
    candidates << 0.4, 2.0, 3.6;
    BOOST_OUTCOME_TRY(auto prediction, model.predict(candidates));
    const Bounds bounds{Vector::Constant(1, 0.0), Vector::Constant(1, 4.0)};
    Matrix starts(17, 1);
    starts.col(0) = Vector::LinSpaced(17, 0.0, 4.0);
    BOOST_OUTCOME_TRY(auto next, optimize_acquisition(model, EI(y.minCoeff()), starts, bounds));
    BOOST_OUTCOME_TRY(auto lml, model.log_marginal_likelihood());
    std::cout << "LML: " << lml << '\n'
              << "Mean:\n"
              << prediction.mean << '\n'
              << "Standard deviation:\n"
              << prediction.standard_deviation << '\n'
              << "Next EI point: " << next.x.transpose() << " (EI=" << next.value << ")\n";
    return boost::outcome_v2::success();
}
} // namespace

int main() {
    const auto result = run_example();
    if (!result) {
        std::cerr << "GPR example failed (GPRError " << static_cast<int>(result.error()) << ")\n";
        return 1;
    }
    return 0;
}
