#include <iostream>
#include <super_gpr/acquisition.hpp>

int main() try {
    using namespace super_gpr;
    Matrix x(6, 1);
    x << 0.0, 0.8, 1.6, 2.4, 3.2, 4.0;
    const Vector y = x.col(0).array().sin();
    auto kernel = ConstantKernel({1.0, 1e-3, 1e3}) * RBF({1.0, 0.1, 10.0}) + WhiteKernel({1e-4, 1e-8, 0.1});
    GPROptions options;
    options.normalize_y = true;
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    model.fit(x, y);
    Matrix candidates(3, 1);
    candidates << 0.4, 2.0, 3.6;
    const auto prediction = model.predict(candidates);
    const Bounds bounds{Vector::Constant(1, 0.0), Vector::Constant(1, 4.0)};
    Matrix starts(17, 1);
    starts.col(0) = Vector::LinSpaced(17, 0.0, 4.0);
    const auto next = optimize_acquisition(model, EI(y.minCoeff()), starts, bounds);
    std::cout << "LML: " << model.log_marginal_likelihood() << '\n'
              << "Mean:\n"
              << prediction.mean << '\n'
              << "Standard deviation:\n"
              << prediction.standard_deviation << '\n'
              << "Next EI point: " << next.x.transpose() << " (EI=" << next.value << ")\n";
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
