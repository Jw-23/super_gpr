#include <super_gpr/acquisition.hpp>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
using namespace super_gpr;

void emit(const std::string& name, const Matrix& data) {
    std::cout << name << ' ' << data.size();
    for (Eigen::Index row = 0; row < data.rows(); ++row) {
        for (Eigen::Index col = 0; col < data.cols(); ++col) { std::cout << ' ' << data(row, col); }
    }
    std::cout << '\n';
}
void emit_scalar(const std::string& name, double value) { emit(name, Matrix::Constant(1, 1, value)); }
Matrix training_x() {
    Matrix x(6, 2);
    x << 0.0, 0.1, 0.4, -0.2, 0.9, 0.5, 1.3, 0.7, 1.8, -0.1, 2.4, 0.4;
    return x;
}
Matrix query_x() {
    Matrix x(3, 2);
    x << 0.2, -0.1, 0.8, 0.4, 2.0, 0.2;
    return x;
}
void report_kernel(const std::string& name, const BoxedKernel& kernel) {
    const Matrix x = training_x();
    emit(name + ".train", covariance(kernel, x));
    emit(name + ".cross", covariance(kernel, x, query_x()));
    emit(name + ".theta", kernel.theta());
    Matrix derivatives(x.rows() * x.rows(), kernel.parameter_count());
    Vector gradient(kernel.parameter_count());
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index col = 0; col < x.rows(); ++col) {
            kernel.gradient(x.row(row).transpose(), x.row(col).transpose(), row == col, gradient);
            derivatives.row(row * x.rows() + col) = gradient.transpose();
        }
    }
    emit(name + ".gradient", derivatives);
}
void report_model(const std::string& name, const BoxedKernel& kernel, const Matrix& x, const Matrix& y,
                  const Matrix& query, const GPROptions& options) {
    GaussianProcessRegressor<BoxedKernel> model(kernel, options);
    model.fit(x, y);
    emit(name + ".theta", model.kernel().theta());
    emit_scalar(name + ".lml", model.log_marginal_likelihood());
    emit(name + ".gradient", model.log_marginal_likelihood(model.kernel().theta()).gradient);
    const auto result = model.predict(query, true);
    emit(name + ".mean", result.mean);
    emit(name + ".std", result.standard_deviation);
    for (std::size_t output = 0; output < result.covariance.size(); ++output) {
        emit(name + ".cov" + std::to_string(output), result.covariance[output]);
    }
}
} // namespace

int main() try {
    using namespace super_gpr;
    std::cout << std::setprecision(17);
    const std::vector<std::pair<std::string, BoxedKernel>> kernels{
        {"rbf", BoxedKernel(RBF({0.7}))},
        {"ard", BoxedKernel(RBF(std::vector<Parameter>{{0.7}, {1.3}}))},
        {"matern05", BoxedKernel(Matern({0.7}, 0.5))},
        {"matern15", BoxedKernel(Matern({0.7}, 1.5))},
        {"matern25", BoxedKernel(Matern(std::vector<Parameter>{{0.7}, {1.3}}, 2.5))},
        {"materninf", BoxedKernel(Matern({0.7}, std::numeric_limits<double>::infinity()))},
        {"rq", BoxedKernel(RationalQuadratic({0.8}, {1.4}))},
        {"periodic", BoxedKernel(ExpSineSquared({0.8}, {1.7}))},
        {"dot", BoxedKernel(DotProduct({0.6}))},
        {"constant", BoxedKernel(ConstantKernel({1.2}))},
        {"white", BoxedKernel(WhiteKernel({0.2}))},
        {"composite", BoxedKernel(1.7 * (RBF{} + DotProduct{}) * (ConstantKernel{} + WhiteKernel{}))},
        {"fixed", BoxedKernel(ConstantKernel(Parameter::fixed(2.0)) *
                    RBF(std::vector<Parameter>{Parameter::fixed(0.7), {1.3}}))}
    };
    for (const auto& entry : kernels) { report_kernel(entry.first, entry.second); }
    Matrix targets(6, 1);
    targets << 1.2, 0.7, -0.1, -0.8, -0.3, 0.4;
    const BoxedKernel kernel(ConstantKernel({1.2}) * RBF(std::vector<Parameter>{{0.7}, {1.1}}) + WhiteKernel({0.03}));
    GPROptions options;
    options.optimize = false;
    options.alpha = 0.02;
    report_model("raw", kernel, training_x(), targets, query_x(), options);
    options.normalize_y = true;
    report_model("normalized", kernel, training_x(), targets, query_x(), options);
    Matrix multi(6, 3);
    multi.col(0) = targets;
    multi.col(1) = 3.0 * targets.array() + 7.0;
    multi.col(2).setConstant(4.0);
    options.sample_noise = Vector::LinSpaced(6, 0.01, 0.04);
    report_model("multi", kernel, training_x(), multi, query_x(), options);
    Matrix x(12, 1);
    x.col(0) = Vector::LinSpaced(12, -2.0, 3.0);
    Matrix y(12, 1);
    y.col(0) = (1.5 * x.col(0).array()).sin() + 0.1 * (8.0 * x.col(0).array()).cos();
    const BoxedKernel optimized(ConstantKernel({1.3, 1e-3, 10.0}) * RBF({0.7, 0.1, 5.0}) + WhiteKernel({0.02, 1e-5, 0.5}));
    options = GPROptions{};
    options.optimizer.gradient_tolerance = 1e-8;
    options.optimizer.function_tolerance = 1e-12;
    report_model("optimized", optimized, x, y, x, options);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
