#include <gtest/gtest.h>
#include <super_gpr/acquisition.hpp>
#include <super_gpr/additive.hpp>

namespace {
using namespace super_gpr;

TEST(AdditiveKernel, EqualsAnExplicitSumAndPreservesTypes) {
    auto component = ConstantKernel({1.3}) * RBF({0.7});
    auto additive = make_additive_kernel({{0, 1}, {2, 3}}, component);
    auto sum = SubspaceKernel<decltype(component)>(component, {0, 1}) +
               SubspaceKernel<decltype(component)>(component, {2, 3});
    static_assert(IsKernel<decltype(additive)>::value, "Additive kernels implement the Kernel interface");
    static_assert(std::is_same<decltype(additive), AdditiveKernel<decltype(component)>>::value,
                  "Additive kernels retain concrete component types");
    Matrix x(3, 4);
    x << 0.1, 0.3, -0.2, 0.5, -0.4, 0.8, 0.7, 0.1, 0.2, -0.1, 0.6, -0.5;
    EXPECT_TRUE(covariance(additive, x).isApprox(covariance(sum, x)));
    EXPECT_TRUE(additive.theta().isApprox(sum.theta()));
    EXPECT_EQ(additive.parameter_count(), 4);
    Vector gradient(4);
    Vector expected(4);
    additive.gradient(x.row(0).transpose(), x.row(1).transpose(), false, gradient);
    sum.gradient(x.row(0).transpose(), x.row(1).transpose(), false, expected);
    EXPECT_TRUE(gradient.isApprox(expected));
}

TEST(AdditiveKernel, AnalyticDerivativesForNoncontiguousAndOverlappingGroups) {
    auto kernel = make_additive_kernel({{2, 0}, {1, 2}}, ConstantKernel({1.3}) * RBF({0.7}));
    Vector x = (Vector(4) << 0.2, -0.3, 0.7, 0.1).finished();
    const Vector y = (Vector(4) << 0.8, 0.4, -0.1, -0.4).finished();
    const Vector theta = kernel.theta();
    Vector hyper(theta.size());
    Vector input(x.size());
    kernel.gradient(x, y, false, hyper);
    kernel.input_gradient(x, y, input);
    constexpr double step = 1e-6;
    for (Eigen::Index index = 0; index < theta.size(); ++index) {
        Vector shifted = theta;
        shifted[index] += step;
        kernel.set_theta(shifted);
        const double upper = kernel.value(x, y, false);
        shifted[index] -= 2.0 * step;
        kernel.set_theta(shifted);
        const double lower = kernel.value(x, y, false);
        EXPECT_NEAR(hyper[index], (upper - lower) / (2.0 * step), 1e-8);
        kernel.set_theta(theta);
    }
    for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
        x[axis] += step;
        const double upper = kernel.value(x, y, false);
        x[axis] -= 2.0 * step;
        const double lower = kernel.value(x, y, false);
        x[axis] += step;
        EXPECT_NEAR(input[axis], (upper - lower) / (2.0 * step), 1e-8);
    }
    EXPECT_DOUBLE_EQ(input[3], 0.0);
    const SubspaceKernel<DotProduct> linear(DotProduct{}, {2, 0});
    linear.diagonal_gradient(x, input);
    EXPECT_NEAR(input[0], 2.0 * x[0], 1e-14);
    EXPECT_NEAR(input[2], 2.0 * x[2], 1e-14);
    EXPECT_DOUBLE_EQ(input[1], 0.0);
    EXPECT_DOUBLE_EQ(input[3], 0.0);
}

TEST(AdditiveKernel, FixedParametersAndInvalidGroups) {
    auto kernel = make_additive_kernel({{0}, {1}}, RBF(Parameter::fixed(0.7)));
    EXPECT_EQ(kernel.theta().size(), 0);
    EXPECT_EQ(kernel.bounds().lower.size(), 0);
    kernel.set_theta(Vector{});
    EXPECT_THROW(make_additive_kernel({}, RBF{}), std::invalid_argument);
    EXPECT_THROW(SubspaceKernel<RBF>(RBF{}, {}), std::invalid_argument);
    EXPECT_THROW(SubspaceKernel<RBF>(RBF{}, {1, 1}), std::invalid_argument);
    EXPECT_THROW(SubspaceKernel<RBF>(RBF{}, {-1}), std::invalid_argument);
    const Vector point = Vector::Zero(1);
    EXPECT_THROW(kernel.value(point, point, true), std::invalid_argument);
}

TEST(AdditiveGP, SharedObservationInfersComponentsAndTheirPosteriorDependence) {
    using Component = SubspaceKernel<ConstantKernel>;
    AdditiveKernel<ConstantKernel> kernel(
        std::vector<Component>{Component(ConstantKernel(Parameter::fixed(4.0)), {0}),
                               Component(ConstantKernel(Parameter::fixed(1.0)), {1})});
    GPROptions options;
    options.alpha = 0.0;
    options.optimize = false;
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    const Matrix x = Matrix::Zero(1, 2);
    const Vector y = Vector::Constant(1, 3.0);
    model.fit(x, y);
    const auto &components = model.kernel().components();
    const auto first = model.predict_component(components[0], x, true);
    const auto second = model.predict_component(components[1], x, true);
    const auto cross = model.component_cross_covariance(components[0], components[1], x);
    EXPECT_NEAR(first.mean(0, 0), 2.4, 1e-14);
    EXPECT_NEAR(second.mean(0, 0), 0.6, 1e-14);
    EXPECT_NEAR(first.covariance[0](0, 0), 0.8, 1e-14);
    EXPECT_NEAR(second.covariance[0](0, 0), 0.8, 1e-14);
    EXPECT_NEAR(cross[0](0, 0), -0.8, 1e-14);
    EXPECT_NEAR(model.predict(x, true).covariance[0](0, 0),
                first.covariance[0](0, 0) + second.covariance[0](0, 0) + 2.0 * cross[0](0, 0), 1e-14);
}

TEST(AdditiveGP, ComponentsReconstructNormalizedMultiOutputMeanAndCovariance) {
    const auto additive = make_additive_kernel({{0}, {1}}, ConstantKernel({1.2}) * RBF({0.7}));
    GPROptions options;
    options.optimize = false;
    options.normalize_y = true;
    auto model = make_additive_gp(additive, Parameter::fixed(0.03), options);
    Matrix x(5, 2);
    x << 0.0, 0.1, 0.5, 0.8, 1.0, -0.3, 1.5, 0.4, 2.0, -0.5;
    Matrix y(5, 2);
    y.col(0) = x.col(0).array().sin() + x.col(1).array().cos() + 4.0;
    y.col(1) = 2.0 * y.col(0).array() - 3.0;
    model.fit(x, y);
    const auto &components = model.kernel().left().components();
    const auto first = model.predict_component(components[0], x, true);
    const auto second = model.predict_component(components[1], x, true);
    const auto noise = model.predict_component(model.kernel().right(), x, true);
    const auto cross = model.component_cross_covariance(components[0], components[1], x);
    const auto total = model.predict(x, true);
    Matrix reconstructed = first.mean + second.mean;
    reconstructed.rowwise() += model.mean_offset().transpose();
    EXPECT_TRUE(total.mean.isApprox(reconstructed));
    for (std::size_t output = 0; output < total.covariance.size(); ++output) {
        const Matrix covariance = first.covariance[output] + second.covariance[output] +
                                  noise.covariance[output] + cross[output] + cross[output].transpose();
        EXPECT_TRUE(total.covariance[output].isApprox(covariance, 1e-10));
    }
}

TEST(AdditiveGP, JointLMLAndPredictionGradientsIncludeEveryGroup) {
    auto kernel =
        make_additive_kernel({{0, 1}, {2}}, ConstantKernel({0.8, 0.01, 10.0}) * RBF({0.7, 0.1, 5.0}));
    GPROptions options;
    options.normalize_y = true;
    options.optimize = false;
    auto model = make_additive_gp(kernel, {0.01, 1e-6, 0.5}, options);
    Matrix x(6, 3);
    x << 0.0, 0.1, -0.2, 0.4, -0.2, 0.8, 0.9, 0.5, 0.3, 1.3, 0.7, -0.4, 1.8, -0.1, 0.6, 2.4, 0.4, 0.2;
    const Vector y = x.col(0).array().sin() + x.col(1).array().cos() + x.col(2).array().square();
    model.fit(x, y);
    const Vector theta = model.kernel().theta();
    const auto likelihood = model.log_marginal_likelihood(theta);
    constexpr double step = 1e-6;
    for (Eigen::Index index = 0; index < theta.size(); ++index) {
        Vector shifted = theta;
        shifted[index] += step;
        const double upper = model.log_marginal_likelihood(shifted).value;
        shifted[index] -= 2.0 * step;
        const double lower = model.log_marginal_likelihood(shifted).value;
        EXPECT_NEAR(likelihood.gradient[index], (upper - lower) / (2.0 * step), 2e-7);
    }
    Vector point = Vector::Constant(3, 0.3);
    const auto posterior = model.predict_with_gradients(point);
    for (Eigen::Index axis = 0; axis < point.size(); ++axis) {
        point[axis] += step;
        const auto upper = model.predict_with_gradients(point);
        point[axis] -= 2.0 * step;
        const auto lower = model.predict_with_gradients(point);
        point[axis] += step;
        EXPECT_NEAR(posterior.mean_gradient[axis], (upper.mean - lower.mean) / (2.0 * step), 1e-7);
        EXPECT_NEAR(posterior.standard_deviation_gradient[axis],
                    (upper.standard_deviation - lower.standard_deviation) / (2.0 * step), 1e-7);
    }
    options.optimize = true;
    auto optimized = make_additive_gp(kernel, {0.01, 1e-6, 0.5}, options);
    optimized.fit(x, y);
    EXPECT_GE(optimized.log_marginal_likelihood(), model.log_marginal_likelihood());
    EXPECT_FALSE(optimized.kernel().theta().isApprox(theta));
    EXPECT_EQ(optimized.kernel().parameter_count(), 5);
}
} // namespace
