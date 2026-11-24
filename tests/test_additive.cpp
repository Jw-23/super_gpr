#include <gtest/gtest.h>
#include <super_gpr/acquisition.hpp>
#include <super_gpr/additive.hpp>

namespace {
using namespace super_gpr;

TEST(AdditiveKernel, EqualsAnExplicitSumAndPreservesTypes) {
    auto component = ConstantKernel::create({1.3}).value() * RBF::create(Parameter{0.7}).value();
    auto additive = make_additive_kernel({{0, 1}, {2, 3}}, component).value();
    auto sum = SubspaceKernel<decltype(component)>::create(component, {0, 1}).value() +
               SubspaceKernel<decltype(component)>::create(component, {2, 3}).value();
    static_assert(IsKernel<decltype(additive)>::value, "Additive kernels implement the Kernel interface");
    static_assert(std::is_same<decltype(additive), AdditiveKernel<decltype(component)>>::value,
                  "Additive kernels retain concrete component types");
    Matrix x(3, 4);
    x << 0.1, 0.3, -0.2, 0.5, -0.4, 0.8, 0.7, 0.1, 0.2, -0.1, 0.6, -0.5;
    EXPECT_TRUE(covariance(additive, x).value().isApprox(covariance(sum, x).value()));
    EXPECT_TRUE(additive.theta().isApprox(sum.theta()));
    EXPECT_EQ(additive.parameter_count(), 4);
    Vector gradient(4);
    Vector expected(4);
    additive.gradient(x.row(0).transpose(), x.row(1).transpose(), false, gradient).value();
    sum.gradient(x.row(0).transpose(), x.row(1).transpose(), false, expected).value();
    EXPECT_TRUE(gradient.isApprox(expected));
}

TEST(AdditiveKernel, AnalyticDerivativesForNoncontiguousAndOverlappingGroups) {
    auto kernel = make_additive_kernel({{2, 0}, {1, 2}}, ConstantKernel::create({1.3}).value() * RBF::create(Parameter{0.7}).value()).value();
    Vector x = (Vector(4) << 0.2, -0.3, 0.7, 0.1).finished();
    const Vector y = (Vector(4) << 0.8, 0.4, -0.1, -0.4).finished();
    const Vector theta = kernel.theta();
    Vector hyper(theta.size());
    Vector input(x.size());
    kernel.gradient(x, y, false, hyper).value();
    kernel.input_gradient(x, y, input).value();
    constexpr double step = 1e-6;
    for (Eigen::Index index = 0; index < theta.size(); ++index) {
        Vector shifted = theta;
        shifted[index] += step;
        kernel.set_theta(shifted).value();
        const double upper = kernel.value(x, y, false).value();
        shifted[index] -= 2.0 * step;
        kernel.set_theta(shifted).value();
        const double lower = kernel.value(x, y, false).value();
        EXPECT_NEAR(hyper[index], (upper - lower) / (2.0 * step), 1e-8);
        kernel.set_theta(theta).value();
    }
    for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
        x[axis] += step;
        const double upper = kernel.value(x, y, false).value();
        x[axis] -= 2.0 * step;
        const double lower = kernel.value(x, y, false).value();
        x[axis] += step;
        EXPECT_NEAR(input[axis], (upper - lower) / (2.0 * step), 1e-8);
    }
    EXPECT_DOUBLE_EQ(input[3], 0.0);
    const auto linear = SubspaceKernel<DotProduct>::create(DotProduct{}, {2, 0}).value();
    linear.diagonal_gradient(x, input).value();
    EXPECT_NEAR(input[0], 2.0 * x[0], 1e-14);
    EXPECT_NEAR(input[2], 2.0 * x[2], 1e-14);
    EXPECT_DOUBLE_EQ(input[1], 0.0);
    EXPECT_DOUBLE_EQ(input[3], 0.0);
}

TEST(AdditiveKernel, FixedParametersAndInvalidGroups) {
    auto kernel = make_additive_kernel({{0}, {1}}, RBF::create(Parameter::fixed(0.7)).value()).value();
    EXPECT_EQ(kernel.theta().size(), 0);
    EXPECT_EQ(kernel.bounds().lower.size(), 0);
    kernel.set_theta(Vector{}).value();
    EXPECT_EQ(make_additive_kernel({}, RBF{}).error(), GPRError::EmptyAdditiveKernel);
    EXPECT_EQ(SubspaceKernel<RBF>::create(RBF{}, {}).error(), GPRError::InvalidFeatureGroup);
    EXPECT_EQ(SubspaceKernel<RBF>::create(RBF{}, {1, 1}).error(), GPRError::InvalidFeatureGroup);
    EXPECT_EQ(SubspaceKernel<RBF>::create(RBF{}, {-1}).error(), GPRError::InvalidFeatureGroup);
    const Vector point = Vector::Zero(1);
    EXPECT_EQ(kernel.value(point, point, true).error(), GPRError::InvalidKernelInput);
    EXPECT_EQ(BoxedKernel(kernel).value(point, point, true).error(), GPRError::InvalidKernelInput);
    Vector gradient = Vector::Ones(1);
    EXPECT_EQ(kernel.components()[1].diagonal_gradient(point, gradient).error(), GPRError::InvalidKernelInput);
    EXPECT_TRUE(gradient.isOnes());
}

TEST(AdditiveGP, SharedObservationInfersComponentsAndTheirPosteriorDependence) {
    using Component = SubspaceKernel<ConstantKernel>;
    auto kernel = AdditiveKernel<ConstantKernel>::create(
        std::vector<Component>{Component::create(ConstantKernel::fixed(4.0), {0}).value(),
                               Component::create(ConstantKernel::fixed(1.0), {1}).value()}).value();
    GPROptions options;
    options.alpha = 0.0;
    options.optimize = false;
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    const Matrix x = Matrix::Zero(1, 2);
    const Vector y = Vector::Constant(1, 3.0);
    model.fit(x, y).value();
    const auto &components = model.kernel().components();
    const auto first = model.predict_component(components[0], x, true).value();
    const auto second = model.predict_component(components[1], x, true).value();
    const auto cross = model.component_cross_covariance(components[0], components[1], x).value();
    EXPECT_NEAR(first.mean(0, 0), 2.4, 1e-14);
    EXPECT_NEAR(second.mean(0, 0), 0.6, 1e-14);
    EXPECT_NEAR(first.covariance[0](0, 0), 0.8, 1e-14);
    EXPECT_NEAR(second.covariance[0](0, 0), 0.8, 1e-14);
    EXPECT_NEAR(cross[0](0, 0), -0.8, 1e-14);
    EXPECT_NEAR(model.predict(x, true).value().covariance[0](0, 0),
                first.covariance[0](0, 0) + second.covariance[0](0, 0) + 2.0 * cross[0](0, 0), 1e-14);
}

TEST(AdditiveGP, ComponentsReconstructNormalizedMultiOutputMeanAndCovariance) {
    const auto additive = make_additive_kernel({{0}, {1}}, ConstantKernel::create({1.2}).value() * RBF::create(Parameter{0.7}).value()).value();
    GPROptions options;
    options.optimize = false;
    options.normalize_y = true;
    auto model = make_additive_gp(additive, Parameter::fixed(0.03), options).value();
    Matrix x(5, 2);
    x << 0.0, 0.1, 0.5, 0.8, 1.0, -0.3, 1.5, 0.4, 2.0, -0.5;
    Matrix y(5, 2);
    y.col(0) = x.col(0).array().sin() + x.col(1).array().cos() + 4.0;
    y.col(1) = 2.0 * y.col(0).array() - 3.0;
    model.fit(x, y).value();
    const auto &components = model.kernel().left().components();
    const auto first = model.predict_component(components[0], x, true).value();
    const auto second = model.predict_component(components[1], x, true).value();
    const auto noise = model.predict_component(model.kernel().right(), x, true).value();
    const auto cross = model.component_cross_covariance(components[0], components[1], x).value();
    const auto total = model.predict(x, true).value();
    Matrix reconstructed = first.mean + second.mean;
    reconstructed.rowwise() += model.mean_offset().value().transpose();
    EXPECT_TRUE(total.mean.isApprox(reconstructed));
    for (std::size_t output = 0; output < total.covariance.size(); ++output) {
        const Matrix covariance = first.covariance[output] + second.covariance[output] +
                                  noise.covariance[output] + cross[output] + cross[output].transpose();
        EXPECT_TRUE(total.covariance[output].isApprox(covariance, 1e-10));
    }
}

TEST(AdditiveGP, BatchedTrainingMatchesPairwiseKernelsWithOverlappingGroups) {
    Matrix x(5, 3);
    x << 0.0, 0.1, 0.3, 0.5, 0.8, 0.2, 1.0, -0.3, 0.7, 1.5, 0.4, 0.1, 2.0, -0.5, 0.3;
    Matrix y(5, 2);
    y.col(0) = x.col(0).array().sin();
    y.col(1) = x.col(2).array().cos();
    GPROptions options;
    options.optimize = false;
    options.normalize_y = true;
    options.sample_noise = Vector::LinSpaced(x.rows(), 0.01, 0.05);
    const std::vector<FeatureGroup> groups{{2, 0}, {0, 1}};
    const auto subkernel = ConstantKernel::create(Parameter{1.2}).value() *
                           Matern::create(std::vector<Parameter>{{0.7}, Parameter::fixed(1.1)}, 2.5).value();
    const auto white = WhiteKernel::create(Parameter{0.05}).value();
    const auto verify = [&](const auto &kernel) {
        using Kernel = std::decay_t<decltype(kernel)>;
        GaussianProcessRegressor<Kernel> batch(kernel, options);
        GaussianProcessRegressor<BoxedKernel> pairwise(BoxedKernel(kernel), options);
        batch.fit(x, y).value();
        pairwise.fit(x, y).value();
        const Vector theta = kernel.theta().array() + 0.1;
        const auto result = batch.log_marginal_likelihood(theta).value();
        const auto expected = pairwise.log_marginal_likelihood(theta).value();
        EXPECT_NEAR(result.value, expected.value, 1e-11);
        EXPECT_TRUE(result.gradient.isApprox(expected.gradient, 1e-11));
        const auto prediction = batch.predict(x, true).value();
        const auto reference = pairwise.predict(x, true).value();
        EXPECT_TRUE(prediction.mean.isApprox(reference.mean, 1e-11));
        EXPECT_TRUE(prediction.standard_deviation.isApprox(reference.standard_deviation, 1e-11));
        for (std::size_t j = 0; j < prediction.covariance.size(); ++j) {
            EXPECT_TRUE(prediction.covariance[j].isApprox(reference.covariance[j], 1e-11));
        }
    };
    verify(make_additive_kernel(groups, subkernel).value() + white);
    verify(white + make_shared_additive_kernel(groups, subkernel).value());
}

TEST(AdditiveGP, JointLMLAndPredictionGradientsIncludeEveryGroup) {
    auto kernel =
        make_additive_kernel({{0, 1}, {2}}, ConstantKernel::create({0.8, 0.01, 10.0}).value() * RBF::create({0.7, 0.1, 5.0}).value()).value();
    GPROptions options;
    options.normalize_y = true;
    options.optimize = false;
    auto model = make_additive_gp(kernel, {0.01, 1e-6, 0.5}, options).value();
    Matrix x(6, 3);
    x << 0.0, 0.1, -0.2, 0.4, -0.2, 0.8, 0.9, 0.5, 0.3, 1.3, 0.7, -0.4, 1.8, -0.1, 0.6, 2.4, 0.4, 0.2;
    const Vector y = x.col(0).array().sin() + x.col(1).array().cos() + x.col(2).array().square();
    model.fit(x, y).value();
    const Vector theta = model.kernel().theta();
    const auto likelihood = model.log_marginal_likelihood(theta).value();
    constexpr double step = 1e-6;
    for (Eigen::Index index = 0; index < theta.size(); ++index) {
        Vector shifted = theta;
        shifted[index] += step;
        const double upper = model.log_marginal_likelihood(shifted).value().value;
        shifted[index] -= 2.0 * step;
        const double lower = model.log_marginal_likelihood(shifted).value().value;
        EXPECT_NEAR(likelihood.gradient[index], (upper - lower) / (2.0 * step), 2e-7);
    }
    Vector point = Vector::Constant(3, 0.3);
    const auto posterior = model.predict_with_gradients(point).value();
    for (Eigen::Index axis = 0; axis < point.size(); ++axis) {
        point[axis] += step;
        const auto upper = model.predict_with_gradients(point).value();
        point[axis] -= 2.0 * step;
        const auto lower = model.predict_with_gradients(point).value();
        point[axis] += step;
        EXPECT_NEAR(posterior.mean_gradient[axis], (upper.mean - lower.mean) / (2.0 * step), 1e-7);
        EXPECT_NEAR(posterior.standard_deviation_gradient[axis],
                    (upper.standard_deviation - lower.standard_deviation) / (2.0 * step), 1e-7);
    }
    options.optimize = true;
    auto optimized = make_additive_gp(kernel, {0.01, 1e-6, 0.5}, options).value();
    optimized.fit(x, y).value();
    EXPECT_GE(optimized.log_marginal_likelihood().value(), model.log_marginal_likelihood().value());
    EXPECT_FALSE(optimized.kernel().theta().isApprox(theta));
    EXPECT_EQ(optimized.kernel().parameter_count(), 5);
}

TEST(SharedAdditiveKernel, TiedParametersPreserveCovarianceAndAnalyticGradients) {
    const auto prototype = ConstantKernel::create({1.3}).value() * Matern::create(std::vector<Parameter>{{0.7}, {1.2}}, 2.5).value();
    auto kernel = make_shared_additive_kernel({{0, 2}, {1, 3}}, prototype).value();
    const auto independent = make_additive_kernel({{0, 2}, {1, 3}}, prototype).value();
    static_assert(IsKernel<decltype(kernel)>::value, "Shared additive kernels implement Kernel");
    EXPECT_EQ(kernel.parameter_count(), 3);
    EXPECT_EQ(independent.parameter_count(), 6);
    Vector x = (Vector(4) << 0.2, -0.3, 0.7, 0.1).finished();
    const Vector y = (Vector(4) << 0.8, 0.4, -0.1, -0.4).finished();
    EXPECT_DOUBLE_EQ(kernel.value(x, y, false).value(), independent.value(x, y, false).value());
    const Vector theta = kernel.theta();
    Vector gradient(theta.size());
    kernel.gradient(x, y, false, gradient).value();
    constexpr double step = 1e-6;
    for (Eigen::Index index = 0; index < theta.size(); ++index) {
        Vector shifted = theta;
        shifted[index] += step;
        kernel.set_theta(shifted).value();
        const double upper = kernel.value(x, y, false).value();
        shifted[index] -= 2.0 * step;
        kernel.set_theta(shifted).value();
        const double lower = kernel.value(x, y, false).value();
        EXPECT_NEAR(gradient[index], (upper - lower) / (2.0 * step), 1e-8);
        kernel.set_theta(theta).value();
    }
    Vector spatial(x.size());
    kernel.input_gradient(x, y, spatial).value();
    for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
        x[axis] += step;
        const double upper = kernel.value(x, y, false).value();
        x[axis] -= 2.0 * step;
        const double lower = kernel.value(x, y, false).value();
        x[axis] += step;
        EXPECT_NEAR(spatial[axis], (upper - lower) / (2.0 * step), 1e-8);
    }
    const Vector updated = theta.array() + 0.1;
    kernel.set_theta(updated).value();
    for (const auto &component : kernel.components()) {
        EXPECT_TRUE(component.theta().isApprox(updated));
    }
    auto fixed = make_shared_additive_kernel({{0}, {1}}, RBF::create(Parameter::fixed(0.7)).value()).value();
    fixed.set_theta(Vector{}).value();
    EXPECT_EQ(fixed.parameter_count(), 0);
}

TEST(SharedAdditiveGP, FittedComponentsReconstructTheJointPrediction) {
    auto kernel = make_shared_additive_kernel({{0}, {1}}, ConstantKernel::create({0.5}).value() * RBF::create(Parameter{0.7}).value()).value();
    GPROptions options;
    options.normalize_y = true;
    auto model = make_additive_gp(kernel, {0.01, 1e-6, 0.5}, options).value();
    Matrix x(5, 2);
    x << 0.0, 0.1, 0.5, 0.8, 1.0, -0.3, 1.5, 0.4, 2.0, -0.5;
    const Vector y = x.col(0).array().sin() + x.col(1).array().cos();
    model.fit(x, y).value();
    EXPECT_EQ(model.kernel().parameter_count(), 3);
    const auto &components = model.kernel().left().components();
    Matrix reconstructed =
        model.predict_component(components[0], x).value().mean + model.predict_component(components[1], x).value().mean;
    reconstructed.rowwise() += model.mean_offset().value().transpose();
    EXPECT_TRUE(reconstructed.isApprox(model.predict(x).value().mean));
}

TEST(GPRPrior, PosteriorGradientMatchesFiniteDifferencesAndLikelihoodStaysUnpenalized) {
    auto kernel = ConstantKernel::create({1.2}).value() * RBF::create(Parameter{0.7}).value() + WhiteKernel::create({0.03}).value();
    GPROptions options;
    options.optimize = false;
    options.prior.mean = kernel.theta().array() + 0.2;
    options.prior.standard_deviation =
        (Vector(3) << 0.5, 1.0, std::numeric_limits<double>::infinity()).finished();
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    Matrix x(5, 1);
    x << 0.0, 0.3, 0.7, 1.1, 1.8;
    const Vector y = x.col(0).array().sin();
    model.fit(x, y).value();
    const Vector theta = model.kernel().theta();
    const auto likelihood = model.log_marginal_likelihood(theta).value();
    const auto posterior = model.log_posterior(theta).value();
    EXPECT_NEAR(likelihood.value - posterior.value, 0.1, 1e-12);
    EXPECT_NEAR(posterior.gradient[0] - likelihood.gradient[0], 0.8, 1e-12);
    EXPECT_NEAR(posterior.gradient[1] - likelihood.gradient[1], 0.2, 1e-12);
    EXPECT_DOUBLE_EQ(posterior.gradient[2], likelihood.gradient[2]);
    EXPECT_NEAR(model.log_posterior().value(), posterior.value, 1e-12);
    EXPECT_NEAR(model.log_marginal_likelihood().value(), likelihood.value, 1e-12);
    constexpr double step = 1e-6;
    for (Eigen::Index index = 0; index < theta.size(); ++index) {
        Vector shifted = theta;
        shifted[index] += step;
        const double upper = model.log_posterior(shifted).value().value;
        shifted[index] -= 2.0 * step;
        const double lower = model.log_posterior(shifted).value().value;
        EXPECT_NEAR(posterior.gradient[index], (upper - lower) / (2.0 * step), 1e-7);
    }
}

TEST(GPRPrior, MAPOptimizationImprovesThePenalizedObjectiveAndRejectsInvalidPrior) {
    const auto kernel = ConstantKernel::create({1.2}).value() * RBF::create(Parameter{0.7}).value() + WhiteKernel::create({0.03}).value();
    GPROptions options;
    options.prior.mean = kernel.theta();
    options.prior.standard_deviation = Vector::Constant(3, 0.01);
    Matrix x(5, 1);
    x << 0.0, 0.3, 0.7, 1.1, 1.8;
    const Vector y = x.col(0).array().sin();
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    model.fit(x, y).value();
    EXPECT_GE(model.log_posterior().value(), model.log_posterior(kernel.theta()).value().value);
    EXPECT_LT((model.kernel().theta() - kernel.theta()).norm(), 0.05);
    options.prior.standard_deviation = Vector::Ones(2);
    GaussianProcessRegressor<decltype(kernel)> invalid(kernel, options);
    const auto failure = invalid.fit(x, y);
    ASSERT_FALSE(failure);
    EXPECT_EQ(failure.error(), GPRError::InvalidPrior);
}
} // namespace
