#include <gtest/gtest.h>
#include <super_gpr/acquisition.hpp>

namespace {
using namespace super_gpr;

Matrix training_x() {
    Matrix x(6, 2);
    x << 0.0, 0.1, 0.4, -0.2, 0.9, 0.5, 1.3, 0.7, 1.8, -0.1, 2.4, 0.4;
    return x;
}
Vector training_y() {
    Vector y(6);
    y << 1.2, 0.7, -0.1, -0.8, -0.3, 0.4;
    return y;
}
std::vector<BoxedKernel> kernels() {
    return {BoxedKernel(RBF::create(Parameter{0.7}).value()),
            BoxedKernel(RBF::create(std::vector<Parameter>{{0.7}, {1.3}}).value()),
            BoxedKernel(Matern::create(Parameter{0.7}, 0.5).value()),
            BoxedKernel(Matern::create(Parameter{0.7}, 1.5).value()),
            BoxedKernel(Matern::create(std::vector<Parameter>{{0.7}, {1.3}}, 2.5).value()),
            BoxedKernel(Matern::create(Parameter{0.7}, std::numeric_limits<double>::infinity()).value()),
            BoxedKernel(RationalQuadratic::create({0.8}, {1.4}).value()),
            BoxedKernel(ExpSineSquared::create({0.8}, {1.7}).value()),
            BoxedKernel(DotProduct::create({0.6}).value()),
            BoxedKernel(ConstantKernel::create({1.2}).value()),
            BoxedKernel(WhiteKernel::create({0.2}).value()),
            BoxedKernel(1.7 * (RBF{} + DotProduct{}) * (ConstantKernel{} + WhiteKernel{}))};
}

TEST(Kernel, OperatorsPreserveConcreteTemplateTypes) {
    auto sum = RBF{} + WhiteKernel{};
    auto product = ConstantKernel{} * RBF{};
    auto nested = (RBF{} + DotProduct{}) * WhiteKernel{};
    static_assert(std::is_same<decltype(sum), AddKernel<RBF, WhiteKernel>>::value,
                  "Sum preserves operand types");
    static_assert(std::is_same<decltype(product), Multiple<ConstantKernel, RBF>>::value,
                  "Product preserves operand types");
    static_assert(std::is_same<decltype(nested), Multiple<AddKernel<RBF, DotProduct>, WhiteKernel>>::value,
                  "Nested types are preserved");
    EXPECT_EQ(sum.parameter_count(), 2);
    EXPECT_EQ(product.parameter_count(), 2);
    EXPECT_EQ(nested.parameter_count(), 3);
}

TEST(Kernel, AnalyticHyperparameterDerivatives) {
    const Vector x = (Vector(2) << 0.3, -0.4).finished();
    const Vector y = (Vector(2) << 1.2, 0.6).finished();
    constexpr double step = 1e-6;
    for (auto kernel : kernels()) {
        const Vector theta = kernel.theta();
        Vector gradient(theta.size());
        for (bool same : {false, true}) {
            kernel.gradient(x, y, same, gradient).value();
            for (Eigen::Index index = 0; index < theta.size(); ++index) {
                Vector shifted = theta;
                shifted[index] += step;
                kernel.set_theta(shifted).value();
                const double upper = kernel.value(x, y, same).value();
                shifted[index] -= 2.0 * step;
                kernel.set_theta(shifted).value();
                const double lower = kernel.value(x, y, same).value();
                EXPECT_NEAR(gradient[index], (upper - lower) / (2.0 * step), 1e-8);
                kernel.set_theta(theta).value();
            }
        }
    }
}

TEST(Kernel, AnalyticInputAndDiagonalDerivatives) {
    Vector x = (Vector(2) << 0.3, -0.4).finished();
    const Vector y = (Vector(2) << 1.2, 0.6).finished();
    constexpr double step = 1e-6;
    for (const auto &kernel : kernels()) {
        Vector input(2);
        Vector diagonal(2);
        kernel.input_gradient(x, y, input).value();
        kernel.diagonal_gradient(x, diagonal).value();
        for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
            x[axis] += step;
            const double upper = kernel.value(x, y, false).value();
            const double diagonal_upper = kernel.value(x, x, true).value();
            x[axis] -= 2.0 * step;
            const double lower = kernel.value(x, y, false).value();
            const double diagonal_lower = kernel.value(x, x, true).value();
            x[axis] += step;
            EXPECT_NEAR(input[axis], (upper - lower) / (2.0 * step), 1e-8);
            EXPECT_NEAR(diagonal[axis], (diagonal_upper - diagonal_lower) / (2.0 * step), 1e-8);
        }
    }
}

TEST(Kernel, WhiteNoiseDistinguishesObservationsWithIdenticalFeatures) {
    const auto kernel = WhiteKernel::create({0.3}).value();
    const Matrix x = Matrix::Zero(3, 2);
    EXPECT_TRUE(covariance(kernel, x).value().isApprox(0.3 * Matrix::Identity(3, 3)));
    EXPECT_TRUE(covariance(kernel, x, x).value().isZero());
}

TEST(Kernel, FixedParametersAndScalars) {
    auto kernel = 2.0 * (ConstantKernel::create(Parameter::fixed(3.0)).value() * RBF::create({0.7, 0.1, 2.0}).value());
    EXPECT_EQ(kernel.parameter_count(), 1);
    EXPECT_NEAR(kernel.theta()[0], std::log(0.7), 1e-15);
    EXPECT_NEAR(kernel.bounds().lower[0], std::log(0.1), 1e-15);
    const Vector point = Vector::Zero(2);
    EXPECT_DOUBLE_EQ(kernel.value(point, point, true).value(), 6.0);
    auto fixed = ConstantKernel::create(Parameter::fixed(2.0)).value() * ConstantKernel::create(Parameter::fixed(3.0)).value();
    EXPECT_EQ(fixed.theta().size(), 0);
    EXPECT_EQ(fixed.bounds().lower.size(), 0);
    fixed.set_theta(Vector{}).value();
}

TEST(Kernel, RejectsMismatchedARDFeatureCounts) {
    const auto kernel = RBF::create(std::vector<Parameter>{{0.7}, {1.3}}).value();
    const Matrix x = Matrix::Zero(2, 3);
    EXPECT_EQ(covariance(kernel, x).error(), GPRError::InvalidKernelInput);
    EXPECT_EQ(covariance(RBF{}, x, Matrix::Zero(2, 2)).error(), GPRError::InvalidPredictionShape);
    const Vector wrong = Vector::Zero(3);
    const Vector point = Vector::Zero(2);
    Vector gradient = Vector::Ones(2);
    Vector wrong_gradient = Vector::Ones(3);
    EXPECT_EQ(kernel.value(wrong, wrong, true).error(), GPRError::InvalidKernelInput);
    EXPECT_EQ(kernel.gradient(point, wrong, false, gradient).error(), GPRError::InvalidKernelInput);
    EXPECT_EQ(kernel.value_and_gradient(point, point, true, wrong_gradient).error(), GPRError::InvalidKernelInput);
    EXPECT_EQ(kernel.input_gradient(point, point, wrong_gradient).error(), GPRError::InvalidKernelInput);
    EXPECT_TRUE(gradient.isOnes());
    EXPECT_TRUE(wrong_gradient.isOnes());
}

TEST(Kernel, ReportsInvalidParametersAndTheta) {
    EXPECT_EQ(RBF::create(Parameter{0.0, 0.1, 1.0}).error(), GPRError::InvalidKernelParameters);
    EXPECT_EQ(RBF::create(std::vector<Parameter>{}).error(), GPRError::InvalidKernelParameters);
    EXPECT_EQ(Matern::create(Parameter{}, 2.0).error(), GPRError::InvalidMaternNu);
    auto kernel = ConstantKernel{} * RBF{};
    const auto theta = kernel.theta();
    EXPECT_EQ(kernel.set_theta(Vector::Zero(1)).error(), GPRError::InvalidThetaSize);
    EXPECT_TRUE(kernel.theta().isApprox(theta));
}

TEST(Kernel, BoxedCopiesHaveIndependentParameters) {
    const BoxedKernel original(RBF{} + WhiteKernel{});
    BoxedKernel copied = original;
    copied.set_theta(Vector::Constant(2, std::log(2.0))).value();
    EXPECT_TRUE(original.theta().isZero());
    EXPECT_FALSE(copied.theta().isZero());
}

TEST(Kernel, UserSuppliedAnalyticCallbacksCompose) {
    auto kernel = make_analytic_kernel(
        Vector::Constant(1, std::log(2.0)), Bounds{Vector::Constant(1, -5.0), Vector::Constant(1, 5.0)},
        [](Point x, Point y, bool /*same*/, Point theta) { return std::exp(theta[0]) * x.dot(y); },
        [](Point x, Point y, bool /*same*/, Point theta, Gradient out) {
            out[0] = std::exp(theta[0]) * x.dot(y);
        },
        [](Point /*x*/, Point y, Point theta, Gradient out) { out = std::exp(theta[0]) * y; },
        [](Point x, Point theta, Gradient out) { out = 2.0 * std::exp(theta[0]) * x; });
    static_assert(IsKernel<decltype(kernel)>::value, "Analytic callbacks implement Kernel");
    const Vector x = Vector::Ones(2);
    const auto combined = kernel * RBF{};
    EXPECT_DOUBLE_EQ(combined.value(x, x, true).value(), 4.0);
    Vector gradient(2);
    combined.gradient(x, x, true, gradient).value();
    EXPECT_DOUBLE_EQ(gradient[0], 4.0);
    EXPECT_DOUBLE_EQ(gradient[1], 0.0);
}

TEST(Optimizer, BoundsAndFixedCoordinates) {
    const Objective quadratic = [](const Vector &x, Vector &gradient) {
        gradient = 2.0 * (x.array() - 3.0).matrix();
        return (x.array() - 3.0).square().sum();
    };
    const Bounds bounds{Vector::Zero(3), (Vector(3) << 2.0, 0.0, 5.0).finished()};
    const auto result = minimize_lbfgsb(quadratic, Vector::Constant(3, -1.0), bounds).value();
    EXPECT_TRUE(result.converged());
    EXPECT_NEAR(result.x[0], 2.0, 1e-8);
    EXPECT_DOUBLE_EQ(result.x[1], 0.0);
    EXPECT_NEAR(result.x[2], 3.0, 1e-8);
    EXPECT_NEAR(result.value, 10.0, 1e-8);
}

TEST(Optimizer, RosenbrockAndInfiniteBounds) {
    const Objective function = [](const Vector &x, Vector &gradient) {
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        gradient.resize(2);
        gradient << -2.0 * a - 400.0 * x[0] * b, 200.0 * b;
        return a * a + 100.0 * b * b;
    };
    const double infinity = std::numeric_limits<double>::infinity();
    const auto result = minimize_lbfgsb(function, (Vector(2) << -1.2, 1.0).finished(),
                                        {Vector::Constant(2, -infinity), Vector::Constant(2, infinity)}).value();
    EXPECT_TRUE(result.converged());
    EXPECT_LT(result.value, 1e-12);
    EXPECT_TRUE(result.x.isApprox(Vector::Ones(2), 1e-5));
}

TEST(Optimizer, ReportsIterationLimitAndCallbackErrors) {
    LBFGSBOptions options;
    options.max_iterations = 1;
    options.function_tolerance = 0.0;
    const Bounds bounds{Vector::Constant(2, -10.0), Vector::Constant(2, 10.0)};
    const Objective function = [](const Vector &x, Vector &gradient) {
        gradient = 2.0 * x;
        gradient[1] *= 10.0;
        return x[0] * x[0] + 10.0 * x[1] * x[1];
    };
    const auto result = minimize_lbfgsb(function, Vector::Constant(2, 2.0), bounds, options).value();
    EXPECT_EQ(result.status, OptimizationStatus::IterationLimit);
    const auto failed = minimize_lbfgsb(
        [](const Vector &, Vector &) -> ObjectiveResult {
            return boost::outcome_v2::failure(OptimizationError::CallbackFailed);
        }, Vector::Ones(2), bounds);
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error(), OptimizationError::CallbackFailed);
    int evaluations = 0;
    options.max_iterations = 100;
    const auto interrupted = minimize_lbfgsb(
        [&](const Vector &x, Vector &gradient) -> ObjectiveResult {
            if (++evaluations == 2) {
                return boost::outcome_v2::failure(OptimizationError::CallbackFailed);
            }
            return function(x, gradient);
        }, Vector::Ones(2), bounds, options);
    ASSERT_FALSE(interrupted);
    EXPECT_EQ(interrupted.error(), OptimizationError::CallbackFailed);
    EXPECT_EQ(evaluations, 2);
    options.memory = 0;
    const auto invalid = minimize_lbfgsb(function, Vector::Ones(2), bounds, options);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error(), OptimizationError::InvalidOptions);
}

TEST(GPR, LMLGradientAndPosteriorInputDerivatives) {
    GPROptions options;
    options.optimize = false;
    options.normalize_y = true;
    auto kernel = ConstantKernel::create({1.2}).value() * RBF::create(std::vector<Parameter>{{0.7}, {1.1}}).value() + WhiteKernel::create({0.03}).value();
    GaussianProcessRegressor<decltype(kernel)> model(kernel, options);
    model.fit(training_x(), training_y()).value();
    const Vector theta = model.kernel().theta();
    const auto likelihood = model.log_marginal_likelihood(theta).value();
    constexpr double step = 1e-6;
    for (Eigen::Index index = 0; index < theta.size(); ++index) {
        Vector shifted = theta;
        shifted[index] += step;
        const double upper = model.log_marginal_likelihood(shifted).value().value;
        shifted[index] -= 2.0 * step;
        const double lower = model.log_marginal_likelihood(shifted).value().value;
        EXPECT_NEAR(likelihood.gradient[index], (upper - lower) / (2.0 * step), 1e-7);
    }
    Vector point = (Vector(2) << 0.6, 0.3).finished();
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
    EXPECT_TRUE(model.kernel().theta().isApprox(theta));
}

TEST(GPR, OptimizationImprovesLikelihoodAndRestartsAreDeterministic) {
    const auto kernel = ConstantKernel{} * RBF::create({0.2, 0.05, 5.0}).value() + WhiteKernel::create({0.01, 1e-6, 1.0}).value();
    GPROptions fixed;
    fixed.optimize = false;
    GaussianProcessRegressor<decltype(kernel)> baseline(kernel, fixed);
    baseline.fit(training_x(), training_y()).value();
    GPROptions options;
    options.optimizer_restarts = 2;
    GaussianProcessRegressor<decltype(kernel)> first(kernel, options);
    GaussianProcessRegressor<decltype(kernel)> second(kernel, options);
    first.fit(training_x(), training_y()).value();
    second.fit(training_x(), training_y()).value();
    EXPECT_GT(first.log_marginal_likelihood().value(), baseline.log_marginal_likelihood().value());
    EXPECT_TRUE(first.kernel().theta().isApprox(second.kernel().theta()));
    EXPECT_EQ(first.optimization_results().size(), 3);
}

TEST(GPR, MultiOutputNoiseAndCovarianceConsistency) {
    GPROptions options;
    options.optimize = false;
    options.normalize_y = true;
    options.sample_noise = Vector::Constant(6, 0.01);
    Matrix targets(6, 2);
    targets.col(0) = training_y();
    targets.col(1) = 3.0 * training_y().array() + 7.0;
    GaussianProcessRegressor<RBF> model(RBF{}, options);
    model.fit(training_x(), targets).value();
    const auto prediction = model.predict(training_x(), true).value();
    EXPECT_TRUE(prediction.mean.col(1).isApprox((3.0 * prediction.mean.col(0).array() + 7.0).matrix()));
    EXPECT_TRUE(prediction.standard_deviation.col(1).isApprox(3.0 * prediction.standard_deviation.col(0)));
    EXPECT_TRUE(prediction.covariance[1].isApprox(9.0 * prediction.covariance[0]));
    EXPECT_TRUE(prediction.covariance[0].diagonal().isApprox(
        prediction.standard_deviation.col(0).array().square().matrix()));
    const auto empty = model.predict(Matrix(0, 2), true).value();
    EXPECT_EQ(empty.mean.rows(), 0);
    EXPECT_EQ(empty.mean.cols(), targets.cols());
    EXPECT_EQ(empty.standard_deviation.rows(), 0);
    ASSERT_EQ(empty.covariance.size(), 2u);
    EXPECT_EQ(empty.covariance[0].size(), 0);
    const Vector theta = model.kernel().theta();
    const auto likelihood = model.log_marginal_likelihood(theta).value();
    const double step = 1e-6;
    EXPECT_NEAR(likelihood.gradient[0],
                (model.log_marginal_likelihood(theta.array() + step).value().value -
                 model.log_marginal_likelihood(theta.array() - step).value().value) /
                    (2.0 * step),
                1e-7);
}

TEST(GPR, PriorConstantTargetsAndNumericalFailures) {
    GaussianProcessRegressor<RBF> prior(RBF{});
    const auto prediction = prior.predict(training_x(), true).value();
    EXPECT_TRUE(prediction.mean.isZero());
    EXPECT_TRUE(prediction.standard_deviation.isOnes());
    GPROptions options;
    options.normalize_y = true;
    options.optimize = false;
    GaussianProcessRegressor<RBF> model(RBF{}, options);
    const Vector targets = Vector::Constant(6, 7.0);
    model.fit(training_x(), targets).value();
    EXPECT_TRUE(model.predict(training_x()).value().mean.isApprox(targets));
    options.alpha = 0.0;
    GaussianProcessRegressor<RBF> singular(RBF{}, options);
    const Matrix duplicate = Matrix::Zero(2, 1);
    const Vector y = Vector::Ones(2);
    const auto failed = singular.fit(duplicate, y);
    ASSERT_FALSE(failed);
    EXPECT_EQ(failed.error(), GPRError::NonPositiveDefiniteCovariance);
}

TEST(GPR, OutcomeReportsKnownErrors) {
    Matrix x(2, 1);
    x << 0.0, 1.0;
    const Vector y = Vector::Ones(2);
    GPROptions options;
    options.optimize = false;
    options.alpha = -0.1;
    GaussianProcessRegressor<RBF> model(RBF::create(Parameter{0.1}).value(), options);
    const Vector initial = model.kernel().theta();
    const auto unfitted = model.log_marginal_likelihood(initial);
    ASSERT_FALSE(unfitted);
    EXPECT_EQ(unfitted.error(), GPRError::NotFitted);
    const auto unfitted_prediction = model.predict_with_gradients(Vector::Zero(1));
    ASSERT_FALSE(unfitted_prediction);
    EXPECT_EQ(unfitted_prediction.error(), GPRError::NotFitted);

    ASSERT_TRUE(model.fit(x, y));
    const auto likelihood = model.log_marginal_likelihood(initial);
    ASSERT_TRUE(likelihood);
    EXPECT_DOUBLE_EQ(likelihood.value().value, model.log_marginal_likelihood().value());

    const Vector empty;
    const auto invalid_fit = model.fit(x, empty);
    ASSERT_FALSE(invalid_fit);
    EXPECT_EQ(invalid_fit.error(), GPRError::InvalidTrainingShape);
    EXPECT_DOUBLE_EQ(model.log_marginal_likelihood().value(), likelihood.value().value);
    const auto invalid_prediction = model.predict(Matrix::Zero(1, 2));
    ASSERT_FALSE(invalid_prediction);
    EXPECT_EQ(invalid_prediction.error(), GPRError::InvalidPredictionShape);
    GaussianProcessRegressor<RBF> ard(RBF::create(std::vector<Parameter>{{0.7}, {1.1}}).value(), options);
    const auto invalid_kernel = ard.fit(x, y);
    ASSERT_FALSE(invalid_kernel);
    EXPECT_EQ(invalid_kernel.error(), GPRError::InvalidKernelInput);

    const auto invalid_theta = model.log_marginal_likelihood(empty);
    ASSERT_FALSE(invalid_theta);
    EXPECT_EQ(invalid_theta.error(), GPRError::InvalidThetaSize);
    const Vector long_scale = Vector::Constant(1, std::log(10.0));
    const auto indefinite = model.log_marginal_likelihood(long_scale);
    ASSERT_FALSE(indefinite);
    EXPECT_EQ(indefinite.error(), GPRError::NonPositiveDefiniteCovariance);

    options.alpha = 0.0;
    const Matrix duplicate = Matrix::Zero(2, 1);
    for (const bool optimize : {false, true}) {
        options.optimize = optimize;
        GaussianProcessRegressor<RBF> singular(RBF{}, options);
        const auto result = singular.fit(duplicate, y);
        ASSERT_FALSE(result);
        EXPECT_EQ(result.error(), GPRError::NonPositiveDefiniteCovariance);
    }
}

TEST(Optimizer, RejectsNonFiniteObjectiveWithAllCoordinatesFixed) {
    const Bounds bounds{Vector::Zero(1), Vector::Zero(1)};
    const auto objective = [](const Vector &, Vector &gradient) {
        gradient.setZero();
        return std::numeric_limits<double>::quiet_NaN();
    };
    const auto result = minimize_lbfgsb(objective, bounds.lower, bounds);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), OptimizationError::NoFiniteObjective);
    const auto invalid_bounds = minimize_lbfgsb(objective, Vector::Zero(2), bounds);
    ASSERT_FALSE(invalid_bounds);
    EXPECT_EQ(invalid_bounds.error(), OptimizationError::InvalidBounds);
}

TEST(Acquisition, ValuesAndPartialDerivatives) {
    const PredictedMean mean;
    const LCB lcb(2.0);
    const ConservativeUCB ucb(2.0);
    const EI ei(0.3, 0.01);
    const PI pi(0.3, 0.01);
    EXPECT_DOUBLE_EQ(mean.evaluate(1.0, 0.5).value, 1.0);
    EXPECT_DOUBLE_EQ(lcb.evaluate(1.0, 0.5).value, 0.0);
    EXPECT_DOUBLE_EQ(ucb.evaluate(1.0, 0.5).value, 2.0);
    constexpr double step = 1e-6;
    for (const AcquisitionFunction *function :
         std::vector<const AcquisitionFunction *>{&mean, &lcb, &ucb, &ei, &pi}) {
        const auto value = function->evaluate(0.2, 0.4);
        EXPECT_NEAR(value.mean_derivative,
                    (function->evaluate(0.2 + step, 0.4).value - function->evaluate(0.2 - step, 0.4).value) /
                        (2.0 * step),
                    1e-8);
        EXPECT_NEAR(value.deviation_derivative,
                    (function->evaluate(0.2, 0.4 + step).value - function->evaluate(0.2, 0.4 - step).value) /
                        (2.0 * step),
                    1e-8);
    }
    EXPECT_DOUBLE_EQ(ei.evaluate(0.1, 0.0).value, 0.19);
    EXPECT_DOUBLE_EQ(pi.evaluate(0.1, 0.0).value, 1.0);
    EXPECT_DOUBLE_EQ(PI(0.0).evaluate(0.0, 0.0).value, 0.0);
    const auto tail = PI(1.0).evaluate(0.0, std::numeric_limits<double>::min());
    EXPECT_DOUBLE_EQ(tail.value, 1.0);
    EXPECT_DOUBLE_EQ(tail.mean_derivative, 0.0);
    EXPECT_DOUBLE_EQ(tail.deviation_derivative, 0.0);
}

TEST(Acquisition, OptimizationMatchesDenseGridAndNativeSign) {
    Matrix x(5, 1);
    x << -2.0, -1.0, 0.0, 1.0, 2.0;
    const Vector y = (x.col(0).array() - 0.3).square();
    GPROptions options;
    options.optimize = false;
    GaussianProcessRegressor<RBF> model(RBF{}, options);
    model.fit(x, y).value();
    const Bounds bounds{Vector::Constant(1, -2.0), Vector::Constant(1, 2.0)};
    const PredictedMean mean;
    const auto no_starts = optimize_acquisition(model, mean, Matrix(0, 1), bounds);
    ASSERT_FALSE(no_starts);
    EXPECT_EQ(no_starts.error(), GPRError::NoAcquisitionStarts);
    const auto invalid_bounds =
        optimize_acquisition(model, mean, Matrix::Zero(1, 1), {Vector::Ones(1), Vector::Zero(1)});
    ASSERT_FALSE(invalid_bounds);
    EXPECT_EQ(invalid_bounds.error(), GPRError::InvalidOptimizerBounds);
    const EI ei(y.minCoeff(), 0.01);
    const LCB lcb;
    const PI pi(y.minCoeff(), 0.01);
    const ConservativeUCB ucb;
    for (const AcquisitionFunction *function :
         std::vector<const AcquisitionFunction *>{&mean, &ei, &lcb, &pi, &ucb}) {
        Matrix starts(17, 1);
        starts.col(0) = Vector::LinSpaced(17, -2.0, 2.0);
        const auto result = optimize_acquisition(model, *function, starts, bounds).value();
        const auto posterior = model.predict_with_gradients(result.x).value();
        EXPECT_NEAR(result.value, function->evaluate(posterior.mean, posterior.standard_deviation).value,
                    1e-12);
        Matrix grid(2001, 1);
        grid.col(0) = Vector::LinSpaced(2001, -2.0, 2.0);
        const auto prediction = model.predict(grid).value();
        double best = function->maximize() ? -1e100 : 1e100;
        for (Eigen::Index row = 0; row < grid.rows(); ++row) {
            const double value =
                function->evaluate(prediction.mean(row, 0), prediction.standard_deviation(row, 0)).value;
            best = function->maximize() ? std::max(best, value) : std::min(best, value);
        }
        EXPECT_LE((result.value - best) * (function->maximize() ? -1.0 : 1.0), 1e-5);
    }
}

TEST(ParameterGrid, EnumeratesInFeatureOrderAndValidatesInputs) {
    const auto grid = ParameterGrid::create({{0.5, 0.0}, {3.0}, {5.0, 1.0}}).value();
    ASSERT_EQ(grid.dimensions(), 3);
    ASSERT_EQ(grid.size(), 4);
    Matrix expected(4, 3);
    expected << 0.5, 3.0, 5.0, 0.5, 3.0, 1.0, 0.0, 3.0, 5.0, 0.0, 3.0, 1.0;
    for (Eigen::Index index = 0; index < grid.size(); ++index) {
        EXPECT_TRUE(grid[index].value().isApprox(expected.row(index).transpose()));
    }
    EXPECT_EQ(grid[-1].error(), GPRError::InvalidGridIndex);
    EXPECT_EQ(grid[grid.size()].error(), GPRError::InvalidGridIndex);
    EXPECT_EQ(ParameterGrid::create({}).error(), GPRError::EmptyParameterGrid);
    EXPECT_EQ(ParameterGrid::create({{0.0}, {}}).error(), GPRError::EmptyGridAxis);
    EXPECT_EQ(ParameterGrid::create({{std::numeric_limits<double>::infinity()}}).error(),
              GPRError::NonFiniteCandidates);
    const std::vector<std::vector<double>> oversized(std::numeric_limits<Eigen::Index>::digits, {0.0, 1.0});
    EXPECT_EQ(ParameterGrid::create(oversized).error(), GPRError::GridSizeOverflow);
}

TEST(Acquisition, SuggestMatchesExhaustiveScoringForCandidatesAndGrid) {
    const Matrix x = training_x();
    const Vector y = training_y();
    GPROptions options;
    options.normalize_y = true;
    options.optimize = false;
    GaussianProcessRegressor<RBF> model(RBF{}, options);
    ASSERT_TRUE(model.fit(x, y));

    const auto grid = ParameterGrid::create({{0.0, 0.4, 1.3}, {-0.2, 0.7}}).value();
    Matrix candidates(6, 2);
    candidates << 0.0, -0.2, 0.0, 0.7, 0.4, -0.2, 0.4, 0.7, 1.3, -0.2, 1.3, 0.7;
    const auto prediction = model.predict(candidates).value();
    const EI ei(y.minCoeff(), 0.01);
    const LCB lcb;
    SuggestOptions search;
    search.excluded_indices = {2, 0, 2};
    for (const AcquisitionFunction *function : std::vector<const AcquisitionFunction *>{&ei, &lcb}) {
        Eigen::Index expected_index = -1;
        double expected_value = 0.0;
        for (const Eigen::Index row : {1, 3, 4, 5}) {
            const double value =
                function->evaluate(prediction.mean(row, 0), prediction.standard_deviation(row, 0)).value;
            if (expected_index < 0 ||
                (function->maximize() ? value > expected_value : value < expected_value)) {
                expected_index = row;
                expected_value = value;
            }
        }
        ASSERT_GE(expected_index, 4);
        for (const Eigen::Index batch_size : {2, 3}) {
            search.batch_size = batch_size;
            for (const auto &next :
                 {suggest(model, *function, candidates, search).value(),
                  suggest(model, *function, grid, search).value()}) {
                ASSERT_TRUE(next);
                EXPECT_EQ(next.index, expected_index);
                EXPECT_TRUE(next.x.isApprox(candidates.row(expected_index).transpose()));
                EXPECT_NEAR(next.value, expected_value, 1e-12);
            }
        }
    }
}

TEST(Acquisition, SuggestUsesFirstAvailableTieAndReportsExhaustion) {
    const Matrix x = training_x();
    const Vector y = Vector::Zero(x.rows());
    GPROptions options;
    options.optimize = false;
    GaussianProcessRegressor<RBF> model(RBF{}, options);
    ASSERT_TRUE(model.fit(x, y));

    const auto grid = ParameterGrid::create({{0.0, 1.0, 2.0}, {0.0}}).value();
    const PredictedMean acquisition;
    const auto first = suggest(model, acquisition, grid).value();
    ASSERT_TRUE(first);
    EXPECT_EQ(first.index, 0);

    SuggestOptions search;
    search.batch_size = 1;
    search.excluded_indices = {first.index};
    const auto next = suggest(model, acquisition, grid, search).value();
    ASSERT_TRUE(next);
    EXPECT_EQ(next.index, 1);
    EXPECT_TRUE(next.x.isApprox(grid[next.index].value()));

    search.excluded_indices = {2, next.index, first.index};
    EXPECT_FALSE(suggest(model, acquisition, grid, search).value());
    EXPECT_FALSE(suggest(model, acquisition, Matrix(0, x.cols())).value());
}

TEST(Acquisition, SuggestValidatesModelAndCandidates) {
    const Matrix x = training_x();
    const Vector y = training_y();
    const LCB acquisition;
    GPROptions options;
    options.optimize = false;
    GaussianProcessRegressor<RBF> model(RBF{}, options);
    EXPECT_EQ(suggest(model, acquisition, x).error(), GPRError::NotFitted);
    ASSERT_TRUE(model.fit(x, y));
    EXPECT_EQ(suggest(model, acquisition, Matrix(0, 1)).error(), GPRError::InvalidPredictionShape);
    EXPECT_EQ(suggest(model, acquisition, ParameterGrid::create({{0.0}}).value()).error(),
              GPRError::InvalidPredictionShape);

    SuggestOptions search;
    search.excluded_indices = {x.rows()};
    EXPECT_EQ(suggest(model, acquisition, x, search).error(), GPRError::InvalidExcludedIndex);
    search.excluded_indices.clear();
    search.batch_size = 0;
    EXPECT_EQ(suggest(model, acquisition, x, search).error(), GPRError::InvalidBatchSize);

    Matrix invalid = x;
    invalid(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(suggest(model, acquisition, invalid).error(), GPRError::NonFiniteCandidates);
    EXPECT_EQ(suggest(model, LCB(std::numeric_limits<double>::infinity()), x).error(),
              GPRError::NonFiniteAcquisition);

    Matrix targets(x.rows(), 2);
    targets.col(0) = y;
    targets.col(1) = y;
    ASSERT_TRUE(model.fit(x, targets));
    EXPECT_EQ(suggest(model, acquisition, x).error(), GPRError::NonScalarTarget);
}
} // namespace
