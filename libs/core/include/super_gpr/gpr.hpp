#pragma once

#include <boost/outcome/try.hpp>
#include <cstdint>
#include <random>
#include <super_gpr/kernel.hpp>
#include <super_gpr/optimizer.hpp>

namespace super_gpr {

namespace detail {

inline GPRError optimizer_error(OptimizationError error) {
    switch (error) {
    case OptimizationError::InvalidBounds: return GPRError::InvalidOptimizerBounds;
    case OptimizationError::InvalidOptions: return GPRError::InvalidOptimizerOptions;
    case OptimizationError::NoFiniteObjective: return GPRError::NoFiniteObjective;
    case OptimizationError::CallbackFailed: return GPRError::CallbackFailed;
    }
    return GPRError::NoFiniteObjective;
}

// LAPACKE uses the lower triangle of a column-major buffer.
class Cholesky {
public:
    GPRResult<void> compute(Eigen::MatrixXd matrix);
    Eigen::MatrixXd solve(Eigen::MatrixXd rhs) const;
    Eigen::MatrixXd solve_lower(Eigen::MatrixXd rhs) const;
    double log_diagonal() const;
    Eigen::MatrixXd take_inverse_lower() &&;

private:
    Eigen::MatrixXd factor_;
};

using TrainingPoints = Eigen::Ref<const Matrix, 0, Eigen::OuterStride<>>;

// The training matrix needs only its lower triangle for LLT. The same
// triangle is used for the symmetric likelihood contraction.
template <class K> struct PairwiseTrainingKernel {
    static constexpr bool split_covariance = false;

    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        for (Eigen::Index col = 0; col < x.rows(); ++col) {
            const auto second = x.row(col).transpose();
            for (Eigen::Index row = col; row < x.rows(); ++row) {
                matrix(row, col) += kernel.value_unchecked(x.row(row).transpose(), second, row == col);
            }
        }
    }

    static void contract(const K &kernel, TrainingPoints x, const Eigen::MatrixXd &contraction,
                         Gradient out) {
        if (out.size() == 0) {
            return;
        }
        Vector scratch(out.size());
        for (Eigen::Index row = 0; row < x.rows(); ++row) {
            const auto first = x.row(row).transpose();
            for (Eigen::Index col = 0; col <= row; ++col) {
                kernel.gradient_unchecked(first, x.row(col).transpose(), row == col, scratch);
                out.noalias() += scratch * (contraction(row, col) * (row == col ? 0.5 : 1.0));
            }
        }
    }
};

template <class K> struct TrainingKernel : PairwiseTrainingKernel<K> {};

template <> struct TrainingKernel<WhiteKernel> {
    static constexpr bool split_covariance = false;

    static void add_covariance(const WhiteKernel &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        const auto point = x.row(0).transpose();
        matrix.diagonal().array() += kernel.value_unchecked(point, point, true);
    }

    static void contract(const WhiteKernel &kernel, TrainingPoints x, const Eigen::MatrixXd &contraction,
                         Gradient out) {
        if (out.size() != 0) {
            const auto point = x.row(0).transpose();
            out[0] += (0.5 * kernel.value_unchecked(point, point, true)) * contraction.diagonal().sum();
        }
    }
};

template <class K1, class K2> struct TrainingKernel<AddKernel<K1, K2>> {
    using K = AddKernel<K1, K2>;
    static constexpr bool split_covariance =
        TrainingKernel<K1>::split_covariance || TrainingKernel<K2>::split_covariance;

    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        add_covariance(kernel, x, matrix, std::integral_constant<bool, split_covariance>{});
    }

    static void contract(const K &kernel, TrainingPoints x, const Eigen::MatrixXd &contraction,
                         Gradient out) {
        TrainingKernel<K1>::contract(kernel.left(), x, contraction,
                                     out.head(kernel.left().parameter_count()));
        TrainingKernel<K2>::contract(kernel.right(), x, contraction,
                                     out.tail(kernel.right().parameter_count()));
    }

private:
    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix,
                               std::true_type) {
        TrainingKernel<K1>::add_covariance(kernel.left(), x, matrix);
        TrainingKernel<K2>::add_covariance(kernel.right(), x, matrix);
    }
    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix,
                               std::false_type) {
        PairwiseTrainingKernel<K>::add_covariance(kernel, x, matrix);
    }
};

inline GPRResult<double> posterior_variance(double prior, double reduction, Eigen::Index observations) {
    const double variance = prior - reduction;
    if (!std::isfinite(variance)) {
        return boost::outcome_v2::failure(GPRError::NonFinitePrediction);
    }
    const double terms = static_cast<double>(std::max<Eigen::Index>(1, observations));
    const double tolerance = (8.0 * std::numeric_limits<double>::epsilon() * terms) *
        std::max(std::abs(prior), std::abs(reduction));
    if (variance < -tolerance) {
        return boost::outcome_v2::failure(GPRError::NegativePredictiveVariance);
    }
    return std::max(0.0, variance);
}

} // namespace detail

struct GaussianLogPrior {
    Vector mean;
    Vector standard_deviation;
};

struct GPROptions {
    double alpha = 1e-10;
    Vector sample_noise;
    bool normalize_y = false;
    bool optimize = true;
    int optimizer_restarts = 0;
    std::uint64_t random_seed = 0;
    LBFGSBOptions optimizer;
    GaussianLogPrior prior;
};

struct Likelihood {
    double value = -std::numeric_limits<double>::infinity();
    Vector gradient;
};

enum class PredictionTarget { Observation, Latent };

struct Prediction {
    Matrix mean;
    Matrix standard_deviation;
    std::vector<Matrix> covariance;
};

struct PosteriorPoint {
    double mean = 0.0;
    double standard_deviation = 0.0;
    Vector mean_gradient;
    Vector standard_deviation_gradient;
};

class AcquisitionFunction;
template <class K> class GaussianProcessRegressor;

template <class K>
GPRResult<OptimizationResult> optimize_acquisition(const GaussianProcessRegressor<K> &model,
                                                   const AcquisitionFunction &acquisition,
                                                   const Matrix &starts, const Bounds &bounds,
                                                   const LBFGSBOptions &options = {});

template <class Kernel> class GaussianProcessRegressor {
    using K = std::decay_t<Kernel>;
    static_assert(IsKernel<K>::value, "K must satisfy the Kernel interface");

public:
    explicit GaussianProcessRegressor(K kernel, GPROptions options = {})
        : GaussianProcessRegressor(std::make_shared<K>(std::move(kernel)), std::move(options)) {}

    GPRResult<void> fit(const Matrix &x, const Vector &y) { return fit(x, Matrix(y)); }
    GPRResult<void> fit(const Matrix &x, const Matrix &y) {
        if (x.rows() == 0 || x.cols() == 0 || y.cols() == 0 || x.rows() != y.rows()) {
            return boost::outcome_v2::failure(GPRError::InvalidTrainingShape);
        }
        if (!x.allFinite() || !y.allFinite()) {
            return boost::outcome_v2::failure(GPRError::NonFiniteTrainingData);
        }
        if (options_.sample_noise.size() != 0 && options_.sample_noise.size() != x.rows()) {
            return boost::outcome_v2::failure(GPRError::InvalidSampleNoiseSize);
        }
        if (options_.sample_noise.size() == 0
                ? (!std::isfinite(options_.alpha) || options_.alpha < 0.0)
                : (!options_.sample_noise.allFinite() || (options_.sample_noise.array() < 0.0).any())) {
            return boost::outcome_v2::failure(GPRError::InvalidTrainingNoise);
        }
        if (options_.optimizer_restarts < 0) {
            return boost::outcome_v2::failure(GPRError::NegativeOptimizerRestarts);
        }
        const auto prior_size = options_.prior.mean.size();
        if (prior_size != options_.prior.standard_deviation.size() ||
            (prior_size != 0 && prior_size != kernel().parameter_count()) ||
            !options_.prior.mean.allFinite() ||
            !(options_.prior.standard_deviation.array() > 0.0).all()) {
            return boost::outcome_v2::failure(GPRError::InvalidPrior);
        }
        BOOST_OUTCOME_TRY(detail::validate_kernel_input(kernel(), x.cols(), 0));
        GaussianProcessRegressor candidate(kernel_, options_);
        BOOST_OUTCOME_TRY(candidate.fit_validated(x, y));
        *this = std::move(candidate);
        return boost::outcome_v2::success();
    }

    const K &kernel() const { return *kernel_; }
    GPRResult<double> log_marginal_likelihood() const {
        BOOST_OUTCOME_TRY(require_fitted());
        return likelihood_;
    }
    const std::vector<OptimizationResult> &optimization_results() const { return optimization_results_; }

    GPRResult<Likelihood> log_marginal_likelihood(Point theta) const {
        if (!fitted_) {
            return boost::outcome_v2::failure(GPRError::NotFitted);
        }
        if (theta.size() != kernel().parameter_count()) {
            return boost::outcome_v2::failure(GPRError::InvalidThetaSize);
        }
        K candidate = kernel();
        BOOST_OUTCOME_TRY(candidate.set_theta(theta));
        return evaluate_likelihood(candidate);
    }

    // Gaussian density in log-parameter coordinates, up to a theta-independent constant.
    GPRResult<double> log_posterior() const {
        BOOST_OUTCOME_TRY(require_fitted());
        const double value = likelihood_ - prior_penalty(kernel().theta());
        if (!std::isfinite(value)) {
            return boost::outcome_v2::failure(GPRError::NonFiniteTrainingResult);
        }
        return value;
    }
    GPRResult<Likelihood> log_posterior(Point theta) const {
        BOOST_OUTCOME_TRY(auto result, log_marginal_likelihood(theta));
        apply_prior(theta, result);
        if (!std::isfinite(result.value) || !result.gradient.allFinite()) {
            return boost::outcome_v2::failure(GPRError::NonFiniteTrainingResult);
        }
        return result;
    }

    GPRResult<Prediction> predict(const Matrix &x, bool return_covariance = false,
                                  PredictionTarget target = PredictionTarget::Observation) const {
        return predict_kernel(kernel(), x, return_covariance, true, target);
    }

    GPRResult<Eigen::Index> n_features_in() const {
        BOOST_OUTCOME_TRY(require_fitted());
        return x_.cols();
    }

    // Condition a constituent prior on the same joint observations and factorization.
    template <class Component>
    GPRResult<Prediction> predict_component(const Component &component, const Matrix &x,
                                 bool return_covariance = false) const {
        BOOST_OUTCOME_TRY(require_fitted());
        BOOST_OUTCOME_TRY(require_component(detail::KernelIdentity(component)));
        return predict_kernel(component, x, return_covariance, false, PredictionTarget::Observation);
    }

    GPRResult<Vector> mean_offset() const {
        BOOST_OUTCOME_TRY(require_fitted());
        return mean_;
    }

    // Distinct additive components are independent a priori, but correlated a posteriori.
    template <class First, class Second>
    GPRResult<std::vector<Matrix>> component_cross_covariance(const First &first, const Second &second,
                                                   const Matrix &x) const {
        BOOST_OUTCOME_TRY(require_fitted());
        if (x.cols() != x_.cols()) {
            return boost::outcome_v2::failure(GPRError::InvalidPredictionShape);
        }
        if (!x.allFinite()) {
            return boost::outcome_v2::failure(GPRError::NonFinitePrediction);
        }
        const detail::KernelIdentity first_id(first);
        const detail::KernelIdentity second_id(second);
        BOOST_OUTCOME_TRY(require_component(first_id));
        BOOST_OUTCOME_TRY(require_component(second_id));
        if (detail::contains_component(first, second_id) || detail::contains_component(second, first_id)) {
            return boost::outcome_v2::failure(GPRError::DependentKernelComponents);
        }
        BOOST_OUTCOME_TRY(auto first_cross, covariance(first, x, x_));
        BOOST_OUTCOME_TRY(auto second_cross, covariance(second, x, x_));
        const Eigen::MatrixXd first_solved = cholesky_.solve_lower(first_cross.transpose());
        const Eigen::MatrixXd second_solved = cholesky_.solve_lower(second_cross.transpose());
        const Matrix cross = -first_solved.transpose() * second_solved;
        std::vector<Matrix> result;
        result.reserve(static_cast<std::size_t>(y_.cols()));
        for (Eigen::Index output = 0; output < y_.cols(); ++output) {
            result.emplace_back(cross * (scale_[output] * scale_[output]));
        }
        return result;
    }

    GPRResult<PosteriorPoint> predict_with_gradients(
        Point point, PredictionTarget target = PredictionTarget::Observation) const {
        BOOST_OUTCOME_TRY(validate_point_prediction(point.size()));
        return posterior_with_gradients(point, target);
    }

private:
    template <class T>
    friend GPRResult<OptimizationResult> optimize_acquisition(const GaussianProcessRegressor<T> &,
                                                              const AcquisitionFunction &,
                                                              const Matrix &, const Bounds &,
                                                              const LBFGSBOptions &);

    GPRResult<void> validate_point_prediction(Eigen::Index features) const {
        BOOST_OUTCOME_TRY(require_fitted());
        if (features != x_.cols()) {
            return boost::outcome_v2::failure(GPRError::InvalidPredictionShape);
        }
        if (y_.cols() != 1) {
            return boost::outcome_v2::failure(GPRError::NonScalarTarget);
        }
        return boost::outcome_v2::success();
    }

    // Exact spatial derivatives for scalar-output Bayesian optimization.
    GPRResult<PosteriorPoint> posterior_with_gradients(Point point, PredictionTarget target) const {
        if (!point.allFinite()) {
            return boost::outcome_v2::failure(GPRError::NonFinitePrediction);
        }
        const auto &covariance_kernel = kernel();
        const bool same = target == PredictionTarget::Observation;
        Vector cross(x_.rows());
        for (Eigen::Index row = 0; row < x_.rows(); ++row) {
            cross[row] = covariance_kernel.value_unchecked(point, x_.row(row).transpose(), false);
        }
        const Eigen::MatrixXd solved = cholesky_.solve(cross);
        BOOST_OUTCOME_TRY(auto variance, detail::posterior_variance(
            covariance_kernel.value_unchecked(point, point, same), cross.dot(solved.col(0)), x_.rows()));
        const double stddev = std::sqrt(variance);
        PosteriorPoint result;
        result.mean = cross.dot(weights_.col(0)) * scale_[0] + mean_[0];
        result.standard_deviation = stddev * scale_[0];
        result.mean_gradient = Vector::Zero(point.size());
        result.standard_deviation_gradient = Vector::Zero(point.size());
        Vector scratch(point.size());
        // Contract the spatial derivatives directly without materializing an n-by-d Jacobian.
        for (Eigen::Index row = 0; row < x_.rows(); ++row) {
            covariance_kernel.input_gradient_unchecked(point, x_.row(row).transpose(), scratch);
            result.mean_gradient.noalias() += weights_(row, 0) * scratch;
            result.standard_deviation_gradient.noalias() += solved(row, 0) * scratch;
        }
        result.mean_gradient *= scale_[0];
        detail::kernel_diagonal_gradient(covariance_kernel, point, scratch, same, 0);
        if (stddev > 0.0) {
            result.standard_deviation_gradient =
                (scratch - 2.0 * result.standard_deviation_gradient) * (scale_[0] / (2.0 * stddev));
        } else {
            result.standard_deviation_gradient.setZero();
        }
        if (!std::isfinite(result.mean) || !std::isfinite(result.standard_deviation) ||
            !result.mean_gradient.allFinite() || !result.standard_deviation_gradient.allFinite()) {
            return boost::outcome_v2::failure(GPRError::NonFinitePrediction);
        }
        return result;
    }

    template <class Component>
    GPRResult<Prediction> predict_kernel(const Component &component, const Matrix &x, bool return_covariance,
                                      bool include_mean, PredictionTarget target) const {
        if (fitted_ && x.cols() != x_.cols()) {
            return boost::outcome_v2::failure(GPRError::InvalidPredictionShape);
        }
        if (!x.allFinite()) {
            return boost::outcome_v2::failure(GPRError::NonFinitePrediction);
        }
        BOOST_OUTCOME_TRY(detail::validate_kernel_input(component, x.cols(), 0));
        const bool same = target == PredictionTarget::Observation;
        const Eigen::Index outputs = fitted_ ? y_.cols() : 1;
        Prediction result{Matrix::Zero(x.rows(), outputs), Matrix(x.rows(), outputs), {}};
        Eigen::MatrixXd solved;
        if (fitted_) {
            const Matrix cross = detail::covariance_unchecked(component, x, x_);
            result.mean.noalias() = cross * weights_;
            solved = cholesky_.solve_lower(cross.transpose());
        }
        Vector variance(x.rows());
        for (Eigen::Index row = 0; row < x.rows(); ++row) {
            const auto point = x.row(row).transpose();
            const double prior = component.value_unchecked(point, point, same);
            const double reduction = fitted_ ? solved.col(row).squaredNorm() : 0.0;
            BOOST_OUTCOME_TRY(auto value, detail::posterior_variance(prior, reduction, solved.rows()));
            variance[row] = value;
        }
        Matrix posterior;
        if (return_covariance) {
            posterior = detail::covariance_unchecked(component, x, same);
            if (fitted_) {
                posterior.noalias() -= solved.transpose() * solved;
            }
            posterior.diagonal() = variance;
        }
        for (Eigen::Index output = 0; output < outputs; ++output) {
            const double scale = fitted_ ? scale_[output] : 1.0;
            const double mean = fitted_ && include_mean ? mean_[output] : 0.0;
            result.mean.col(output) = result.mean.col(output).array() * scale + mean;
            result.standard_deviation.col(output) = variance.array().sqrt() * scale;
            if (return_covariance) {
                result.covariance.emplace_back(posterior * (scale * scale));
                if (!result.covariance.back().allFinite()) {
                    return boost::outcome_v2::failure(GPRError::NonFinitePrediction);
                }
            }
        }
        if (!result.mean.allFinite() || !result.standard_deviation.allFinite()) {
            return boost::outcome_v2::failure(GPRError::NonFinitePrediction);
        }
        return result;
    }

    GPRResult<void> require_fitted() const {
        if (!fitted_) {
            return boost::outcome_v2::failure(GPRError::NotFitted);
        }
        return boost::outcome_v2::success();
    }
    GPRResult<void> require_component(detail::KernelIdentity component) const {
        if (!detail::has_additive_component(kernel(), component, 0)) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelComponent);
        }
        return boost::outcome_v2::success();
    }

    GaussianProcessRegressor(std::shared_ptr<const K> kernel, GPROptions options)
        : kernel_(std::move(kernel)), options_(std::move(options)) {}

    GPRResult<void> fit_validated(const Matrix &x, const Matrix &y) {
        x_ = x;
        y_ = y;
        mean_ = Vector::Zero(y.cols());
        scale_ = Vector::Ones(y.cols());
        if (options_.normalize_y) {
            for (Eigen::Index output = 0; output < y.cols(); ++output) {
                mean_[output] = y.col(output).mean();
                const double stddev = std::sqrt((y.col(output).array() - mean_[output]).square().mean());
                if (!std::isfinite(stddev)) {
                    return boost::outcome_v2::failure(GPRError::NonFiniteTrainingResult);
                }
                scale_[output] = stddev < 10.0 * std::numeric_limits<double>::epsilon() ? 1.0 : stddev;
                y_.col(output) = (y.col(output).array() - mean_[output]) / scale_[output];
            }
        }
        if (options_.optimize && kernel().parameter_count() > 0) {
            BOOST_OUTCOME_TRY(optimize_kernel());
        }
        BOOST_OUTCOME_TRY(cholesky_.compute(training_covariance(kernel())));
        weights_ = cholesky_.solve(y_);
        likelihood_ = likelihood_value(cholesky_, weights_);
        if (!std::isfinite(likelihood_)) {
            return boost::outcome_v2::failure(GPRError::NonFiniteTrainingResult);
        }
        fitted_ = true;
        return boost::outcome_v2::success();
    }

    Eigen::MatrixXd training_covariance(const K &kernel) const {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(x_.rows(), x_.rows());
        detail::TrainingKernel<K>::add_covariance(kernel, x_, matrix);
        if (options_.sample_noise.size() == 0) {
            matrix.diagonal().array() += options_.alpha;
        } else {
            matrix.diagonal() += options_.sample_noise;
        }
        return matrix;
    }
    double likelihood_value(const detail::Cholesky &factor, const Eigen::MatrixXd &weights) const {
        const auto outputs = static_cast<double>(y_.cols());
        return -0.5 * (y_.array() * weights.array()).sum() -
               outputs * factor.log_diagonal() -
               0.5 * static_cast<double>(y_.rows()) * outputs * std::log(2.0 * std::acos(-1.0));
    }
    GPRResult<Likelihood> evaluate_likelihood(const K &kernel) const {
        detail::Cholesky factor;
        BOOST_OUTCOME_TRY(factor.compute(training_covariance(kernel)));
        Likelihood result;
        result.gradient = Vector::Zero(kernel.parameter_count());
        const Eigen::MatrixXd weights = factor.solve(y_);
        result.value = likelihood_value(factor, weights);
        if (!std::isfinite(result.value)) {
            return boost::outcome_v2::failure(GPRError::NonFiniteTrainingResult);
        }
        // Contract one derivative at a time, avoiding an O(n^2 * p) tensor.
        if (result.gradient.size() != 0) {
            Eigen::MatrixXd contraction = std::move(factor).take_inverse_lower();
            contraction.triangularView<Eigen::Lower>() *= -static_cast<double>(y_.cols());
            contraction.selfadjointView<Eigen::Lower>().rankUpdate(weights);
            detail::TrainingKernel<K>::contract(kernel, x_, contraction, result.gradient);
            if (!result.gradient.allFinite()) {
                return boost::outcome_v2::failure(GPRError::NonFiniteTrainingResult);
            }
        }
        return result;
    }
    double prior_penalty(Point theta) const {
        if (options_.prior.mean.size() == 0) {
            return 0.0;
        }
        return 0.5 * ((theta - options_.prior.mean).array() / options_.prior.standard_deviation.array())
                         .square()
                         .sum();
    }
    void apply_prior(Point theta, Likelihood &result) const {
        if (options_.prior.mean.size() != 0) {
            result.value -= prior_penalty(theta);
            result.gradient.array() -=
                (theta - options_.prior.mean).array() / options_.prior.standard_deviation.array().square();
        }
    }
    GPRResult<void> optimize_kernel() {
        const auto bounds = kernel().bounds();
        if (options_.optimizer_restarts > 0 && (!bounds.lower.allFinite() || !bounds.upper.allFinite())) {
            return boost::outcome_v2::failure(GPRError::NonFiniteRestartBounds);
        }
        K candidate = kernel();
        GPRError trial_error = GPRError::NonPositiveDefiniteCovariance;
        GPRError callback_error = GPRError::CallbackFailed;
        auto objective = [&](const Vector &theta, Vector &gradient) -> ObjectiveResult {
            const auto updated = candidate.set_theta(theta);
            if (!updated) {
                callback_error = updated.error();
                return boost::outcome_v2::failure(OptimizationError::CallbackFailed);
            }
            auto result = evaluate_likelihood(candidate);
            if (!result) {
                if (result.error() != GPRError::NonPositiveDefiniteCovariance) {
                    trial_error = result.error();
                }
                gradient.setZero(theta.size());
                return std::numeric_limits<double>::infinity();
            }
            auto &likelihood = result.value();
            apply_prior(theta, likelihood);
            if (!std::isfinite(likelihood.value) || !likelihood.gradient.allFinite()) {
                trial_error = GPRError::NonFiniteTrainingResult;
                gradient.setZero(theta.size());
                return std::numeric_limits<double>::infinity();
            }
            gradient = -likelihood.gradient;
            return -likelihood.value;
        };
        std::mt19937_64 random(options_.random_seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        Vector initial = kernel().theta();
        Vector best_theta = initial;
        double best = std::numeric_limits<double>::infinity();
        GPRError last_error = GPRError::NoFiniteObjective;
        for (std::int64_t restart = 0; restart <= options_.optimizer_restarts; ++restart) {
            if (restart > 0) {
                for (Eigen::Index index = 0; index < initial.size(); ++index) {
                    const double weight = uniform(random);
                    initial[index] = (1.0 - weight) * bounds.lower[index] + weight * bounds.upper[index];
                }
            }
            trial_error = GPRError::NonPositiveDefiniteCovariance;
            callback_error = GPRError::CallbackFailed;
            auto outcome = minimize_lbfgsb(objective, initial, bounds, options_.optimizer);
            if (!outcome) {
                const auto error = outcome.error();
                switch (error) {
                case OptimizationError::NoFiniteObjective:
                    last_error = trial_error;
                    break;
                case OptimizationError::CallbackFailed:
                    last_error = callback_error;
                    break;
                default:
                    return boost::outcome_v2::failure(detail::optimizer_error(error));
                }
                continue;
            }
            auto &result = outcome.value();
            if (result.value < best) {
                best = result.value;
                best_theta = result.x;
            }
            optimization_results_.push_back(std::move(result));
        }
        if (optimization_results_.empty()) {
            return boost::outcome_v2::failure(last_error);
        }
        BOOST_OUTCOME_TRY(candidate.set_theta(best_theta));
        kernel_ = std::make_shared<K>(std::move(candidate));
        return boost::outcome_v2::success();
    }
    std::shared_ptr<const K> kernel_;
    GPROptions options_;
    Matrix x_;
    Matrix y_;
    Eigen::MatrixXd weights_;
    Vector mean_;
    Vector scale_;
    detail::Cholesky cholesky_;
    std::vector<OptimizationResult> optimization_results_;
    double likelihood_ = 0.0;
    bool fitted_ = false;
};

} // namespace super_gpr
