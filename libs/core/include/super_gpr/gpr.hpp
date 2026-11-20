#pragma once

#include <boost/outcome/std_result.hpp>
#include <boost/outcome/try.hpp>
#include <cstdint>
#include <random>
#include <super_gpr/kernel.hpp>
#include <super_gpr/optimizer.hpp>

namespace super_gpr {

enum class GPRError {
    InvalidTrainingShape,
    InvalidSampleNoiseSize,
    NegativeOptimizerRestarts,
    InvalidPrior,
    NonFiniteRestartBounds,
    NonPositiveDefiniteCovariance,
    NotFitted,
    InvalidThetaSize,
    InvalidOptimizerBounds,
    InvalidOptimizerOptions,
    InvalidPredictionShape,
    NonScalarTarget,
    NoAcquisitionStarts,
    InvalidKernelInput,
    NoFiniteObjective,
};

template <class T> using GPRResult = boost::outcome_v2::std_checked<T, GPRError>;

namespace detail {

inline GPRError optimizer_error(OptimizationError error) {
    switch (error) {
    case OptimizationError::InvalidBounds: return GPRError::InvalidOptimizerBounds;
    case OptimizationError::InvalidOptions: return GPRError::InvalidOptimizerOptions;
    case OptimizationError::NoFiniteObjective: return GPRError::NoFiniteObjective;
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
                matrix(row, col) += kernel.value(x.row(row).transpose(), second, row == col);
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
                kernel.gradient(first, x.row(col).transpose(), row == col, scratch);
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
        matrix.diagonal().array() += kernel.value(point, point, true);
    }

    static void contract(const WhiteKernel &kernel, TrainingPoints x, const Eigen::MatrixXd &contraction,
                         Gradient out) {
        if (out.size() != 0) {
            const auto point = x.row(0).transpose();
            out[0] += (0.5 * kernel.value(point, point, true)) * contraction.diagonal().sum();
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
        : kernel_(std::move(kernel)), options_(std::move(options)) {}

    GPRResult<void> fit(const Matrix &x, const Vector &y) { return fit(x, Matrix(y)); }
    GPRResult<void> fit(const Matrix &x, const Matrix &y) try {
        if (x.rows() == 0 || x.cols() == 0 || y.cols() == 0 || x.rows() != y.rows()) {
            return boost::outcome_v2::failure(GPRError::InvalidTrainingShape);
        }
        if (options_.sample_noise.size() != 0 && options_.sample_noise.size() != x.rows()) {
            return boost::outcome_v2::failure(GPRError::InvalidSampleNoiseSize);
        }
        if (options_.optimizer_restarts < 0) {
            return boost::outcome_v2::failure(GPRError::NegativeOptimizerRestarts);
        }
        const auto prior_size = options_.prior.mean.size();
        if (prior_size != options_.prior.standard_deviation.size() ||
            (prior_size != 0 && prior_size != kernel_.parameter_count()) ||
            (options_.prior.standard_deviation.array() <= 0.0).any()) {
            return boost::outcome_v2::failure(GPRError::InvalidPrior);
        }
        fitted_ = false;
        x_ = x;
        y_ = y;
        mean_ = Vector::Zero(y.cols());
        scale_ = Vector::Ones(y.cols());
        if (options_.normalize_y) {
            for (Eigen::Index output = 0; output < y.cols(); ++output) {
                mean_[output] = y.col(output).mean();
                const double stddev = std::sqrt((y.col(output).array() - mean_[output]).square().mean());
                scale_[output] = stddev < 10.0 * std::numeric_limits<double>::epsilon() ? 1.0 : stddev;
                y_.col(output) = (y.col(output).array() - mean_[output]) / scale_[output];
            }
        }
        optimization_results_.clear();
        if (options_.optimize && kernel_.parameter_count() > 0) {
            BOOST_OUTCOME_TRY(optimize_kernel());
        }
        BOOST_OUTCOME_TRY(cholesky_.compute(training_covariance(kernel_)));
        weights_ = cholesky_.solve(y_);
        likelihood_ = likelihood_value(cholesky_, weights_);
        fitted_ = true;
        return boost::outcome_v2::success();
    } catch (const detail::InvalidKernelInput &) {
        return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
    }

    const K &kernel() const { return kernel_; }
    GPRResult<double> log_marginal_likelihood() const {
        BOOST_OUTCOME_TRY(require_fitted());
        return likelihood_;
    }
    const std::vector<OptimizationResult> &optimization_results() const { return optimization_results_; }

    GPRResult<Likelihood> log_marginal_likelihood(Point theta) const try {
        if (!fitted_) {
            return boost::outcome_v2::failure(GPRError::NotFitted);
        }
        if (theta.size() != kernel_.parameter_count()) {
            return boost::outcome_v2::failure(GPRError::InvalidThetaSize);
        }
        K candidate = kernel_;
        candidate.set_theta(theta);
        return evaluate_likelihood(candidate);
    } catch (const detail::InvalidKernelInput &) {
        return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
    }

    // Gaussian density in log-parameter coordinates, up to a theta-independent constant.
    GPRResult<double> log_posterior() const {
        BOOST_OUTCOME_TRY(require_fitted());
        return likelihood_ - prior_penalty(kernel_.theta());
    }
    GPRResult<Likelihood> log_posterior(Point theta) const {
        BOOST_OUTCOME_TRY(auto result, log_marginal_likelihood(theta));
        apply_prior(theta, result);
        return result;
    }

    GPRResult<Prediction> predict(const Matrix &x, bool return_covariance = false) const {
        return predict_kernel(kernel_, x, return_covariance, true);
    }

    // Condition a constituent prior on the same joint observations and factorization.
    template <class Component>
    GPRResult<Prediction> predict_component(const Component &component, const Matrix &x,
                                 bool return_covariance = false) const {
        BOOST_OUTCOME_TRY(require_fitted());
        return predict_kernel(component, x, return_covariance, false);
    }

    GPRResult<Vector> mean_offset() const {
        BOOST_OUTCOME_TRY(require_fitted());
        return mean_;
    }

    // Distinct additive components are independent a priori, but correlated a posteriori.
    template <class First, class Second>
    GPRResult<std::vector<Matrix>> component_cross_covariance(const First &first, const Second &second,
                                                   const Matrix &x) const try {
        BOOST_OUTCOME_TRY(require_fitted());
        if (x.cols() != x_.cols()) {
            return boost::outcome_v2::failure(GPRError::InvalidPredictionShape);
        }
        const Matrix first_cross = covariance(first, x, x_);
        const Matrix second_cross = covariance(second, x, x_);
        const Eigen::MatrixXd first_solved = cholesky_.solve_lower(first_cross.transpose());
        const Eigen::MatrixXd second_solved = cholesky_.solve_lower(second_cross.transpose());
        const Matrix cross = -first_solved.transpose() * second_solved;
        std::vector<Matrix> result;
        result.reserve(static_cast<std::size_t>(y_.cols()));
        for (Eigen::Index output = 0; output < y_.cols(); ++output) {
            result.emplace_back(cross * (scale_[output] * scale_[output]));
        }
        return result;
    } catch (const detail::InvalidKernelInput &) {
        return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
    }

    GPRResult<PosteriorPoint> predict_with_gradients(Point point) const try {
        BOOST_OUTCOME_TRY(validate_point_prediction(point.size()));
        return posterior_with_gradients(point);
    } catch (const detail::InvalidKernelInput &) {
        return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
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
    PosteriorPoint posterior_with_gradients(Point point) const {
        Vector cross(x_.rows());
        for (Eigen::Index row = 0; row < x_.rows(); ++row) {
            cross[row] = kernel_.value(point, x_.row(row).transpose(), false);
        }
        const Eigen::MatrixXd solved = cholesky_.solve(cross);
        const double variance = std::max(0.0, kernel_.value(point, point, true) - cross.dot(solved.col(0)));
        const double stddev = std::sqrt(variance);
        PosteriorPoint result;
        result.mean = cross.dot(weights_.col(0)) * scale_[0] + mean_[0];
        result.standard_deviation = stddev * scale_[0];
        result.mean_gradient = Vector::Zero(point.size());
        result.standard_deviation_gradient = Vector::Zero(point.size());
        Vector scratch(point.size());
        // Contract the spatial derivatives directly without materializing an n-by-d Jacobian.
        for (Eigen::Index row = 0; row < x_.rows(); ++row) {
            kernel_.input_gradient(point, x_.row(row).transpose(), scratch);
            result.mean_gradient.noalias() += weights_(row, 0) * scratch;
            result.standard_deviation_gradient.noalias() += solved(row, 0) * scratch;
        }
        result.mean_gradient *= scale_[0];
        kernel_.diagonal_gradient(point, scratch);
        if (stddev > 0.0) {
            result.standard_deviation_gradient =
                (scratch - 2.0 * result.standard_deviation_gradient) * (scale_[0] / (2.0 * stddev));
        } else {
            result.standard_deviation_gradient.setZero();
        }
        return result;
    }

    template <class Component>
    GPRResult<Prediction> predict_kernel(const Component &component, const Matrix &x, bool return_covariance,
                              bool include_mean) const try {
        if (fitted_ && x.cols() != x_.cols()) {
            return boost::outcome_v2::failure(GPRError::InvalidPredictionShape);
        }
        const Eigen::Index outputs = fitted_ ? y_.cols() : 1;
        Prediction result{Matrix::Zero(x.rows(), outputs), Matrix(x.rows(), outputs), {}};
        Eigen::MatrixXd solved;
        if (fitted_) {
            const Matrix cross = covariance(component, x, x_);
            result.mean.noalias() = cross * weights_;
            solved = cholesky_.solve_lower(cross.transpose());
        }
        Vector variance(x.rows());
        for (Eigen::Index row = 0; row < x.rows(); ++row) {
            const auto point = x.row(row).transpose();
            const double prior = component.value(point, point, true);
            variance[row] = std::max(0.0, prior - (fitted_ ? solved.col(row).squaredNorm() : 0.0));
        }
        Matrix posterior;
        if (return_covariance) {
            posterior = covariance(component, x);
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
            }
        }
        return result;
    } catch (const detail::InvalidKernelInput &) {
        return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
    }

    GPRResult<void> require_fitted() const {
        if (!fitted_) {
            return boost::outcome_v2::failure(GPRError::NotFitted);
        }
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
    static Likelihood invalid_likelihood(Eigen::Index parameters) {
        return {-std::numeric_limits<double>::infinity(), Vector::Zero(parameters)};
    }
    GPRResult<Likelihood> evaluate_likelihood(const K &kernel) const {
        detail::Cholesky factor;
        BOOST_OUTCOME_TRY(factor.compute(training_covariance(kernel)));
        Likelihood result;
        result.gradient = Vector::Zero(kernel.parameter_count());
        const Eigen::MatrixXd weights = factor.solve(y_);
        result.value = likelihood_value(factor, weights);
        // Contract one derivative at a time, avoiding an O(n^2 * p) tensor.
        if (result.gradient.size() != 0) {
            Eigen::MatrixXd contraction = std::move(factor).take_inverse_lower();
            contraction.triangularView<Eigen::Lower>() *= -static_cast<double>(y_.cols());
            contraction.selfadjointView<Eigen::Lower>().rankUpdate(weights);
            detail::TrainingKernel<K>::contract(kernel, x_, contraction, result.gradient);
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
        const auto bounds = kernel_.bounds();
        if (options_.optimizer_restarts > 0 && (!bounds.lower.allFinite() || !bounds.upper.allFinite())) {
            return boost::outcome_v2::failure(GPRError::NonFiniteRestartBounds);
        }
        K candidate = kernel_;
        bool factored = false;
        auto objective = [&](const Vector &theta, Vector &gradient) {
            candidate.set_theta(theta);
            auto result = evaluate_likelihood(candidate);
            factored |= result.has_value();
            auto likelihood = result ? std::move(result).value() : invalid_likelihood(theta.size());
            apply_prior(theta, likelihood);
            gradient = -likelihood.gradient;
            return -likelihood.value;
        };
        std::mt19937_64 random(options_.random_seed);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        Vector initial = kernel_.theta();
        Vector best_theta = initial;
        double best = std::numeric_limits<double>::infinity();
        for (int restart = 0; restart <= options_.optimizer_restarts; ++restart) {
            if (restart > 0) {
                for (Eigen::Index index = 0; index < initial.size(); ++index) {
                    initial[index] =
                        bounds.lower[index] + uniform(random) * (bounds.upper[index] - bounds.lower[index]);
                }
            }
            factored = false;
            auto outcome = minimize_lbfgsb(objective, initial, bounds, options_.optimizer);
            if (!outcome) {
                const auto error = outcome.error();
                return boost::outcome_v2::failure(
                    error == OptimizationError::NoFiniteObjective && !factored
                        ? GPRError::NonPositiveDefiniteCovariance : detail::optimizer_error(error));
            }
            auto &result = outcome.value();
            if (result.value < best) {
                best = result.value;
                best_theta = result.x;
            }
            optimization_results_.push_back(std::move(result));
        }
        kernel_.set_theta(best_theta);
        return boost::outcome_v2::success();
    }
    K kernel_;
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
