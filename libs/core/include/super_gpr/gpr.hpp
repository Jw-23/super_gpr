#pragma once

#include <Eigen/Cholesky>
#include <cstdint>
#include <random>
#include <super_gpr/kernel.hpp>
#include <super_gpr/optimizer.hpp>

namespace super_gpr {

struct GPROptions {
    double alpha = 1e-10;
    Vector sample_noise;
    bool normalize_y = false;
    bool optimize = true;
    int optimizer_restarts = 0;
    std::uint64_t random_seed = 0;
    LBFGSBOptions optimizer;
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

template <class Kernel> class GaussianProcessRegressor {
    using K = std::decay_t<Kernel>;
    static_assert(IsKernel<K>::value, "K must satisfy the Kernel interface");
    using Cholesky = Eigen::LLT<Eigen::MatrixXd>;

public:
    explicit GaussianProcessRegressor(K kernel, GPROptions options = {})
        : kernel_(std::move(kernel)), options_(std::move(options)) {}

    void fit(const Matrix &x, const Vector &y) { fit(x, Matrix(y)); }
    void fit(const Matrix &x, const Matrix &y) {
        if (x.rows() == 0 || x.cols() == 0 || y.cols() == 0 || x.rows() != y.rows()) {
            throw std::invalid_argument("Training data must be nonempty with matching sample counts");
        }
        if (options_.sample_noise.size() != 0 && options_.sample_noise.size() != x.rows()) {
            throw std::invalid_argument("sample_noise must have one variance per training observation");
        }
        if (options_.optimizer_restarts < 0) {
            throw std::invalid_argument("optimizer_restarts must be nonnegative");
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
            optimize_kernel();
        }
        Matrix matrix = training_covariance(kernel_);
        cholesky_.compute(matrix);
        if (cholesky_.info() != Eigen::Success) {
            throw std::runtime_error(
                "Training covariance is not positive definite; check the kernel or increase alpha");
        }
        weights_ = cholesky_.solve(y_);
        likelihood_ = likelihood_value(cholesky_, weights_);
        fitted_ = true;
    }

    const K &kernel() const { return kernel_; }
    double log_marginal_likelihood() const {
        require_fitted();
        return likelihood_;
    }
    const std::vector<OptimizationResult> &optimization_results() const { return optimization_results_; }

    Likelihood log_marginal_likelihood(Point theta) const {
        require_fitted();
        K candidate = kernel_;
        candidate.set_theta(theta);
        return evaluate_likelihood(candidate);
    }

    Prediction predict(const Matrix &x, bool return_covariance = false) const {
        if (fitted_ && x.cols() != x_.cols()) {
            throw std::invalid_argument("Wrong prediction feature count");
        }
        const Eigen::Index outputs = fitted_ ? y_.cols() : 1;
        Prediction result{Matrix::Zero(x.rows(), outputs), Matrix(x.rows(), outputs), {}};
        Matrix solved;
        if (fitted_) {
            const Matrix cross = covariance(kernel_, x, x_);
            result.mean.noalias() = cross * weights_;
            solved = cholesky_.matrixL().solve(cross.transpose());
        }
        Vector variance(x.rows());
        for (Eigen::Index row = 0; row < x.rows(); ++row) {
            const auto point = x.row(row).transpose();
            const double prior = kernel_.value(point, point, true);
            variance[row] = std::max(0.0, prior - (fitted_ ? solved.col(row).squaredNorm() : 0.0));
        }
        Matrix posterior;
        if (return_covariance) {
            posterior = covariance(kernel_, x);
            if (fitted_) {
                posterior.noalias() -= solved.transpose() * solved;
            }
            posterior.diagonal() = variance;
        }
        for (Eigen::Index output = 0; output < outputs; ++output) {
            const double scale = fitted_ ? scale_[output] : 1.0;
            const double mean = fitted_ ? mean_[output] : 0.0;
            result.mean.col(output) = result.mean.col(output).array() * scale + mean;
            result.standard_deviation.col(output) = variance.array().sqrt() * scale;
            if (return_covariance) {
                result.covariance.emplace_back(posterior * (scale * scale));
            }
        }
        return result;
    }

    // Exact spatial derivatives for scalar-output Bayesian optimization.
    PosteriorPoint predict_with_gradients(Point point) const {
        require_fitted();
        if (point.size() != x_.cols() || y_.cols() != 1) {
            throw std::invalid_argument("Point prediction requires matching features and a scalar target");
        }
        Vector cross(x_.rows());
        Matrix jacobian(x_.rows(), point.size());
        Vector scratch(point.size());
        for (Eigen::Index row = 0; row < x_.rows(); ++row) {
            const auto training = x_.row(row).transpose();
            cross[row] = kernel_.value(point, training, false);
            kernel_.input_gradient(point, training, scratch);
            jacobian.row(row) = scratch.transpose();
        }
        const Vector solved = cholesky_.solve(cross);
        const double variance = std::max(0.0, kernel_.value(point, point, true) - cross.dot(solved));
        const double stddev = std::sqrt(variance);
        PosteriorPoint result;
        result.mean = cross.dot(weights_.col(0)) * scale_[0] + mean_[0];
        result.standard_deviation = stddev * scale_[0];
        result.mean_gradient = jacobian.transpose() * weights_.col(0) * scale_[0];
        kernel_.diagonal_gradient(point, scratch);
        result.standard_deviation_gradient = Vector::Zero(point.size());
        if (stddev > 0.0) {
            result.standard_deviation_gradient =
                (scratch - 2.0 * jacobian.transpose() * solved) * (scale_[0] / (2.0 * stddev));
        }
        return result;
    }

private:
    void require_fitted() const {
        if (!fitted_) {
            throw std::logic_error("Fit the Gaussian process first");
        }
    }
    Matrix training_covariance(const K &kernel) const {
        Matrix matrix = covariance(kernel, x_);
        if (options_.sample_noise.size() == 0) {
            matrix.diagonal().array() += options_.alpha;
        } else {
            matrix.diagonal() += options_.sample_noise;
        }
        return matrix;
    }
    double likelihood_value(const Cholesky &factor, const Matrix &weights) const {
        const auto outputs = static_cast<double>(y_.cols());
        return -0.5 * (y_.array() * weights.array()).sum() -
               outputs * factor.matrixLLT().diagonal().array().log().sum() -
               0.5 * static_cast<double>(y_.rows()) * outputs * std::log(2.0 * std::acos(-1.0));
    }
    Likelihood evaluate_likelihood(const K &kernel) const {
        const Matrix matrix = training_covariance(kernel);
        const Cholesky factor(matrix);
        Likelihood result;
        result.gradient = Vector::Zero(kernel.parameter_count());
        if (factor.info() != Eigen::Success) {
            return result;
        }
        const Matrix weights = factor.solve(y_);
        result.value = likelihood_value(factor, weights);
        // Contract one derivative at a time, avoiding an O(n^2 * p) tensor.
        const Matrix contraction =
            weights * weights.transpose() -
            static_cast<double>(y_.cols()) * factor.solve(Matrix::Identity(x_.rows(), x_.rows()));
        Vector scratch(kernel.parameter_count());
        for (Eigen::Index row = 0; row < x_.rows(); ++row) {
            for (Eigen::Index col = 0; col <= row; ++col) {
                kernel.gradient(x_.row(row).transpose(), x_.row(col).transpose(), row == col, scratch);
                result.gradient.noalias() += scratch * (contraction(row, col) * (row == col ? 0.5 : 1.0));
            }
        }
        return result;
    }
    void optimize_kernel() {
        const auto bounds = kernel_.bounds();
        if (options_.optimizer_restarts > 0 && (!bounds.lower.allFinite() || !bounds.upper.allFinite())) {
            throw std::invalid_argument("Optimizer restarts require finite bounds");
        }
        K candidate = kernel_;
        auto objective = [&](const Vector &theta, Vector &gradient) {
            candidate.set_theta(theta);
            const auto likelihood = evaluate_likelihood(candidate);
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
            auto result = minimize_lbfgsb(objective, initial, bounds, options_.optimizer);
            if (result.value < best) {
                best = result.value;
                best_theta = result.x;
            }
            optimization_results_.push_back(std::move(result));
        }
        kernel_.set_theta(best_theta);
    }
    K kernel_;
    GPROptions options_;
    Matrix x_;
    Matrix y_;
    Matrix weights_;
    Vector mean_;
    Vector scale_;
    Cholesky cholesky_;
    std::vector<OptimizationResult> optimization_results_;
    double likelihood_ = 0.0;
    bool fitted_ = false;
};

} // namespace super_gpr
