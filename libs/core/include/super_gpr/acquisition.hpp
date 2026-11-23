#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <super_gpr/gpr.hpp>
#include <super_gpr/parameter_grid.hpp>
#include <utility>
#include <vector>

namespace super_gpr {

struct AcquisitionValue {
    double value = 0.0;
    double mean_derivative = 0.0;
    double deviation_derivative = 0.0;
};

class AcquisitionFunction {
public:
    AcquisitionFunction() = default;
    AcquisitionFunction(const AcquisitionFunction &) = default;
    AcquisitionFunction &operator=(const AcquisitionFunction &) = default;
    AcquisitionFunction(AcquisitionFunction &&) = default;
    AcquisitionFunction &operator=(AcquisitionFunction &&) = default;
    virtual ~AcquisitionFunction() = default;
    virtual AcquisitionValue evaluate(double mean, double standard_deviation) const = 0;
    virtual bool maximize() const { return false; }
};

class PredictedMean final : public AcquisitionFunction {
public:
    AcquisitionValue evaluate(double mean, double /*standard_deviation*/) const override {
        return {mean, 1.0, 0.0};
    }
};

class LCB final : public AcquisitionFunction {
public:
    explicit LCB(double kappa = 2.0) : kappa_(kappa) {}
    AcquisitionValue evaluate(double mean, double standard_deviation) const override {
        return {mean - kappa_ * standard_deviation, 1.0, -kappa_};
    }

private:
    double kappa_;
};

class ConservativeUCB final : public AcquisitionFunction {
public:
    explicit ConservativeUCB(double kappa = 2.0) : kappa_(kappa) {}
    AcquisitionValue evaluate(double mean, double standard_deviation) const override {
        return {mean + kappa_ * standard_deviation, 1.0, kappa_};
    }

private:
    double kappa_;
};

inline double normal_pdf(double z) {
    return std::exp(-0.5 * z * z) / std::sqrt(2.0 * std::acos(-1.0));
}
inline double normal_cdf(double z) {
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

// Improvement is defined relative to the smallest observed target.
class ExpectedImprovement final : public AcquisitionFunction {
public:
    explicit ExpectedImprovement(double best, double xi = 0.0) : best_(best), xi_(xi) {}
    bool maximize() const override { return true; }
    AcquisitionValue evaluate(double mean, double standard_deviation) const override {
        const double improvement = best_ - mean - xi_;
        if (standard_deviation <= 0.0) {
            return {std::max(0.0, improvement), improvement > 0.0 ? -1.0 : 0.0, 0.0};
        }
        const double z = improvement / standard_deviation;
        const double cdf = normal_cdf(z);
        const double pdf = normal_pdf(z);
        return {std::max(0.0, improvement * cdf + standard_deviation * pdf), -cdf, pdf};
    }

private:
    double best_;
    double xi_;
};

class ProbabilityOfImprovement final : public AcquisitionFunction {
public:
    explicit ProbabilityOfImprovement(double best, double xi = 0.0) : best_(best), xi_(xi) {}
    bool maximize() const override { return true; }
    AcquisitionValue evaluate(double mean, double standard_deviation) const override {
        const double improvement = best_ - mean - xi_;
        if (standard_deviation <= 0.0) {
            return {improvement > 0.0 ? 1.0 : 0.0, 0.0, 0.0};
        }
        const double z = improvement / standard_deviation;
        const double density = normal_pdf(z);
        if (density == 0.0) {
            return {normal_cdf(z), 0.0, 0.0};
        }
        return {normal_cdf(z), -density / standard_deviation, -density * z / standard_deviation};
    }

private:
    double best_;
    double xi_;
};

using EI = ExpectedImprovement;
using PI = ProbabilityOfImprovement;

struct SuggestOptions {
    std::vector<Eigen::Index> excluded_indices;
    Eigen::Index batch_size = 256;
};

struct Suggestion {
    Vector x;
    double value = 0.0;
    Eigen::Index index = -1;

    explicit operator bool() const { return index >= 0; }
};

namespace detail {

inline std::vector<Eigen::Index> sorted_exclusions(const SuggestOptions &options,
                                                   Eigen::Index candidate_count) {
    auto excluded = options.excluded_indices;
    std::sort(excluded.begin(), excluded.end());
    if (!excluded.empty() && (excluded.front() < 0 || excluded.back() >= candidate_count)) {
        throw std::invalid_argument("Excluded candidate index is out of range");
    }
    excluded.erase(std::unique(excluded.begin(), excluded.end()), excluded.end());
    return excluded;
}

template <class K, class CopyPoint>
Suggestion suggest_discrete(const GaussianProcessRegressor<K> &model, const AcquisitionFunction &acquisition,
                            Eigen::Index candidate_count, Eigen::Index dimensions,
                            const SuggestOptions &options, const CopyPoint &copy_point) {
    if (dimensions != model.n_features_in() || model.mean_offset().size() != 1) {
        throw std::invalid_argument("Suggestion requires matching features and a scalar target");
    }
    if (options.batch_size <= 0) {
        throw std::invalid_argument("Suggestion batch size must be positive");
    }
    const auto excluded = sorted_exclusions(options, candidate_count);
    auto next_exclusion = excluded.begin();
    Eigen::Index remaining = candidate_count - static_cast<Eigen::Index>(excluded.size());
    if (remaining == 0) {
        return {};
    }

    Matrix batch(std::min(remaining, options.batch_size), dimensions);
    std::vector<Eigen::Index> indices(static_cast<std::size_t>(batch.rows()));
    Eigen::Index candidate = 0;
    const bool maximize = acquisition.maximize();
    Suggestion best;
    while (remaining > 0) {
        if (remaining < batch.rows()) {
            batch.conservativeResize(remaining, Eigen::NoChange);
        }
        for (Eigen::Index row = 0; row < batch.rows(); ++row) {
            while (next_exclusion != excluded.end() && *next_exclusion == candidate) {
                ++next_exclusion;
                ++candidate;
            }
            copy_point(candidate, batch.row(row));
            indices[static_cast<std::size_t>(row)] = candidate;
            ++candidate;
        }

        const auto prediction = model.predict(batch);
        for (Eigen::Index row = 0; row < batch.rows(); ++row) {
            const double value =
                acquisition.evaluate(prediction.mean(row, 0), prediction.standard_deviation(row, 0)).value;
            if (!std::isfinite(value)) {
                throw std::domain_error("Suggestion requires finite acquisition values");
            }
            if (!best || (maximize ? value > best.value : value < best.value)) {
                best.x = batch.row(row).transpose();
                best.value = value;
                best.index = indices[static_cast<std::size_t>(row)];
            }
        }
        remaining -= batch.rows();
    }
    return best;
}

} // namespace detail

// The result index is the original zero-based candidate row, with ties resolved in row order.
template <class K>
Suggestion suggest(const GaussianProcessRegressor<K> &model, const AcquisitionFunction &acquisition,
                   const Matrix &candidates, const SuggestOptions &options = {}) {
    if (!candidates.allFinite()) {
        throw std::invalid_argument("Suggestion candidates must be finite");
    }
    return detail::suggest_discrete(
        model, acquisition, candidates.rows(), candidates.cols(), options,
        [&candidates](Eigen::Index index, auto row) { row = candidates.row(index); });
}

// The result index addresses the grid directly. The grid is generated one batch at a time.
template <class K>
Suggestion suggest(const GaussianProcessRegressor<K> &model, const AcquisitionFunction &acquisition,
                   const ParameterGrid &grid, const SuggestOptions &options = {}) {
    return detail::suggest_discrete(model, acquisition, grid.size(), grid.dimensions(), options,
                                    [&grid](Eigen::Index index, auto row) { row = grid[index].transpose(); });
}

// result.value and result.gradient use the native acquisition sign, including EI/PI.
template <class K>
GPRResult<OptimizationResult> optimize_acquisition(const GaussianProcessRegressor<K> &model,
                                                   const AcquisitionFunction &acquisition,
                                                   const Matrix &starts, const Bounds &bounds,
                                                   const LBFGSBOptions &options) try {
    if (starts.rows() == 0) {
        return boost::outcome_v2::failure(GPRError::NoAcquisitionStarts);
    }
    BOOST_OUTCOME_TRY(model.validate_point_prediction(starts.cols()));
    const double sign = acquisition.maximize() ? -1.0 : 1.0;
    auto objective = [&](const Vector &point, Vector &gradient) {
        const auto posterior = model.posterior_with_gradients(point);
        const auto value = acquisition.evaluate(posterior.mean, posterior.standard_deviation);
        gradient = sign * (value.mean_derivative * posterior.mean_gradient +
                           value.deviation_derivative * posterior.standard_deviation_gradient);
        return sign * value.value;
    };
    OptimizationResult best;
    best.value = std::numeric_limits<double>::infinity();
    for (Eigen::Index row = 0; row < starts.rows(); ++row) {
        auto outcome = minimize_lbfgsb(objective, starts.row(row).transpose(), bounds, options);
        if (!outcome) {
            return boost::outcome_v2::failure(detail::optimizer_error(outcome.error()));
        }
        auto &result = outcome.value();
        if (result.value < best.value) {
            best = std::move(result);
        }
    }
    best.value *= sign;
    best.gradient *= sign;
    return best;
} catch (const detail::InvalidKernelInput &) {
    return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
}

} // namespace super_gpr
