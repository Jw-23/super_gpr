#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <super_gpr/gpr.hpp>
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

// Cartesian product in feature order. The last feature varies fastest.
class ParameterGrid {
public:
    static GPRResult<ParameterGrid> create(std::vector<std::vector<double>> axes) {
        if (axes.empty()) {
            return boost::outcome_v2::failure(GPRError::EmptyParameterGrid);
        }
        Eigen::Index size = 1;
        for (const auto &axis : axes) {
            if (axis.empty()) {
                return boost::outcome_v2::failure(GPRError::EmptyGridAxis);
            }
            if (axis.size() > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max() / size)) {
                return boost::outcome_v2::failure(GPRError::GridSizeOverflow);
            }
            for (const double value : axis) {
                if (!std::isfinite(value)) {
                    return boost::outcome_v2::failure(GPRError::NonFiniteCandidates);
                }
            }
            size *= static_cast<Eigen::Index>(axis.size());
        }
        return ParameterGrid(std::move(axes), size);
    }

    Eigen::Index size() const { return size_; }
    Eigen::Index dimensions() const { return static_cast<Eigen::Index>(axes_.size()); }

    GPRResult<Vector> operator[](Eigen::Index index) const {
        if (index < 0 || index >= size()) {
            return boost::outcome_v2::failure(GPRError::InvalidGridIndex);
        }
        Vector point(dimensions());
        for (Eigen::Index feature = point.size(); feature > 0; --feature) {
            const auto &axis = axes_[static_cast<std::size_t>(feature - 1)];
            const auto length = static_cast<Eigen::Index>(axis.size());
            point[feature - 1] = axis[static_cast<std::size_t>(index % length)];
            index /= length;
        }
        return point;
    }

private:
    ParameterGrid(std::vector<std::vector<double>> axes, Eigen::Index size)
        : axes_(std::move(axes)), size_(size) {}

    std::vector<std::vector<double>> axes_;
    Eigen::Index size_;
};

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

inline GPRResult<std::vector<Eigen::Index>> sorted_exclusions(const SuggestOptions &options,
                                                            Eigen::Index candidate_count) {
    auto excluded = options.excluded_indices;
    std::sort(excluded.begin(), excluded.end());
    if (!excluded.empty() && (excluded.front() < 0 || excluded.back() >= candidate_count)) {
        return boost::outcome_v2::failure(GPRError::InvalidExcludedIndex);
    }
    excluded.erase(std::unique(excluded.begin(), excluded.end()), excluded.end());
    return excluded;
}

template <class K, class CopyPoint>
GPRResult<Suggestion> suggest_discrete(const GaussianProcessRegressor<K> &model,
                                       const AcquisitionFunction &acquisition, Eigen::Index candidate_count,
                                       Eigen::Index dimensions, const SuggestOptions &options,
                                       const CopyPoint &copy_point) {
    BOOST_OUTCOME_TRY(auto features, model.n_features_in());
    if (dimensions != features) {
        return boost::outcome_v2::failure(GPRError::InvalidPredictionShape);
    }
    BOOST_OUTCOME_TRY(auto offset, model.mean_offset());
    if (offset.size() != 1) {
        return boost::outcome_v2::failure(GPRError::NonScalarTarget);
    }
    if (options.batch_size <= 0) {
        return boost::outcome_v2::failure(GPRError::InvalidBatchSize);
    }
    BOOST_OUTCOME_TRY(auto excluded, sorted_exclusions(options, candidate_count));
    auto next_exclusion = excluded.begin();
    Eigen::Index remaining = candidate_count - static_cast<Eigen::Index>(excluded.size());
    if (remaining == 0) {
        return Suggestion{};
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
            BOOST_OUTCOME_TRY(copy_point(candidate, batch.row(row)));
            indices[static_cast<std::size_t>(row)] = candidate;
            ++candidate;
        }

        BOOST_OUTCOME_TRY(auto prediction, model.predict(batch));
        for (Eigen::Index row = 0; row < batch.rows(); ++row) {
            const double value =
                acquisition.evaluate(prediction.mean(row, 0), prediction.standard_deviation(row, 0)).value;
            if (!std::isfinite(value)) {
                return boost::outcome_v2::failure(GPRError::NonFiniteAcquisition);
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
GPRResult<Suggestion> suggest(const GaussianProcessRegressor<K> &model, const AcquisitionFunction &acquisition,
                             const Matrix &candidates, const SuggestOptions &options = {}) {
    if (!candidates.allFinite()) {
        return boost::outcome_v2::failure(GPRError::NonFiniteCandidates);
    }
    return detail::suggest_discrete(
        model, acquisition, candidates.rows(), candidates.cols(), options,
        [&candidates](Eigen::Index index, auto row) -> GPRResult<void> {
            row = candidates.row(index);
            return boost::outcome_v2::success();
        });
}

// The result index addresses the grid directly. The grid is generated one batch at a time.
template <class K>
GPRResult<Suggestion> suggest(const GaussianProcessRegressor<K> &model, const AcquisitionFunction &acquisition,
                             const ParameterGrid &grid, const SuggestOptions &options = {}) {
    return detail::suggest_discrete(model, acquisition, grid.size(), grid.dimensions(), options,
        [&grid](Eigen::Index index, auto row) -> GPRResult<void> {
            BOOST_OUTCOME_TRY(auto point, grid[index]);
            row = point.transpose();
            return boost::outcome_v2::success();
        });
}

// result.value and result.gradient use the native acquisition sign, including EI/PI.
template <class K>
GPRResult<OptimizationResult> optimize_acquisition(const GaussianProcessRegressor<K> &model,
                                                   const AcquisitionFunction &acquisition,
                                                   const Matrix &starts, const Bounds &bounds,
                                                   const LBFGSBOptions &options) {
    if (starts.rows() == 0) {
        return boost::outcome_v2::failure(GPRError::NoAcquisitionStarts);
    }
    BOOST_OUTCOME_TRY(model.validate_point_prediction(starts.cols()));
    const double sign = acquisition.maximize() ? -1.0 : 1.0;
    auto objective = [&](const Vector &point, Vector &gradient) -> ObjectiveResult {
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
}

} // namespace super_gpr
