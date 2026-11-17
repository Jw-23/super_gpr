#pragma once

#include <super_gpr/gpr.hpp>

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

// result.value and result.gradient use the native acquisition sign, including EI/PI.
template <class K>
OptimizationResult optimize_acquisition(const GaussianProcessRegressor<K> &model,
                                        const AcquisitionFunction &acquisition, const Matrix &starts,
                                        const Bounds &bounds, const LBFGSBOptions &options = {}) {
    if (starts.rows() == 0) {
        throw std::invalid_argument("Provide at least one acquisition starting point");
    }
    const double sign = acquisition.maximize() ? -1.0 : 1.0;
    auto objective = [&](const Vector &point, Vector &gradient) {
        const auto posterior = model.predict_with_gradients(point);
        const auto value = acquisition.evaluate(posterior.mean, posterior.standard_deviation);
        gradient = sign * (value.mean_derivative * posterior.mean_gradient +
                           value.deviation_derivative * posterior.standard_deviation_gradient);
        return sign * value.value;
    };
    OptimizationResult best;
    best.value = std::numeric_limits<double>::infinity();
    for (Eigen::Index row = 0; row < starts.rows(); ++row) {
        auto result = minimize_lbfgsb(objective, starts.row(row).transpose(), bounds, options);
        if (result.value < best.value) {
            best = std::move(result);
        }
    }
    best.value *= sign;
    best.gradient *= sign;
    return best;
}

} // namespace super_gpr
