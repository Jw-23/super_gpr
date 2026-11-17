#include <LBFGSB.h>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <super_gpr/optimizer.hpp>
#include <utility>
#include <vector>

namespace super_gpr {
namespace {
std::vector<Eigen::Index> active_coordinates(const Bounds &bounds) {
    std::vector<Eigen::Index> active;
    for (Eigen::Index axis = 0; axis < bounds.lower.size(); ++axis) {
        if (bounds.lower[axis] != bounds.upper[axis]) {
            active.push_back(axis);
        }
    }
    return active;
}
} // namespace

OptimizationResult minimize_lbfgsb(const Objective &objective, Vector initial, const Bounds &bounds,
                                   const LBFGSBOptions &options) {
    if (bounds.lower.size() != initial.size() || bounds.upper.size() != initial.size() ||
        (bounds.lower.array() > bounds.upper.array()).any()) {
        throw std::invalid_argument("Bounds must match x and satisfy lower <= upper");
    }
    initial = initial.cwiseMax(bounds.lower).cwiseMin(bounds.upper);
    const auto active = active_coordinates(bounds);
    OptimizationResult result;
    result.x = initial;
    result.gradient.resize(initial.size());
    if (active.empty()) {
        result.value = objective(result.x, result.gradient);
        result.evaluations = 1;
        return result;
    }
    const auto size = static_cast<Eigen::Index>(active.size());
    Vector reduced(size);
    Vector lower(size);
    Vector upper(size);
    for (Eigen::Index index = 0; index < size; ++index) {
        const auto axis = active[static_cast<std::size_t>(index)];
        reduced[index] = initial[axis];
        lower[index] = bounds.lower[axis];
        upper[index] = bounds.upper[axis];
    }
    Vector point = initial;
    Vector full_gradient(initial.size());
    double best = std::numeric_limits<double>::infinity();
    std::exception_ptr callback_error;
    auto evaluate = [&](const Vector &x, Vector &gradient) {
        for (Eigen::Index index = 0; index < size; ++index) {
            point[active[static_cast<std::size_t>(index)]] = x[index];
        }
        double value = 0.0;
        try {
            value = objective(point, full_gradient);
        } catch (...) {
            callback_error = std::current_exception();
            throw;
        }
        ++result.evaluations;
        gradient.resize(size);
        for (Eigen::Index index = 0; index < size; ++index) {
            gradient[index] = full_gradient[active[static_cast<std::size_t>(index)]];
        }
        if (std::isfinite(value) && full_gradient.allFinite() && value < best) {
            best = value;
            result.x = point;
            result.value = value;
            result.gradient = full_gradient;
        }
        return value;
    };
    LBFGSpp::LBFGSBParam<double> parameters;
    parameters.m = options.memory;
    parameters.max_iterations = options.max_iterations;
    parameters.max_linesearch = options.max_line_search;
    parameters.epsilon = options.gradient_tolerance;
    parameters.epsilon_rel = 0.0;
    parameters.past = options.function_tolerance > 0.0 ? 1 : 0;
    parameters.delta = options.function_tolerance;
    LBFGSpp::LBFGSBSolver<double> solver(parameters);
    double value = 0.0;
    try {
        result.iterations = solver.minimize(evaluate, reduced, value, lower, upper);
    } catch (const std::runtime_error &) {
        if (callback_error) {
            std::rethrow_exception(callback_error);
        }
        if (!std::isfinite(best)) {
            throw;
        }
        result.status = OptimizationStatus::LineSearchFailure;
        return result;
    }
    if (!std::isfinite(best)) {
        throw std::runtime_error("Objective has no finite value and gradient");
    }
    const double projected =
        ((result.x - result.gradient).cwiseMax(bounds.lower).cwiseMin(bounds.upper) - result.x)
            .cwiseAbs()
            .maxCoeff();
    if (projected <= options.gradient_tolerance) {
        result.status = OptimizationStatus::ProjectedGradient;
    } else if (options.max_iterations > 0 && result.iterations >= options.max_iterations) {
        result.status = OptimizationStatus::IterationLimit;
    } else {
        result.status = OptimizationStatus::FunctionTolerance;
    }
    return result;
}

} // namespace super_gpr
