#pragma once

#include <functional>
#include <super_gpr/types.hpp>

namespace super_gpr {

struct LBFGSBOptions {
    int memory = 10;
    int max_iterations = 1000;
    int max_line_search = 40;
    double gradient_tolerance = 1e-5;
    double function_tolerance = 2.220446049250313e-9;
};

enum class OptimizationStatus { ProjectedGradient, FunctionTolerance, IterationLimit, LineSearchFailure };

struct OptimizationResult {
    Vector x;
    double value = 0.0;
    Vector gradient;
    int iterations = 0;
    int evaluations = 0;
    OptimizationStatus status = OptimizationStatus::ProjectedGradient;

    bool converged() const {
        return status == OptimizationStatus::ProjectedGradient ||
               status == OptimizationStatus::FunctionTolerance;
    }
};

// Minimize f(x). The callback must write df/dx into gradient.
using Objective = std::function<double(const Vector &, Vector &)>;
OptimizationResult minimize_lbfgsb(const Objective &objective, Vector initial, const Bounds &bounds,
                                   const LBFGSBOptions &options = {});

} // namespace super_gpr
