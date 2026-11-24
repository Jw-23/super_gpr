#include <LBFGSB.h>
#include <boost/outcome/try.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <lapacke.h>
#include <limits>
#include <stdexcept>
#include <super_gpr/gpr.hpp>
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

LBFGSBResult minimize_lbfgsb(const Objective &objective, Vector initial, const Bounds &bounds,
                             const LBFGSBOptions &options) {
    if (bounds.lower.size() != initial.size() || bounds.upper.size() != initial.size() ||
        (bounds.lower.array() > bounds.upper.array()).any()) {
        return boost::outcome_v2::failure(OptimizationError::InvalidBounds);
    }
    initial = initial.cwiseMax(bounds.lower).cwiseMin(bounds.upper);
    const auto invoke = [&](const Vector &point, Vector &gradient) -> ObjectiveResult {
        try {
            return objective(point, gradient);
        } catch (...) {
            return boost::outcome_v2::failure(OptimizationError::CallbackFailed);
        }
    };
    const auto active = active_coordinates(bounds);
    OptimizationResult result;
    result.x = initial;
    result.gradient.resize(initial.size());
    if (active.empty()) {
        BOOST_OUTCOME_TRY(auto value, invoke(result.x, result.gradient));
        result.value = value;
        result.evaluations = 1;
        if (!std::isfinite(result.value) || !result.gradient.allFinite()) {
            return boost::outcome_v2::failure(OptimizationError::NoFiniteObjective);
        }
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
    bool callback_failed = false;
    OptimizationError callback_error = OptimizationError::CallbackFailed;
    auto evaluate = [&](const Vector &x, Vector &gradient) {
        for (Eigen::Index index = 0; index < size; ++index) {
            point[active[static_cast<std::size_t>(index)]] = x[index];
        }
        if (callback_failed) {
            gradient.setZero(size);
            return std::numeric_limits<double>::infinity();
        }
        const auto outcome = invoke(point, full_gradient);
        ++result.evaluations;
        if (!outcome) {
            callback_error = outcome.error();
            callback_failed = true;
            gradient.setZero(size);
            return std::numeric_limits<double>::infinity();
        }
        const double value = outcome.value();
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
    double value = 0.0;
    bool line_search_failed = false;
    try {
        LBFGSpp::LBFGSBSolver<double> solver(parameters);
        result.iterations = solver.minimize(evaluate, reduced, value, lower, upper);
    } catch (const std::invalid_argument &) {
        return boost::outcome_v2::failure(callback_failed ? callback_error : OptimizationError::InvalidOptions);
    } catch (const std::logic_error &) {
        line_search_failed = true;
    } catch (const std::runtime_error &) {
        line_search_failed = true;
    }
    if (callback_failed) {
        return boost::outcome_v2::failure(callback_error);
    }
    if (!std::isfinite(best)) {
        return boost::outcome_v2::failure(OptimizationError::NoFiniteObjective);
    }
    if (line_search_failed) {
        result.status = OptimizationStatus::LineSearchFailure;
        return result;
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

namespace detail {
namespace {

lapack_int lapack_size(Eigen::Index size) {
    assert(size >= 0 && static_cast<std::uintmax_t>(size) <=
                            static_cast<std::uintmax_t>(std::numeric_limits<lapack_int>::max()));
    return static_cast<lapack_int>(size);
}

// Solves and inversion use a successful factorization and internally sized buffers.
void assert_success(lapack_int status) {
    assert(status == 0);
    (void)status;
}

void solve_triangle(const Eigen::MatrixXd &factor, Eigen::MatrixXd &rhs, char transpose) {
    const lapack_int n = lapack_size(factor.rows());
    assert_success(LAPACKE_dtrtrs_work(LAPACK_COL_MAJOR, 'L', transpose, 'N', n, lapack_size(rhs.cols()),
                                       factor.data(), n, rhs.data(), n));
}

} // namespace

GPRResult<void> Cholesky::compute(Eigen::MatrixXd matrix) {
    assert(matrix.rows() > 0 && matrix.rows() == matrix.cols());
    const lapack_int n = lapack_size(matrix.rows());
    const lapack_int status = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, matrix.data(), n);
    // A nonpositive pivot or NaN (-4 from LAPACKE) invalidates the factor.
    if (status != 0) {
        factor_.resize(0, 0);
        return boost::outcome_v2::failure(GPRError::NonPositiveDefiniteCovariance);
    }
    factor_ = std::move(matrix);
    return boost::outcome_v2::success();
}

Eigen::MatrixXd Cholesky::solve(Eigen::MatrixXd rhs) const {
    assert(factor_.rows() > 0 && factor_.rows() == rhs.rows());
    // Two triangular solves avoid the single-RHS DPOTRS overhead on OpenBLAS.
    if (rhs.cols() == 1) {
        solve_triangle(factor_, rhs, 'N');
        solve_triangle(factor_, rhs, 'T');
    } else if (rhs.cols() != 0) {
        const lapack_int n = lapack_size(factor_.rows());
        assert_success(LAPACKE_dpotrs_work(LAPACK_COL_MAJOR, 'L', n, lapack_size(rhs.cols()), factor_.data(),
                                           n, rhs.data(), n));
    }
    return rhs;
}

Eigen::MatrixXd Cholesky::solve_lower(Eigen::MatrixXd rhs) const {
    assert(factor_.rows() > 0 && factor_.rows() == rhs.rows());
    if (rhs.cols() != 0) {
        solve_triangle(factor_, rhs, 'N');
    }
    return rhs;
}

double Cholesky::log_diagonal() const {
    return factor_.diagonal().array().log().sum();
}

Eigen::MatrixXd Cholesky::take_inverse_lower() && {
    const lapack_int n = lapack_size(factor_.rows());
    assert_success(LAPACKE_dpotri_work(LAPACK_COL_MAJOR, 'L', n, factor_.data(), n));
    return std::move(factor_);
}

} // namespace detail

} // namespace super_gpr
