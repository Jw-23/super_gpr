#pragma once

#include <super_gpr/types.hpp>

namespace super_gpr {
namespace detail {

// LAPACKE uses the lower triangle of a column-major buffer.
class Cholesky {
public:
    bool compute(Eigen::MatrixXd matrix);
    Eigen::MatrixXd solve(Eigen::MatrixXd rhs) const;
    Eigen::MatrixXd solve_lower(Eigen::MatrixXd rhs) const;
    double log_diagonal() const;
    Eigen::MatrixXd take_inverse_lower() &&;

private:
    Eigen::MatrixXd factor_;
};

} // namespace detail
} // namespace super_gpr
