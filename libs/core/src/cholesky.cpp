#include <lapacke.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <super_gpr/gpr.hpp>
#include <utility>

namespace super_gpr {
namespace detail {
namespace {

lapack_int lapack_size(Eigen::Index size) {
    if (size < 0 || static_cast<std::uintmax_t>(size) >
                        static_cast<std::uintmax_t>(std::numeric_limits<lapack_int>::max())) {
        throw std::length_error("Matrix size exceeds the LAPACKE integer range");
    }
    return static_cast<lapack_int>(size);
}

void check_status(const char *operation, lapack_int status) {
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed with LAPACKE info " +
                                 std::to_string(status));
    }
}

void check_rhs(const Eigen::MatrixXd &factor, const Eigen::MatrixXd &rhs) {
    if (factor.rows() != rhs.rows() || factor.rows() == 0) {
        throw std::invalid_argument("The right-hand side must match a nonempty Cholesky factor");
    }
}

void solve_triangle(const Eigen::MatrixXd &factor, Eigen::MatrixXd &rhs, char transpose) {
    const lapack_int n = lapack_size(factor.rows());
    check_status("DTRTRS", LAPACKE_dtrtrs_work(LAPACK_COL_MAJOR, 'L', transpose, 'N', n,
                                              lapack_size(rhs.cols()), factor.data(), n, rhs.data(), n));
}

} // namespace

bool Cholesky::compute(Eigen::MatrixXd matrix) {
    if (matrix.rows() == 0 || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("Cholesky requires a nonempty square matrix");
    }
    const lapack_int n = lapack_size(matrix.rows());
    const lapack_int status = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, matrix.data(), n);
    // The LAPACKE front end reports NaNs in its matrix argument as -4.
    if (status > 0 || status == -4) {
        factor_.resize(0, 0);
        return false;
    }
    check_status("DPOTRF", status);
    factor_ = std::move(matrix);
    return true;
}

Eigen::MatrixXd Cholesky::solve(Eigen::MatrixXd rhs) const {
    check_rhs(factor_, rhs);
    // Two triangular solves avoid the single-RHS DPOTRS overhead on OpenBLAS.
    if (rhs.cols() == 1) {
        solve_triangle(factor_, rhs, 'N');
        solve_triangle(factor_, rhs, 'T');
    } else if (rhs.cols() != 0) {
        const lapack_int n = lapack_size(factor_.rows());
        check_status("DPOTRS", LAPACKE_dpotrs_work(LAPACK_COL_MAJOR, 'L', n, lapack_size(rhs.cols()),
                                                  factor_.data(), n, rhs.data(), n));
    }
    return rhs;
}

Eigen::MatrixXd Cholesky::solve_lower(Eigen::MatrixXd rhs) const {
    check_rhs(factor_, rhs);
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
    check_status("DPOTRI", LAPACKE_dpotri_work(LAPACK_COL_MAJOR, 'L', n, factor_.data(), n));
    return std::move(factor_);
}

} // namespace detail
} // namespace super_gpr
