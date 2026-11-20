#include <lapacke.h>
#include <cassert>
#include <cstdint>
#include <limits>
#include <super_gpr/gpr.hpp>
#include <utility>

namespace super_gpr {
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
    assert_success(LAPACKE_dtrtrs_work(LAPACK_COL_MAJOR, 'L', transpose, 'N', n,
                                      lapack_size(rhs.cols()), factor.data(), n, rhs.data(), n));
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
        assert_success(LAPACKE_dpotrs_work(LAPACK_COL_MAJOR, 'L', n, lapack_size(rhs.cols()),
                                          factor_.data(), n, rhs.data(), n));
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
