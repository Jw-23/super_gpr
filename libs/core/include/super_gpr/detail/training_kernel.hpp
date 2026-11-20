#pragma once

#include <super_gpr/kernel.hpp>

namespace super_gpr {
namespace detail {

using TrainingPoints = Eigen::Ref<const Matrix, 0, Eigen::OuterStride<>>;

// The training matrix needs only its lower triangle for LLT. The same
// triangle is used for the symmetric likelihood contraction.
template <class K> struct PairwiseTrainingKernel {
    static constexpr bool split_covariance = false;

    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        for (Eigen::Index col = 0; col < x.rows(); ++col) {
            const auto second = x.row(col).transpose();
            for (Eigen::Index row = col; row < x.rows(); ++row) {
                matrix(row, col) += kernel.value(x.row(row).transpose(), second, row == col);
            }
        }
    }

    static void contract(const K &kernel, TrainingPoints x, const Eigen::MatrixXd &contraction,
                         Gradient out) {
        if (out.size() == 0) {
            return;
        }
        Vector scratch(out.size());
        for (Eigen::Index row = 0; row < x.rows(); ++row) {
            const auto first = x.row(row).transpose();
            for (Eigen::Index col = 0; col <= row; ++col) {
                kernel.gradient(first, x.row(col).transpose(), row == col, scratch);
                out.noalias() += scratch * (contraction(row, col) * (row == col ? 0.5 : 1.0));
            }
        }
    }
};

template <class K> struct TrainingKernel : PairwiseTrainingKernel<K> {};

template <> struct TrainingKernel<WhiteKernel> {
    static constexpr bool split_covariance = false;

    static void add_covariance(const WhiteKernel &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        const auto point = x.row(0).transpose();
        matrix.diagonal().array() += kernel.value(point, point, true);
    }

    static void contract(const WhiteKernel &kernel, TrainingPoints x, const Eigen::MatrixXd &contraction,
                         Gradient out) {
        if (out.size() != 0) {
            const auto point = x.row(0).transpose();
            out[0] += (0.5 * kernel.value(point, point, true)) * contraction.diagonal().sum();
        }
    }
};

template <class K1, class K2> struct TrainingKernel<AddKernel<K1, K2>> {
    using K = AddKernel<K1, K2>;
    static constexpr bool split_covariance =
        TrainingKernel<K1>::split_covariance || TrainingKernel<K2>::split_covariance;

    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        add_covariance(kernel, x, matrix, std::integral_constant<bool, split_covariance>{});
    }

    static void contract(const K &kernel, TrainingPoints x, const Eigen::MatrixXd &contraction,
                         Gradient out) {
        TrainingKernel<K1>::contract(kernel.left(), x, contraction,
                                     out.head(kernel.left().parameter_count()));
        TrainingKernel<K2>::contract(kernel.right(), x, contraction,
                                     out.tail(kernel.right().parameter_count()));
    }

private:
    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix,
                               std::true_type) {
        TrainingKernel<K1>::add_covariance(kernel.left(), x, matrix);
        TrainingKernel<K2>::add_covariance(kernel.right(), x, matrix);
    }
    static void add_covariance(const K &kernel, TrainingPoints x, Eigen::MatrixXd &matrix,
                               std::false_type) {
        PairwiseTrainingKernel<K>::add_covariance(kernel, x, matrix);
    }
};

} // namespace detail
} // namespace super_gpr
