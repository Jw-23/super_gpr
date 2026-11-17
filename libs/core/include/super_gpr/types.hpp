#pragma once

#include <Eigen/Core>

namespace super_gpr {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Point = Eigen::Ref<const Vector>;
using Gradient = Eigen::Ref<Vector>;

struct Bounds {
    Vector lower;
    Vector upper;
};

} // namespace super_gpr
