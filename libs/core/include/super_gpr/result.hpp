#pragma once

#include <boost/outcome/std_result.hpp>

namespace super_gpr {

enum class GPRError {
    InvalidTrainingShape,
    InvalidSampleNoiseSize,
    NegativeOptimizerRestarts,
    InvalidPrior,
    NonFiniteRestartBounds,
    NonPositiveDefiniteCovariance,
    NotFitted,
    InvalidThetaSize,
};

template <class T> using GPRResult = boost::outcome_v2::std_checked<T, GPRError>;

} // namespace super_gpr
