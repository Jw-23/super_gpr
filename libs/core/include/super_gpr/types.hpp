#pragma once

#include <Eigen/Core>
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
    InvalidOptimizerBounds,
    InvalidOptimizerOptions,
    InvalidPredictionShape,
    NonScalarTarget,
    NoAcquisitionCandidates,
    InvalidKernelInput,
    NoFiniteObjective,
    EmptyParameterGrid,
    EmptyGridAxis,
    GridSizeOverflow,
    NonFiniteCandidates,
    InvalidGridIndex,
    InvalidExcludedIndex,
    InvalidBatchSize,
    NonFiniteAcquisition,
    InvalidKernelParameters,
    InvalidMaternNu,
    InvalidFeatureGroup,
    EmptyAdditiveKernel,
    CallbackFailed,
    NonFiniteTrainingData,
    InvalidTrainingNoise,
    NonFiniteTrainingResult,
    InvalidTheta,
    NonFinitePrediction,
    NegativePredictiveVariance,
    InvalidKernelComponent,
    DependentKernelComponents,
};

template <class T> using GPRResult = boost::outcome_v2::std_checked<T, GPRError>;

using Vector = Eigen::VectorXd;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Point = Eigen::Ref<const Vector>;
using Gradient = Eigen::Ref<Vector>;

struct Bounds {
    Vector lower;
    Vector upper;
};

} // namespace super_gpr
