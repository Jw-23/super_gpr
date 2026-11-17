#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <super_gpr/additive.hpp>
#include <vector>

namespace super_gpr_experiments {
using namespace super_gpr;

struct Dataset {
    Matrix train_x;
    Vector train_y;
    Matrix test_x;
    Vector test_y;
};

inline Matrix sample_inputs(Eigen::Index count, Eigen::Index dimension, std::mt19937_64 &random) {
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    Matrix result(count, dimension);
    for (Eigen::Index row = 0; row < count; ++row) {
        for (Eigen::Index axis = 0; axis < dimension; ++axis) {
            result(row, axis) = uniform(random);
        }
    }
    return result;
}

inline Vector additive_target(const Matrix &x) {
    const Eigen::Index groups = x.cols() / 4;
    Vector result = Vector::Zero(x.rows());
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index group = 0; group < groups; ++group) {
            const auto start = 4 * group;
            const double weight = 0.6 + 0.2 * static_cast<double>(group % 5);
            const double a = x(row, start);
            const double b = x(row, start + 1);
            const double c = x(row, start + 2);
            const double d = x(row, start + 3);
            result[row] +=
                weight * (std::sin(2.5 * a + 1.2 * b) + 0.6 * std::cos(2.0 * c - 0.7 * d) + 0.5 * a * c);
        }
    }
    return result / std::sqrt(static_cast<double>(groups));
}

// Dense Fourier features create cross-group interactions without using test labels.
inline Vector interaction_target(const Matrix &x, std::uint64_t seed) {
    std::mt19937_64 random(seed + 1234567);
    std::normal_distribution<double> normal(0.0, 2.0 / std::sqrt(static_cast<double>(x.cols())));
    std::uniform_real_distribution<double> phase(0.0, 2.0 * std::acos(-1.0));
    Vector result = Vector::Zero(x.rows());
    constexpr int feature_count = 64;
    Vector frequency(x.cols());
    for (int feature = 0; feature < feature_count; ++feature) {
        for (Eigen::Index axis = 0; axis < x.cols(); ++axis) {
            frequency[axis] = normal(random);
        }
        const double offset = phase(random);
        result.array() += ((x * frequency).array() + offset).cos();
    }
    return result * std::sqrt(2.0 / static_cast<double>(feature_count));
}

inline Dataset make_dataset(Eigen::Index dimension, Eigen::Index train_count, Eigen::Index test_count,
                            std::uint64_t seed, bool interactions) {
    std::mt19937_64 random(seed);
    Dataset result;
    result.train_x = sample_inputs(train_count, dimension, random);
    result.test_x = sample_inputs(test_count, dimension, random);
    result.train_y =
        interactions ? interaction_target(result.train_x, seed) : additive_target(result.train_x);
    result.test_y = interactions ? interaction_target(result.test_x, seed) : additive_target(result.test_x);
    std::normal_distribution<double> noise(0.0, 0.03);
    for (Eigen::Index row = 0; row < train_count; ++row) {
        result.train_y[row] += noise(random);
    }
    return result;
}

inline std::vector<FeatureGroup> feature_groups(Eigen::Index dimension) {
    std::vector<FeatureGroup> result;
    for (Eigen::Index start = 0; start < dimension; start += 4) {
        result.push_back({start, start + 1, start + 2, start + 3});
    }
    return result;
}

} // namespace super_gpr_experiments
