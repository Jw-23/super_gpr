#pragma once

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <super_gpr/types.hpp>
#include <utility>
#include <vector>

namespace super_gpr {

// Cartesian product in feature order. The last feature varies fastest.
class ParameterGrid {
public:
    ParameterGrid(std::initializer_list<std::vector<double>> axes)
        : ParameterGrid(std::vector<std::vector<double>>(axes)) {}

    explicit ParameterGrid(std::vector<std::vector<double>> axes) : axes_(std::move(axes)) {
        if (axes_.empty()) {
            throw std::invalid_argument("Parameter grid must have at least one feature");
        }
        for (const auto &axis : axes_) {
            if (axis.empty()) {
                throw std::invalid_argument("Each grid feature must have at least one value");
            }
            if (axis.size() > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max() / size_)) {
                throw std::overflow_error("Parameter grid size exceeds the candidate index range");
            }
            for (const double value : axis) {
                if (!std::isfinite(value)) {
                    throw std::invalid_argument("Parameter grid values must be finite");
                }
            }
            size_ *= static_cast<Eigen::Index>(axis.size());
        }
    }

    Eigen::Index size() const { return size_; }
    Eigen::Index dimensions() const { return static_cast<Eigen::Index>(axes_.size()); }

    Vector operator[](Eigen::Index index) const {
        if (index < 0 || index >= size()) {
            throw std::out_of_range("Parameter grid index is out of range");
        }
        Vector point(dimensions());
        for (Eigen::Index feature = point.size(); feature > 0; --feature) {
            const auto &axis = axes_[static_cast<std::size_t>(feature - 1)];
            const auto length = static_cast<Eigen::Index>(axis.size());
            point[feature - 1] = axis[static_cast<std::size_t>(index % length)];
            index /= length;
        }
        return point;
    }

private:
    std::vector<std::vector<double>> axes_;
    Eigen::Index size_ = 1;
};

} // namespace super_gpr
