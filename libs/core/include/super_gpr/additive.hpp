#pragma once

#include <super_gpr/gpr.hpp>

namespace super_gpr {

using FeatureGroup = std::vector<Eigen::Index>;

// Restrict a kernel to selected coordinates without erasing its concrete type.
template <class K> class SubspaceKernel {
    static_assert(IsKernel<K>::value, "K must satisfy the Kernel interface");

public:
    SubspaceKernel(K kernel, FeatureGroup features)
        : kernel_(std::move(kernel)), features_(std::move(features)) {
        if (features_.empty()) {
            throw std::invalid_argument("A kernel subspace must contain at least one feature");
        }
        auto sorted = features_;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.front() < 0 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            throw std::invalid_argument("Subspace features must be nonnegative and unique");
        }
        required_features_ = sorted.back() + 1;
        for (std::size_t index = 1; index < features_.size(); ++index) {
            contiguous_ =
                contiguous_ && features_[index] == features_.front() + static_cast<Eigen::Index>(index);
        }
    }

    const K &kernel() const { return kernel_; }
    const FeatureGroup &features() const { return features_; }
    Eigen::Index parameter_count() const { return kernel_.parameter_count(); }
    Vector theta() const { return kernel_.theta(); }
    Bounds bounds() const { return kernel_.bounds(); }
    void set_theta(Point theta) { kernel_.set_theta(theta); }

    double value(Point x, Point y, bool same) const {
        return with_points(
            x, y, [&](Point local_x, Point local_y) { return kernel_.value(local_x, local_y, same); });
    }
    void gradient(Point x, Point y, bool same, Gradient out) const {
        with_points(x, y,
                    [&](Point local_x, Point local_y) { kernel_.gradient(local_x, local_y, same, out); });
    }
    void input_gradient(Point x, Point y, Gradient out) const {
        out.setZero();
        if (contiguous_) {
            with_points(x, y, [&](Point local_x, Point local_y) {
                kernel_.input_gradient(local_x, local_y, out.segment(features_.front(), dimension()));
            });
        } else {
            Vector local(dimension());
            with_points(
                x, y, [&](Point local_x, Point local_y) { kernel_.input_gradient(local_x, local_y, local); });
            scatter(local, out);
        }
    }
    void diagonal_gradient(Point x, Gradient out) const {
        check_features(x);
        out.setZero();
        if (contiguous_) {
            kernel_.diagonal_gradient(x.segment(features_.front(), dimension()),
                                      out.segment(features_.front(), dimension()));
        } else {
            const Vector local_x = project(x);
            Vector local(dimension());
            kernel_.diagonal_gradient(local_x, local);
            scatter(local, out);
        }
    }

private:
    Eigen::Index dimension() const { return static_cast<Eigen::Index>(features_.size()); }
    void check_features(Point point) const {
        if (point.size() < required_features_) {
            throw std::invalid_argument("Subspace feature exceeds input dimension");
        }
    }
    Vector project(Point point) const {
        Vector result(dimension());
        for (Eigen::Index axis = 0; axis < dimension(); ++axis) {
            result[axis] = point[features_[static_cast<std::size_t>(axis)]];
        }
        return result;
    }
    void scatter(Point local, Gradient out) const {
        for (Eigen::Index axis = 0; axis < dimension(); ++axis) {
            out[features_[static_cast<std::size_t>(axis)]] = local[axis];
        }
    }
    template <class Function>
    auto with_points(Point x, Point y, Function function) const -> decltype(function(x, y)) {
        check_features(x);
        check_features(y);
        if (contiguous_) {
            return function(x.segment(features_.front(), dimension()),
                            y.segment(features_.front(), dimension()));
        }
        const Vector local_x = project(x);
        const Vector local_y = project(y);
        return function(local_x, local_y);
    }
    K kernel_;
    FeatureGroup features_;
    Eigen::Index required_features_ = 0;
    bool contiguous_ = true;
};

// A runtime number of subspaces, each retaining its statically typed kernel.
template <class K> class AdditiveKernel {
public:
    explicit AdditiveKernel(std::vector<SubspaceKernel<K>> components) : components_(std::move(components)) {
        if (components_.empty()) {
            throw std::invalid_argument("An additive kernel needs at least one component");
        }
        offsets_.reserve(components_.size() + 1);
        offsets_.push_back(0);
        for (const auto &component : components_) {
            offsets_.push_back(offsets_.back() + component.parameter_count());
        }
    }
    AdditiveKernel(const std::vector<FeatureGroup> &groups, const K &prototype)
        : AdditiveKernel(make_components(groups, prototype)) {}

    const std::vector<SubspaceKernel<K>> &components() const { return components_; }
    Eigen::Index parameter_count() const { return offsets_.back(); }
    Vector theta() const {
        Vector result(parameter_count());
        for (std::size_t index = 0; index < components_.size(); ++index) {
            result.segment(offsets_[index], components_[index].parameter_count()) =
                components_[index].theta();
        }
        return result;
    }
    Bounds bounds() const {
        Bounds result{Vector(parameter_count()), Vector(parameter_count())};
        for (std::size_t index = 0; index < components_.size(); ++index) {
            const auto bounds = components_[index].bounds();
            result.lower.segment(offsets_[index], bounds.lower.size()) = bounds.lower;
            result.upper.segment(offsets_[index], bounds.upper.size()) = bounds.upper;
        }
        return result;
    }
    void set_theta(Point theta) {
        if (theta.size() != parameter_count()) {
            throw std::invalid_argument("Wrong additive theta size");
        }
        for (std::size_t index = 0; index < components_.size(); ++index) {
            components_[index].set_theta(
                theta.segment(offsets_[index], components_[index].parameter_count()));
        }
    }
    double value(Point x, Point y, bool same) const {
        double result = 0.0;
        for (const auto &component : components_) {
            result += component.value(x, y, same);
        }
        return result;
    }
    void gradient(Point x, Point y, bool same, Gradient out) const {
        for (std::size_t index = 0; index < components_.size(); ++index) {
            components_[index].gradient(x, y, same,
                                        out.segment(offsets_[index], components_[index].parameter_count()));
        }
    }
    void input_gradient(Point x, Point y, Gradient out) const {
        out.setZero();
        Vector scratch(x.size());
        for (const auto &component : components_) {
            component.input_gradient(x, y, scratch);
            out += scratch;
        }
    }
    void diagonal_gradient(Point x, Gradient out) const {
        out.setZero();
        Vector scratch(x.size());
        for (const auto &component : components_) {
            component.diagonal_gradient(x, scratch);
            out += scratch;
        }
    }

private:
    static std::vector<SubspaceKernel<K>> make_components(const std::vector<FeatureGroup> &groups,
                                                          const K &prototype) {
        std::vector<SubspaceKernel<K>> components;
        components.reserve(groups.size());
        for (const auto &group : groups) {
            components.emplace_back(prototype, group);
        }
        return components;
    }
    std::vector<SubspaceKernel<K>> components_;
    std::vector<Eigen::Index> offsets_;
};

template <class K>
AdditiveKernel<K> make_additive_kernel(const std::vector<FeatureGroup> &groups, const K &prototype) {
    return AdditiveKernel<K>(groups, prototype);
}

template <class K> using AdditiveGP = GaussianProcessRegressor<AddKernel<AdditiveKernel<K>, WhiteKernel>>;

template <class K>
AdditiveGP<K> make_additive_gp(AdditiveKernel<K> kernel, Parameter noise = {1e-4, 1e-8, 1.0},
                               GPROptions options = {}) {
    return AdditiveGP<K>(std::move(kernel) + WhiteKernel(noise), std::move(options));
}

} // namespace super_gpr
