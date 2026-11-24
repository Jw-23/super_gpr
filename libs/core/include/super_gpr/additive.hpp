#pragma once

#include <super_gpr/gpr.hpp>

namespace super_gpr {

using FeatureGroup = std::vector<Eigen::Index>;

// Restrict a kernel to selected coordinates without erasing its concrete type.
template <class K> class SubspaceKernel : public detail::CheckedKernelEvaluation<SubspaceKernel<K>> {
    static_assert(IsKernel<K>::value, "K must satisfy the Kernel interface");

public:
    static GPRResult<SubspaceKernel> create(K kernel, FeatureGroup features) {
        if (features.empty()) {
            return boost::outcome_v2::failure(GPRError::InvalidFeatureGroup);
        }
        auto sorted = features;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.front() < 0 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            return boost::outcome_v2::failure(GPRError::InvalidFeatureGroup);
        }
        BOOST_OUTCOME_TRY(detail::validate_kernel_input(kernel, static_cast<Eigen::Index>(features.size()), 0));
        return SubspaceKernel(std::move(kernel), std::move(features), sorted.back());
    }

    GPRResult<void> validate_input(Eigen::Index dimensions) const {
        if (last_feature_ >= dimensions) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
        }
        return boost::outcome_v2::success();
    }

    const K &kernel() const { return kernel_; }
    const FeatureGroup &features() const { return features_; }
    Eigen::Index parameter_count() const { return kernel_.parameter_count(); }
    Vector theta() const { return kernel_.theta(); }
    Bounds bounds() const { return kernel_.bounds(); }
    GPRResult<void> set_theta(Point theta) { return kernel_.set_theta(theta); }

    double value_unchecked(Point x, Point y, bool same) const {
        return with_points(
            x, y, [&](Point local_x, Point local_y) { return kernel_.value_unchecked(local_x, local_y, same); });
    }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        with_points(x, y,
                    [&](Point local_x, Point local_y) { kernel_.gradient_unchecked(local_x, local_y, same, out); });
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const {
        out.setZero();
        if (contiguous_) {
            with_points(x, y, [&](Point local_x, Point local_y) {
                kernel_.input_gradient_unchecked(local_x, local_y, out.segment(features_.front(), dimension()));
            });
        } else {
            Vector local(dimension());
            with_points(
                x, y, [&](Point local_x, Point local_y) { kernel_.input_gradient_unchecked(local_x, local_y, local); });
            scatter(local, out);
        }
    }
    void diagonal_gradient_unchecked(Point x, Gradient out) const {
        assert(last_feature_ < x.size());
        out.setZero();
        if (contiguous_) {
            kernel_.diagonal_gradient_unchecked(contiguous_point(x),
                                      out.segment(features_.front(), dimension()));
        } else {
            const Vector local_x = project(x);
            Vector local(dimension());
            kernel_.diagonal_gradient_unchecked(local_x, local);
            scatter(local, out);
        }
    }

private:
    friend struct detail::TrainingKernel<SubspaceKernel<K>>;
    template <class> friend class AdditiveKernel;

    SubspaceKernel(K kernel, FeatureGroup features, Eigen::Index last_feature)
        : kernel_(std::move(kernel)), features_(std::move(features)), last_feature_(last_feature) {
        for (std::size_t index = 1; index < features_.size(); ++index) {
            contiguous_ = contiguous_ && features_[index] - features_.front() == static_cast<Eigen::Index>(index);
        }
    }

    Eigen::Index dimension() const { return static_cast<Eigen::Index>(features_.size()); }
    Eigen::Map<const Vector> contiguous_point(Point point) const {
        return Eigen::Map<const Vector>(point.data() + features_.front(), dimension());
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
    void add_local_gradient(Point local, Gradient out) const {
        if (contiguous_) {
            out.segment(features_.front(), dimension()) += local;
            return;
        }
        for (Eigen::Index axis = 0; axis < dimension(); ++axis) {
            out[features_[static_cast<std::size_t>(axis)]] += local[axis];
        }
    }
    void add_input_gradient(Point x, Point y, Gradient out, Gradient scratch) const {
        auto local = scratch.head(dimension());
        with_points(x, y, [&](Point local_x, Point local_y) {
            kernel_.input_gradient_unchecked(local_x, local_y, local);
        });
        add_local_gradient(local, out);
    }
    void add_diagonal_gradient(Point x, Gradient out, Gradient scratch) const {
        assert(last_feature_ < x.size());
        auto local = scratch.head(dimension());
        if (contiguous_) {
            kernel_.diagonal_gradient_unchecked(contiguous_point(x), local);
        } else {
            kernel_.diagonal_gradient_unchecked(project(x), local);
        }
        add_local_gradient(local, out);
    }
    template <class Function>
    auto with_points(Point x, Point y, Function function) const -> decltype(function(x, y)) {
        assert(last_feature_ < x.size());
        assert(last_feature_ < y.size());
        if (contiguous_) {
            return function(contiguous_point(x), contiguous_point(y));
        }
        const Vector local_x = project(x);
        const Vector local_y = project(y);
        return function(local_x, local_y);
    }
    template <class Function> void with_training_points(detail::TrainingPoints x, Function function) const {
        assert(last_feature_ < x.cols());
        if (contiguous_) {
            function(x.middleCols(features_.front(), dimension()));
            return;
        }
        Matrix projected(x.rows(), dimension());
        for (Eigen::Index row = 0; row < x.rows(); ++row) {
            for (Eigen::Index axis = 0; axis < dimension(); ++axis) {
                projected(row, axis) = x(row, features_[static_cast<std::size_t>(axis)]);
            }
        }
        function(projected);
    }
    K kernel_;
    FeatureGroup features_;
    Eigen::Index last_feature_;
    bool contiguous_ = true;
};

// A runtime number of subspaces, each retaining its statically typed kernel.
template <class K> class AdditiveKernel : public detail::CheckedKernelEvaluation<AdditiveKernel<K>> {
public:
    static GPRResult<AdditiveKernel> create(std::vector<SubspaceKernel<K>> components) {
        if (components.empty()) {
            return boost::outcome_v2::failure(GPRError::EmptyAdditiveKernel);
        }
        return AdditiveKernel(std::move(components));
    }
    static GPRResult<AdditiveKernel> create(const std::vector<FeatureGroup> &groups, const K &prototype) {
        std::vector<SubspaceKernel<K>> components;
        components.reserve(groups.size());
        for (const auto &group : groups) {
            BOOST_OUTCOME_TRY(auto component, SubspaceKernel<K>::create(prototype, group));
            components.push_back(std::move(component));
        }
        return create(std::move(components));
    }
    GPRResult<void> validate_input(Eigen::Index dimensions) const {
        for (const auto &component : components_) {
            BOOST_OUTCOME_TRY(component.validate_input(dimensions));
        }
        return boost::outcome_v2::success();
    }

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
    GPRResult<void> set_theta(Point theta) {
        if (theta.size() != parameter_count()) {
            return boost::outcome_v2::failure(GPRError::InvalidThetaSize);
        }
        for (std::size_t index = 0; index < components_.size(); ++index) {
            BOOST_OUTCOME_TRY(components_[index].set_theta(
                theta.segment(offsets_[index], components_[index].parameter_count())));
        }
        return boost::outcome_v2::success();
    }
    double value_unchecked(Point x, Point y, bool same) const {
        double result = 0.0;
        for (const auto &component : components_) {
            result += component.value_unchecked(x, y, same);
        }
        return result;
    }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        for (std::size_t index = 0; index < components_.size(); ++index) {
            components_[index].gradient_unchecked(x, y, same,
                                        out.segment(offsets_[index], components_[index].parameter_count()));
        }
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const {
        out.setZero();
        Vector scratch(max_group_size_);
        for (const auto &component : components_) {
            component.add_input_gradient(x, y, out, scratch);
        }
    }
    void diagonal_gradient_unchecked(Point x, Gradient out) const {
        out.setZero();
        Vector scratch(max_group_size_);
        for (const auto &component : components_) {
            component.add_diagonal_gradient(x, out, scratch);
        }
    }

private:
    explicit AdditiveKernel(std::vector<SubspaceKernel<K>> components) : components_(std::move(components)) {
        offsets_.reserve(components_.size() + 1);
        offsets_.push_back(0);
        for (const auto &component : components_) {
            offsets_.push_back(offsets_.back() + component.parameter_count());
            max_group_size_ = std::max(max_group_size_, component.dimension());
        }
    }

    std::vector<SubspaceKernel<K>> components_;
    std::vector<Eigen::Index> offsets_;
    Eigen::Index max_group_size_ = 0;
};

template <class K>
GPRResult<AdditiveKernel<K>> make_additive_kernel(const std::vector<FeatureGroup> &groups, const K &prototype) {
    return AdditiveKernel<K>::create(groups, prototype);
}

// Every component uses the same theta; the covariance remains a sum of kernels.
template <class K> class SharedAdditiveKernel : public detail::CheckedKernelEvaluation<SharedAdditiveKernel<K>> {
public:
    static GPRResult<SharedAdditiveKernel> create(const std::vector<FeatureGroup> &groups,
                                                 const K &prototype) {
        BOOST_OUTCOME_TRY(auto sum, AdditiveKernel<K>::create(groups, prototype));
        return SharedAdditiveKernel(std::move(sum));
    }
    GPRResult<void> validate_input(Eigen::Index dimensions) const { return sum_.validate_input(dimensions); }

    const std::vector<SubspaceKernel<K>> &components() const { return sum_.components(); }
    Eigen::Index parameter_count() const { return components().front().parameter_count(); }
    Vector theta() const { return components().front().theta(); }
    Bounds bounds() const { return components().front().bounds(); }
    GPRResult<void> set_theta(Point theta) {
        if (theta.size() != parameter_count()) {
            return boost::outcome_v2::failure(GPRError::InvalidThetaSize);
        }
        Vector expanded(sum_.parameter_count());
        Eigen::Index offset = 0;
        for (const auto &component : components()) {
            expanded.segment(offset, component.parameter_count()) = theta;
            offset += component.parameter_count();
        }
        return sum_.set_theta(expanded);
    }
    double value_unchecked(Point x, Point y, bool same) const { return sum_.value_unchecked(x, y, same); }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        out.setZero();
        Vector scratch(parameter_count());
        for (const auto &component : components()) {
            component.gradient_unchecked(x, y, same, scratch);
            out += scratch;
        }
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const { sum_.input_gradient_unchecked(x, y, out); }
    void diagonal_gradient_unchecked(Point x, Gradient out) const { sum_.diagonal_gradient_unchecked(x, out); }

private:
    explicit SharedAdditiveKernel(AdditiveKernel<K> sum) : sum_(std::move(sum)) {}

    AdditiveKernel<K> sum_;
};

template <class K>
GPRResult<SharedAdditiveKernel<K>> make_shared_additive_kernel(const std::vector<FeatureGroup> &groups,
                                                    const K &prototype) {
    return SharedAdditiveKernel<K>::create(groups, prototype);
}

namespace detail {

template <class K> struct TrainingKernel<SubspaceKernel<K>> {
    static constexpr bool split_covariance = true;

    static void add_covariance(const SubspaceKernel<K> &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        kernel.with_training_points(x, [&](TrainingPoints projected) {
            TrainingKernel<K>::add_covariance(kernel.kernel(), projected, matrix);
        });
    }

    static void contract(const SubspaceKernel<K> &kernel, TrainingPoints x,
                         const Eigen::MatrixXd &contraction, Gradient out) {
        kernel.with_training_points(x, [&](TrainingPoints projected) {
            TrainingKernel<K>::contract(kernel.kernel(), projected, contraction, out);
        });
    }
};

template <class K> struct AdditiveTrainingCovariance {
    static constexpr bool split_covariance = true;

    template <class Sum>
    static void add_covariance(const Sum &kernel, TrainingPoints x, Eigen::MatrixXd &matrix) {
        for (const auto &component : kernel.components()) {
            TrainingKernel<SubspaceKernel<K>>::add_covariance(component, x, matrix);
        }
    }
};

template <class K> struct TrainingKernel<AdditiveKernel<K>> : AdditiveTrainingCovariance<K> {
    static void contract(const AdditiveKernel<K> &kernel, TrainingPoints x,
                         const Eigen::MatrixXd &contraction, Gradient out) {
        Eigen::Index offset = 0;
        for (const auto &component : kernel.components()) {
            const Eigen::Index count = component.parameter_count();
            TrainingKernel<SubspaceKernel<K>>::contract(component, x, contraction, out.segment(offset, count));
            offset += count;
        }
    }
};

template <class K> struct TrainingKernel<SharedAdditiveKernel<K>> : AdditiveTrainingCovariance<K> {
    static void contract(const SharedAdditiveKernel<K> &kernel, TrainingPoints x,
                         const Eigen::MatrixXd &contraction, Gradient out) {
        for (const auto &component : kernel.components()) {
            TrainingKernel<SubspaceKernel<K>>::contract(component, x, contraction, out);
        }
    }
};

} // namespace detail

template <class K> using AdditiveGP = GaussianProcessRegressor<AddKernel<AdditiveKernel<K>, WhiteKernel>>;
template <class K>
using SharedAdditiveGP = GaussianProcessRegressor<AddKernel<SharedAdditiveKernel<K>, WhiteKernel>>;

template <class K>
GPRResult<AdditiveGP<K>> make_additive_gp(AdditiveKernel<K> kernel, Parameter noise = {1e-4, 1e-8, 1.0},
                               GPROptions options = {}) {
    BOOST_OUTCOME_TRY(auto white, WhiteKernel::create(noise));
    return AdditiveGP<K>(std::move(kernel) + std::move(white), std::move(options));
}

template <class K>
GPRResult<SharedAdditiveGP<K>> make_additive_gp(SharedAdditiveKernel<K> kernel, Parameter noise = {1e-4, 1e-8, 1.0},
                                     GPROptions options = {}) {
    BOOST_OUTCOME_TRY(auto white, WhiteKernel::create(noise));
    return SharedAdditiveGP<K>(std::move(kernel) + std::move(white), std::move(options));
}

} // namespace super_gpr
