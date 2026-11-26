#pragma once

#include <algorithm>
#include <boost/outcome/try.hpp>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <super_gpr/types.hpp>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

namespace super_gpr {

struct Parameter {
    double value = 1.0;
    double lower = 1e-5;
    double upper = 1e5;

    static Parameter fixed(double value) { return {value, value, value}; }
};

// Fixed parameters are omitted from theta, bounds, and hyperparameter gradients.
class ParameterSet {
public:
    Eigen::Index parameter_count() const { return size_; }

    Vector theta() const {
        Vector result(size_);
        Eigen::Index index = 0;
        for (const auto &parameter : parameters_) {
            if (parameter.lower != parameter.upper) {
                result[index++] = std::log(parameter.value);
            }
        }
        return result;
    }

    Bounds bounds() const {
        Bounds result{Vector(size_), Vector(size_)};
        Eigen::Index index = 0;
        for (const auto &parameter : parameters_) {
            if (parameter.lower != parameter.upper) {
                result.lower[index] = std::log(parameter.lower);
                result.upper[index++] = std::log(parameter.upper);
            }
        }
        return result;
    }

    GPRResult<void> set_theta(Point theta) {
        if (theta.size() != size_) {
            return boost::outcome_v2::failure(GPRError::InvalidThetaSize);
        }
        // Validate before mutating, including exp overflow and underflow.
        const double min_theta = std::log(std::numeric_limits<double>::denorm_min());
        const double max_theta = std::log(std::numeric_limits<double>::max());
        if (!((theta.array() >= min_theta) && (theta.array() <= max_theta)).all()) {
            return boost::outcome_v2::failure(GPRError::InvalidTheta);
        }
        Eigen::Index index = 0;
        for (auto &parameter : parameters_) {
            if (parameter.lower != parameter.upper) {
                parameter.value = std::exp(theta[index++]);
            }
        }
        return boost::outcome_v2::success();
    }

protected:
    explicit ParameterSet(std::vector<Parameter> parameters) : parameters_(std::move(parameters)) {
        for (const auto &parameter : parameters_) {
            if (parameter.lower != parameter.upper) {
                ++size_;
            }
        }
    }

    template <class K> static GPRResult<K> checked_parameters(K kernel, bool allow_fixed_zero = false) {
        for (const auto &parameter : kernel.parameters_) {
            const bool fixed = parameter.lower == parameter.upper;
            const bool valid_value = std::isfinite(parameter.value) &&
                (parameter.value > 0.0 || (fixed && allow_fixed_zero && parameter.value == 0.0));
            const bool valid_bounds = fixed ? parameter.value == parameter.lower
                : parameter.lower > 0.0 && std::isfinite(parameter.lower) && parameter.lower <= parameter.upper;
            if (!valid_value || !valid_bounds) {
                return boost::outcome_v2::failure(GPRError::InvalidKernelParameters);
            }
        }
        return kernel;
    }

    double parameter(std::size_t index) const { return parameters_[index].value; }
    std::size_t stored_parameter_count() const { return parameters_.size(); }
    void write_gradient(std::size_t index, double value, Gradient gradient, Eigen::Index &offset) const {
        if (parameters_[index].lower != parameters_[index].upper) {
            gradient[offset++] = value;
        }
    }

private:
    std::vector<Parameter> parameters_;
    Eigen::Index size_ = 0;
};

template <class...> using Void = void;
template <class K, class = void> struct IsKernel : std::false_type {};
template <class K>
struct IsKernel<
    K, Void<decltype(std::declval<const K &>().value_unchecked(std::declval<Point>(), std::declval<Point>(), true)),
            decltype(std::declval<const K &>().gradient_unchecked(std::declval<Point>(), std::declval<Point>(), true,
                                                        std::declval<Gradient>())),
            decltype(std::declval<const K &>().input_gradient_unchecked(std::declval<Point>(), std::declval<Point>(),
                                                              std::declval<Gradient>())),
            decltype(std::declval<const K &>().diagonal_gradient_unchecked(std::declval<Point>(),
                                                                 std::declval<Gradient>())),
            decltype(std::declval<const K &>().theta()), decltype(std::declval<const K &>().bounds()),
            decltype(std::declval<const K &>().parameter_count()),
            decltype(std::declval<K &>().set_theta(std::declval<Point>()))>>
    : std::is_same<decltype(std::declval<K &>().set_theta(std::declval<Point>())), GPRResult<void>> {};

namespace detail {

inline bool valid_optimizer_bounds(const Bounds &bounds, Eigen::Index size) {
    const double infinity = std::numeric_limits<double>::infinity();
    return bounds.lower.size() == size && bounds.upper.size() == size &&
           (bounds.lower.array() <= bounds.upper.array()).all() &&
           (bounds.lower.array() < infinity).all() && (bounds.upper.array() > -infinity).all();
}

template <class K>
auto validate_kernel_input(const K &kernel, Eigen::Index dimensions, int)
    -> decltype(kernel.validate_input(dimensions)) {
    return kernel.validate_input(dimensions);
}

template <class K>
GPRResult<void> validate_kernel_input(const K &, Eigen::Index, long) {
    return boost::outcome_v2::success();
}

// same=false selects the latent diagonal. Smooth kernels can use the
// same derivative for either target.
template <class K>
auto kernel_diagonal_gradient(const K &kernel, Point x, Gradient out, bool same, int)
    -> decltype(kernel.diagonal_gradient_unchecked(x, out, same)) {
    kernel.diagonal_gradient_unchecked(x, out, same);
}

template <class K>
void kernel_diagonal_gradient(const K &kernel, Point x, Gradient out, bool same, long) {
    if (same) {
        kernel.diagonal_gradient_unchecked(x, out);
    } else {
        // By symmetry, both arguments contribute the same derivative on the diagonal.
        kernel.input_gradient_unchecked(x, x, out);
        out *= 2.0;
    }
}

struct KernelIdentity {
    const void *address;
    const std::type_info &type;

    template <class K> explicit KernelIdentity(const K &kernel)
        : address(std::addressof(kernel)), type(typeid(K)) {}

    bool operator==(const KernelIdentity &other) const {
        return address == other.address && type == other.type;
    }
};

template <class K>
auto has_additive_component(const K &kernel, KernelIdentity component, int)
    -> decltype(kernel.has_additive_component(component)) {
    return kernel.has_additive_component(component);
}

template <class K>
bool has_additive_component(const K &, KernelIdentity, long) {
    return false;
}

template <class K> bool contains_component(const K &kernel, KernelIdentity component) {
    return KernelIdentity(kernel) == component || has_additive_component(kernel, component, 0);
}

// Opt in library kernels by exact type so user subclasses with custom value or
// gradient methods continue to use their original interface.
template <class K> struct ValueGradient {
    static double evaluate(const K &kernel, Point x, Point y, bool same, Gradient out) {
        kernel.gradient_unchecked(x, y, same, out);
        return kernel.value_unchecked(x, y, same);
    }
};

template <class K> struct FusedValueGradient {
    static double evaluate(const K &kernel, Point x, Point y, bool same, Gradient out) {
        return kernel.value_and_gradient_unchecked(x, y, same, out);
    }
};

// Public point evaluations validate once. Training and prediction use the
// unchecked kernel interface after validating the shape of the entire batch.
template <class K> class CheckedKernelEvaluation {
public:
    GPRResult<double> value(Point x, Point y, bool same) const {
        BOOST_OUTCOME_TRY(validate_points(x, y));
        return kernel().value_unchecked(x, y, same);
    }
    GPRResult<void> gradient(Point x, Point y, bool same, Gradient out) const {
        BOOST_OUTCOME_TRY(validate_gradient(x, y, out.size()));
        kernel().gradient_unchecked(x, y, same, out);
        return boost::outcome_v2::success();
    }
    GPRResult<double> value_and_gradient(Point x, Point y, bool same, Gradient out) const {
        BOOST_OUTCOME_TRY(validate_gradient(x, y, out.size()));
        return ValueGradient<K>::evaluate(kernel(), x, y, same, out);
    }
    GPRResult<void> input_gradient(Point x, Point y, Gradient out) const {
        if (out.size() != x.size()) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
        }
        BOOST_OUTCOME_TRY(validate_points(x, y));
        kernel().input_gradient_unchecked(x, y, out);
        return boost::outcome_v2::success();
    }
    GPRResult<void> diagonal_gradient(Point x, Gradient out) const {
        if (out.size() != x.size()) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
        }
        BOOST_OUTCOME_TRY(validate_kernel_input(kernel(), x.size(), 0));
        kernel().diagonal_gradient_unchecked(x, out);
        return boost::outcome_v2::success();
    }

private:
    const K &kernel() const { return static_cast<const K &>(*this); }
    GPRResult<void> validate_points(Point x, Point y) const {
        if (x.size() != y.size()) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
        }
        return validate_kernel_input(kernel(), x.size(), 0);
    }
    GPRResult<void> validate_gradient(Point x, Point y, Eigen::Index size) const {
        if (size != kernel().parameter_count()) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
        }
        return validate_points(x, y);
    }
};

} // namespace detail

class ConstantKernel : public ParameterSet, public detail::CheckedKernelEvaluation<ConstantKernel> {
public:
    ConstantKernel() : ConstantKernel(Parameter{}) {}
    static GPRResult<ConstantKernel> fixed(double value) { return create(Parameter::fixed(value)); }
    static GPRResult<ConstantKernel> create(Parameter constant) {
        return checked_parameters(ConstantKernel(constant), true);
    }
    double value_unchecked(Point /*x*/, Point /*y*/, bool /*same*/) const { return parameter(0); }
    void gradient_unchecked(Point /*x*/, Point /*y*/, bool /*same*/, Gradient out) const {
        Eigen::Index offset = 0;
        write_gradient(0, parameter(0), out, offset);
    }
    static void input_gradient_unchecked(Point /*x*/, Point /*y*/, Gradient out) { out.setZero(); }
    static void diagonal_gradient_unchecked(Point /*x*/, Gradient out, bool /*same*/ = true) { out.setZero(); }

private:
    explicit ConstantKernel(Parameter constant) : ParameterSet({constant}) {}
};

class WhiteKernel : public ParameterSet, public detail::CheckedKernelEvaluation<WhiteKernel> {
public:
    WhiteKernel() : WhiteKernel(Parameter{}) {}
    static GPRResult<WhiteKernel> create(Parameter noise) {
        return checked_parameters(WhiteKernel(noise), true);
    }
    double value_unchecked(Point /*x*/, Point /*y*/, bool same) const { return same ? parameter(0) : 0.0; }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        Eigen::Index offset = 0;
        write_gradient(0, value_unchecked(x, y, same), out, offset);
    }
    static void input_gradient_unchecked(Point /*x*/, Point /*y*/, Gradient out) { out.setZero(); }
    static void diagonal_gradient_unchecked(Point /*x*/, Gradient out, bool /*same*/ = true) { out.setZero(); }

private:
    explicit WhiteKernel(Parameter noise) : ParameterSet({noise}) {}
};

// A single length scale is isotropic; multiple scales enable ARD.
class RadialKernel : public ParameterSet {
public:
    GPRResult<void> validate_input(Eigen::Index dimensions) const {
        if (stored_parameter_count() != 1 &&
            stored_parameter_count() != static_cast<std::size_t>(dimensions)) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
        }
        return boost::outcome_v2::success();
    }
    static void diagonal_gradient_unchecked(Point /*x*/, Gradient out, bool /*same*/ = true) { out.setZero(); }

protected:
    explicit RadialKernel(Parameter scale) : ParameterSet({scale}) {}
    explicit RadialKernel(std::vector<Parameter> scales) : ParameterSet(std::move(scales)) {}

    double scale(Eigen::Index axis) const {
        return parameter(stored_parameter_count() == 1 ? 0 : static_cast<std::size_t>(axis));
    }
    double squared_distance(Point x, Point y) const {
        if (stored_parameter_count() == 1) {
            return ((x - y) / parameter(0)).squaredNorm();
        }
        assert(stored_parameter_count() == static_cast<std::size_t>(x.size()));
        double result = 0.0;
        for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
            const double delta = (x[axis] - y[axis]) / parameter(static_cast<std::size_t>(axis));
            result += delta * delta;
        }
        return result;
    }
    double squared_distance_and_gradient(Point x, Point y, Gradient out) const {
        Eigen::Index offset = 0;
        if (stored_parameter_count() == 1) {
            const double distance = squared_distance(x, y);
            write_gradient(0, distance, out, offset);
            return distance;
        }
        assert(stored_parameter_count() == static_cast<std::size_t>(x.size()));
        double distance = 0.0;
        for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
            const auto index = static_cast<std::size_t>(axis);
            const double delta = (x[axis] - y[axis]) / parameter(index);
            const double square = delta * delta;
            distance += square;
            write_gradient(index, square, out, offset);
        }
        return distance;
    }
    void radial_gradient(Point x, Point y, double factor, Gradient out) const {
        squared_distance_and_gradient(x, y, out);
        out *= factor;
    }
    void radial_input_gradient(Point x, Point y, double factor, Gradient out) const {
        for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
            out[axis] = -factor * (x[axis] - y[axis]) / (scale(axis) * scale(axis));
        }
    }
};

class RBF : public RadialKernel, public detail::CheckedKernelEvaluation<RBF> {
public:
    RBF() : RBF(Parameter{}) {}
    static GPRResult<RBF> create(Parameter scale) { return checked_parameters(RBF(scale)); }
    static GPRResult<RBF> create(std::vector<Parameter> scales) {
        if (scales.empty()) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelParameters);
        }
        return checked_parameters(RBF(std::move(scales)));
    }
    double value_unchecked(Point x, Point y, bool /*same*/) const { return std::exp(-0.5 * squared_distance(x, y)); }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        value_and_gradient_unchecked(x, y, same, out);
    }
    double value_and_gradient_unchecked(Point x, Point y, bool /*same*/, Gradient out) const {
        const double kernel_value = std::exp(-0.5 * squared_distance_and_gradient(x, y, out));
        out *= kernel_value;
        return kernel_value;
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const {
        radial_input_gradient(x, y, value_unchecked(x, y, false), out);
    }
private:
    explicit RBF(Parameter scale) : RadialKernel(scale) {}
    explicit RBF(std::vector<Parameter> scales) : RadialKernel(std::move(scales)) {}
};

class Matern : public RadialKernel, public detail::CheckedKernelEvaluation<Matern> {
public:
    Matern() : Matern(Parameter{}, 1.5) {}
    static GPRResult<Matern> create(Parameter scale, double nu = 1.5) {
        BOOST_OUTCOME_TRY(check_nu(nu));
        return checked_parameters(Matern(scale, nu));
    }
    static GPRResult<Matern> create(std::vector<Parameter> scales, double nu = 1.5) {
        BOOST_OUTCOME_TRY(check_nu(nu));
        if (scales.empty()) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelParameters);
        }
        return checked_parameters(Matern(std::move(scales), nu));
    }
    double value_unchecked(Point x, Point y, bool /*same*/) const {
        return radial_value(std::sqrt(squared_distance(x, y))).value;
    }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        value_and_gradient_unchecked(x, y, same, out);
    }
    double value_and_gradient_unchecked(Point x, Point y, bool /*same*/, Gradient out) const {
        const auto radial = radial_value(std::sqrt(squared_distance_and_gradient(x, y, out)));
        out *= radial.gradient_factor;
        return radial.value;
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const {
        radial_input_gradient(x, y, radial_value(std::sqrt(squared_distance(x, y))).gradient_factor, out);
    }

private:
    Matern(Parameter scale, double nu) : RadialKernel(scale), nu_(nu) {}
    Matern(std::vector<Parameter> scales, double nu) : RadialKernel(std::move(scales)), nu_(nu) {}

    static GPRResult<void> check_nu(double nu) {
        if (nu != 0.5 && nu != 1.5 && nu != 2.5 && nu != std::numeric_limits<double>::infinity()) {
            return boost::outcome_v2::failure(GPRError::InvalidMaternNu);
        }
        return boost::outcome_v2::success();
    }
    struct RadialValue {
        double value;
        double gradient_factor;
    };

    RadialValue radial_value(double distance) const {
        if (nu_ == 0.5) {
            const double decay = std::exp(-distance);
            return {decay, distance == 0.0 ? 0.0 : decay / distance};
        }
        if (nu_ == 1.5) {
            const double r = std::sqrt(3.0) * distance;
            const double decay = std::exp(-r);
            return {(1.0 + r) * decay, 3.0 * decay};
        }
        if (nu_ == 2.5) {
            const double r = std::sqrt(5.0) * distance;
            const double decay = std::exp(-r);
            return {(1.0 + r + r * r / 3.0) * decay, (5.0 / 3.0) * (1.0 + r) * decay};
        }
        const double decay = std::exp(-0.5 * distance * distance);
        return {decay, decay};
    }
    double nu_;
};

namespace detail {
template <> struct ValueGradient<RBF> : FusedValueGradient<RBF> {};
template <> struct ValueGradient<Matern> : FusedValueGradient<Matern> {};
} // namespace detail

class RationalQuadratic : public ParameterSet, public detail::CheckedKernelEvaluation<RationalQuadratic> {
public:
    RationalQuadratic() : RationalQuadratic(Parameter{}, Parameter{}) {}
    static GPRResult<RationalQuadratic> create(Parameter scale, Parameter alpha = {}) {
        return checked_parameters(RationalQuadratic(scale, alpha));
    }
    double value_unchecked(Point x, Point y, bool /*same*/) const { return std::pow(base(x, y), -parameter(0)); }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        const double b = base(x, y);
        const double k = value_unchecked(x, y, same);
        Eigen::Index offset = 0;
        write_gradient(0, k * parameter(0) * (-std::log(b) + (b - 1.0) / b), out, offset);
        write_gradient(1, k * (x - y).squaredNorm() / (parameter(1) * parameter(1) * b), out, offset);
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const {
        out = -(x - y) * (value_unchecked(x, y, false) / (parameter(1) * parameter(1) * base(x, y)));
    }
    static void diagonal_gradient_unchecked(Point /*x*/, Gradient out, bool /*same*/ = true) { out.setZero(); }

private:
    RationalQuadratic(Parameter scale, Parameter alpha) : ParameterSet({alpha, scale}) {}

    double base(Point x, Point y) const {
        return 1.0 + (x - y).squaredNorm() / (2.0 * parameter(0) * parameter(1) * parameter(1));
    }
};

class ExpSineSquared : public ParameterSet, public detail::CheckedKernelEvaluation<ExpSineSquared> {
public:
    ExpSineSquared() : ExpSineSquared(Parameter{}, Parameter{}) {}
    static GPRResult<ExpSineSquared> create(Parameter scale, Parameter periodicity = {}) {
        return checked_parameters(ExpSineSquared(scale, periodicity));
    }
    GPRResult<void> validate_input(Eigen::Index dimensions) const {
        if (dimensions != 1) {
            return boost::outcome_v2::failure(GPRError::InvalidKernelInput);
        }
        return boost::outcome_v2::success();
    }
    double value_unchecked(Point x, Point y, bool /*same*/) const {
        const double sine = std::sin(angle(x, y));
        return std::exp(-2.0 * sine * sine / (parameter(0) * parameter(0)));
    }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        const double a = angle(x, y);
        const double factor = 4.0 * value_unchecked(x, y, same) / (parameter(0) * parameter(0));
        Eigen::Index offset = 0;
        write_gradient(0, factor * std::sin(a) * std::sin(a), out, offset);
        write_gradient(1, factor * a * std::sin(a) * std::cos(a), out, offset);
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const {
        const double distance = (x - y).norm();
        if (distance == 0.0) {
            out.setZero();
            return;
        }
        const double a = angle(x, y);
        out = (x - y) * (-4.0 * value_unchecked(x, y, false) * std::sin(a) * std::cos(a) * pi() /
                         (parameter(0) * parameter(0) * parameter(1) * distance));
    }
    static void diagonal_gradient_unchecked(Point /*x*/, Gradient out, bool /*same*/ = true) { out.setZero(); }

private:
    ExpSineSquared(Parameter scale, Parameter periodicity) : ParameterSet({scale, periodicity}) {}

    static double pi() { return std::acos(-1.0); }
    double angle(Point x, Point y) const { return pi() * (x - y).norm() / parameter(1); }
};

class DotProduct : public ParameterSet, public detail::CheckedKernelEvaluation<DotProduct> {
public:
    DotProduct() : DotProduct(Parameter{}) {}
    static GPRResult<DotProduct> create(Parameter sigma) {
        return checked_parameters(DotProduct(sigma), true);
    }
    double value_unchecked(Point x, Point y, bool /*same*/) const { return x.dot(y) + parameter(0) * parameter(0); }
    void gradient_unchecked(Point /*x*/, Point /*y*/, bool /*same*/, Gradient out) const {
        Eigen::Index offset = 0;
        write_gradient(0, 2.0 * parameter(0) * parameter(0), out, offset);
    }
    static void input_gradient_unchecked(Point /*x*/, Point y, Gradient out) { out = y; }
    static void diagonal_gradient_unchecked(Point x, Gradient out, bool /*same*/ = true) { out = 2.0 * x; }

private:
    explicit DotProduct(Parameter sigma) : ParameterSet({sigma}) {}
};

struct AddOperation {
    static double apply(double left, double right) { return left + right; }
    static double left_factor(double /*left*/, double /*right*/) { return 1.0; }
    static double right_factor(double /*left*/, double /*right*/) { return 1.0; }
};
struct MultiplyOperation {
    static double apply(double left, double right) { return left * right; }
    static double left_factor(double /*left*/, double right) { return right; }
    static double right_factor(double left, double /*right*/) { return left; }
};
template <class K1, class K2, class Operation> class BinaryKernel
    : public detail::CheckedKernelEvaluation<BinaryKernel<K1, K2, Operation>> {
    static_assert(IsKernel<K1>::value && IsKernel<K2>::value, "Operands must satisfy the Kernel interface");

public:
    BinaryKernel(K1 left, K2 right) : left_(std::move(left)), right_(std::move(right)) {}
    const K1 &left() const { return left_; }
    const K2 &right() const { return right_; }
    Eigen::Index parameter_count() const { return left_.parameter_count() + right_.parameter_count(); }
    Vector theta() const {
        Vector out(parameter_count());
        out << left_.theta(), right_.theta();
        return out;
    }
    Bounds bounds() const {
        const auto left = left_.bounds();
        const auto right = right_.bounds();
        Bounds out{Vector(parameter_count()), Vector(parameter_count())};
        out.lower << left.lower, right.lower;
        out.upper << left.upper, right.upper;
        return out;
    }
    GPRResult<void> set_theta(Point theta) {
        if (theta.size() != parameter_count()) {
            return boost::outcome_v2::failure(GPRError::InvalidThetaSize);
        }
        BOOST_OUTCOME_TRY(left_.set_theta(theta.head(left_.parameter_count())));
        return right_.set_theta(theta.tail(right_.parameter_count()));
    }
    GPRResult<void> validate_input(Eigen::Index dimensions) const {
        BOOST_OUTCOME_TRY(detail::validate_kernel_input(left_, dimensions, 0));
        return detail::validate_kernel_input(right_, dimensions, 0);
    }
    double value_unchecked(Point x, Point y, bool same) const {
        return Operation::apply(left_.value_unchecked(x, y, same), right_.value_unchecked(x, y, same));
    }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        gradient_impl(x, y, same, out, std::is_same<Operation, AddOperation>{});
    }
    double value_and_gradient_unchecked(Point x, Point y, bool same, Gradient out) const {
        auto left = out.head(left_.parameter_count());
        auto right = out.tail(right_.parameter_count());
        const double a = detail::ValueGradient<K1>::evaluate(left_, x, y, same, left);
        const double b = detail::ValueGradient<K2>::evaluate(right_, x, y, same, right);
        left *= Operation::left_factor(a, b);
        right *= Operation::right_factor(a, b);
        return Operation::apply(a, b);
    }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const {
        spatial_gradient(x, y, false, out,
                         [&](const auto &kernel, Gradient target) { kernel.input_gradient_unchecked(x, y, target); });
    }
    void diagonal_gradient_unchecked(Point x, Gradient out, bool same = true) const {
        spatial_gradient(x, x, same, out, [&](const auto &kernel, Gradient target) {
            detail::kernel_diagonal_gradient(kernel, x, target, same, 0);
        });
    }

private:
    template <class Derivative>
    void spatial_gradient(Point x, Point y, bool same, Gradient out, Derivative derivative) const {
        constexpr bool is_sum = std::is_same<Operation, AddOperation>::value;
        constexpr bool is_product = std::is_same<Operation, MultiplyOperation>::value;
        constexpr bool left_constant =
            std::is_same<K1, ConstantKernel>::value || std::is_same<K1, WhiteKernel>::value;
        constexpr bool right_constant =
            std::is_same<K2, ConstantKernel>::value || std::is_same<K2, WhiteKernel>::value;
        if ((is_sum || is_product) && left_constant) {
            derivative(right_, out);
            if (is_product) {
                out *= left_.value_unchecked(x, y, same);
            }
            return;
        }
        derivative(left_, out);
        if ((is_sum || is_product) && right_constant) {
            if (is_product) {
                out *= right_.value_unchecked(x, y, same);
            }
            return;
        }
        Vector scratch(out.size());
        derivative(right_, scratch);
        if (is_sum) {
            out += scratch;
            return;
        }
        const double a = left_.value_unchecked(x, y, same);
        const double b = right_.value_unchecked(x, y, same);
        out = Operation::left_factor(a, b) * out + Operation::right_factor(a, b) * scratch;
    }

    void gradient_impl(Point x, Point y, bool same, Gradient out, std::true_type) const {
        left_.gradient_unchecked(x, y, same, out.head(left_.parameter_count()));
        right_.gradient_unchecked(x, y, same, out.tail(right_.parameter_count()));
    }
    void gradient_impl(Point x, Point y, bool same, Gradient out, std::false_type) const {
        value_and_gradient_unchecked(x, y, same, out);
    }

    K1 left_;
    K2 right_;
};

template <class K1, class K2> class AddKernel final : public BinaryKernel<K1, K2, AddOperation> {
public:
    AddKernel(K1 left, K2 right) : BinaryKernel<K1, K2, AddOperation>(std::move(left), std::move(right)) {}

    bool has_additive_component(detail::KernelIdentity component) const {
        return detail::contains_component(this->left(), component) ||
               detail::contains_component(this->right(), component);
    }
};

template <class K1, class K2> class Multiple final : public BinaryKernel<K1, K2, MultiplyOperation> {
public:
    Multiple(K1 left, K2 right)
        : BinaryKernel<K1, K2, MultiplyOperation>(std::move(left), std::move(right)) {}
};

template <class K1, class K2> using Add = AddKernel<K1, K2>;
template <class K1, class K2> using Multiply = Multiple<K1, K2>;

namespace detail {
template <class K1, class K2, class Operation> struct ValueGradient<BinaryKernel<K1, K2, Operation>>
    : FusedValueGradient<BinaryKernel<K1, K2, Operation>> {};
template <class K1, class K2> struct ValueGradient<AddKernel<K1, K2>>
    : FusedValueGradient<AddKernel<K1, K2>> {};
template <class K1, class K2> struct ValueGradient<Multiple<K1, K2>>
    : FusedValueGradient<Multiple<K1, K2>> {};
} // namespace detail

template <class K1, class K2, std::enable_if_t<IsKernel<K1>::value && IsKernel<K2>::value, int> = 0>
AddKernel<K1, K2> operator+(K1 left, K2 right) {
    return {std::move(left), std::move(right)};
}
template <class K1, class K2, std::enable_if_t<IsKernel<K1>::value && IsKernel<K2>::value, int> = 0>
Multiple<K1, K2> operator*(K1 left, K2 right) {
    return {std::move(left), std::move(right)};
}

// Scalar factors are fixed ConstantKernels and introduce no optimization parameters.
template <class K, std::enable_if_t<IsKernel<K>::value, int> = 0>
GPRResult<Multiple<ConstantKernel, K>> operator*(double factor, K kernel) {
    BOOST_OUTCOME_TRY(auto constant, ConstantKernel::fixed(factor));
    return std::move(constant) * std::move(kernel);
}
template <class K, std::enable_if_t<IsKernel<K>::value, int> = 0>
GPRResult<Multiple<K, ConstantKernel>> operator*(K kernel, double factor) {
    BOOST_OUTCOME_TRY(auto constant, ConstantKernel::fixed(factor));
    return std::move(kernel) * std::move(constant);
}

class BoxedKernel : public detail::CheckedKernelEvaluation<BoxedKernel> {
    struct Interface {
        Interface() = default;
        Interface(const Interface &) = default;
        Interface &operator=(const Interface &) = default;
        Interface(Interface &&) = default;
        Interface &operator=(Interface &&) = default;
        virtual ~Interface() = default;
        virtual std::unique_ptr<Interface> clone() const = 0;
        virtual Eigen::Index parameter_count() const = 0;
        virtual Vector theta() const = 0;
        virtual Bounds bounds() const = 0;
        virtual GPRResult<void> set_theta(Point theta) = 0;
        virtual GPRResult<void> validate_input(Eigen::Index dimensions) const = 0;
        virtual double value_unchecked(Point x, Point y, bool same) const = 0;
        virtual void gradient_unchecked(Point x, Point y, bool same, Gradient out) const = 0;
        virtual void input_gradient_unchecked(Point x, Point y, Gradient out) const = 0;
        virtual void diagonal_gradient_unchecked(Point x, Gradient out, bool same) const = 0;
    };
    template <class K> struct Model final : Interface {
        explicit Model(K kernel) : kernel(std::move(kernel)) {}
        std::unique_ptr<Interface> clone() const override { return std::make_unique<Model>(kernel); }
        Eigen::Index parameter_count() const override { return kernel.parameter_count(); }
        Vector theta() const override { return kernel.theta(); }
        Bounds bounds() const override { return kernel.bounds(); }
        GPRResult<void> set_theta(Point theta) override { return kernel.set_theta(theta); }
        GPRResult<void> validate_input(Eigen::Index dimensions) const override {
            return detail::validate_kernel_input(kernel, dimensions, 0);
        }
        double value_unchecked(Point x, Point y, bool same) const override { return kernel.value_unchecked(x, y, same); }
        void gradient_unchecked(Point x, Point y, bool same, Gradient out) const override {
            kernel.gradient_unchecked(x, y, same, out);
        }
        void input_gradient_unchecked(Point x, Point y, Gradient out) const override {
            kernel.input_gradient_unchecked(x, y, out);
        }
        void diagonal_gradient_unchecked(Point x, Gradient out, bool same) const override {
            detail::kernel_diagonal_gradient(kernel, x, out, same, 0);
        }
        K kernel;
    };

public:
    template <class K, std::enable_if_t<IsKernel<K>::value && !std::is_same<K, BoxedKernel>::value, int> = 0>
    explicit BoxedKernel(K kernel) : kernel_(std::make_unique<Model<K>>(std::move(kernel))) {}
    BoxedKernel(const BoxedKernel &other) : kernel_(other.kernel_->clone()) {}
    BoxedKernel &operator=(const BoxedKernel &other) {
        if (this != &other) {
            kernel_ = other.kernel_->clone();
        }
        return *this;
    }
    BoxedKernel(BoxedKernel &&) noexcept = default;
    BoxedKernel &operator=(BoxedKernel &&) noexcept = default;
    ~BoxedKernel() = default;
    Eigen::Index parameter_count() const { return kernel_->parameter_count(); }
    Vector theta() const { return kernel_->theta(); }
    Bounds bounds() const { return kernel_->bounds(); }
    GPRResult<void> set_theta(Point theta) { return kernel_->set_theta(theta); }
    GPRResult<void> validate_input(Eigen::Index dimensions) const {
        return kernel_->validate_input(dimensions);
    }
    double value_unchecked(Point x, Point y, bool same) const { return kernel_->value_unchecked(x, y, same); }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const { kernel_->gradient_unchecked(x, y, same, out); }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const { kernel_->input_gradient_unchecked(x, y, out); }
    void diagonal_gradient_unchecked(Point x, Gradient out, bool same = true) const {
        kernel_->diagonal_gradient_unchecked(x, out, same);
    }

private:
    std::unique_ptr<Interface> kernel_;
};

// User callbacks receive theta in log coordinates. All derivatives are supplied analytically.
template <class Value, class HyperGradient, class InputGradient, class DiagonalGradient>
class AnalyticKernel
    : public detail::CheckedKernelEvaluation<AnalyticKernel<Value, HyperGradient, InputGradient, DiagonalGradient>> {
public:
    static GPRResult<AnalyticKernel> create(Vector theta, Bounds bounds, Value value, HyperGradient hyper,
                                           InputGradient input, DiagonalGradient diagonal) {
        if (!detail::valid_optimizer_bounds(bounds, theta.size())) {
            return boost::outcome_v2::failure(GPRError::InvalidOptimizerBounds);
        }
        if (!theta.allFinite()) {
            return boost::outcome_v2::failure(GPRError::InvalidTheta);
        }
        return AnalyticKernel(std::move(theta), std::move(bounds), std::move(value), std::move(hyper),
                              std::move(input), std::move(diagonal));
    }
    Eigen::Index parameter_count() const { return theta_.size(); }
    Vector theta() const { return theta_; }
    Bounds bounds() const { return bounds_; }
    GPRResult<void> set_theta(Point theta) {
        if (theta.size() != theta_.size()) {
            return boost::outcome_v2::failure(GPRError::InvalidThetaSize);
        }
        if (!theta.allFinite()) {
            return boost::outcome_v2::failure(GPRError::InvalidTheta);
        }
        theta_ = theta;
        return boost::outcome_v2::success();
    }
    double value_unchecked(Point x, Point y, bool same) const { return value_(x, y, same, theta_); }
    void gradient_unchecked(Point x, Point y, bool same, Gradient out) const { hyper_(x, y, same, theta_, out); }
    void input_gradient_unchecked(Point x, Point y, Gradient out) const { input_(x, y, theta_, out); }
    void diagonal_gradient_unchecked(Point x, Gradient out) const { diagonal_(x, theta_, out); }

private:
    AnalyticKernel(Vector theta, Bounds bounds, Value value, HyperGradient hyper, InputGradient input,
                   DiagonalGradient diagonal)
        : theta_(std::move(theta)), bounds_(std::move(bounds)), value_(std::move(value)),
          hyper_(std::move(hyper)), input_(std::move(input)), diagonal_(std::move(diagonal)) {}

    Vector theta_;
    Bounds bounds_;
    Value value_;
    HyperGradient hyper_;
    InputGradient input_;
    DiagonalGradient diagonal_;
};

template <class V, class H, class I, class D>
GPRResult<AnalyticKernel<V, H, I, D>> make_analytic_kernel(Vector theta, Bounds bounds, V value, H hyper,
                                                          I input, D diagonal) {
    return AnalyticKernel<V, H, I, D>::create(std::move(theta), std::move(bounds), std::move(value),
                                              std::move(hyper), std::move(input), std::move(diagonal));
}

namespace detail {

template <class K> Matrix covariance_unchecked(const K &kernel, const Matrix &x, bool same = true) {
    Matrix result(x.rows(), x.rows());
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index col = 0; col <= row; ++col) {
            const double value = kernel.value_unchecked(x.row(row).transpose(), x.row(col).transpose(), same && row == col);
            result(row, col) = value;
            // NOLINTNEXTLINE(readability-suspicious-call-argument) -- Symmetric entry.
            result(col, row) = value;
        }
    }
    return result;
}

template <class K> Matrix covariance_unchecked(const K &kernel, const Matrix &x, const Matrix &y) {
    Matrix result(x.rows(), y.rows());
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index col = 0; col < y.rows(); ++col) {
            result(row, col) = kernel.value_unchecked(x.row(row).transpose(), y.row(col).transpose(), false);
        }
    }
    return result;
}

} // namespace detail

template <class K> GPRResult<Matrix> covariance(const K &kernel, const Matrix &x) {
    static_assert(IsKernel<K>::value, "K must satisfy the Kernel interface");
    BOOST_OUTCOME_TRY(detail::validate_kernel_input(kernel, x.cols(), 0));
    return detail::covariance_unchecked(kernel, x);
}

template <class K> GPRResult<Matrix> covariance(const K &kernel, const Matrix &x, const Matrix &y) {
    if (x.cols() != y.cols()) {
        return boost::outcome_v2::failure(GPRError::InvalidPredictionShape);
    }
    BOOST_OUTCOME_TRY(detail::validate_kernel_input(kernel, x.cols(), 0));
    return detail::covariance_unchecked(kernel, x, y);
}

} // namespace super_gpr
