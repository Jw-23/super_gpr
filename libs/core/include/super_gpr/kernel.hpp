#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <super_gpr/types.hpp>
#include <type_traits>
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
    explicit ParameterSet(std::vector<Parameter> parameters) : parameters_(std::move(parameters)) {
        for (const auto &parameter : parameters_) {
            if (parameter.lower != parameter.upper &&
                (!(parameter.value > 0.0) || !(parameter.lower > 0.0) || parameter.lower > parameter.upper)) {
                throw std::invalid_argument("Kernel parameters and bounds must be positive and ordered");
            }
            if (parameter.lower != parameter.upper) {
                ++size_;
            }
        }
    }

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

    void set_theta(Point theta) {
        if (theta.size() != size_) {
            throw std::invalid_argument("Wrong theta size");
        }
        Eigen::Index index = 0;
        for (auto &parameter : parameters_) {
            if (parameter.lower != parameter.upper) {
                parameter.value = std::exp(theta[index++]);
            }
        }
    }

protected:
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
    K, Void<decltype(std::declval<const K &>().value(std::declval<Point>(), std::declval<Point>(), true)),
            decltype(std::declval<const K &>().gradient(std::declval<Point>(), std::declval<Point>(), true,
                                                        std::declval<Gradient>())),
            decltype(std::declval<const K &>().input_gradient(std::declval<Point>(), std::declval<Point>(),
                                                              std::declval<Gradient>())),
            decltype(std::declval<const K &>().diagonal_gradient(std::declval<Point>(),
                                                                 std::declval<Gradient>())),
            decltype(std::declval<const K &>().theta()), decltype(std::declval<const K &>().bounds()),
            decltype(std::declval<const K &>().parameter_count()),
            decltype(std::declval<K &>().set_theta(std::declval<Point>()))>> : std::true_type {};

class ConstantKernel : public ParameterSet {
public:
    explicit ConstantKernel(Parameter constant = {}) : ParameterSet({constant}) {}
    double value(Point /*x*/, Point /*y*/, bool /*same*/) const { return parameter(0); }
    void gradient(Point /*x*/, Point /*y*/, bool /*same*/, Gradient out) const {
        Eigen::Index offset = 0;
        write_gradient(0, parameter(0), out, offset);
    }
    static void input_gradient(Point /*x*/, Point /*y*/, Gradient out) { out.setZero(); }
    static void diagonal_gradient(Point /*x*/, Gradient out) { out.setZero(); }
};

class WhiteKernel : public ParameterSet {
public:
    explicit WhiteKernel(Parameter noise = {}) : ParameterSet({noise}) {}
    double value(Point /*x*/, Point /*y*/, bool same) const { return same ? parameter(0) : 0.0; }
    void gradient(Point x, Point y, bool same, Gradient out) const {
        Eigen::Index offset = 0;
        write_gradient(0, value(x, y, same), out, offset);
    }
    static void input_gradient(Point /*x*/, Point /*y*/, Gradient out) { out.setZero(); }
    static void diagonal_gradient(Point /*x*/, Gradient out) { out.setZero(); }
};

// A single length scale is isotropic; multiple scales enable ARD.
class RadialKernel : public ParameterSet {
public:
    explicit RadialKernel(Parameter scale) : ParameterSet({scale}) {}
    explicit RadialKernel(std::vector<Parameter> scales) : ParameterSet(std::move(scales)) {
        if (stored_parameter_count() == 0) {
            throw std::invalid_argument("Length scales cannot be empty");
        }
    }
    static void diagonal_gradient(Point /*x*/, Gradient out) { out.setZero(); }

protected:
    double scale(Eigen::Index axis) const {
        return parameter(stored_parameter_count() == 1 ? 0 : static_cast<std::size_t>(axis));
    }
    double squared_distance(Point x, Point y) const {
        if (stored_parameter_count() != 1 && stored_parameter_count() != static_cast<std::size_t>(x.size())) {
            throw std::invalid_argument("ARD length scales must match the feature count");
        }
        double result = 0.0;
        for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
            const double delta = (x[axis] - y[axis]) / scale(axis);
            result += delta * delta;
        }
        return result;
    }
    void radial_gradient(Point x, Point y, double factor, Gradient out) const {
        Eigen::Index offset = 0;
        if (stored_parameter_count() == 1) {
            write_gradient(0, factor * squared_distance(x, y), out, offset);
        } else {
            for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
                const double delta = (x[axis] - y[axis]) / scale(axis);
                write_gradient(static_cast<std::size_t>(axis), factor * delta * delta, out, offset);
            }
        }
    }
    void radial_input_gradient(Point x, Point y, double factor, Gradient out) const {
        for (Eigen::Index axis = 0; axis < x.size(); ++axis) {
            out[axis] = -factor * (x[axis] - y[axis]) / (scale(axis) * scale(axis));
        }
    }
};

class RBF : public RadialKernel {
public:
    explicit RBF(Parameter scale = {}) : RadialKernel(scale) {}
    explicit RBF(std::vector<Parameter> scales) : RadialKernel(std::move(scales)) {}
    double value(Point x, Point y, bool /*same*/) const { return std::exp(-0.5 * squared_distance(x, y)); }
    void gradient(Point x, Point y, bool same, Gradient out) const {
        radial_gradient(x, y, value(x, y, same), out);
    }
    void input_gradient(Point x, Point y, Gradient out) const {
        radial_input_gradient(x, y, value(x, y, false), out);
    }
};

class Matern : public RadialKernel {
public:
    explicit Matern(Parameter scale = {}, double nu = 1.5) : RadialKernel(scale), nu_(nu) { check_nu(); }
    explicit Matern(std::vector<Parameter> scales, double nu = 1.5)
        : RadialKernel(std::move(scales)), nu_(nu) {
        check_nu();
    }
    double value(Point x, Point y, bool /*same*/) const {
        const double distance = std::sqrt(squared_distance(x, y));
        if (nu_ == 0.5) {
            return std::exp(-distance);
        }
        if (nu_ == 1.5) {
            const double r = std::sqrt(3.0) * distance;
            return (1.0 + r) * std::exp(-r);
        }
        if (nu_ == 2.5) {
            const double r = std::sqrt(5.0) * distance;
            return (1.0 + r + r * r / 3.0) * std::exp(-r);
        }
        return std::exp(-0.5 * distance * distance);
    }
    void gradient(Point x, Point y, bool /*same*/, Gradient out) const {
        radial_gradient(x, y, factor(x, y), out);
    }
    void input_gradient(Point x, Point y, Gradient out) const {
        radial_input_gradient(x, y, factor(x, y), out);
    }

private:
    void check_nu() const {
        if (nu_ != 0.5 && nu_ != 1.5 && nu_ != 2.5 && nu_ != std::numeric_limits<double>::infinity()) {
            throw std::invalid_argument("Matern supports nu = 0.5, 1.5, 2.5, or infinity");
        }
    }
    double factor(Point x, Point y) const {
        const double distance = std::sqrt(squared_distance(x, y));
        if (nu_ == 0.5) {
            return distance == 0.0 ? 0.0 : std::exp(-distance) / distance;
        }
        if (nu_ == 1.5) {
            return 3.0 * std::exp(-std::sqrt(3.0) * distance);
        }
        if (nu_ == 2.5) {
            const double r = std::sqrt(5.0) * distance;
            return (5.0 / 3.0) * (1.0 + r) * std::exp(-r);
        }
        return std::exp(-0.5 * distance * distance);
    }
    double nu_;
};

class RationalQuadratic : public ParameterSet {
public:
    explicit RationalQuadratic(Parameter scale = {}, Parameter alpha = {}) : ParameterSet({alpha, scale}) {}
    double value(Point x, Point y, bool /*same*/) const { return std::pow(base(x, y), -parameter(0)); }
    void gradient(Point x, Point y, bool same, Gradient out) const {
        const double b = base(x, y);
        const double k = value(x, y, same);
        Eigen::Index offset = 0;
        write_gradient(0, k * parameter(0) * (-std::log(b) + (b - 1.0) / b), out, offset);
        write_gradient(1, k * (x - y).squaredNorm() / (parameter(1) * parameter(1) * b), out, offset);
    }
    void input_gradient(Point x, Point y, Gradient out) const {
        out = -(x - y) * (value(x, y, false) / (parameter(1) * parameter(1) * base(x, y)));
    }
    static void diagonal_gradient(Point /*x*/, Gradient out) { out.setZero(); }

private:
    double base(Point x, Point y) const {
        return 1.0 + (x - y).squaredNorm() / (2.0 * parameter(0) * parameter(1) * parameter(1));
    }
};

class ExpSineSquared : public ParameterSet {
public:
    explicit ExpSineSquared(Parameter scale = {}, Parameter periodicity = {})
        : ParameterSet({scale, periodicity}) {}
    double value(Point x, Point y, bool /*same*/) const {
        const double sine = std::sin(angle(x, y));
        return std::exp(-2.0 * sine * sine / (parameter(0) * parameter(0)));
    }
    void gradient(Point x, Point y, bool same, Gradient out) const {
        const double a = angle(x, y);
        const double factor = 4.0 * value(x, y, same) / (parameter(0) * parameter(0));
        Eigen::Index offset = 0;
        write_gradient(0, factor * std::sin(a) * std::sin(a), out, offset);
        write_gradient(1, factor * a * std::sin(a) * std::cos(a), out, offset);
    }
    void input_gradient(Point x, Point y, Gradient out) const {
        const double distance = (x - y).norm();
        if (distance == 0.0) {
            out.setZero();
            return;
        }
        const double a = angle(x, y);
        out = (x - y) * (-4.0 * value(x, y, false) * std::sin(a) * std::cos(a) * pi() /
                         (parameter(0) * parameter(0) * parameter(1) * distance));
    }
    static void diagonal_gradient(Point /*x*/, Gradient out) { out.setZero(); }

private:
    static double pi() { return std::acos(-1.0); }
    double angle(Point x, Point y) const { return pi() * (x - y).norm() / parameter(1); }
};

class DotProduct : public ParameterSet {
public:
    explicit DotProduct(Parameter sigma = {}) : ParameterSet({sigma}) {}
    double value(Point x, Point y, bool /*same*/) const { return x.dot(y) + parameter(0) * parameter(0); }
    void gradient(Point /*x*/, Point /*y*/, bool /*same*/, Gradient out) const {
        Eigen::Index offset = 0;
        write_gradient(0, 2.0 * parameter(0) * parameter(0), out, offset);
    }
    static void input_gradient(Point /*x*/, Point y, Gradient out) { out = y; }
    static void diagonal_gradient(Point x, Gradient out) { out = 2.0 * x; }
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
template <class K1, class K2, class Operation> class BinaryKernel {
    static_assert(IsKernel<K1>::value && IsKernel<K2>::value, "Operands must satisfy the Kernel interface");

public:
    BinaryKernel(K1 left, K2 right) : left_(std::move(left)), right_(std::move(right)) {}
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
    void set_theta(Point theta) {
        if (theta.size() != parameter_count()) {
            throw std::invalid_argument("Wrong theta size");
        }
        left_.set_theta(theta.head(left_.parameter_count()));
        right_.set_theta(theta.tail(right_.parameter_count()));
    }
    double value(Point x, Point y, bool same) const {
        return Operation::apply(left_.value(x, y, same), right_.value(x, y, same));
    }
    void gradient(Point x, Point y, bool same, Gradient out) const {
        auto left = out.head(left_.parameter_count());
        auto right = out.tail(right_.parameter_count());
        left_.gradient(x, y, same, left);
        right_.gradient(x, y, same, right);
        const double a = left_.value(x, y, same);
        const double b = right_.value(x, y, same);
        left *= Operation::left_factor(a, b);
        right *= Operation::right_factor(a, b);
    }
    void input_gradient(Point x, Point y, Gradient out) const {
        Vector scratch(x.size());
        left_.input_gradient(x, y, out);
        right_.input_gradient(x, y, scratch);
        const double a = left_.value(x, y, false);
        const double b = right_.value(x, y, false);
        out = Operation::left_factor(a, b) * out + Operation::right_factor(a, b) * scratch;
    }
    void diagonal_gradient(Point x, Gradient out) const {
        Vector scratch(x.size());
        left_.diagonal_gradient(x, out);
        right_.diagonal_gradient(x, scratch);
        const double a = left_.value(x, x, true);
        const double b = right_.value(x, x, true);
        out = Operation::left_factor(a, b) * out + Operation::right_factor(a, b) * scratch;
    }

private:
    K1 left_;
    K2 right_;
};

template <class K1, class K2> class AddKernel final : public BinaryKernel<K1, K2, AddOperation> {
public:
    AddKernel(K1 left, K2 right) : BinaryKernel<K1, K2, AddOperation>(std::move(left), std::move(right)) {}
};

template <class K1, class K2> class Multiple final : public BinaryKernel<K1, K2, MultiplyOperation> {
public:
    Multiple(K1 left, K2 right)
        : BinaryKernel<K1, K2, MultiplyOperation>(std::move(left), std::move(right)) {}
};

template <class K1, class K2> using Add = AddKernel<K1, K2>;
template <class K1, class K2> using Multiply = Multiple<K1, K2>;

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
Multiple<ConstantKernel, K> operator*(double factor, K kernel) {
    return {ConstantKernel(Parameter::fixed(factor)), std::move(kernel)};
}
template <class K, std::enable_if_t<IsKernel<K>::value, int> = 0>
Multiple<K, ConstantKernel> operator*(K kernel, double factor) {
    return {std::move(kernel), ConstantKernel(Parameter::fixed(factor))};
}

class BoxedKernel {
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
        virtual void set_theta(Point theta) = 0;
        virtual double value(Point x, Point y, bool same) const = 0;
        virtual void gradient(Point x, Point y, bool same, Gradient out) const = 0;
        virtual void input_gradient(Point x, Point y, Gradient out) const = 0;
        virtual void diagonal_gradient(Point x, Gradient out) const = 0;
    };
    template <class K> struct Model final : Interface {
        explicit Model(K kernel) : kernel(std::move(kernel)) {}
        std::unique_ptr<Interface> clone() const override { return std::make_unique<Model>(kernel); }
        Eigen::Index parameter_count() const override { return kernel.parameter_count(); }
        Vector theta() const override { return kernel.theta(); }
        Bounds bounds() const override { return kernel.bounds(); }
        void set_theta(Point theta) override { kernel.set_theta(theta); }
        double value(Point x, Point y, bool same) const override { return kernel.value(x, y, same); }
        void gradient(Point x, Point y, bool same, Gradient out) const override {
            kernel.gradient(x, y, same, out);
        }
        void input_gradient(Point x, Point y, Gradient out) const override {
            kernel.input_gradient(x, y, out);
        }
        void diagonal_gradient(Point x, Gradient out) const override { kernel.diagonal_gradient(x, out); }
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
    void set_theta(Point theta) { kernel_->set_theta(theta); }
    double value(Point x, Point y, bool same) const { return kernel_->value(x, y, same); }
    void gradient(Point x, Point y, bool same, Gradient out) const { kernel_->gradient(x, y, same, out); }
    void input_gradient(Point x, Point y, Gradient out) const { kernel_->input_gradient(x, y, out); }
    void diagonal_gradient(Point x, Gradient out) const { kernel_->diagonal_gradient(x, out); }

private:
    std::unique_ptr<Interface> kernel_;
};

// User callbacks receive theta in log coordinates. All derivatives are supplied analytically.
template <class Value, class HyperGradient, class InputGradient, class DiagonalGradient>
class AnalyticKernel {
public:
    AnalyticKernel(Vector theta, Bounds bounds, Value value, HyperGradient hyper, InputGradient input,
                   DiagonalGradient diagonal)
        : theta_(std::move(theta)), bounds_(std::move(bounds)), value_(std::move(value)),
          hyper_(std::move(hyper)), input_(std::move(input)), diagonal_(std::move(diagonal)) {}
    Eigen::Index parameter_count() const { return theta_.size(); }
    Vector theta() const { return theta_; }
    Bounds bounds() const { return bounds_; }
    void set_theta(Point theta) { theta_ = theta; }
    double value(Point x, Point y, bool same) const { return value_(x, y, same, theta_); }
    void gradient(Point x, Point y, bool same, Gradient out) const { hyper_(x, y, same, theta_, out); }
    void input_gradient(Point x, Point y, Gradient out) const { input_(x, y, theta_, out); }
    void diagonal_gradient(Point x, Gradient out) const { diagonal_(x, theta_, out); }

private:
    Vector theta_;
    Bounds bounds_;
    Value value_;
    HyperGradient hyper_;
    InputGradient input_;
    DiagonalGradient diagonal_;
};

template <class V, class H, class I, class D>
AnalyticKernel<V, H, I, D> make_analytic_kernel(Vector theta, Bounds bounds, V value, H hyper, I input,
                                                D diagonal) {
    return {std::move(theta), std::move(bounds), std::move(value),
            std::move(hyper), std::move(input),  std::move(diagonal)};
}

template <class K> Matrix covariance(const K &kernel, const Matrix &x) {
    static_assert(IsKernel<K>::value, "K must satisfy the Kernel interface");
    Matrix result(x.rows(), x.rows());
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index col = 0; col <= row; ++col) {
            const double value = kernel.value(x.row(row).transpose(), x.row(col).transpose(), row == col);
            result(row, col) = value;
            // NOLINTNEXTLINE(readability-suspicious-call-argument) -- Symmetric entry.
            result(col, row) = value;
        }
    }
    return result;
}

template <class K> Matrix covariance(const K &kernel, const Matrix &x, const Matrix &y) {
    if (x.cols() != y.cols()) {
        throw std::invalid_argument("Cross covariance requires matching feature counts");
    }
    Matrix result(x.rows(), y.rows());
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index col = 0; col < y.rows(); ++col) {
            result(row, col) = kernel.value(x.row(row).transpose(), y.row(col).transpose(), false);
        }
    }
    return result;
}

} // namespace super_gpr
