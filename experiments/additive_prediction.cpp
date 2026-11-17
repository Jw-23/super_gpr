#include "problems.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <super_gpr/additive.hpp>

namespace {
using namespace super_gpr;
using namespace super_gpr_experiments;

template <class K>
void fit_and_report(const std::string &function, const std::string &family, const std::string &name,
                    const Dataset &data, std::uint64_t seed, const K &kernel) {
    GPROptions options;
    options.normalize_y = true;
    options.optimizer.max_iterations = 500;
    options.optimizer.function_tolerance = 1e-8;
    GaussianProcessRegressor<K> model(kernel, options);
    const auto started = std::chrono::steady_clock::now();
    model.fit(data.train_x, data.train_y);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const auto prediction = model.predict(data.test_x);
    const Vector residual = prediction.mean.col(0) - data.test_y;
    const double squared_error = residual.squaredNorm();
    const double total_variance = (data.test_y.array() - data.test_y.mean()).square().sum();
    const double rmse = std::sqrt(squared_error / static_cast<double>(data.test_y.size()));
    int iterations = 0;
    int evaluations = 0;
    bool converged = true;
    for (const auto &result : model.optimization_results()) {
        iterations += result.iterations;
        evaluations += result.evaluations;
        converged = converged && result.converged();
    }
    const double initial_lml = model.log_marginal_likelihood(kernel.theta()).value;
    std::cout << function << ',' << family << ',' << data.train_x.cols() << ',' << data.train_x.rows() << ','
              << data.test_x.rows() << ',' << seed << ',' << name << ',' << model.kernel().parameter_count()
              << ',' << rmse << ',' << std::sqrt(squared_error / total_variance) << ','
              << 1.0 - squared_error / total_variance << ',' << seconds << ',' << iterations << ','
              << evaluations << ',' << converged << ',' << initial_lml << ','
              << model.log_marginal_likelihood() << ',';
    const Vector parameters = model.kernel().theta().array().exp();
    for (Eigen::Index index = 0; index < parameters.size(); ++index) {
        if (index != 0) {
            std::cout << ';';
        }
        std::cout << parameters[index];
    }
    std::cout << '\n' << std::flush;
    std::cerr << function << ' ' << family << " d=" << data.train_x.cols() << " n=" << data.train_x.rows()
              << " seed=" << seed << ' ' << name << " R2=" << 1.0 - squared_error / total_variance
              << " seconds=" << seconds << " converged=" << converged << '\n';
}

template <class Factory>
void compare_models(Eigen::Index dimension, Eigen::Index train_count, Eigen::Index test_count,
                    std::uint64_t seed, bool interactions, const std::string &family, const Factory &radial) {
    const auto data = make_dataset(dimension, train_count, test_count, seed, interactions);
    const std::string function = interactions ? "interaction" : "additive";
    const auto groups = feature_groups(dimension);
    auto component = ConstantKernel({1.0 / static_cast<double>(groups.size()), 1e-5, 100.0}) *
                     radial(Parameter{0.7, 0.1, 20.0});
    auto additive = make_additive_kernel(groups, component) + WhiteKernel({0.001, 1e-8, 0.5});
    fit_and_report(function, family, "additive", data, seed, additive);
    const double initial_scale = 0.7 * std::sqrt(static_cast<double>(groups.size()));
    auto isotropic = ConstantKernel({1.0, 1e-5, 100.0}) * radial(Parameter{initial_scale, 0.1, 100.0}) +
                     WhiteKernel({0.001, 1e-8, 0.5});
    fit_and_report(function, family, "full_iso", data, seed, isotropic);
    auto ard =
        ConstantKernel({1.0, 1e-5, 100.0}) *
            radial(std::vector<Parameter>(static_cast<std::size_t>(dimension), {initial_scale, 0.1, 100.0})) +
        WhiteKernel({0.001, 1e-8, 0.5});
    fit_and_report(function, family, "full_ard", data, seed, ard);
}
struct ExperimentOptions {
    bool quick = false;
    std::string kernel = "rbf";
};

ExperimentOptions parse_options(int argc, char **argv) {
    ExperimentOptions options;
    for (int index = 1; index < argc; ++index) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- argc bounds argv access.
        const std::string argument(argv[index]);
        if (argument == "--quick") {
            options.quick = true;
        } else if (argument == "--kernel=rbf" || argument == "--kernel=matern15" ||
                   argument == "--kernel=matern25") {
            options.kernel = argument.substr(std::string("--kernel=").size());
        } else {
            throw std::invalid_argument(
                "Usage: additive_experiment [--quick] [--kernel=rbf|matern15|matern25]");
        }
    }
    return options;
}

template <class Factory> void run_experiment(const ExperimentOptions &options, const Factory &radial) {
    const bool quick = options.quick;
    std::cout
        << std::setprecision(12)
        << "function,kernel,dimension,train_count,test_count,seed,model,parameters,rmse,nrmse,r2,fit_seconds,"
           "iterations,evaluations,converged,initial_lml,lml,physical_parameters\n";
    const Eigen::Index test_count = quick ? 256 : 1024;
    const Eigen::Index control_train_count = quick ? 64 : 256;
    const std::vector<Eigen::Index> dimensions =
        quick ? std::vector<Eigen::Index>{4, 20} : std::vector<Eigen::Index>{4, 20, 80};
    const std::vector<Eigen::Index> train_counts =
        quick ? std::vector<Eigen::Index>{64} : std::vector<Eigen::Index>{128, 256};
    const std::vector<std::uint64_t> seeds =
        quick ? std::vector<std::uint64_t>{11} : std::vector<std::uint64_t>{11, 29, 47};
    for (auto dimension : dimensions) {
        for (auto train_count : train_counts) {
            for (auto seed : seeds) {
                compare_models(dimension, train_count, test_count, seed, false, options.kernel, radial);
            }
        }
    }
    for (auto dimension : dimensions) {
        if (dimension == 4) {
            continue;
        }
        for (auto seed : seeds) {
            compare_models(dimension, control_train_count, test_count, seed, true, options.kernel, radial);
        }
    }
}
} // namespace

int main(int argc, char **argv) try {
    const auto options = parse_options(argc, argv);
    if (options.kernel == "rbf") {
        run_experiment(options, [](auto scales) { return RBF(std::move(scales)); });
    } else {
        const double nu = options.kernel == "matern15" ? 1.5 : 2.5;
        run_experiment(options, [nu](auto scales) { return Matern(std::move(scales), nu); });
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
