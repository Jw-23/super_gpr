#include "problems.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
using namespace super_gpr;
using namespace super_gpr_experiments;

struct Options {
    bool quick = false;
    bool plain_matern = false;
    bool suite = false;
    std::string bbob_directory;
    std::vector<std::uint64_t> seeds{5003, 6007, 7001};
};
Options parse_options(int argc, char **argv) {
    Options result;
    for (int index = 1; index < argc; ++index) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- argc bounds argv access.
        const std::string argument(argv[index]);
        if (argument == "--quick") {
            result.quick = true;
        } else if (argument == "--suite") {
            result.suite = true;
        } else if (argument.compare(0, 12, "--bbob-data=") == 0) {
            result.bbob_directory = argument.substr(12);
        } else if (argument == "--plain-matern25") {
            result.plain_matern = true;
        } else if (argument.compare(0, 7, "--seed=") == 0) {
            const auto value = argument.substr(7);
            std::size_t end = 0;
            const auto seed = std::stoull(value, &end);
            if (end != value.size()) {
                throw std::invalid_argument("Invalid seed");
            }
            result.seeds = {seed};
        } else {
            throw std::invalid_argument("Usage: sample_learning_curve [--quick] [--seed=integer] "
                                        "[--plain-matern25] [--suite] [--bbob-data=directory]");
        }
    }
    if (result.quick && result.seeds.size() != 1) {
        result.seeds = {101};
    } else if (result.suite && result.seeds.size() != 1) {
        result.seeds = {11003, 12007, 13001};
    }
    return result;
}

void hash_value(double value, std::uint64_t &hash) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "Checksums require 64-bit doubles");
    std::memcpy(&bits, &value, sizeof(bits));
    hash = (hash ^ bits) * 1099511628211ULL;
}
std::uint64_t checksum(const Matrix &x, const Vector &y) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index axis = 0; axis < x.cols(); ++axis) {
            hash_value(x(row, axis), hash);
        }
        hash_value(y[row], hash);
    }
    return hash;
}

// Independent RNG streams keep the test set and existing noisy labels fixed as N grows.
Dataset nested_dataset(Eigen::Index dimension, Eigen::Index max_count, Eigen::Index test_count,
                       std::uint64_t seed, const FunctionCase &function) {
    std::mt19937_64 training(seed);
    std::mt19937_64 testing(seed + 1000000);
    std::mt19937_64 observation_noise(seed + 2000000);
    Dataset data;
    data.train_x = sample_inputs(max_count, dimension, training);
    data.test_x = sample_inputs(test_count, dimension, testing);
    data.train_y = function.target(data.train_x);
    data.test_y = function.target(data.test_x);
    std::normal_distribution<double> noise(0.0, 0.03);
    for (Eigen::Index row = 0; row < max_count; ++row) {
        data.train_y[row] += noise(observation_noise);
    }
    return data;
}

struct Diagnostic {
    Matrix x;
    Vector y;
    int instance = 0;
};

template <class K>
void evaluate(const std::string &name, std::uint64_t seed, const Dataset &data, Eigen::Index count,
              const K &kernel, std::uint64_t training_checksum, std::uint64_t test_checksum,
              const std::string &function, const Diagnostic &diagnostic = {}) {
    const Matrix x = data.train_x.topRows(count);
    const Vector y = data.train_y.head(count);
    GPROptions options;
    options.normalize_y = true;
    options.optimizer.max_iterations = 500;
    options.optimizer.function_tolerance = 1e-8;
    GaussianProcessRegressor<K> model(kernel, options);
    std::cerr << "Starting seed=" << seed << " n=" << count << " model=" << name << '\n';
    const auto start = std::chrono::steady_clock::now();
    model.fit(x, y).value();
    const double fit_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto predict_start = std::chrono::steady_clock::now();
    const auto prediction = model.predict(data.test_x).value();
    const double predict_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - predict_start).count();
    const Vector residual = prediction.mean.col(0) - data.test_y;
    const double error = residual.squaredNorm();
    const double variance = (data.test_y.array() - data.test_y.mean()).square().sum();
    const double rmse = std::sqrt(error / static_cast<double>(residual.size()));
    std::vector<double> absolute(static_cast<std::size_t>(residual.size()));
    for (Eigen::Index index = 0; index < residual.size(); ++index) {
        absolute[static_cast<std::size_t>(index)] = std::abs(residual[index]);
    }
    std::sort(absolute.begin(), absolute.end());
    const auto quantile_index =
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(absolute.size()))) - 1;
    const double train_rmse =
        std::sqrt((model.predict(x).value().mean.col(0) - y).squaredNorm() / static_cast<double>(count));
    const Vector parameters = model.kernel().theta().array().exp();
    const double noise_std =
        std::sqrt(parameters[parameters.size() - 1] * (y.array() - y.mean()).square().mean());
    int iterations = 0;
    int evaluations = 0;
    bool converged = true;
    for (const auto &result : model.optimization_results()) {
        iterations += result.iterations;
        evaluations += result.evaluations;
        converged = converged && result.converged();
    }
    std::cout << data.train_x.cols() << ',' << count << ',' << data.test_x.rows() << ',' << seed << ','
              << name << ',' << parameters.size() << ',' << rmse << ',' << std::sqrt(error / variance) << ','
              << 1.0 - error / variance << ',' << residual.cwiseAbs().mean() << ','
              << absolute[quantile_index] << ',' << absolute.back() << ',' << train_rmse << ',' << noise_std
              << ',' << fit_seconds << ',' << predict_seconds << ',' << iterations << ',' << evaluations
              << ',' << converged << ',' << model.log_marginal_likelihood().value() << ',' << training_checksum << ','
              << test_checksum << ',';
    for (Eigen::Index index = 0; index < parameters.size(); ++index) {
        if (index != 0) {
            std::cout << ';';
        }
        std::cout << parameters[index];
    }
    std::cout << ',' << function << ',' << diagnostic.instance;
    if (diagnostic.x.rows() > 0) {
        const Vector local_error = model.predict(diagnostic.x).value().mean.col(0) - diagnostic.y;
        const double local_mse = local_error.squaredNorm() / static_cast<double>(local_error.size());
        const double global_variance = variance / static_cast<double>(residual.size());
        const double local_variance = (diagnostic.y.array() - diagnostic.y.mean()).square().mean();
        const double constant_mse = (diagnostic.y.array() - y.mean()).square().mean();
        std::cout << ',' << std::sqrt(local_mse) << ',' << std::sqrt(local_mse / global_variance) << ','
                  << 1.0 - local_mse / local_variance << ',' << std::sqrt(constant_mse / global_variance)
                  << ',' << checksum(diagnostic.x, diagnostic.y);
    } else {
        std::cout << ",,,,,";
    }
    std::cout << '\n' << std::flush;
    std::cerr << "Finished seed=" << seed << " n=" << count << " model=" << name
              << " R2=" << 1.0 - error / variance << " RMSE=" << rmse << " seconds=" << fit_seconds
              << " converged=" << converged << '\n';
}

void evaluate_models(std::uint64_t seed, const Dataset &data, const std::vector<Eigen::Index> &counts,
                     bool plain_matern, bool suite, const std::string &function,
                     const Diagnostic &diagnostic = {}) {
    const auto training_checksum = checksum(data.train_x, data.train_y);
    const auto test_checksum = checksum(data.test_x, data.test_y);
    if (plain_matern || suite) {
        const auto kernel = ConstantKernel({1.0, 1e-5, 100.0}) * Matern({1.0, 0.1, 100.0}, 2.5) +
                            WhiteKernel({0.001, 1e-8, 0.5});
        for (auto count : counts) {
            evaluate("plain_matern25", seed, data, count, kernel, training_checksum, test_checksum, function,
                     diagnostic);
        }
        if (!suite) {
            return;
        }
    }
    const auto groups = feature_groups(data.train_x.cols());
    const auto group_count = static_cast<double>(groups.size());
    const WhiteKernel noise({0.001, 1e-8, 0.5});
    const auto independent = make_additive_kernel(groups, ConstantKernel({1.0 / group_count, 1e-5, 100.0}) *
                                                              RBF({0.7, 0.1, 20.0})) +
                             noise;
    const auto shared =
        make_shared_additive_kernel(
            groups, ConstantKernel({1.0 / group_count, 1e-5 / group_count, 100.0 / group_count}) *
                        RBF(std::vector<Parameter>(4, {0.7, 0.1, 20.0}))) +
        noise;
    for (auto count : counts) {
        evaluate("independent", seed, data, count, independent, training_checksum, test_checksum, function,
                 diagnostic);
        evaluate("shared_ard", seed, data, count, shared, training_checksum, test_checksum, function,
                 diagnostic);
    }
}
void run(std::uint64_t seed, bool quick, bool plain_matern, bool suite, const FunctionCase &function) {
    const Eigen::Index dimension = quick ? 20 : 80;
    const Eigen::Index test_count = quick ? 128 : 2048;
    auto counts =
        suite ? std::vector<Eigen::Index>{128, 256, 512} : std::vector<Eigen::Index>{128, 256, 512, 1024};
    if (quick) {
        counts = {32, 64};
    }
    const auto max_count = counts.back();
    const auto data = nested_dataset(dimension, max_count, test_count, seed, function);
    evaluate_models(seed, data, counts, plain_matern, suite, function.name);
}

void read_observations(std::istream &source, Eigen::Index count, Eigen::Index dimension, Matrix &x,
                       Vector &y) {
    x.resize(count, dimension);
    y.resize(count);
    for (Eigen::Index row = 0; row < count; ++row) {
        for (Eigen::Index axis = 0; axis < dimension; ++axis) {
            source >> x(row, axis);
        }
        source >> y[row];
    }
}

void run_bbob(const Options &options) {
    std::ifstream manifest(options.bbob_directory + "/cases.txt");
    manifest.exceptions(std::ios::badbit);
    std::uint64_t seed = 0;
    Diagnostic diagnostic;
    if (!(manifest >> seed >> diagnostic.instance)) {
        throw std::runtime_error("Cannot read BBOB manifest");
    }
    const auto counts =
        options.quick ? std::vector<Eigen::Index>{32, 64} : std::vector<Eigen::Index>{128, 256, 512};
    std::string function;
    std::string file;
    while (manifest >> function >> file) {
        std::ifstream source(options.bbob_directory + "/" + file);
        source.exceptions(std::ios::failbit | std::ios::badbit);
        Eigen::Index dimension = 0;
        Eigen::Index training = 0;
        Eigen::Index testing = 0;
        Eigen::Index local = 0;
        source >> dimension >> training >> testing >> local;
        Dataset data;
        read_observations(source, training, dimension, data.train_x, data.train_y);
        read_observations(source, testing, dimension, data.test_x, data.test_y);
        read_observations(source, local, dimension, diagnostic.x, diagnostic.y);
        if (counts.back() > training) {
            throw std::runtime_error("BBOB dataset has too few training observations");
        }
        std::cerr << "Function=" << function << " instance=" << diagnostic.instance << '\n';
        evaluate_models(seed, data, counts, false, true, function, diagnostic);
    }
}
} // namespace

int main(int argc, char **argv) try {
    Eigen::setNbThreads(1);
    const auto options = parse_options(argc, argv);
    std::cout << std::setprecision(12)
              << "dimension,train_count,test_count,seed,model,parameters,rmse,nrmse,r2,mae,p95_absolute_"
                 "error,max_absolute_error,train_rmse,noise_std,fit_seconds,predict_seconds,iterations,"
                 "evaluations,converged,lml,training_pool_checksum,test_checksum,physical_parameters,"
                 "function,instance,local_rmse,local_nrmse,local_r2,constant_local_nrmse,local_checksum\n";
    if (!options.bbob_directory.empty()) {
        run_bbob(options);
        return 0;
    }
    const auto functions =
        options.suite ? function_suite() : std::vector<FunctionCase>{{"original", additive_target}};
    for (auto seed : options.seeds) {
        for (const auto &function : functions) {
            std::cerr << "Function=" << function.name << '\n';
            run(seed, options.quick, options.plain_matern, options.suite, function);
        }
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
