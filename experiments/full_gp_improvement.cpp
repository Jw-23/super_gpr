#include "problems.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {
using namespace super_gpr;
using namespace super_gpr_experiments;

struct Candidate {
    std::string name;
    bool multistart = false;
    bool quadratic = false;
    bool warped = false;
};
const std::vector<Candidate> &candidates() {
    static const std::vector<Candidate> result{{"baseline", false, false, false},
                                               {"multistart", true, false, false},
                                               {"quadratic", true, true, false},
                                               {"asinh_matern", true, false, true},
                                               {"asinh_quadratic", true, true, true}};
    return result;
}

struct Observations {
    Matrix x;
    Vector y;
};
Observations read_observations(std::istream &source, Eigen::Index count, Eigen::Index dimension) {
    Observations result{Matrix(count, dimension), Vector(count)};
    for (Eigen::Index row = 0; row < count; ++row) {
        for (Eigen::Index axis = 0; axis < dimension; ++axis) {
            source >> result.x(row, axis);
        }
        source >> result.y[row];
    }
    return result;
}
std::uint64_t checksum(const Observations &data) {
    std::uint64_t result = 14695981039346656037ULL;
    const auto mix = [&result](double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "Checksums require 64-bit doubles");
        std::memcpy(&bits, &value, sizeof(bits));
        result = (result ^ bits) * 1099511628211ULL;
    };
    for (Eigen::Index row = 0; row < data.x.rows(); ++row) {
        for (Eigen::Index axis = 0; axis < data.x.cols(); ++axis) {
            mix(data.x(row, axis));
        }
        mix(data.y[row]);
    }
    return result;
}

struct Warp {
    double center = 0.0;
    double scale = 1.0;
    bool enabled = false;
    explicit Warp(const Vector &y, bool use_warp) : enabled(use_warp) {
        if (enabled) {
            std::vector<double> ordered(y.begin(), y.end());
            std::sort(ordered.begin(), ordered.end());
            center = ordered[ordered.size() / 2];
            scale = ordered[3 * ordered.size() / 4] - ordered[ordered.size() / 4];
            if (scale <= 0.0) {
                scale = std::sqrt((y.array() - y.mean()).square().mean());
            }
            if (scale == 0.0) {
                scale = 1.0;
            }
        }
    }
    Vector forward(const Vector &y) const {
        if (!enabled) {
            return y;
        }
        return ((y.array() - center) / scale).unaryExpr([](double value) { return std::asinh(value); });
    }
    Vector mean(const Prediction &prediction) const {
        if (!enabled) {
            return prediction.mean.col(0);
        }
        // Integrate the inverse transform under the Gaussian predictive distribution.
        return center + scale * (0.5 * prediction.standard_deviation.col(0).array().square()).exp() *
                            prediction.mean.col(0).array().sinh();
    }
};

struct Fit {
    Vector prediction;
    Vector parameters;
    double seconds = 0.0;
    double lml = -std::numeric_limits<double>::infinity();
    double train_rmse = 0.0;
    double warp_center = 0.0;
    double warp_scale = 1.0;
    int iterations = 0;
    int evaluations = 0;
    int starts = 0;
    int converged_starts = 0;
    bool best_converged = false;
};

template <class K>
Fit fit_kernel(const K &kernel, const Observations &training, const Matrix &query, const Warp &warp,
               bool multistart) {
    GPROptions options;
    options.normalize_y = true;
    options.optimizer.max_iterations = 500;
    options.optimizer.function_tolerance = 1e-8;
    const Vector target = warp.forward(training.y);
    Fit result;
    const auto begin = std::chrono::steady_clock::now();
    const int start_count = multistart ? 3 : 1;
    for (int start = 0; start < start_count; ++start) {
        auto initial = kernel;
        auto theta = initial.theta();
        if (start > 0) {
            theta[1] = std::log(static_cast<double>(start) *
                                std::sqrt(static_cast<double>(training.x.cols()) / 3.0));
            theta[theta.size() - 1] = std::log(start == 1 ? 0.05 : 0.2);
            initial.set_theta(theta);
        }
        GaussianProcessRegressor<K> model(initial, options);
        model.fit(training.x, target).value();
        const auto &optimization = model.optimization_results().front();
        result.iterations += optimization.iterations;
        result.evaluations += optimization.evaluations;
        result.converged_starts += static_cast<int>(optimization.converged());
        ++result.starts;
        const double lml = model.log_marginal_likelihood().value();
        if (lml > result.lml) {
            result.lml = lml;
            result.parameters = model.kernel().theta().array().exp();
            result.best_converged = optimization.converged();
            result.prediction = warp.mean(model.predict(query).value());
            const Vector train_error = warp.mean(model.predict(training.x).value()) - training.y;
            result.train_rmse = std::sqrt(train_error.squaredNorm() / static_cast<double>(target.size()));
        }
    }
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    result.warp_center = warp.center;
    result.warp_scale = warp.scale;
    if (!result.prediction.allFinite()) {
        throw std::runtime_error("Non-finite back-transformed prediction");
    }
    return result;
}

Fit fit(const Candidate &candidate, const Observations &training, const Matrix &query) {
    const Warp warp(training.y, candidate.warped);
    const auto stationary = ConstantKernel({1.0, 1e-5, 100.0}) * Matern({1.0, 0.1, 100.0}, 2.5);
    const WhiteKernel noise({0.001, 1e-8, 0.5});
    if (candidate.quadratic) {
        const double dimension_scale = static_cast<double>(training.x.cols()) / 3.0;
        const auto dot = (1.0 / dimension_scale) * DotProduct(Parameter::fixed(std::sqrt(dimension_scale)));
        const auto kernel = stationary + ConstantKernel({1.0, 1e-5, 100.0}) * dot * dot + noise;
        return fit_kernel(kernel, training, query, warp, candidate.multistart);
    }
    return fit_kernel(stationary + noise, training, query, warp, candidate.multistart);
}

struct Fold {
    Observations training;
    Observations validation;
};
std::vector<Fold> folds(const Observations &data, std::uint64_t seed) {
    std::vector<Eigen::Index> order(static_cast<std::size_t>(data.x.rows()));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 random(seed + 424242);
    std::shuffle(order.begin(), order.end(), random);
    std::vector<Fold> result;
    for (Eigen::Index fold = 0; fold < 3; ++fold) {
        const Eigen::Index validation_count = (data.x.rows() + 2 - fold) / 3;
        const Eigen::Index training_count = data.x.rows() - validation_count;
        Fold split{{Matrix(training_count, data.x.cols()), Vector(training_count)},
                   {Matrix(validation_count, data.x.cols()), Vector(validation_count)}};
        Eigen::Index train = 0;
        Eigen::Index valid = 0;
        for (Eigen::Index index = 0; index < data.x.rows(); ++index) {
            const auto row = order[static_cast<std::size_t>(index)];
            auto &block = index % 3 == fold ? split.validation : split.training;
            auto &position = index % 3 == fold ? valid : train;
            block.x.row(position) = data.x.row(row);
            block.y[position++] = data.y[row];
        }
        result.push_back(std::move(split));
    }
    return result;
}

struct Scenario {
    std::string function;
    int instance = 0;
    std::uint64_t seed = 0;
    Eigen::Index count = 0;
};
void emit(const Scenario &scenario, const Candidate &candidate, const std::string &phase, int fold,
          bool selected, double cv_nrmse, const Observations &training, const Observations &query,
          const Fit &result, const Observations &local = {}) {
    const Vector error = result.prediction.head(query.y.size()) - query.y;
    const double variance = (query.y.array() - query.y.mean()).square().mean();
    const double mse = error.squaredNorm() / static_cast<double>(error.size());
    std::cout << scenario.function << ',' << scenario.instance << ',' << scenario.seed << ','
              << training.x.cols() << ',' << scenario.count << ',' << candidate.name << ',' << phase << ','
              << fold << ',' << selected << ',' << cv_nrmse << ',' << training.x.rows() << ','
              << query.x.rows() << ',' << std::sqrt(mse) << ',' << std::sqrt(mse / variance) << ','
              << 1.0 - mse / variance << ',' << result.train_rmse << ',' << result.seconds << ','
              << result.lml << ',' << result.iterations << ',' << result.evaluations << ',' << result.starts
              << ',' << result.converged_starts << ',' << result.best_converged << ',' << result.warp_center
              << ',' << result.warp_scale << ',' << checksum(training) << ',' << checksum(query) << ',';
    if (local.y.size() > 0) {
        const Vector local_error = result.prediction.tail(local.y.size()) - local.y;
        const double local_mse = local_error.squaredNorm() / static_cast<double>(local.y.size());
        const double local_variance = (local.y.array() - local.y.mean()).square().mean();
        std::cout << std::sqrt(local_mse) << ',' << std::sqrt(local_mse / variance) << ','
                  << 1.0 - local_mse / local_variance << ',' << checksum(local);
    } else {
        std::cout << ",,,";
    }
    std::cout << ',';
    for (Eigen::Index index = 0; index < result.parameters.size(); ++index) {
        if (index != 0) {
            std::cout << ';';
        }
        std::cout << result.parameters[index];
    }
    std::cout << '\n' << std::flush;
}

void evaluate(const Scenario &scenario, const Observations &training, const Observations &testing,
              const Observations &local) {
    const auto split = folds(training, scenario.seed);
    const double train_variance = (training.y.array() - training.y.mean()).square().mean();
    std::vector<double> scores;
    for (const auto &candidate : candidates()) {
        double error = 0.0;
        int fold_index = 0;
        for (const auto &fold : split) {
            const auto result = fit(candidate, fold.training, fold.validation.x);
            error += (result.prediction - fold.validation.y).squaredNorm();
            emit(scenario, candidate, "cv", fold_index++, false, 0.0, fold.training, fold.validation, result);
        }
        scores.push_back(std::sqrt(error / static_cast<double>(training.y.size()) / train_variance));
    }
    const auto best = static_cast<std::size_t>(
        std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
    std::cerr << scenario.function << " instance=" << scenario.instance
              << " selected=" << candidates()[best].name << " cv_nrmse=" << scores[best] << '\n';
    // Lock the candidate using training folds before querying either test set.
    Matrix query(testing.x.rows() + local.x.rows(), training.x.cols());
    query << testing.x, local.x;
    for (std::size_t index = 0; index < candidates().size(); ++index) {
        const auto &candidate = candidates()[index];
        const auto result = fit(candidate, training, query);
        emit(scenario, candidate, "test", -1, index == best, scores[index], training, testing, result, local);
        const double variance = (testing.y.array() - testing.y.mean()).square().sum();
        const double error = (result.prediction.head(testing.y.size()) - testing.y).squaredNorm();
        std::cerr << "Finished " << candidate.name << " R2=" << 1.0 - error / variance
                  << " seconds=" << result.seconds << '\n';
    }
}
} // namespace

int main(int argc, char **argv) try {
    Eigen::setNbThreads(1);
    if (argc != 2) {
        throw std::invalid_argument("Usage: full_gp_improvement BBOB_DATA_DIRECTORY");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- argc guards argv access.
    const std::string directory(argv[1]);
    std::ifstream manifest(directory + "/cases.txt");
    Scenario scenario;
    if (!(manifest >> scenario.seed >> scenario.instance)) {
        throw std::runtime_error("Cannot read BBOB manifest");
    }
    std::cout << std::setprecision(12)
              << "function,instance,seed,dimension,train_count,candidate,phase,fold,selected,cv_nrmse,"
                 "fit_count,query_count,rmse,nrmse,r2,train_rmse,fit_predict_seconds,lml,iterations,"
                 "evaluations,starts,converged_starts,best_converged,warp_center,warp_scale,"
                 "training_checksum,query_checksum,local_rmse,local_nrmse,local_r2,local_checksum,"
                 "physical_parameters\n";
    std::string file;
    while (manifest >> scenario.function >> file) {
        std::ifstream source(std::string(directory).append("/").append(file));
        source.exceptions(std::ios::failbit | std::ios::badbit);
        Eigen::Index dimension = 0;
        Eigen::Index testing = 0;
        Eigen::Index local = 0;
        source >> dimension >> scenario.count >> testing >> local;
        const auto training = read_observations(source, scenario.count, dimension);
        const auto test_data = read_observations(source, testing, dimension);
        const auto local_data = read_observations(source, local, dimension);
        std::cerr << "Starting " << scenario.function << " instance=" << scenario.instance << '\n';
        evaluate(scenario, training, test_data, local_data);
    }
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
