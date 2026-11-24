#include "problems.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <super_gpr/additive.hpp>

namespace {
using namespace super_gpr;
using namespace super_gpr_experiments;

enum class Mode { Independent, SharedIsotropic, SharedARD, Regularized };
struct Candidate {
    std::string name;
    Mode mode;
    double prior_scale = 0.0;
};
struct Scenario {
    std::string function;
    Eigen::Index dimension = 0;
    Eigen::Index train_count = 0;
    std::uint64_t seed = 0;
};
struct FitResult {
    Vector prediction;
    Vector parameters;
    double train_rmse = 0.0;
    double noise_std = 0.0;
    double effective_df = 0.0;
    double seconds = 0.0;
    double lml = 0.0;
    double objective = 0.0;
    int iterations = 0;
    int evaluations = 0;
    bool converged = true;
};

const std::vector<Candidate> &candidates() {
    static const std::vector<Candidate> choices{
        {"independent", Mode::Independent, 0.0}, {"shared_iso", Mode::SharedIsotropic, 0.0},
        {"shared_ard", Mode::SharedARD, 0.0},    {"map_0.5", Mode::Regularized, 0.5},
        {"map_1.0", Mode::Regularized, 1.0},     {"map_2.0", Mode::Regularized, 2.0}};
    return choices;
}

// A control with variable group shapes tests the stronger parameter-sharing assumption.
Vector heterogeneous_target(const Matrix &x) {
    const Eigen::Index groups = x.cols() / 4;
    Vector result = Vector::Zero(x.rows());
    for (Eigen::Index row = 0; row < x.rows(); ++row) {
        for (Eigen::Index group = 0; group < groups; ++group) {
            const auto start = group * 4;
            const double a = x(row, start) * (0.5 + 0.4 * static_cast<double>(group % 5));
            const double b = x(row, start + 1) * (0.5 + 0.4 * static_cast<double>((group + 1) % 5));
            const double c = x(row, start + 2) * (0.5 + 0.4 * static_cast<double>((group + 2) % 5));
            const double d = x(row, start + 3) * (0.5 + 0.4 * static_cast<double>((group + 3) % 5));
            const double weight = 0.6 + 0.2 * static_cast<double>(group % 5);
            result[row] +=
                weight * (std::sin(2.5 * a + 1.2 * b) + 0.6 * std::cos(2.0 * c - 0.7 * d) + 0.5 * a * c);
        }
    }
    return result / std::sqrt(static_cast<double>(groups));
}

Dataset dataset(const Scenario &scenario, Eigen::Index test_count) {
    auto data = make_dataset(scenario.dimension, scenario.train_count, test_count, scenario.seed,
                             scenario.function == "interaction");
    if (scenario.function == "heterogeneous") {
        data.train_y += heterogeneous_target(data.train_x) - additive_target(data.train_x);
        data.test_y = heterogeneous_target(data.test_x);
    }
    return data;
}

template <class K>
ExperimentResult<FitResult> fit_kernel(K kernel, const GPROptions &options, const Matrix &x, const Vector &y,
                     const Matrix &query) {
    GaussianProcessRegressor<K> model(std::move(kernel), options);
    const auto start = std::chrono::steady_clock::now();
    BOOST_OUTCOME_TRY(model.fit(x, y));
    FitResult result;
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    BOOST_OUTCOME_TRY(auto prediction, model.predict(query));
    BOOST_OUTCOME_TRY(auto training, model.predict(x));
    result.prediction = prediction.mean.col(0);
    result.train_rmse =
        std::sqrt((training.mean.col(0) - y).squaredNorm() / static_cast<double>(y.size()));
    result.parameters = model.kernel().theta().array().exp();
    const double noise = result.parameters[result.parameters.size() - 1];
    result.noise_std = std::sqrt(noise * (y.array() - y.mean()).square().mean());
    BOOST_OUTCOME_TRY(auto covariance, super_gpr::covariance(model.kernel(), x));
    covariance.diagonal().array() += options.alpha;
    detail::Cholesky factor;
    BOOST_OUTCOME_TRY(factor.compute(std::move(covariance)));
    const Eigen::MatrixXd inverse = std::move(factor).take_inverse_lower();
    result.effective_df = static_cast<double>(x.rows()) - (noise + options.alpha) * inverse.trace();
    BOOST_OUTCOME_TRY(auto lml, model.log_marginal_likelihood());
    BOOST_OUTCOME_TRY(auto objective, model.log_posterior());
    result.lml = lml;
    result.objective = objective;
    for (const auto &optimization : model.optimization_results()) {
        result.iterations += optimization.iterations;
        result.evaluations += optimization.evaluations;
        result.converged = result.converged && optimization.converged();
    }
    return result;
}

ExperimentResult<FitResult> fit(const Candidate &candidate, const Matrix &x, const Vector &y, const Matrix &query) {
    GPROptions options;
    options.normalize_y = true;
    options.optimizer.max_iterations = 500;
    options.optimizer.function_tolerance = 1e-8;
    const auto groups = feature_groups(x.cols());
    const auto count = static_cast<double>(groups.size());
    const Parameter scale{0.7, 0.1, 20.0};
    BOOST_OUTCOME_TRY(auto noise, WhiteKernel::create({0.001, 1e-8, 0.5}));
    if (candidate.mode == Mode::SharedIsotropic || candidate.mode == Mode::SharedARD) {
        const std::size_t dimensions = candidate.mode == Mode::SharedARD ? 4 : 1;
        BOOST_OUTCOME_TRY(auto amplitude, ConstantKernel::create({1.0 / count, 1e-5 / count, 100.0 / count}));
        BOOST_OUTCOME_TRY(auto radial, RBF::create(std::vector<Parameter>(dimensions, scale)));
        BOOST_OUTCOME_TRY(auto shared, make_shared_additive_kernel(groups, amplitude * radial));
        return fit_kernel(shared + noise, options, x, y, query);
    }
    BOOST_OUTCOME_TRY(auto amplitude, ConstantKernel::create({1.0 / count, 1e-5, 100.0}));
    BOOST_OUTCOME_TRY(auto radial, RBF::create(scale));
    BOOST_OUTCOME_TRY(auto additive, make_additive_kernel(groups, amplitude * radial));
    auto kernel = additive + noise;
    if (candidate.mode == Mode::Regularized) {
        options.prior.mean = kernel.theta();
        options.prior.standard_deviation = Vector::Constant(kernel.parameter_count(), candidate.prior_scale);
        options.prior.standard_deviation.tail(1).setConstant(1.5);
    }
    return fit_kernel(std::move(kernel), options, x, y, query);
}

void emit(const Scenario &scenario, const Candidate &candidate, const std::string &phase, int fold,
          bool selected, double validation_rmse, Eigen::Index fit_count, const Vector &actual,
          const FitResult &result) {
    const double error = (result.prediction - actual).squaredNorm();
    const double variance = (actual.array() - actual.mean()).square().sum();
    std::cout << scenario.function << ',' << scenario.dimension << ',' << scenario.train_count << ','
              << scenario.seed << ',' << candidate.name << ',' << phase << ',' << fold << ',' << selected
              << ',' << validation_rmse << ',' << fit_count << ',' << actual.size() << ','
              << result.parameters.size() << ',' << std::sqrt(error / static_cast<double>(actual.size()))
              << ',' << std::sqrt(error / variance) << ',' << 1.0 - error / variance << ','
              << result.train_rmse << ',' << result.noise_std << ',' << result.effective_df << ','
              << result.seconds << ',' << result.iterations << ',' << result.evaluations << ','
              << result.converged << ',' << result.lml << ',' << result.objective << ',';
    for (Eigen::Index index = 0; index < result.parameters.size(); ++index) {
        if (index != 0) {
            std::cout << ';';
        }
        std::cout << result.parameters[index];
    }
    std::cout << '\n' << std::flush;
}

struct FoldData {
    Matrix train_x;
    Vector train_y;
    Matrix validation_x;
    Vector validation_y;
};
std::vector<FoldData> folds(const Matrix &x, const Vector &y, std::uint64_t seed) {
    std::vector<Eigen::Index> order(static_cast<std::size_t>(x.rows()));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 random(seed + 424242);
    std::shuffle(order.begin(), order.end(), random);
    constexpr Eigen::Index fold_count = 3;
    std::vector<FoldData> result;
    result.reserve(static_cast<std::size_t>(fold_count));
    for (Eigen::Index fold = 0; fold < fold_count; ++fold) {
        Eigen::Index validation_count = 0;
        for (Eigen::Index index = fold; index < x.rows(); index += fold_count) {
            ++validation_count;
        }
        FoldData data{Matrix(x.rows() - validation_count, x.cols()), Vector(x.rows() - validation_count),
                      Matrix(validation_count, x.cols()), Vector(validation_count)};
        Eigen::Index train = 0;
        Eigen::Index validation = 0;
        for (Eigen::Index index = 0; index < x.rows(); ++index) {
            const auto row = order[static_cast<std::size_t>(index)];
            if (index % fold_count == fold) {
                data.validation_x.row(validation) = x.row(row);
                data.validation_y[validation++] = y[row];
            } else {
                data.train_x.row(train) = x.row(row);
                data.train_y[train++] = y[row];
            }
        }
        result.push_back(std::move(data));
    }
    return result;
}

ExperimentResult<std::vector<double>> cross_validate(const Scenario &scenario, const Matrix &x, const Vector &y) {
    const auto split = folds(x, y, scenario.seed);
    std::vector<double> scores;
    scores.reserve(candidates().size());
    for (const auto &candidate : candidates()) {
        double squared_error = 0.0;
        int fold_index = 0;
        for (const auto &fold : split) {
            BOOST_OUTCOME_TRY(auto result, fit(candidate, fold.train_x, fold.train_y, fold.validation_x));
            squared_error += (result.prediction - fold.validation_y).squaredNorm();
            emit(scenario, candidate, "cv", fold_index++, false, 0.0, fold.train_x.rows(), fold.validation_y,
                 result);
        }
        scores.push_back(std::sqrt(squared_error / static_cast<double>(y.size())));
    }
    return scores;
}

ExperimentResult<void> evaluate(const Scenario &scenario, Eigen::Index test_count) {
    const auto data = dataset(scenario, test_count);
    BOOST_OUTCOME_TRY(auto scores, cross_validate(scenario, data.train_x, data.train_y));
    const auto best = static_cast<std::size_t>(
        std::distance(scores.begin(), std::min_element(scores.begin(), scores.end())));
    // The winner is locked using training folds before any test predictions are evaluated.
    for (std::size_t index = 0; index < candidates().size(); ++index) {
        const auto &candidate = candidates()[index];
        BOOST_OUTCOME_TRY(auto result, fit(candidate, data.train_x, data.train_y, data.test_x));
        emit(scenario, candidate, "test", -1, index == best, scores[index], data.train_x.rows(), data.test_y,
             result);
    }
    std::cerr << scenario.function << " d=" << scenario.dimension << " n=" << scenario.train_count
              << " seed=" << scenario.seed << " selected=" << candidates()[best].name
              << " cv_rmse=" << scores[best] << '\n';
    return boost::outcome_v2::success();
}
ExperimentResult<void> run(int argc, char **argv) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) -- argc guards argv access.
    const bool quick = argc == 2 && std::string(argv[1]) == "--quick";
    if (argc != 1 && !quick) {
        return boost::outcome_v2::failure(ExperimentError("Usage: additive_improvement [--quick]"));
    }
    std::cout << std::setprecision(12)
              << "function,dimension,train_count,seed,candidate,phase,fold,selected,validation_rmse,fit_"
                 "count,query_count,parameters,rmse,nrmse,r2,train_rmse,noise_std,effective_df,fit_seconds,"
                 "iterations,evaluations,converged,lml,objective,physical_parameters\n";
    const std::vector<std::uint64_t> seeds =
        quick ? std::vector<std::uint64_t>{101} : std::vector<std::uint64_t>{1001, 2003, 3007};
    const std::vector<Eigen::Index> dimensions =
        quick ? std::vector<Eigen::Index>{20, 80} : std::vector<Eigen::Index>{4, 20, 80};
    const std::vector<Eigen::Index> counts =
        quick ? std::vector<Eigen::Index>{64} : std::vector<Eigen::Index>{128, 256};
    const Eigen::Index test_count = quick ? 256 : 1024;
    for (auto dimension : dimensions) {
        for (auto count : counts) {
            for (auto seed : seeds) {
                BOOST_OUTCOME_TRY(evaluate({"additive", dimension, count, seed}, test_count));
            }
        }
    }
    if (!quick) {
        for (auto count : counts) {
            for (auto seed : seeds) {
                BOOST_OUTCOME_TRY(evaluate({"heterogeneous", 80, count, seed}, test_count));
            }
        }
        for (auto seed : seeds) {
            BOOST_OUTCOME_TRY(evaluate({"interaction", 80, 256, seed}, test_count));
        }
    }
    return boost::outcome_v2::success();
}
} // namespace

int main(int argc, char **argv) {
    return super_gpr_experiments::report_experiment(run(argc, argv));
}
