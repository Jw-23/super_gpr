"""Compare the C++ probe with scikit-learn using identical data and parameters."""

import subprocess
import sys
import warnings

import numpy as np
import sklearn
from numpy.testing import assert_allclose
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, ConstantKernel, DotProduct, ExpSineSquared, Matern,
    RationalQuadratic, WhiteKernel,
)


def main():
    output = subprocess.check_output([sys.argv[1]], text=True)
    actual = {}
    for line in output.splitlines():
        name, count, *values = line.split()
        assert len(values) == int(count), name
        actual[name] = np.array(values, dtype=float)
    checks = 0
    max_error = 0.0

    def compare(name, expected, atol=2e-10, rtol=2e-10):
        nonlocal checks, max_error
        expected = np.asarray(expected).ravel()
        assert_allclose(actual[name], expected, atol=atol, rtol=rtol, err_msg=name)
        if expected.size:
            max_error = max(max_error, float(np.max(np.abs(actual[name] - expected))))
        checks += 1

    x = np.array([[0, .1], [.4, -.2], [.9, .5], [1.3, .7], [1.8, -.1], [2.4, .4]])
    query = np.array([[.2, -.1], [.8, .4], [2., .2]])
    kernels = {
        "rbf": RBF(.7), "ard": RBF([.7, 1.3]),
        "matern05": Matern(.7, nu=.5), "matern15": Matern(.7, nu=1.5),
        "matern25": Matern([.7, 1.3], nu=2.5), "materninf": Matern(.7, nu=np.inf),
        "rq": RationalQuadratic(.8, 1.4), "periodic": ExpSineSquared(.8, 1.7),
        "dot": DotProduct(.6), "constant": ConstantKernel(1.2), "white": WhiteKernel(.2),
        "composite": ConstantKernel(1.7, "fixed") * (RBF() + DotProduct()) * (ConstantKernel() + WhiteKernel()),
        "fixed": ConstantKernel(2., "fixed") * RBF([.7, 1.3], [[.7, .7], [1e-5, 1e5]]),
    }
    for name, kernel in kernels.items():
        features = 1 if name == "periodic" else x.shape[1]
        inputs = x[:, :features]
        matrix, gradient = kernel(inputs, eval_gradient=True)
        # Equal numeric bounds fix individual ARD axes in super_gpr.
        # sklearn only supports fixing the entire ARD parameter group.
        if name == "fixed":
            gradient = gradient[:, :, 1:]
            theta = kernel.theta[1:]
        else:
            theta = kernel.theta
        compare(name + ".train", matrix)
        compare(name + ".cross", kernel(inputs, query[:, :features]))
        compare(name + ".theta", theta)
        compare(name + ".gradient", gradient)

    def model_case(name, kernel, inputs, targets, queries, **kwargs):
        model = GaussianProcessRegressor(kernel, **kwargs).fit(inputs, targets)
        mean, std = model.predict(queries, return_std=True)
        _, covariance = model.predict(queries, return_cov=True)
        compare(name + ".theta", model.kernel_.theta, atol=2e-5, rtol=2e-5)
        compare(name + ".lml", model.log_marginal_likelihood_value_, atol=2e-9)
        _, gradient = model.log_marginal_likelihood(model.kernel_.theta, eval_gradient=True)
        compare(name + ".gradient", gradient, atol=2e-5)
        tolerance = 2e-6 if name == "optimized" else 2e-10
        compare(name + ".mean", mean, atol=tolerance, rtol=tolerance)
        compare(name + ".std", std, atol=tolerance, rtol=tolerance)
        if covariance.ndim == 2:
            covariance = covariance[:, :, None]
        for index in range(covariance.shape[2]):
            compare(name + f".cov{index}", covariance[:, :, index], atol=tolerance, rtol=tolerance)

    kernel = ConstantKernel(1.2) * RBF([.7, 1.1]) + WhiteKernel(.03)
    y = np.array([1.2, .7, -.1, -.8, -.3, .4])
    model_case("raw", kernel, x, y, query, optimizer=None, alpha=.02)
    model_case("normalized", kernel, x, y, query, optimizer=None, alpha=.02, normalize_y=True)
    targets = np.column_stack([y, 3 * y + 7, np.full(6, 4.)])
    model_case("multi", kernel, x, targets, query, optimizer=None,
               alpha=np.linspace(.01, .04, 6), normalize_y=True)
    inputs = np.linspace(-2, 3, 12)[:, None]
    targets = np.sin(1.5 * inputs[:, 0]) + .1 * np.cos(8 * inputs[:, 0])

    def optimizer(objective, initial_theta, bounds):
        from scipy.optimize import minimize
        result = minimize(objective, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds,
                          options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 1000, "maxls": 40})
        assert result.success, result.message
        return result.x, result.fun

    kernel = ConstantKernel(1.3, (1e-3, 10.)) * RBF(.7, (.1, 5.)) + WhiteKernel(.02, (1e-5, .5))
    model_case("optimized", kernel, inputs, targets, inputs, optimizer=optimizer, alpha=1e-10)
    assert checks == len(actual), f"Unchecked records: {len(actual) - checks}"
    print(f"scikit-learn {sklearn.__version__}: {checks} comparisons passed; maximum absolute error {max_error:.3g}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        main()
