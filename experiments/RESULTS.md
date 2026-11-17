# Prediction results

72 fitted models on 24 paired datasets.

Values are mean ± sample standard deviation across seeds; higher R² is better.

## additive

| Dimension | Training samples | Additive GP R² | Full RBF GP R² | Full ARD GP R² |
| --- | --- | --- | --- | --- |
| 4 | 128 | 0.985 ± 0.004 | 0.985 ± 0.004 | 0.993 ± 0.001 |
| 4 | 256 | 0.995 ± 0.001 | 0.995 ± 0.001 | 0.997 ± 0.000 |
| 20 | 128 | 0.698 ± 0.045 | 0.438 ± 0.024 | 0.438 ± 0.075 |
| 20 | 256 | 0.951 ± 0.004 | 0.511 ± 0.013 | 0.597 ± 0.040 |
| 80 | 128 | 0.183 ± 0.086 | 0.263 ± 0.047 | -0.170 ± 0.211 |
| 80 | 256 | 0.444 ± 0.108 | 0.381 ± 0.039 | 0.117 ± 0.029 |

## interaction

| Dimension | Training samples | Additive GP R² | Full RBF GP R² | Full ARD GP R² |
| --- | --- | --- | --- | --- |
| 20 | 256 | 0.425 ± 0.083 | 0.644 ± 0.065 | 0.588 ± 0.095 |
| 80 | 256 | 0.181 ± 0.114 | 0.352 ± 0.084 | -0.032 ± 0.067 |

## NRMSE and optimizer diagnostics

NRMSE = RMSE / test-target population standard deviation; lower is better.

| Function | Dimension | Samples | Model | NRMSE | Parameters | Converged | Mean fit seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| additive | 4 | 128 | additive | 0.121 | 3 | 3/3 | 0.016 |
| additive | 4 | 128 | full_ard | 0.081 | 6 | 3/3 | 0.009 |
| additive | 4 | 128 | full_rbf | 0.121 | 3 | 3/3 | 0.009 |
| additive | 4 | 256 | additive | 0.069 | 3 | 3/3 | 0.039 |
| additive | 4 | 256 | full_ard | 0.050 | 6 | 3/3 | 0.035 |
| additive | 4 | 256 | full_rbf | 0.069 | 3 | 3/3 | 0.029 |
| additive | 20 | 128 | additive | 0.549 | 11 | 3/3 | 0.157 |
| additive | 20 | 128 | full_ard | 0.748 | 22 | 3/3 | 0.066 |
| additive | 20 | 128 | full_rbf | 0.750 | 3 | 3/3 | 0.011 |
| additive | 20 | 256 | additive | 0.222 | 11 | 3/3 | 0.903 |
| additive | 20 | 256 | full_ard | 0.635 | 22 | 3/3 | 0.357 |
| additive | 20 | 256 | full_rbf | 0.699 | 3 | 3/3 | 0.050 |
| additive | 80 | 128 | additive | 0.903 | 41 | 3/3 | 1.799 |
| additive | 80 | 128 | full_ard | 1.079 | 82 | 3/3 | 0.562 |
| additive | 80 | 128 | full_rbf | 0.858 | 3 | 3/3 | 0.010 |
| additive | 80 | 256 | additive | 0.743 | 41 | 3/3 | 7.058 |
| additive | 80 | 256 | full_ard | 0.939 | 82 | 3/3 | 2.235 |
| additive | 80 | 256 | full_rbf | 0.787 | 3 | 3/3 | 0.053 |
| interaction | 20 | 256 | additive | 0.757 | 11 | 3/3 | 1.165 |
| interaction | 20 | 256 | full_ard | 0.639 | 22 | 3/3 | 0.152 |
| interaction | 20 | 256 | full_rbf | 0.595 | 3 | 3/3 | 0.049 |
| interaction | 80 | 256 | additive | 0.904 | 41 | 3/3 | 8.261 |
| interaction | 80 | 256 | full_ard | 1.016 | 82 | 3/3 | 2.583 |
| interaction | 80 | 256 | full_rbf | 0.804 | 3 | 3/3 | 0.047 |

These results use known variable groups and a single optimizer start per model.
A finite iteration budget and non-convex LML optimization do not guarantee a global optimum.
Fits that did not converge are retained and reported rather than omitted.
