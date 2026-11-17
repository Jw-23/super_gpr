# Prediction results

216 fitted models on 24 paired datasets across 3 kernel families.

Values are mean ± sample standard deviation across seeds. Kernel families share identical dataset keys.

## additive: R2 (higher is better)

| Dimension | Training samples | Model | RBF | Matérn 1.5 | Matérn 2.5 |
| --- | --- | --- | --- | --- | --- |
| 4 | 128 | Additive | 0.9852 ± 0.0038 | 0.9736 ± 0.0049 | 0.9845 ± 0.0022 |
| 4 | 128 | Full isotropic | 0.9852 ± 0.0038 | 0.9736 ± 0.0049 | 0.9845 ± 0.0022 |
| 4 | 128 | Full ARD | 0.9934 ± 0.0014 | 0.9905 ± 0.0048 | 0.9924 ± 0.0028 |
| 4 | 256 | Additive | 0.9953 ± 0.0006 | 0.9926 ± 0.0006 | 0.9943 ± 0.0004 |
| 4 | 256 | Full isotropic | 0.9953 ± 0.0006 | 0.9926 ± 0.0006 | 0.9943 ± 0.0004 |
| 4 | 256 | Full ARD | 0.9975 ± 0.0002 | 0.9960 ± 0.0002 | 0.9968 ± 0.0001 |
| 20 | 128 | Additive | 0.6976 ± 0.0450 | 0.6778 ± 0.0344 | 0.6834 ± 0.0417 |
| 20 | 128 | Full isotropic | 0.4378 ± 0.0244 | 0.4561 ± 0.0300 | 0.4495 ± 0.0310 |
| 20 | 128 | Full ARD | 0.4382 ± 0.0749 | 0.4526 ± 0.0954 | 0.4657 ± 0.0912 |
| 20 | 256 | Additive | 0.9505 ± 0.0043 | 0.8980 ± 0.0081 | 0.9249 ± 0.0087 |
| 20 | 256 | Full isotropic | 0.5109 ± 0.0135 | 0.5310 ± 0.0064 | 0.5276 ± 0.0068 |
| 20 | 256 | Full ARD | 0.5967 ± 0.0402 | 0.6152 ± 0.0382 | 0.6171 ± 0.0331 |
| 80 | 128 | Additive | 0.1825 ± 0.0862 | 0.2553 ± 0.0826 | 0.2271 ± 0.0984 |
| 80 | 128 | Full isotropic | 0.2633 ± 0.0467 | 0.2710 ± 0.0485 | 0.2699 ± 0.0481 |
| 80 | 128 | Full ARD | -0.1696 ± 0.2114 | 0.0495 ± 0.1163 | -0.0301 ± 0.0881 |
| 80 | 256 | Additive | 0.4439 ± 0.1079 | 0.4569 ± 0.1010 | 0.4505 ± 0.1060 |
| 80 | 256 | Full isotropic | 0.3807 ± 0.0387 | 0.3924 ± 0.0393 | 0.3900 ± 0.0393 |
| 80 | 256 | Full ARD | 0.1173 ± 0.0294 | 0.2203 ± 0.0356 | 0.1685 ± 0.0409 |

## interaction: R2 (higher is better)

| Dimension | Training samples | Model | RBF | Matérn 1.5 | Matérn 2.5 |
| --- | --- | --- | --- | --- | --- |
| 20 | 256 | Additive | 0.4252 ± 0.0826 | 0.4460 ± 0.1230 | 0.4251 ± 0.0905 |
| 20 | 256 | Full isotropic | 0.6437 ± 0.0645 | 0.6359 ± 0.0743 | 0.6415 ± 0.0702 |
| 20 | 256 | Full ARD | 0.5882 ± 0.0946 | 0.6003 ± 0.0920 | 0.5935 ± 0.0920 |
| 80 | 256 | Additive | 0.1806 ± 0.1143 | 0.2218 ± 0.1231 | 0.2106 ± 0.1263 |
| 80 | 256 | Full isotropic | 0.3521 ± 0.0840 | 0.3505 ± 0.0885 | 0.3516 ± 0.0874 |
| 80 | 256 | Full ARD | -0.0324 ± 0.0674 | 0.1102 ± 0.0690 | 0.0300 ± 0.0488 |

## additive: RMSE (lower is better)

| Dimension | Training samples | Model | RBF | Matérn 1.5 | Matérn 2.5 |
| --- | --- | --- | --- | --- | --- |
| 4 | 128 | Additive | 0.0589 ± 0.0071 | 0.0787 ± 0.0070 | 0.0604 ± 0.0040 |
| 4 | 128 | Full isotropic | 0.0589 ± 0.0071 | 0.0787 ± 0.0070 | 0.0604 ± 0.0040 |
| 4 | 128 | Full ARD | 0.0395 ± 0.0040 | 0.0464 ± 0.0114 | 0.0419 ± 0.0074 |
| 4 | 256 | Additive | 0.0334 ± 0.0023 | 0.0420 ± 0.0017 | 0.0367 ± 0.0014 |
| 4 | 256 | Full isotropic | 0.0334 ± 0.0023 | 0.0420 ± 0.0017 | 0.0367 ± 0.0014 |
| 4 | 256 | Full ARD | 0.0245 ± 0.0012 | 0.0306 ± 0.0009 | 0.0275 ± 0.0005 |
| 20 | 128 | Additive | 0.4582 ± 0.0162 | 0.4736 ± 0.0072 | 0.4691 ± 0.0126 |
| 20 | 128 | Full isotropic | 0.6268 ± 0.0236 | 0.6162 ± 0.0161 | 0.6199 ± 0.0159 |
| 20 | 128 | Full ARD | 0.6261 ± 0.0527 | 0.6166 ± 0.0526 | 0.6099 ± 0.0587 |
| 20 | 256 | Additive | 0.1860 ± 0.0037 | 0.2671 ± 0.0029 | 0.2290 ± 0.0068 |
| 20 | 256 | Full isotropic | 0.5856 ± 0.0243 | 0.5734 ± 0.0198 | 0.5755 ± 0.0201 |
| 20 | 256 | Full ARD | 0.5311 ± 0.0273 | 0.5185 ± 0.0117 | 0.5173 ± 0.0075 |
| 80 | 128 | Additive | 0.7532 ± 0.0475 | 0.7188 ± 0.0475 | 0.7321 ± 0.0540 |
| 80 | 128 | Full isotropic | 0.7153 ± 0.0298 | 0.7115 ± 0.0310 | 0.7121 ± 0.0308 |
| 80 | 128 | Full ARD | 0.8990 ± 0.0799 | 0.8117 ± 0.0535 | 0.8454 ± 0.0348 |
| 80 | 256 | Additive | 0.6254 ± 0.0636 | 0.6182 ± 0.0604 | 0.6217 ± 0.0629 |
| 80 | 256 | Full isotropic | 0.6617 ± 0.0260 | 0.6554 ± 0.0265 | 0.6567 ± 0.0265 |
| 80 | 256 | Full ARD | 0.7900 ± 0.0144 | 0.7424 ± 0.0193 | 0.7667 ± 0.0222 |

## interaction: RMSE (lower is better)

| Dimension | Training samples | Model | RBF | Matérn 1.5 | Matérn 2.5 |
| --- | --- | --- | --- | --- | --- |
| 20 | 256 | Additive | 0.6186 ± 0.0403 | 0.6035 ± 0.0166 | 0.6181 ± 0.0361 |
| 20 | 256 | Full isotropic | 0.4857 ± 0.0226 | 0.4900 ± 0.0170 | 0.4866 ± 0.0188 |
| 20 | 256 | Full ARD | 0.5203 ± 0.0348 | 0.5122 ± 0.0192 | 0.5169 ± 0.0289 |
| 80 | 256 | Additive | 0.7611 ± 0.0427 | 0.7413 ± 0.0486 | 0.7466 ± 0.0501 |
| 80 | 256 | Full isotropic | 0.6770 ± 0.0338 | 0.6777 ± 0.0360 | 0.6771 ± 0.0355 |
| 80 | 256 | Full ARD | 0.8562 ± 0.0356 | 0.7944 ± 0.0243 | 0.8300 ± 0.0256 |

## NRMSE and optimizer diagnostics

NRMSE = RMSE / test-target population standard deviation; lower is better.

| Function | Kernel | Dimension | Samples | Model | NRMSE | Parameters | Converged | Mean fit seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| additive | Matérn 1.5 | 4 | 128 | Additive | 0.1619 | 3 | 3/3 | 0.022 |
| additive | Matérn 1.5 | 4 | 128 | Full ARD | 0.0955 | 6 | 3/3 | 0.020 |
| additive | Matérn 1.5 | 4 | 128 | Full isotropic | 0.1619 | 3 | 3/3 | 0.019 |
| additive | Matérn 1.5 | 4 | 256 | Additive | 0.0862 | 3 | 3/3 | 0.080 |
| additive | Matérn 1.5 | 4 | 256 | Full ARD | 0.0629 | 6 | 3/3 | 0.063 |
| additive | Matérn 1.5 | 4 | 256 | Full isotropic | 0.0862 | 3 | 3/3 | 0.069 |
| additive | Matérn 1.5 | 20 | 128 | Additive | 0.5671 | 11 | 3/3 | 0.181 |
| additive | Matérn 1.5 | 20 | 128 | Full ARD | 0.7379 | 22 | 3/3 | 0.130 |
| additive | Matérn 1.5 | 20 | 128 | Full isotropic | 0.7373 | 3 | 3/3 | 0.008 |
| additive | Matérn 1.5 | 20 | 256 | Additive | 0.3193 | 11 | 3/3 | 0.948 |
| additive | Matérn 1.5 | 20 | 256 | Full ARD | 0.6198 | 22 | 3/3 | 0.477 |
| additive | Matérn 1.5 | 20 | 256 | Full isotropic | 0.6848 | 3 | 3/3 | 0.032 |
| additive | Matérn 1.5 | 80 | 128 | Additive | 0.8621 | 41 | 3/3 | 1.725 |
| additive | Matérn 1.5 | 80 | 128 | Full ARD | 0.9737 | 82 | 3/3 | 0.447 |
| additive | Matérn 1.5 | 80 | 128 | Full isotropic | 0.8535 | 3 | 3/3 | 0.013 |
| additive | Matérn 1.5 | 80 | 256 | Additive | 0.7349 | 41 | 3/3 | 7.250 |
| additive | Matérn 1.5 | 80 | 256 | Full ARD | 0.8829 | 82 | 3/3 | 1.660 |
| additive | Matérn 1.5 | 80 | 256 | Full isotropic | 0.7792 | 3 | 3/3 | 0.061 |
| additive | Matérn 2.5 | 4 | 128 | Additive | 0.1242 | 3 | 3/3 | 0.015 |
| additive | Matérn 2.5 | 4 | 128 | Full ARD | 0.0861 | 6 | 3/3 | 0.011 |
| additive | Matérn 2.5 | 4 | 128 | Full isotropic | 0.1242 | 3 | 3/3 | 0.011 |
| additive | Matérn 2.5 | 4 | 256 | Additive | 0.0753 | 3 | 3/3 | 0.051 |
| additive | Matérn 2.5 | 4 | 256 | Full ARD | 0.0564 | 6 | 3/3 | 0.048 |
| additive | Matérn 2.5 | 4 | 256 | Full isotropic | 0.0753 | 3 | 3/3 | 0.037 |
| additive | Matérn 2.5 | 20 | 128 | Additive | 0.5619 | 11 | 3/3 | 0.191 |
| additive | Matérn 2.5 | 20 | 128 | Full ARD | 0.7292 | 22 | 3/3 | 0.095 |
| additive | Matérn 2.5 | 20 | 128 | Full isotropic | 0.7418 | 3 | 3/3 | 0.008 |
| additive | Matérn 2.5 | 20 | 256 | Additive | 0.2737 | 11 | 3/3 | 1.238 |
| additive | Matérn 2.5 | 20 | 256 | Full ARD | 0.6184 | 22 | 3/3 | 0.403 |
| additive | Matérn 2.5 | 20 | 256 | Full isotropic | 0.6873 | 3 | 3/3 | 0.072 |
| additive | Matérn 2.5 | 80 | 128 | Additive | 0.8780 | 41 | 3/3 | 1.817 |
| additive | Matérn 2.5 | 80 | 128 | Full ARD | 1.0143 | 82 | 3/3 | 0.581 |
| additive | Matérn 2.5 | 80 | 128 | Full isotropic | 0.8542 | 3 | 3/3 | 0.013 |
| additive | Matérn 2.5 | 80 | 256 | Additive | 0.7391 | 41 | 3/3 | 8.467 |
| additive | Matérn 2.5 | 80 | 256 | Full ARD | 0.9117 | 82 | 3/3 | 2.434 |
| additive | Matérn 2.5 | 80 | 256 | Full isotropic | 0.7808 | 3 | 3/3 | 0.090 |
| additive | RBF | 4 | 128 | Additive | 0.1212 | 3 | 3/3 | 0.016 |
| additive | RBF | 4 | 128 | Full ARD | 0.0812 | 6 | 3/3 | 0.009 |
| additive | RBF | 4 | 128 | Full isotropic | 0.1212 | 3 | 3/3 | 0.009 |
| additive | RBF | 4 | 256 | Additive | 0.0686 | 3 | 3/3 | 0.039 |
| additive | RBF | 4 | 256 | Full ARD | 0.0503 | 6 | 3/3 | 0.035 |
| additive | RBF | 4 | 256 | Full isotropic | 0.0686 | 3 | 3/3 | 0.029 |
| additive | RBF | 20 | 128 | Additive | 0.5489 | 11 | 3/3 | 0.157 |
| additive | RBF | 20 | 128 | Full ARD | 0.7485 | 22 | 3/3 | 0.066 |
| additive | RBF | 20 | 128 | Full isotropic | 0.7497 | 3 | 3/3 | 0.011 |
| additive | RBF | 20 | 256 | Additive | 0.2223 | 11 | 3/3 | 0.903 |
| additive | RBF | 20 | 256 | Full ARD | 0.6345 | 22 | 3/3 | 0.357 |
| additive | RBF | 20 | 256 | Full isotropic | 0.6993 | 3 | 3/3 | 0.050 |
| additive | RBF | 80 | 128 | Additive | 0.9033 | 41 | 3/3 | 1.799 |
| additive | RBF | 80 | 128 | Full ARD | 1.0786 | 82 | 3/3 | 0.562 |
| additive | RBF | 80 | 128 | Full isotropic | 0.8580 | 3 | 3/3 | 0.010 |
| additive | RBF | 80 | 256 | Additive | 0.7435 | 41 | 3/3 | 7.058 |
| additive | RBF | 80 | 256 | Full ARD | 0.9395 | 82 | 3/3 | 2.235 |
| additive | RBF | 80 | 256 | Full isotropic | 0.7867 | 3 | 3/3 | 0.053 |
| interaction | Matérn 1.5 | 20 | 256 | Additive | 0.7411 | 11 | 3/3 | 0.801 |
| interaction | Matérn 1.5 | 20 | 256 | Full ARD | 0.6292 | 22 | 3/3 | 0.233 |
| interaction | Matérn 1.5 | 20 | 256 | Full isotropic | 0.6012 | 3 | 3/3 | 0.078 |
| interaction | Matérn 1.5 | 80 | 256 | Additive | 0.8802 | 41 | 3/3 | 11.346 |
| interaction | Matérn 1.5 | 80 | 256 | Full ARD | 0.9428 | 82 | 3/3 | 2.411 |
| interaction | Matérn 1.5 | 80 | 256 | Full isotropic | 0.8046 | 3 | 3/3 | 0.061 |
| interaction | Matérn 2.5 | 20 | 256 | Additive | 0.7567 | 11 | 3/3 | 0.638 |
| interaction | Matérn 2.5 | 20 | 256 | Full ARD | 0.6346 | 22 | 3/3 | 0.202 |
| interaction | Matérn 2.5 | 20 | 256 | Full isotropic | 0.5968 | 3 | 3/3 | 0.052 |
| interaction | Matérn 2.5 | 80 | 256 | Additive | 0.8865 | 41 | 3/3 | 8.558 |
| interaction | Matérn 2.5 | 80 | 256 | Full ARD | 0.9847 | 82 | 3/3 | 2.790 |
| interaction | Matérn 2.5 | 80 | 256 | Full isotropic | 0.8040 | 3 | 3/3 | 0.063 |
| interaction | RBF | 20 | 256 | Additive | 0.7568 | 11 | 3/3 | 1.165 |
| interaction | RBF | 20 | 256 | Full ARD | 0.6386 | 22 | 3/3 | 0.152 |
| interaction | RBF | 20 | 256 | Full isotropic | 0.5953 | 3 | 3/3 | 0.049 |
| interaction | RBF | 80 | 256 | Additive | 0.9037 | 41 | 3/3 | 8.261 |
| interaction | RBF | 80 | 256 | Full ARD | 1.0157 | 82 | 3/3 | 2.583 |
| interaction | RBF | 80 | 256 | Full isotropic | 0.8038 | 3 | 3/3 | 0.047 |

These results use known variable groups and a single optimizer start per model.
Matérn nu is fixed; amplitudes, length scales, and noise are optimized jointly.
A finite iteration budget and non-convex LML optimization do not guarantee a global optimum.
Fits that did not converge are retained and reported rather than omitted.
