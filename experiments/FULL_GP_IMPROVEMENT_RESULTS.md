# Full-dimensional GP improvement results

Mean test R² ± sample SD across fresh instance/data pairs. Higher is better.
CV selection uses pooled original-output squared error in three training folds. All candidate results are retained.

| Function | baseline | multistart | quadratic | asinh_matern | asinh_quadratic | cv_selected |
| --- | --- | --- | --- | --- | --- | --- |
| f07_step_ellipsoid | 0.6028 ± 0.0373 | 0.6028 ± 0.0373 | 0.6042 ± 0.0359 | 0.6000 ± 0.0276 | 0.6025 ± 0.0254 | 0.6050 ± 0.0354 |
| f09_rosenbrock_rotated | 0.0411 ± 0.0061 | 0.0411 ± 0.0061 | 0.1317 ± 0.0288 | 0.0359 ± 0.0062 | 0.1336 ± 0.0159 | 0.1289 ± 0.0261 |
| f10_ellipsoid_rotated | 0.3794 ± 0.1291 | 0.3794 ± 0.1291 | 0.3864 ± 0.1265 | 0.3863 ± 0.1299 | 0.3927 ± 0.1270 | 0.3848 ± 0.1300 |
| f11_discus | 0.1627 ± 0.1273 | 0.1628 ± 0.1273 | 0.1624 ± 0.1291 | 0.1856 ± 0.1629 | 0.1891 ± 0.1682 | 0.1889 ± 0.1682 |
| f12_bent_cigar | 0.0362 ± 0.0349 | 0.0472 ± 0.0196 | 0.0460 ± 0.0149 | 0.0738 ± 0.0367 | 0.0767 ± 0.0370 | 0.0767 ± 0.0370 |
| f13_sharp_ridge | 0.7504 ± 0.0441 | 0.7505 ± 0.0440 | 0.7505 ± 0.0440 | 0.7375 ± 0.0423 | 0.7375 ± 0.0423 | 0.7505 ± 0.0440 |
| f15_rastrigin_rotated | 0.3518 ± 0.0260 | 0.3518 ± 0.0260 | 0.3565 ± 0.0259 | 0.3854 ± 0.0055 | 0.3857 ± 0.0048 | 0.3865 ± 0.0045 |
| f16_weierstrass | -0.0047 ± 0.0042 | -0.0047 ± 0.0042 | -0.0047 ± 0.0042 | -0.0041 ± 0.0036 | -0.0041 ± 0.0036 | -0.0047 ± 0.0042 |
| f18_schaffers1000 | 0.2860 ± 0.0857 | 0.2849 ± 0.0865 | 0.2892 ± 0.0862 | 0.3347 ± 0.1124 | 0.3454 ± 0.1188 | 0.3423 ± 0.1198 |
| f21_gallagher101 | 0.0760 ± 0.0009 | 0.0760 ± 0.0009 | 0.1086 ± 0.0097 | 0.1027 ± 0.0171 | 0.1334 ± 0.0218 | 0.1334 ± 0.0218 |
| f23_katsuura | -0.0009 ± 0.0002 | -0.0009 ± 0.0001 | -0.0009 ± 0.0001 | -0.0008 ± 0.0007 | -0.0008 ± 0.0007 | -0.0009 ± 0.0001 |
| f24_lunacek | 0.2163 ± 0.0618 | 0.2164 ± 0.0620 | 0.2882 ± 0.0100 | 0.2168 ± 0.0485 | 0.2892 ± 0.0078 | 0.2884 ± 0.0086 |

Selected versus baseline: 30 lower RMSE, 2 equal, 4 higher.
Selected candidates: {'asinh_quadratic': 16, 'asinh_matern': 4, 'quadratic': 10, 'baseline': 2, 'multistart': 4}.
Fits: 720. Optimizer runs: 1872; converged: 1872. Chosen LML run converged in 720/720 fits.

## Paired selected results

| Function | Instance | Selected | Baseline R² | Selected R² | Baseline RMSE | Selected RMSE | Baseline local NRMSE | Selected local NRMSE | CV fit/predict seconds | Selected fit/predict seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f07_step_ellipsoid | 4 | asinh_quadratic | 0.586164 | 0.588704 | 1265.53 | 1261.65 | 0.77426 | 1.29943 | 27.865 | 7.577 |
| f09_rosenbrock_rotated | 4 | asinh_quadratic | 0.0383196 | 0.144335 | 719556 | 678736 | 2.03699 | 2.46107 | 22.280 | 6.002 |
| f10_ellipsoid_rotated | 4 | asinh_quadratic | 0.341686 | 0.353526 | 2.11278e+07 | 2.0937e+07 | 1.26874 | 1.19587 | 28.354 | 7.383 |
| f11_discus | 4 | asinh_matern | 0.143433 | 0.159 | 1.21135e+07 | 1.20029e+07 | 0.635297 | 0.55762 | 33.673 | 3.708 |
| f12_bent_cigar | 4 | asinh_quadratic | 0.0695539 | 0.119335 | 3.19451e+10 | 3.10788e+10 | 0.736167 | 0.657971 | 30.919 | 8.120 |
| f13_sharp_ridge | 4 | quadratic | 0.741889 | 0.741878 | 178.083 | 178.087 | 6.67524 | 6.67527 | 32.450 | 8.801 |
| f15_rastrigin_rotated | 4 | asinh_quadratic | 0.35708 | 0.391207 | 2371.88 | 2308.07 | 0.846292 | 6.67604 | 24.478 | 8.718 |
| f16_weierstrass | 4 | baseline | -0.00571396 | -0.00571396 | 10.7882 | 10.7882 | 7.22752 | 7.22752 | 21.929 | 0.184 |
| f18_schaffers1000 | 4 | asinh_quadratic | 0.374433 | 0.473413 | 534.051 | 489.983 | 0.83705 | 0.471029 | 26.761 | 7.737 |
| f21_gallagher101 | 4 | asinh_quadratic | 0.0769623 | 0.130388 | 0.141747 | 0.137584 | 585.636 | 585.6 | 22.337 | 5.712 |
| f23_katsuura | 4 | multistart | -0.00105084 | -0.00105081 | 2.40797 | 2.40797 | 1.02022 | 1.02022 | 17.498 | 1.957 |
| f24_lunacek | 4 | asinh_quadratic | 0.145404 | 0.285244 | 275.64 | 252.081 | 8.51663 | 6.53546 | 20.113 | 5.196 |
| f07_step_ellipsoid | 5 | quadratic | 0.57663 | 0.580775 | 1235.93 | 1229.87 | 0.894779 | 1.1762 | 27.684 | 8.837 |
| f09_rosenbrock_rotated | 5 | quadratic | 0.0368739 | 0.0987879 | 682038 | 659752 | 2.56728 | 1.45261 | 22.432 | 4.182 |
| f10_ellipsoid_rotated | 5 | multistart | 0.273258 | 0.273296 | 2.0285e+07 | 2.02845e+07 | 1.43846 | 1.43805 | 28.355 | 2.670 |
| f11_discus | 5 | asinh_quadratic | 0.298553 | 0.370069 | 1.22531e+07 | 1.16117e+07 | 0.047229 | 0.0502402 | 26.779 | 8.581 |
| f12_bent_cigar | 5 | asinh_quadratic | -7.06503e-05 | 0.0526185 | 4.24342e+10 | 4.13012e+10 | 0.487849 | 0.317282 | 28.848 | 8.246 |
| f13_sharp_ridge | 5 | quadratic | 0.71114 | 0.711349 | 187.978 | 187.91 | 6.82472 | 6.82204 | 32.064 | 9.815 |
| f15_rastrigin_rotated | 5 | asinh_quadratic | 0.374773 | 0.382289 | 1777.28 | 1766.56 | 0.436089 | 1.96174 | 24.311 | 8.819 |
| f16_weierstrass | 5 | quadratic | -0.008316 | -0.00830333 | 11.1568 | 11.1567 | 7.01551 | 7.01573 | 22.079 | 5.365 |
| f18_schaffers1000 | 5 | asinh_matern | 0.280207 | 0.314848 | 255.417 | 249.195 | 0.717425 | 1.14036 | 27.572 | 2.148 |
| f21_gallagher101 | 5 | asinh_quadratic | 0.0751061 | 0.113267 | 0.130618 | 0.127895 | 635.317 | 635.176 | 23.726 | 7.763 |
| f23_katsuura | 5 | baseline | -0.000842544 | -0.000842544 | 2.38404 | 2.38404 | 1.0268 | 1.0268 | 18.396 | 0.520 |
| f24_lunacek | 5 | quadratic | 0.259352 | 0.281807 | 256.615 | 252.695 | 7.63269 | 7.40157 | 22.937 | 6.636 |
| f07_step_ellipsoid | 6 | multistart | 0.645482 | 0.645588 | 1250.09 | 1249.91 | 0.0483463 | 0.0502332 | 31.056 | 2.691 |
| f09_rosenbrock_rotated | 6 | quadratic | 0.0480282 | 0.143566 | 671992 | 637380 | 1.25772 | 0.0832723 | 19.184 | 5.565 |
| f10_ellipsoid_rotated | 6 | quadratic | 0.523111 | 0.527649 | 2.04299e+07 | 2.03325e+07 | 0.839052 | 1.02245 | 30.390 | 8.662 |
| f11_discus | 6 | asinh_matern | 0.0461912 | 0.0375794 | 1.22292e+07 | 1.22842e+07 | 0.405362 | 0.376019 | 29.809 | 4.506 |
| f12_bent_cigar | 6 | asinh_quadratic | 0.038994 | 0.0581432 | 1.40032e+11 | 1.3863e+11 | 0.19242 | 0.219022 | 29.452 | 7.634 |
| f13_sharp_ridge | 6 | multistart | 0.798137 | 0.798139 | 165.39 | 165.389 | 5.78985 | 5.78984 | 29.770 | 3.425 |
| f15_rastrigin_rotated | 6 | asinh_matern | 0.323599 | 0.38603 | 4576.18 | 4359.88 | 0.778097 | 5.23809 | 31.087 | 2.650 |
| f16_weierstrass | 6 | quadratic | -1.70736e-05 | -1.77745e-05 | 11.1599 | 11.1599 | 6.87521 | 6.87522 | 17.427 | 4.888 |
| f18_schaffers1000 | 6 | asinh_quadratic | 0.203259 | 0.238569 | 235.238 | 229.966 | 0.367173 | 0.518302 | 28.700 | 7.804 |
| f21_gallagher101 | 6 | asinh_quadratic | 0.0760493 | 0.15652 | 0.132895 | 0.126976 | 624.831 | 624.882 | 23.062 | 5.979 |
| f23_katsuura | 6 | quadratic | -0.000705555 | -0.000826275 | 2.32483 | 2.32497 | 1.04287 | 1.04311 | 17.491 | 5.922 |
| f24_lunacek | 6 | asinh_quadratic | 0.244027 | 0.298144 | 248.45 | 239.392 | 7.58042 | 5.90924 | 20.978 | 6.770 |

Local NRMSE uses the global test target SD. Local targets are never used for candidate selection.
LML values are only compared among starts of the same candidate and training fold, not across output transforms.
