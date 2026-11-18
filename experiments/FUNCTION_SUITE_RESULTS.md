# Paired synthetic-function results

Values are test R², mean ± sample SD across seeds. Higher is better; negative values are retained.

| Function | Samples | Additive, independent | Additive, shared ARD | Plain Matérn 2.5 |
| --- | --- | --- | --- | --- |
| polynomial | 128 | 0.4068 ± 0.1118 | 0.6886 ± 0.0213 | 0.5491 ± 0.0338 |
| polynomial | 256 | 0.7731 ± 0.0117 | 0.8798 ± 0.0156 | 0.7038 ± 0.0328 |
| polynomial | 512 | 0.9603 ± 0.0029 | 0.9939 ± 0.0006 | 0.7773 ± 0.0156 |
| periodic | 128 | -0.2193 ± 0.2033 | 0.2195 ± 0.1521 | 0.0145 ± 0.0332 |
| periodic | 256 | -0.0455 ± 0.0245 | 0.4858 ± 0.0324 | 0.0212 ± 0.0314 |
| periodic | 512 | 0.0124 ± 0.0428 | 0.6202 ± 0.0074 | 0.0406 ± 0.0079 |
| nonsmooth | 128 | -0.0536 ± 0.0742 | 0.3640 ± 0.0389 | 0.0269 ± 0.0301 |
| nonsmooth | 256 | 0.3555 ± 0.0864 | 0.6959 ± 0.0292 | 0.0822 ± 0.0099 |
| nonsmooth | 512 | 0.8169 ± 0.0035 | 0.8665 ± 0.0067 | 0.2035 ± 0.0328 |
| multiscale | 128 | -0.3809 ± 0.3048 | 0.0609 ± 0.0465 | -0.0059 ± 0.0142 |
| multiscale | 256 | -0.0428 ± 0.0346 | 0.1578 ± 0.0825 | 0.0134 ± 0.0132 |
| multiscale | 512 | 0.0772 ± 0.0284 | 0.3748 ± 0.0665 | 0.0289 ± 0.0097 |
| cross_pair | 128 | -0.3648 ± 0.0370 | -0.0513 ± 0.0416 | -0.0092 ± 0.0083 |
| cross_pair | 256 | -0.1705 ± 0.0972 | -0.0055 ± 0.0039 | -0.0018 ± 0.0009 |
| cross_pair | 512 | -0.0623 ± 0.0347 | -0.0110 ± 0.0061 | -0.0000 ± 0.0031 |
| linear | 128 | 0.9961 ± 0.0013 | 0.9961 ± 0.0015 | 0.9973 ± 0.0005 |
| linear | 256 | 0.9992 ± 0.0001 | 0.9992 ± 0.0001 | 0.9995 ± 0.0001 |
| linear | 512 | 0.9995 ± 0.0000 | 0.9996 ± 0.0000 | 0.9996 ± 0.0000 |
| dense_fourier | 128 | -0.1644 ± 0.0505 | 0.0032 ± 0.0809 | 0.2105 ± 0.0702 |
| dense_fourier | 256 | 0.0907 ± 0.1412 | 0.2486 ± 0.0654 | 0.2851 ± 0.0539 |
| dense_fourier | 512 | 0.2276 ± 0.0519 | 0.3200 ± 0.0164 | 0.3588 ± 0.0134 |

Converged: 189/189 fits. All fits are retained.

## polynomial


Dimension: 80. Seeds: 11003, 12007, 13001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | 0.4068 ± 0.1118 | 0.4200 ± 0.0355 | 0.3342 | 0.8325 | 2.4789 | 3/3 |
| 128 | Plain Matérn 2.5 | 0.5491 ± 0.0338 | 0.3672 ± 0.0084 | 0.2924 | 0.7205 | 0.0959 | 3/3 |
| 128 | Additive, shared ARD | 0.6886 ± 0.0213 | 0.3054 ± 0.0141 | 0.2441 | 0.5842 | 0.5659 | 3/3 |
| 256 | Additive, independent | 0.7731 ± 0.0117 | 0.2606 ± 0.0063 | 0.2070 | 0.5116 | 7.5362 | 3/3 |
| 256 | Plain Matérn 2.5 | 0.7038 ± 0.0328 | 0.2976 ± 0.0161 | 0.2368 | 0.5879 | 0.1338 | 3/3 |
| 256 | Additive, shared ARD | 0.8798 ± 0.0156 | 0.1896 ± 0.0152 | 0.1501 | 0.3686 | 1.4799 | 3/3 |
| 512 | Additive, independent | 0.9603 ± 0.0029 | 0.1089 ± 0.0036 | 0.0867 | 0.2149 | 12.8752 | 3/3 |
| 512 | Plain Matérn 2.5 | 0.7773 ± 0.0156 | 0.2581 ± 0.0057 | 0.2055 | 0.5040 | 0.5900 | 3/3 |
| 512 | Additive, shared ARD | 0.9939 ± 0.0006 | 0.0427 ± 0.0020 | 0.0341 | 0.0834 | 3.4041 | 3/3 |

### Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11003 | Additive, independent | 128 | 0.30095 | 0.44897 | 0.00157 | 188 | 197 |
| 11003 | Additive, independent | 256 | 0.76922 | 0.25797 | 0.00274 | 210 | 222 |
| 11003 | Additive, independent | 512 | 0.95925 | 0.10841 | 0.04699 | 68 | 76 |
| 11003 | Plain Matérn 2.5 | 128 | 0.51131 | 0.37539 | 0.01530 | 11 | 54 |
| 11003 | Plain Matérn 2.5 | 256 | 0.69965 | 0.29429 | 0.01583 | 12 | 22 |
| 11003 | Plain Matérn 2.5 | 512 | 0.76298 | 0.26143 | 0.01663 | 12 | 22 |
| 11003 | Additive, shared ARD | 128 | 0.70709 | 0.29062 | 0.24056 | 47 | 62 |
| 11003 | Additive, shared ARD | 256 | 0.89654 | 0.17273 | 0.00014 | 34 | 37 |
| 11003 | Additive, shared ARD | 512 | 0.99395 | 0.04177 | 0.02711 | 19 | 22 |
| 12007 | Additive, independent | 128 | 0.52375 | 0.38033 | 0.00047 | 216 | 225 |
| 12007 | Additive, independent | 256 | 0.76385 | 0.26782 | 0.07851 | 165 | 178 |
| 12007 | Additive, independent | 512 | 0.95811 | 0.11280 | 0.04343 | 74 | 81 |
| 12007 | Plain Matérn 2.5 | 128 | 0.57649 | 0.35865 | 0.00029 | 31 | 77 |
| 12007 | Plain Matérn 2.5 | 256 | 0.67323 | 0.31504 | 0.01707 | 11 | 23 |
| 12007 | Plain Matérn 2.5 | 512 | 0.77495 | 0.26145 | 0.01680 | 11 | 23 |
| 12007 | Additive, shared ARD | 128 | 0.66539 | 0.31880 | 0.23979 | 40 | 52 |
| 12007 | Additive, shared ARD | 256 | 0.86561 | 0.20204 | 0.05160 | 29 | 37 |
| 12007 | Additive, shared ARD | 512 | 0.99332 | 0.04505 | 0.02354 | 17 | 19 |
| 13001 | Additive, independent | 128 | 0.39558 | 0.43070 | 0.00159 | 291 | 316 |
| 13001 | Additive, independent | 256 | 0.78632 | 0.25609 | 0.00115 | 167 | 179 |
| 13001 | Additive, independent | 512 | 0.96364 | 0.10563 | 0.05321 | 77 | 84 |
| 13001 | Plain Matérn 2.5 | 128 | 0.55964 | 0.36763 | 0.17591 | 30 | 84 |
| 13001 | Plain Matérn 2.5 | 256 | 0.73841 | 0.28334 | 0.01787 | 12 | 24 |
| 13001 | Plain Matérn 2.5 | 512 | 0.79387 | 0.25152 | 0.01684 | 11 | 21 |
| 13001 | Additive, shared ARD | 128 | 0.69336 | 0.30677 | 0.24518 | 43 | 59 |
| 13001 | Additive, shared ARD | 256 | 0.87732 | 0.19404 | 0.04453 | 31 | 38 |
| 13001 | Additive, shared ARD | 512 | 0.99444 | 0.04132 | 0.02684 | 19 | 22 |

Test RMSE decreases in 18/18 consecutive within-seed transitions.
Optimizer convergence: 27/27 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.

## periodic


Dimension: 80. Seeds: 11003, 12007, 13001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | -0.2193 ± 0.2033 | 0.9281 ± 0.0842 | 0.7398 | 1.8132 | 1.7008 | 3/3 |
| 128 | Plain Matérn 2.5 | 0.0145 ± 0.0332 | 0.8358 ± 0.0117 | 0.6688 | 1.6283 | 0.0339 | 3/3 |
| 128 | Additive, shared ARD | 0.2195 ± 0.1521 | 0.7413 ± 0.0667 | 0.5904 | 1.4621 | 0.5504 | 3/3 |
| 256 | Additive, independent | -0.0455 ± 0.0245 | 0.8610 ± 0.0182 | 0.6887 | 1.6582 | 8.0344 | 3/3 |
| 256 | Plain Matérn 2.5 | 0.0212 ± 0.0314 | 0.8329 ± 0.0100 | 0.6668 | 1.6301 | 0.1660 | 3/3 |
| 256 | Additive, shared ARD | 0.4858 ± 0.0324 | 0.6037 ± 0.0240 | 0.4817 | 1.1848 | 2.6047 | 3/3 |
| 512 | Additive, independent | 0.0124 ± 0.0428 | 0.8368 ± 0.0259 | 0.6680 | 1.6246 | 27.8516 | 3/3 |
| 512 | Plain Matérn 2.5 | 0.0406 ± 0.0079 | 0.8247 ± 0.0075 | 0.6605 | 1.5942 | 0.8773 | 3/3 |
| 512 | Additive, shared ARD | 0.6202 ± 0.0074 | 0.5189 ± 0.0073 | 0.4138 | 1.0165 | 10.3460 | 3/3 |

### Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11003 | Additive, independent | 128 | -0.09027 | 0.87098 | 0.00750 | 128 | 132 |
| 11003 | Additive, independent | 256 | -0.02183 | 0.84320 | 0.00838 | 182 | 197 |
| 11003 | Additive, independent | 512 | 0.04793 | 0.81391 | 0.00832 | 175 | 181 |
| 11003 | Plain Matérn 2.5 | 128 | -0.02013 | 0.84250 | 0.02316 | 13 | 16 |
| 11003 | Plain Matérn 2.5 | 256 | -0.01348 | 0.83975 | 0.02421 | 15 | 17 |
| 11003 | Plain Matérn 2.5 | 512 | 0.03388 | 0.81990 | 0.02543 | 11 | 52 |
| 11003 | Additive, shared ARD | 128 | 0.05113 | 0.81254 | 0.02316 | 20 | 25 |
| 11003 | Additive, shared ARD | 256 | 0.52314 | 0.57602 | 0.42160 | 47 | 68 |
| 11003 | Additive, shared ARD | 512 | 0.62529 | 0.51061 | 0.43480 | 47 | 63 |
| 12007 | Additive, independent | 128 | -0.45362 | 1.02484 | 0.00522 | 182 | 193 |
| 12007 | Additive, independent | 256 | -0.07077 | 0.87959 | 0.00424 | 127 | 131 |
| 12007 | Additive, independent | 512 | -0.03515 | 0.86483 | 0.02269 | 129 | 139 |
| 12007 | Plain Matérn 2.5 | 128 | 0.01748 | 0.84256 | 0.02996 | 5 | 28 |
| 12007 | Plain Matérn 2.5 | 256 | 0.02912 | 0.83756 | 0.02890 | 5 | 31 |
| 12007 | Plain Matérn 2.5 | 512 | 0.03866 | 0.83343 | 0.02713 | 9 | 28 |
| 12007 | Additive, shared ARD | 128 | 0.26047 | 0.73098 | 0.24665 | 37 | 60 |
| 12007 | Additive, shared ARD | 256 | 0.46989 | 0.61889 | 0.34036 | 51 | 65 |
| 12007 | Additive, shared ARD | 512 | 0.62366 | 0.52146 | 0.43722 | 47 | 60 |
| 13001 | Additive, independent | 128 | -0.11393 | 0.88854 | 0.00490 | 192 | 202 |
| 13001 | Additive, independent | 256 | -0.04391 | 0.86016 | 0.00179 | 268 | 288 |
| 13001 | Additive, independent | 512 | 0.02446 | 0.83152 | 0.00307 | 190 | 198 |
| 13001 | Plain Matérn 2.5 | 128 | 0.04606 | 0.82226 | 0.02648 | 6 | 32 |
| 13001 | Plain Matérn 2.5 | 256 | 0.04782 | 0.82151 | 0.02731 | 11 | 38 |
| 13001 | Plain Matérn 2.5 | 512 | 0.04927 | 0.82088 | 0.02802 | 9 | 19 |
| 13001 | Additive, shared ARD | 128 | 0.34690 | 0.68036 | 0.42524 | 55 | 83 |
| 13001 | Additive, shared ARD | 256 | 0.46450 | 0.61607 | 0.31955 | 50 | 64 |
| 13001 | Additive, shared ARD | 512 | 0.61170 | 0.52460 | 0.45983 | 45 | 64 |

Test RMSE decreases in 18/18 consecutive within-seed transitions.
Optimizer convergence: 27/27 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.

## nonsmooth


Dimension: 80. Seeds: 11003, 12007, 13001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | -0.0536 ± 0.0742 | 0.4264 ± 0.0167 | 0.3393 | 0.8367 | 2.4188 | 3/3 |
| 128 | Plain Matérn 2.5 | 0.0269 ± 0.0301 | 0.4099 ± 0.0109 | 0.3271 | 0.8035 | 0.0824 | 3/3 |
| 128 | Additive, shared ARD | 0.3640 ± 0.0389 | 0.3312 ± 0.0060 | 0.2640 | 0.6591 | 0.2893 | 3/3 |
| 256 | Additive, independent | 0.3555 ± 0.0864 | 0.3329 ± 0.0181 | 0.2647 | 0.6517 | 6.6046 | 3/3 |
| 256 | Plain Matérn 2.5 | 0.0822 ± 0.0099 | 0.3981 ± 0.0073 | 0.3175 | 0.7838 | 0.2692 | 3/3 |
| 256 | Additive, shared ARD | 0.6959 ± 0.0292 | 0.2289 ± 0.0093 | 0.1831 | 0.4440 | 1.6150 | 3/3 |
| 512 | Additive, independent | 0.8169 ± 0.0035 | 0.1778 ± 0.0032 | 0.1415 | 0.3471 | 17.4077 | 3/3 |
| 512 | Plain Matérn 2.5 | 0.2035 ± 0.0328 | 0.3708 ± 0.0119 | 0.2958 | 0.7286 | 0.7337 | 3/3 |
| 512 | Additive, shared ARD | 0.8665 ± 0.0067 | 0.1518 ± 0.0030 | 0.1204 | 0.2982 | 5.5505 | 3/3 |

### Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11003 | Additive, independent | 128 | -0.13926 | 0.44454 | 0.00097 | 294 | 310 |
| 11003 | Additive, independent | 256 | 0.41308 | 0.31907 | 0.00028 | 110 | 118 |
| 11003 | Additive, independent | 512 | 0.81289 | 0.18015 | 0.00043 | 107 | 119 |
| 11003 | Plain Matérn 2.5 | 128 | 0.04613 | 0.40676 | 0.01332 | 5 | 49 |
| 11003 | Plain Matérn 2.5 | 256 | 0.07906 | 0.39968 | 0.01293 | 10 | 51 |
| 11003 | Plain Matérn 2.5 | 512 | 0.17750 | 0.37772 | 0.01255 | 11 | 21 |
| 11003 | Additive, shared ARD | 128 | 0.38433 | 0.32679 | 0.01331 | 15 | 20 |
| 11003 | Additive, shared ARD | 256 | 0.67696 | 0.23671 | 0.08891 | 31 | 40 |
| 11003 | Additive, shared ARD | 512 | 0.86147 | 0.15501 | 0.05197 | 26 | 34 |
| 12007 | Additive, independent | 128 | -0.01150 | 0.42277 | 0.00119 | 162 | 175 |
| 12007 | Additive, independent | 256 | 0.39730 | 0.32634 | 0.00146 | 124 | 130 |
| 12007 | Additive, independent | 512 | 0.81837 | 0.17915 | 0.00173 | 100 | 103 |
| 12007 | Plain Matérn 2.5 | 128 | -0.00772 | 0.42198 | 0.01238 | 11 | 92 |
| 12007 | Plain Matérn 2.5 | 256 | 0.07416 | 0.40447 | 0.01266 | 10 | 49 |
| 12007 | Plain Matérn 2.5 | 512 | 0.19279 | 0.37767 | 0.01314 | 13 | 38 |
| 12007 | Additive, shared ARD | 128 | 0.38852 | 0.32871 | 0.00014 | 39 | 43 |
| 12007 | Additive, shared ARD | 256 | 0.72950 | 0.21862 | 0.11450 | 28 | 39 |
| 12007 | Additive, shared ARD | 512 | 0.87411 | 0.14915 | 0.07751 | 25 | 33 |
| 13001 | Additive, independent | 128 | -0.01005 | 0.41179 | 0.00116 | 237 | 258 |
| 13001 | Additive, independent | 256 | 0.25624 | 0.35336 | 0.00004 | 224 | 244 |
| 13001 | Additive, independent | 512 | 0.81933 | 0.17416 | 0.03697 | 94 | 101 |
| 13001 | Plain Matérn 2.5 | 128 | 0.04241 | 0.40095 | 0.01424 | 8 | 45 |
| 13001 | Plain Matérn 2.5 | 256 | 0.09323 | 0.39017 | 0.01410 | 11 | 39 |
| 13001 | Plain Matérn 2.5 | 512 | 0.24034 | 0.35712 | 0.01362 | 11 | 23 |
| 13001 | Additive, shared ARD | 128 | 0.31923 | 0.33807 | 0.01427 | 20 | 25 |
| 13001 | Additive, shared ARD | 256 | 0.68119 | 0.23135 | 0.12165 | 31 | 43 |
| 13001 | Additive, shared ARD | 512 | 0.86377 | 0.15123 | 0.06257 | 26 | 36 |

Test RMSE decreases in 18/18 consecutive within-seed transitions.
Optimizer convergence: 27/27 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.

## multiscale


Dimension: 80. Seeds: 11003, 12007, 13001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | -0.3809 ± 0.3048 | 0.9478 ± 0.1059 | 0.7572 | 1.8780 | 1.5879 | 3/3 |
| 128 | Plain Matérn 2.5 | -0.0059 ± 0.0142 | 0.8119 ± 0.0111 | 0.6472 | 1.5768 | 0.0324 | 3/3 |
| 128 | Additive, shared ARD | 0.0609 ± 0.0465 | 0.7842 ± 0.0077 | 0.6283 | 1.5182 | 0.2862 | 3/3 |
| 256 | Additive, independent | -0.0428 ± 0.0346 | 0.8268 ± 0.0262 | 0.6601 | 1.6143 | 7.3114 | 3/3 |
| 256 | Plain Matérn 2.5 | 0.0134 ± 0.0132 | 0.8041 ± 0.0158 | 0.6419 | 1.5664 | 0.1750 | 3/3 |
| 256 | Additive, shared ARD | 0.1578 ± 0.0825 | 0.7419 ± 0.0220 | 0.5906 | 1.4521 | 1.1363 | 3/3 |
| 512 | Additive, independent | 0.0772 ± 0.0284 | 0.7778 ± 0.0250 | 0.6212 | 1.5361 | 27.7955 | 3/3 |
| 512 | Plain Matérn 2.5 | 0.0289 ± 0.0097 | 0.7978 ± 0.0126 | 0.6369 | 1.5551 | 1.4327 | 3/3 |
| 512 | Additive, shared ARD | 0.3748 ± 0.0665 | 0.6392 ± 0.0270 | 0.5111 | 1.2567 | 4.0854 | 3/3 |

### Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11003 | Additive, independent | 128 | -0.16745 | 0.89015 | 0.00627 | 218 | 224 |
| 11003 | Additive, independent | 256 | -0.03372 | 0.83761 | 0.00846 | 159 | 168 |
| 11003 | Additive, independent | 512 | 0.07874 | 0.79074 | 0.00698 | 155 | 167 |
| 11003 | Plain Matérn 2.5 | 128 | 0.00848 | 0.82034 | 0.02638 | 13 | 27 |
| 11003 | Plain Matérn 2.5 | 256 | 0.02627 | 0.81295 | 0.02624 | 10 | 45 |
| 11003 | Plain Matérn 2.5 | 512 | 0.03934 | 0.80747 | 0.02674 | 8 | 52 |
| 11003 | Additive, shared ARD | 128 | 0.08373 | 0.78860 | 0.02567 | 29 | 36 |
| 11003 | Additive, shared ARD | 256 | 0.24065 | 0.71790 | 0.02559 | 24 | 26 |
| 11003 | Additive, shared ARD | 512 | 0.45155 | 0.61011 | 0.02406 | 23 | 26 |
| 12007 | Additive, independent | 128 | -0.24528 | 0.88328 | 0.00344 | 143 | 149 |
| 12007 | Additive, independent | 256 | -0.01357 | 0.79687 | 0.00778 | 152 | 165 |
| 12007 | Additive, independent | 512 | 0.10475 | 0.74892 | 0.01400 | 275 | 284 |
| 12007 | Plain Matérn 2.5 | 128 | -0.01997 | 0.79939 | 0.02474 | 12 | 41 |
| 12007 | Plain Matérn 2.5 | 256 | 0.01413 | 0.78591 | 0.02464 | 10 | 42 |
| 12007 | Plain Matérn 2.5 | 512 | 0.02014 | 0.78351 | 0.02469 | 9 | 55 |
| 12007 | Additive, shared ARD | 128 | 0.00740 | 0.78859 | 0.02467 | 18 | 23 |
| 12007 | Additive, shared ARD | 256 | 0.07556 | 0.76103 | 0.02470 | 17 | 18 |
| 12007 | Additive, shared ARD | 512 | 0.33810 | 0.64396 | 0.02521 | 21 | 24 |
| 13001 | Additive, independent | 128 | -0.73004 | 1.06998 | 0.00403 | 113 | 118 |
| 13001 | Additive, independent | 256 | -0.08106 | 0.84581 | 0.00420 | 214 | 228 |
| 13001 | Additive, independent | 512 | 0.04807 | 0.79369 | 0.01909 | 65 | 69 |
| 13001 | Plain Matérn 2.5 | 128 | -0.00623 | 0.81601 | 0.02633 | 2 | 4 |
| 13001 | Plain Matérn 2.5 | 256 | -0.00012 | 0.81353 | 0.02463 | 2 | 4 |
| 13001 | Plain Matérn 2.5 | 512 | 0.02715 | 0.80236 | 0.02520 | 10 | 55 |
| 13001 | Additive, shared ARD | 128 | 0.09160 | 0.77533 | 0.02569 | 22 | 28 |
| 13001 | Additive, shared ARD | 256 | 0.15716 | 0.74683 | 0.00031 | 37 | 41 |
| 13001 | Additive, shared ARD | 512 | 0.33480 | 0.66347 | 0.02509 | 21 | 25 |

Test RMSE decreases in 18/18 consecutive within-seed transitions.
Optimizer convergence: 27/27 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.

## cross_pair


Dimension: 80. Seeds: 11003, 12007, 13001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | -0.3648 ± 0.0370 | 1.1504 ± 0.0194 | 0.9143 | 2.2144 | 1.4774 | 3/3 |
| 128 | Plain Matérn 2.5 | -0.0092 ± 0.0083 | 0.9893 ± 0.0116 | 0.7907 | 1.9198 | 0.0136 | 3/3 |
| 128 | Additive, shared ARD | -0.0513 ± 0.0416 | 1.0096 ± 0.0242 | 0.8085 | 1.9618 | 0.2334 | 3/3 |
| 256 | Additive, independent | -0.1705 ± 0.0972 | 1.0649 ± 0.0485 | 0.8499 | 2.0883 | 6.5834 | 3/3 |
| 256 | Plain Matérn 2.5 | -0.0018 ± 0.0009 | 0.9857 ± 0.0139 | 0.7890 | 1.9033 | 0.1219 | 3/3 |
| 256 | Additive, shared ARD | -0.0055 ± 0.0039 | 0.9875 ± 0.0141 | 0.7901 | 1.9096 | 1.3649 | 3/3 |
| 512 | Additive, independent | -0.0623 ± 0.0347 | 1.0148 ± 0.0082 | 0.8093 | 1.9658 | 23.3526 | 3/3 |
| 512 | Plain Matérn 2.5 | -0.0000 ± 0.0031 | 0.9848 ± 0.0135 | 0.7878 | 1.9072 | 0.9834 | 3/3 |
| 512 | Additive, shared ARD | -0.0110 ± 0.0061 | 0.9902 ± 0.0134 | 0.7925 | 1.9080 | 4.6490 | 3/3 |

### Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11003 | Additive, independent | 128 | -0.32601 | 1.14535 | 0.01366 | 190 | 200 |
| 11003 | Additive, independent | 256 | -0.26552 | 1.11892 | 0.01924 | 153 | 156 |
| 11003 | Additive, independent | 512 | -0.05551 | 1.02187 | 0.01561 | 124 | 131 |
| 11003 | Plain Matérn 2.5 | 128 | -0.01135 | 1.00026 | 0.02773 | 11 | 22 |
| 11003 | Plain Matérn 2.5 | 256 | -0.00239 | 0.99582 | 0.02924 | 13 | 27 |
| 11003 | Plain Matérn 2.5 | 512 | 0.00294 | 0.99317 | 0.03096 | 9 | 55 |
| 11003 | Additive, shared ARD | 128 | -0.08785 | 1.03740 | 0.02785 | 19 | 22 |
| 11003 | Additive, shared ARD | 256 | -0.00955 | 0.99937 | 0.03165 | 13 | 27 |
| 11003 | Additive, shared ARD | 512 | -0.01562 | 1.00237 | 0.00140 | 33 | 37 |
| 12007 | Additive, independent | 128 | -0.36844 | 1.13402 | 0.00265 | 116 | 121 |
| 12007 | Additive, independent | 256 | -0.17473 | 1.05070 | 0.00584 | 163 | 180 |
| 12007 | Additive, independent | 512 | -0.09983 | 1.01665 | 0.02589 | 122 | 135 |
| 12007 | Plain Matérn 2.5 | 128 | -0.01614 | 0.97720 | 0.03424 | 2 | 4 |
| 12007 | Plain Matérn 2.5 | 256 | -0.00082 | 0.96981 | 0.03210 | 11 | 20 |
| 12007 | Plain Matérn 2.5 | 512 | 0.00026 | 0.96928 | 0.03220 | 10 | 48 |
| 12007 | Additive, shared ARD | 128 | -0.06006 | 0.99810 | 0.03447 | 18 | 20 |
| 12007 | Additive, shared ARD | 256 | -0.00502 | 0.97184 | 0.03449 | 21 | 28 |
| 12007 | Additive, shared ARD | 512 | -0.01330 | 0.97584 | 0.06641 | 16 | 19 |
| 13001 | Additive, independent | 128 | -0.39982 | 1.17175 | 0.00508 | 127 | 136 |
| 13001 | Additive, independent | 256 | -0.07118 | 1.02501 | 0.01141 | 155 | 166 |
| 13001 | Additive, independent | 512 | -0.03146 | 1.00583 | 0.01463 | 160 | 173 |
| 13001 | Plain Matérn 2.5 | 128 | 0.00000 | 0.99037 | 0.03040 | 2 | 4 |
| 13001 | Plain Matérn 2.5 | 256 | -0.00218 | 0.99145 | 0.03094 | 14 | 16 |
| 13001 | Plain Matérn 2.5 | 512 | -0.00328 | 0.99199 | 0.02945 | 2 | 4 |
| 13001 | Additive, shared ARD | 128 | -0.00600 | 0.99333 | 0.03085 | 25 | 28 |
| 13001 | Additive, shared ARD | 256 | -0.00182 | 0.99127 | 0.03493 | 23 | 47 |
| 13001 | Additive, shared ARD | 512 | -0.00411 | 0.99240 | 0.07116 | 28 | 30 |

Test RMSE decreases in 13/18 consecutive within-seed transitions.
Optimizer convergence: 27/27 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.

## linear


Dimension: 80. Seeds: 11003, 12007, 13001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | 0.9961 ± 0.0013 | 0.0610 ± 0.0094 | 0.0488 | 0.1198 | 0.9810 | 3/3 |
| 128 | Plain Matérn 2.5 | 0.9973 ± 0.0005 | 0.0515 ± 0.0045 | 0.0415 | 0.0994 | 0.0667 | 3/3 |
| 128 | Additive, shared ARD | 0.9961 ± 0.0015 | 0.0605 ± 0.0104 | 0.0485 | 0.1185 | 0.3438 | 3/3 |
| 256 | Additive, independent | 0.9992 ± 0.0001 | 0.0279 ± 0.0012 | 0.0223 | 0.0546 | 1.7219 | 3/3 |
| 256 | Plain Matérn 2.5 | 0.9995 ± 0.0001 | 0.0229 ± 0.0015 | 0.0183 | 0.0443 | 0.3356 | 3/3 |
| 256 | Additive, shared ARD | 0.9992 ± 0.0001 | 0.0279 ± 0.0014 | 0.0222 | 0.0544 | 0.6646 | 3/3 |
| 512 | Additive, independent | 0.9995 ± 0.0000 | 0.0212 ± 0.0007 | 0.0168 | 0.0413 | 5.3405 | 3/3 |
| 512 | Plain Matérn 2.5 | 0.9996 ± 0.0000 | 0.0189 ± 0.0003 | 0.0150 | 0.0372 | 1.2378 | 3/3 |
| 512 | Additive, shared ARD | 0.9996 ± 0.0000 | 0.0204 ± 0.0006 | 0.0162 | 0.0399 | 2.3903 | 3/3 |

### Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11003 | Additive, independent | 128 | 0.99494 | 0.06885 | 0.00010 | 106 | 109 |
| 11003 | Additive, independent | 256 | 0.99908 | 0.02930 | 0.02629 | 42 | 47 |
| 11003 | Additive, independent | 512 | 0.99955 | 0.02054 | 0.02961 | 25 | 29 |
| 11003 | Plain Matérn 2.5 | 128 | 0.99690 | 0.05386 | 0.00010 | 29 | 45 |
| 11003 | Plain Matérn 2.5 | 256 | 0.99936 | 0.02452 | 0.00010 | 25 | 46 |
| 11003 | Plain Matérn 2.5 | 512 | 0.99962 | 0.01880 | 0.00010 | 28 | 48 |
| 11003 | Additive, shared ARD | 128 | 0.99459 | 0.07122 | 0.00010 | 30 | 35 |
| 11003 | Additive, shared ARD | 256 | 0.99907 | 0.02948 | 0.02620 | 14 | 15 |
| 11003 | Additive, shared ARD | 512 | 0.99958 | 0.01972 | 0.02971 | 13 | 15 |
| 12007 | Additive, independent | 128 | 0.99746 | 0.05062 | 0.00011 | 50 | 53 |
| 12007 | Additive, independent | 256 | 0.99928 | 0.02702 | 0.02442 | 39 | 44 |
| 12007 | Additive, independent | 512 | 0.99956 | 0.02116 | 0.02680 | 31 | 36 |
| 12007 | Plain Matérn 2.5 | 128 | 0.99787 | 0.04634 | 0.00011 | 27 | 38 |
| 12007 | Plain Matérn 2.5 | 256 | 0.99954 | 0.02152 | 0.00010 | 24 | 85 |
| 12007 | Plain Matérn 2.5 | 512 | 0.99966 | 0.01859 | 0.00010 | 27 | 47 |
| 12007 | Additive, shared ARD | 128 | 0.99748 | 0.05042 | 0.00011 | 30 | 35 |
| 12007 | Additive, shared ARD | 256 | 0.99928 | 0.02697 | 0.02417 | 14 | 17 |
| 12007 | Additive, shared ARD | 512 | 0.99958 | 0.02067 | 0.02673 | 13 | 15 |
| 13001 | Additive, independent | 128 | 0.99592 | 0.06352 | 0.00010 | 138 | 141 |
| 13001 | Additive, independent | 256 | 0.99924 | 0.02746 | 0.02555 | 36 | 41 |
| 13001 | Additive, independent | 512 | 0.99951 | 0.02191 | 0.02765 | 31 | 35 |
| 13001 | Plain Matérn 2.5 | 128 | 0.99701 | 0.05434 | 0.00010 | 26 | 67 |
| 13001 | Plain Matérn 2.5 | 256 | 0.99948 | 0.02275 | 0.00010 | 24 | 44 |
| 13001 | Plain Matérn 2.5 | 512 | 0.99963 | 0.01922 | 0.00010 | 27 | 45 |
| 13001 | Additive, shared ARD | 128 | 0.99636 | 0.06000 | 0.00010 | 32 | 35 |
| 13001 | Additive, shared ARD | 256 | 0.99925 | 0.02728 | 0.02577 | 15 | 18 |
| 13001 | Additive, shared ARD | 512 | 0.99956 | 0.02091 | 0.02776 | 13 | 14 |

Test RMSE decreases in 18/18 consecutive within-seed transitions.
Optimizer convergence: 27/27 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.

## dense_fourier


Dimension: 80. Seeds: 11003, 12007, 13001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | -0.1644 ± 0.0505 | 0.8459 ± 0.0319 | 0.6732 | 1.6713 | 1.9573 | 3/3 |
| 128 | Plain Matérn 2.5 | 0.2105 ± 0.0702 | 0.6958 ± 0.0207 | 0.5541 | 1.3632 | 0.0555 | 3/3 |
| 128 | Additive, shared ARD | 0.0032 ± 0.0809 | 0.7820 ± 0.0250 | 0.6216 | 1.5281 | 0.1160 | 3/3 |
| 256 | Additive, independent | 0.0907 ± 0.1412 | 0.7456 ± 0.0522 | 0.5952 | 1.4720 | 7.5494 | 3/3 |
| 256 | Plain Matérn 2.5 | 0.2851 ± 0.0539 | 0.6624 ± 0.0215 | 0.5265 | 1.3143 | 0.3045 | 3/3 |
| 256 | Additive, shared ARD | 0.2486 ± 0.0654 | 0.6789 ± 0.0241 | 0.5394 | 1.3426 | 2.0099 | 3/3 |
| 512 | Additive, independent | 0.2276 ± 0.0519 | 0.6885 ± 0.0161 | 0.5487 | 1.3613 | 24.1594 | 3/3 |
| 512 | Plain Matérn 2.5 | 0.3588 ± 0.0134 | 0.6277 ± 0.0119 | 0.4994 | 1.2463 | 0.5786 | 3/3 |
| 512 | Additive, shared ARD | 0.3200 ± 0.0164 | 0.6464 ± 0.0095 | 0.5144 | 1.2771 | 8.5426 | 3/3 |

### Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11003 | Additive, independent | 128 | -0.17989 | 0.85248 | 0.00399 | 186 | 200 |
| 11003 | Additive, independent | 256 | 0.23076 | 0.68833 | 0.01180 | 183 | 201 |
| 11003 | Additive, independent | 512 | 0.26936 | 0.67083 | 0.01719 | 147 | 161 |
| 11003 | Plain Matérn 2.5 | 128 | 0.24920 | 0.68003 | 0.02813 | 10 | 53 |
| 11003 | Plain Matérn 2.5 | 256 | 0.33999 | 0.63759 | 0.02740 | 10 | 51 |
| 11003 | Plain Matérn 2.5 | 512 | 0.37426 | 0.62081 | 0.02647 | 11 | 22 |
| 11003 | Additive, shared ARD | 128 | 0.07664 | 0.75413 | 0.02813 | 8 | 11 |
| 11003 | Additive, shared ARD | 256 | 0.31022 | 0.65181 | 0.60528 | 38 | 55 |
| 11003 | Additive, shared ARD | 512 | 0.33698 | 0.63904 | 0.57283 | 34 | 51 |
| 12007 | Additive, independent | 128 | -0.10799 | 0.81132 | 0.00259 | 161 | 176 |
| 12007 | Additive, independent | 256 | -0.05161 | 0.79041 | 0.01098 | 178 | 200 |
| 12007 | Additive, independent | 512 | 0.16949 | 0.70242 | 0.01989 | 150 | 160 |
| 12007 | Plain Matérn 2.5 | 128 | 0.12943 | 0.71916 | 0.02545 | 11 | 40 |
| 12007 | Plain Matérn 2.5 | 256 | 0.23235 | 0.67531 | 0.02387 | 9 | 52 |
| 12007 | Plain Matérn 2.5 | 512 | 0.35137 | 0.62076 | 0.02430 | 11 | 21 |
| 12007 | Additive, shared ARD | 128 | -0.08349 | 0.80230 | 0.02551 | 11 | 13 |
| 12007 | Additive, shared ARD | 256 | 0.18003 | 0.69795 | 0.53380 | 36 | 49 |
| 12007 | Additive, shared ARD | 512 | 0.30422 | 0.64292 | 0.54338 | 36 | 63 |
| 13001 | Additive, independent | 128 | -0.20547 | 0.87403 | 0.00312 | 217 | 230 |
| 13001 | Additive, independent | 256 | 0.09293 | 0.75817 | 0.00401 | 164 | 177 |
| 13001 | Additive, independent | 512 | 0.24409 | 0.69212 | 0.02661 | 114 | 131 |
| 13001 | Plain Matérn 2.5 | 128 | 0.25291 | 0.68807 | 0.02845 | 12 | 32 |
| 13001 | Plain Matérn 2.5 | 256 | 0.28283 | 0.67415 | 0.02626 | 10 | 56 |
| 13001 | Plain Matérn 2.5 | 512 | 0.35074 | 0.64144 | 0.02556 | 12 | 22 |
| 13001 | Additive, shared ARD | 128 | 0.01633 | 0.78954 | 0.02850 | 9 | 11 |
| 13001 | Additive, shared ARD | 256 | 0.25557 | 0.68685 | 0.58270 | 35 | 47 |
| 13001 | Additive, shared ARD | 512 | 0.31866 | 0.65710 | 0.57200 | 32 | 44 |

Test RMSE decreases in 18/18 consecutive within-seed transitions.
Optimizer convergence: 27/27 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.
