# Training sample learning curve

Dimension: 80. Seeds: 5003, 6007, 7001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Additive, independent | 0.2328 ± 0.0354 | 0.7288 ± 0.0170 | 0.5854 | 1.4190 | 2.1584 | 3/3 |
| 128 | Plain Matérn 2.5 | 0.2807 ± 0.0150 | 0.7058 ± 0.0018 | 0.5618 | 1.3917 | 0.0736 | 3/3 |
| 128 | Additive, shared ARD | 0.5140 ± 0.0407 | 0.5796 ± 0.0174 | 0.4610 | 1.1285 | 0.3664 | 3/3 |
| 256 | Additive, independent | 0.4346 ± 0.0563 | 0.6252 ± 0.0319 | 0.5006 | 1.2129 | 8.1335 | 3/3 |
| 256 | Plain Matérn 2.5 | 0.3635 ± 0.0192 | 0.6639 ± 0.0111 | 0.5274 | 1.3051 | 0.2446 | 3/3 |
| 256 | Additive, shared ARD | 0.6823 ± 0.0478 | 0.4680 ± 0.0312 | 0.3736 | 0.9218 | 1.6782 | 3/3 |
| 512 | Additive, independent | 0.7458 ± 0.0139 | 0.4195 ± 0.0138 | 0.3328 | 0.8395 | 21.7048 | 3/3 |
| 512 | Plain Matérn 2.5 | 0.4477 ± 0.0204 | 0.6185 ± 0.0181 | 0.4931 | 1.1986 | 0.5686 | 3/3 |
| 512 | Additive, shared ARD | 0.9279 ± 0.0068 | 0.2233 ± 0.0108 | 0.1770 | 0.4377 | 6.4827 | 3/3 |
| 1024 | Additive, independent | 0.9482 ± 0.0029 | 0.1894 ± 0.0031 | 0.1505 | 0.3723 | 110.1911 | 3/3 |
| 1024 | Plain Matérn 2.5 | 0.5002 ± 0.0228 | 0.5883 ± 0.0192 | 0.4694 | 1.1406 | 2.5608 | 3/3 |
| 1024 | Additive, shared ARD | 0.9864 ± 0.0004 | 0.0969 ± 0.0023 | 0.0775 | 0.1892 | 16.9011 | 3/3 |

## Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5003 | Additive, independent | 128 | 0.27362 | 0.71178 | 0.00220 | 185 | 193 |
| 5003 | Additive, independent | 256 | 0.49894 | 0.59117 | 0.00444 | 191 | 204 |
| 5003 | Additive, independent | 512 | 0.75986 | 0.40925 | 0.00087 | 171 | 179 |
| 5003 | Additive, independent | 1024 | 0.94836 | 0.18979 | 0.00040 | 142 | 151 |
| 5003 | Plain Matérn 2.5 | 128 | 0.28675 | 0.70532 | 0.02777 | 10 | 53 |
| 5003 | Plain Matérn 2.5 | 256 | 0.38565 | 0.65459 | 0.02780 | 10 | 50 |
| 5003 | Plain Matérn 2.5 | 512 | 0.42913 | 0.63101 | 0.02597 | 11 | 22 |
| 5003 | Plain Matérn 2.5 | 1024 | 0.47746 | 0.60371 | 0.02622 | 10 | 19 |
| 5003 | Additive, shared ARD | 128 | 0.54223 | 0.56505 | 0.44161 | 36 | 55 |
| 5003 | Additive, shared ARD | 256 | 0.72863 | 0.43506 | 0.00032 | 33 | 35 |
| 5003 | Additive, shared ARD | 512 | 0.93565 | 0.21185 | 0.00008 | 38 | 45 |
| 5003 | Additive, shared ARD | 1024 | 0.98676 | 0.09608 | 0.02513 | 22 | 26 |
| 6007 | Additive, independent | 128 | 0.21125 | 0.72885 | 0.00121 | 181 | 195 |
| 6007 | Additive, independent | 256 | 0.41075 | 0.62996 | 0.00234 | 184 | 193 |
| 6007 | Additive, independent | 512 | 0.74538 | 0.41411 | 0.00240 | 113 | 122 |
| 6007 | Additive, independent | 1024 | 0.94517 | 0.19216 | 0.00027 | 150 | 159 |
| 6007 | Plain Matérn 2.5 | 128 | 0.26354 | 0.70427 | 0.02477 | 9 | 51 |
| 6007 | Plain Matérn 2.5 | 256 | 0.35164 | 0.66080 | 0.02519 | 10 | 53 |
| 6007 | Plain Matérn 2.5 | 512 | 0.46960 | 0.59768 | 0.02701 | 11 | 22 |
| 6007 | Plain Matérn 2.5 | 1024 | 0.52301 | 0.56678 | 0.02676 | 10 | 19 |
| 6007 | Additive, shared ARD | 128 | 0.46737 | 0.59893 | 0.02483 | 16 | 19 |
| 6007 | Additive, shared ARD | 256 | 0.63311 | 0.49709 | 0.30898 | 43 | 55 |
| 6007 | Additive, shared ARD | 512 | 0.92513 | 0.22455 | 0.00011 | 36 | 42 |
| 6007 | Additive, shared ARD | 1024 | 0.98657 | 0.09511 | 0.03039 | 21 | 23 |
| 7001 | Additive, independent | 128 | 0.21344 | 0.74576 | 0.00343 | 239 | 253 |
| 7001 | Additive, independent | 256 | 0.39418 | 0.65449 | 0.00097 | 202 | 217 |
| 7001 | Additive, independent | 512 | 0.73212 | 0.43521 | 0.00854 | 92 | 100 |
| 7001 | Additive, independent | 1024 | 0.95102 | 0.18610 | 0.00011 | 153 | 167 |
| 7001 | Plain Matérn 2.5 | 128 | 0.29167 | 0.70771 | 0.02607 | 11 | 56 |
| 7001 | Plain Matérn 2.5 | 256 | 0.35330 | 0.67622 | 0.02737 | 10 | 24 |
| 7001 | Plain Matérn 2.5 | 512 | 0.44451 | 0.62672 | 0.02599 | 10 | 20 |
| 7001 | Plain Matérn 2.5 | 1024 | 0.50025 | 0.59445 | 0.02596 | 10 | 19 |
| 7001 | Additive, shared ARD | 128 | 0.53253 | 0.57493 | 0.00015 | 37 | 39 |
| 7001 | Additive, shared ARD | 256 | 0.68519 | 0.47180 | 0.22039 | 35 | 39 |
| 7001 | Additive, shared ARD | 512 | 0.92297 | 0.23338 | 0.00010 | 32 | 35 |
| 7001 | Additive, shared ARD | 1024 | 0.98598 | 0.09956 | 0.02160 | 23 | 27 |

Test RMSE decreases in 27/27 consecutive within-seed transitions.
Optimizer convergence: 36/36 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.
