# Training sample learning curve

Dimension: 80. Seeds: 5003, 6007, 7001. Test points per seed: 2048.

Training sets are nested prefixes of one noisy pool. Test inputs and noiseless targets stay fixed for each seed.
Every fit uses the same cold-start settings; test labels do not select hyperparameters or model structure.

Values are mean ± sample standard deviation across seeds.

| Training samples | Model | Test R² | Test RMSE | MAE | 95th percentile absolute error | Fit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | Independent | 0.2328 ± 0.0354 | 0.7288 ± 0.0170 | 0.5854 | 1.4190 | 2.1584 | 3/3 |
| 128 | Shared ARD | 0.5140 ± 0.0407 | 0.5796 ± 0.0174 | 0.4610 | 1.1285 | 0.3664 | 3/3 |
| 256 | Independent | 0.4346 ± 0.0563 | 0.6252 ± 0.0319 | 0.5006 | 1.2129 | 8.1335 | 3/3 |
| 256 | Shared ARD | 0.6823 ± 0.0478 | 0.4680 ± 0.0312 | 0.3736 | 0.9218 | 1.6782 | 3/3 |
| 512 | Independent | 0.7458 ± 0.0139 | 0.4195 ± 0.0138 | 0.3328 | 0.8395 | 21.7048 | 3/3 |
| 512 | Shared ARD | 0.9279 ± 0.0068 | 0.2233 ± 0.0108 | 0.1770 | 0.4377 | 6.4827 | 3/3 |
| 1024 | Independent | 0.9482 ± 0.0029 | 0.1894 ± 0.0031 | 0.1505 | 0.3723 | 110.1911 | 3/3 |
| 1024 | Shared ARD | 0.9864 ± 0.0004 | 0.0969 ± 0.0023 | 0.0775 | 0.1892 | 16.9011 | 3/3 |

## Each seed

| Seed | Model | Training samples | R² | RMSE | Noise std | Iterations | Evaluations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5003 | Independent | 128 | 0.27362 | 0.71178 | 0.00220 | 185 | 193 |
| 5003 | Independent | 256 | 0.49894 | 0.59117 | 0.00444 | 191 | 204 |
| 5003 | Independent | 512 | 0.75986 | 0.40925 | 0.00087 | 171 | 179 |
| 5003 | Independent | 1024 | 0.94836 | 0.18979 | 0.00040 | 142 | 151 |
| 5003 | Shared ARD | 128 | 0.54223 | 0.56505 | 0.44161 | 36 | 55 |
| 5003 | Shared ARD | 256 | 0.72863 | 0.43506 | 0.00032 | 33 | 35 |
| 5003 | Shared ARD | 512 | 0.93565 | 0.21185 | 0.00008 | 38 | 45 |
| 5003 | Shared ARD | 1024 | 0.98676 | 0.09608 | 0.02513 | 22 | 26 |
| 6007 | Independent | 128 | 0.21125 | 0.72885 | 0.00121 | 181 | 195 |
| 6007 | Independent | 256 | 0.41075 | 0.62996 | 0.00234 | 184 | 193 |
| 6007 | Independent | 512 | 0.74538 | 0.41411 | 0.00240 | 113 | 122 |
| 6007 | Independent | 1024 | 0.94517 | 0.19216 | 0.00027 | 150 | 159 |
| 6007 | Shared ARD | 128 | 0.46737 | 0.59893 | 0.02483 | 16 | 19 |
| 6007 | Shared ARD | 256 | 0.63311 | 0.49709 | 0.30898 | 43 | 55 |
| 6007 | Shared ARD | 512 | 0.92513 | 0.22455 | 0.00011 | 36 | 42 |
| 6007 | Shared ARD | 1024 | 0.98657 | 0.09511 | 0.03039 | 21 | 23 |
| 7001 | Independent | 128 | 0.21344 | 0.74576 | 0.00343 | 239 | 253 |
| 7001 | Independent | 256 | 0.39418 | 0.65449 | 0.00097 | 202 | 217 |
| 7001 | Independent | 512 | 0.73212 | 0.43521 | 0.00854 | 92 | 100 |
| 7001 | Independent | 1024 | 0.95102 | 0.18610 | 0.00011 | 153 | 167 |
| 7001 | Shared ARD | 128 | 0.53253 | 0.57493 | 0.00015 | 37 | 39 |
| 7001 | Shared ARD | 256 | 0.68519 | 0.47180 | 0.22039 | 35 | 39 |
| 7001 | Shared ARD | 512 | 0.92297 | 0.23338 | 0.00010 | 32 | 35 |
| 7001 | Shared ARD | 1024 | 0.98598 | 0.09956 | 0.02160 | 23 | 27 |

Test RMSE decreases in 18/18 consecutive within-seed transitions.
Optimizer convergence: 24/24 fits.
All results are retained. More data does not impose monotonic test accuracy after hyperparameter reoptimization.
