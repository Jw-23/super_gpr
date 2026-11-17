# Additive GP improvement results

648 fits on 27 fresh datasets: three training folds and one full refit per candidate.

Model selection uses only training-fold RMSE. Test labels are used after the winner is locked.

R² values are mean ± sample standard deviation across seeds; test targets are noiseless.

| Function | Dimension | Samples | independent | shared_iso | shared_ard | map_0.5 | map_1.0 | map_2.0 | cv_selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| additive | 4 | 128 | 0.988 ± 0.002 | 0.988 ± 0.002 | 0.994 ± 0.001 | 0.988 ± 0.003 | 0.989 ± 0.002 | 0.989 ± 0.002 | 0.994 ± 0.001 |
| additive | 4 | 256 | 0.993 ± 0.002 | 0.993 ± 0.002 | 0.997 ± 0.000 | 0.993 ± 0.001 | 0.993 ± 0.002 | 0.993 ± 0.002 | 0.997 ± 0.000 |
| additive | 20 | 128 | 0.734 ± 0.044 | 0.754 ± 0.048 | 0.920 ± 0.010 | 0.756 ± 0.019 | 0.748 ± 0.028 | 0.743 ± 0.033 | 0.920 ± 0.010 |
| additive | 20 | 256 | 0.947 ± 0.010 | 0.937 ± 0.010 | 0.988 ± 0.001 | 0.942 ± 0.007 | 0.945 ± 0.009 | 0.947 ± 0.010 | 0.988 ± 0.001 |
| additive | 80 | 128 | 0.129 ± 0.036 | 0.369 ± 0.040 | 0.518 ± 0.036 | 0.376 ± 0.053 | 0.324 ± 0.045 | 0.251 ± 0.051 | 0.518 ± 0.036 |
| additive | 80 | 256 | 0.469 ± 0.023 | 0.540 ± 0.041 | 0.709 ± 0.020 | 0.542 ± 0.042 | 0.514 ± 0.030 | 0.492 ± 0.023 | 0.709 ± 0.020 |
| heterogeneous | 80 | 128 | -0.169 ± 0.093 | 0.051 ± 0.042 | 0.184 ± 0.087 | 0.036 ± 0.036 | 0.025 ± 0.043 | -0.025 ± 0.053 | 0.148 ± 0.117 |
| heterogeneous | 80 | 256 | 0.018 ± 0.017 | 0.145 ± 0.042 | 0.413 ± 0.023 | 0.156 ± 0.034 | 0.129 ± 0.023 | 0.082 ± 0.017 | 0.413 ± 0.023 |
| interaction | 80 | 256 | 0.130 ± 0.104 | 0.234 ± 0.161 | 0.226 ± 0.158 | 0.136 ± 0.056 | 0.146 ± 0.063 | 0.143 ± 0.076 | 0.242 ± 0.148 |

## Fit and generalization diagnostics

Effective degrees of freedom: trace(K_signal (K_signal + noise I)^-1).
Noise standard deviations are restored to the original target scale; injected noise is 0.03.

| Function | D | N | Candidate | Test RMSE | Train RMSE | Noise std | Effective df | Refit seconds | Converged |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| additive | 4 | 128 | cv_selected | 0.0365 | 0.0229 | 0.0312 | 59.5558 | 0.0124 | 3/3 |
| additive | 4 | 128 | independent | 0.0515 | 0.0207 | 0.0332 | 78.3326 | 0.0095 | 3/3 |
| additive | 4 | 128 | map_0.5 | 0.0515 | 0.0189 | 0.0315 | 83.1698 | 0.0095 | 3/3 |
| additive | 4 | 128 | map_1.0 | 0.0511 | 0.0198 | 0.0321 | 80.7009 | 0.0093 | 3/3 |
| additive | 4 | 128 | map_2.0 | 0.0512 | 0.0203 | 0.0324 | 79.5109 | 0.0094 | 3/3 |
| additive | 4 | 128 | shared_ard | 0.0365 | 0.0229 | 0.0312 | 59.5558 | 0.0124 | 3/3 |
| additive | 4 | 128 | shared_iso | 0.0515 | 0.0207 | 0.0332 | 78.3326 | 0.0121 | 3/3 |
| additive | 4 | 256 | cv_selected | 0.0250 | 0.0252 | 0.0305 | 81.2160 | 0.0560 | 3/3 |
| additive | 4 | 256 | independent | 0.0394 | 0.0220 | 0.0298 | 116.8893 | 0.0379 | 3/3 |
| additive | 4 | 256 | map_0.5 | 0.0389 | 0.0217 | 0.0295 | 118.7804 | 0.0363 | 3/3 |
| additive | 4 | 256 | map_1.0 | 0.0392 | 0.0219 | 0.0296 | 117.6822 | 0.0371 | 3/3 |
| additive | 4 | 256 | map_2.0 | 0.0394 | 0.0219 | 0.0296 | 117.3120 | 0.0384 | 3/3 |
| additive | 4 | 256 | shared_ard | 0.0250 | 0.0252 | 0.0305 | 81.2160 | 0.0560 | 3/3 |
| additive | 4 | 256 | shared_iso | 0.0394 | 0.0220 | 0.0298 | 116.8893 | 0.0472 | 3/3 |
| additive | 20 | 128 | cv_selected | 0.2382 | 0.0000 | 0.0001 | 128.0000 | 0.1031 | 3/3 |
| additive | 20 | 128 | independent | 0.4329 | 0.0312 | 0.0635 | 117.6235 | 0.1350 | 3/3 |
| additive | 20 | 128 | map_0.5 | 0.4152 | 0.0020 | 0.0268 | 127.2981 | 0.0624 | 3/3 |
| additive | 20 | 128 | map_1.0 | 0.4223 | 0.0021 | 0.0271 | 127.2418 | 0.0712 | 3/3 |
| additive | 20 | 128 | map_2.0 | 0.4259 | 0.0022 | 0.0272 | 127.2190 | 0.0826 | 3/3 |
| additive | 20 | 128 | shared_ard | 0.2382 | 0.0000 | 0.0001 | 128.0000 | 0.1031 | 3/3 |
| additive | 20 | 128 | shared_iso | 0.4164 | 0.0293 | 0.0646 | 119.1875 | 0.0813 | 3/3 |
| additive | 20 | 256 | cv_selected | 0.0907 | 0.0100 | 0.0314 | 230.0760 | 0.2764 | 3/3 |
| additive | 20 | 256 | independent | 0.1924 | 0.0000 | 0.0002 | 255.9995 | 0.7350 | 3/3 |
| additive | 20 | 256 | map_0.5 | 0.2013 | 0.0021 | 0.0213 | 253.1553 | 0.4215 | 3/3 |
| additive | 20 | 256 | map_1.0 | 0.1961 | 0.0021 | 0.0211 | 253.0655 | 0.4319 | 3/3 |
| additive | 20 | 256 | map_2.0 | 0.1938 | 0.0021 | 0.0209 | 253.0360 | 0.4571 | 3/3 |
| additive | 20 | 256 | shared_ard | 0.0907 | 0.0100 | 0.0314 | 230.0760 | 0.2764 | 3/3 |
| additive | 20 | 256 | shared_iso | 0.2100 | 0.0000 | 0.0002 | 255.9998 | 0.3417 | 3/3 |
| additive | 80 | 128 | cv_selected | 0.5845 | 0.0515 | 0.1140 | 115.6667 | 0.2594 | 3/3 |
| additive | 80 | 128 | independent | 0.7865 | 0.0000 | 0.0021 | 127.9969 | 1.8944 | 3/3 |
| additive | 80 | 128 | map_0.5 | 0.6648 | 0.0011 | 0.0272 | 127.7878 | 0.1418 | 3/3 |
| additive | 80 | 128 | map_1.0 | 0.6925 | 0.0011 | 0.0269 | 127.7517 | 0.2217 | 3/3 |
| additive | 80 | 128 | map_2.0 | 0.7285 | 0.0012 | 0.0267 | 127.7129 | 0.4052 | 3/3 |
| additive | 80 | 128 | shared_ard | 0.5845 | 0.0515 | 0.1140 | 115.6667 | 0.2594 | 3/3 |
| additive | 80 | 128 | shared_iso | 0.6690 | 0.0729 | 0.1476 | 114.4061 | 0.1666 | 3/3 |
| additive | 80 | 256 | cv_selected | 0.4542 | 0.0506 | 0.1057 | 228.1229 | 0.8155 | 3/3 |
| additive | 80 | 256 | independent | 0.6138 | 0.0000 | 0.0027 | 255.9886 | 8.1008 | 3/3 |
| additive | 80 | 256 | map_0.5 | 0.5696 | 0.0012 | 0.0264 | 255.3571 | 0.8557 | 3/3 |
| additive | 80 | 256 | map_1.0 | 0.5870 | 0.0013 | 0.0264 | 255.2856 | 1.2042 | 3/3 |
| additive | 80 | 256 | map_2.0 | 0.5999 | 0.0014 | 0.0265 | 255.2473 | 1.9525 | 3/3 |
| additive | 80 | 256 | shared_ard | 0.4542 | 0.0506 | 0.1057 | 228.1229 | 0.8155 | 3/3 |
| additive | 80 | 256 | shared_iso | 0.5709 | 0.0014 | 0.0276 | 255.3687 | 0.3978 | 3/3 |
| heterogeneous | 80 | 128 | cv_selected | 0.8220 | 0.0009 | 0.0260 | 127.8324 | 0.2282 | 3/3 |
| heterogeneous | 80 | 128 | independent | 0.9644 | 0.0000 | 0.0016 | 127.9987 | 1.8086 | 3/3 |
| heterogeneous | 80 | 128 | map_0.5 | 0.8756 | 0.0010 | 0.0265 | 127.8339 | 0.1307 | 3/3 |
| heterogeneous | 80 | 128 | map_1.0 | 0.8807 | 0.0009 | 0.0262 | 127.8401 | 0.1735 | 3/3 |
| heterogeneous | 80 | 128 | map_2.0 | 0.9028 | 0.0009 | 0.0260 | 127.8321 | 0.3496 | 3/3 |
| heterogeneous | 80 | 128 | shared_ard | 0.8051 | 0.0010 | 0.0260 | 127.8159 | 0.2688 | 3/3 |
| heterogeneous | 80 | 128 | shared_iso | 0.8690 | 0.0009 | 0.0262 | 127.8635 | 0.1072 | 3/3 |
| heterogeneous | 80 | 256 | cv_selected | 0.6822 | 0.0008 | 0.0177 | 255.6724 | 0.9623 | 3/3 |
| heterogeneous | 80 | 256 | independent | 0.8823 | 0.0001 | 0.0063 | 255.9777 | 8.7242 | 3/3 |
| heterogeneous | 80 | 256 | map_0.5 | 0.8184 | 0.0011 | 0.0279 | 255.6459 | 0.8458 | 3/3 |
| heterogeneous | 80 | 256 | map_1.0 | 0.8310 | 0.0010 | 0.0275 | 255.6587 | 0.9667 | 3/3 |
| heterogeneous | 80 | 256 | map_2.0 | 0.8532 | 0.0010 | 0.0274 | 255.6506 | 1.6740 | 3/3 |
| heterogeneous | 80 | 256 | shared_ard | 0.6822 | 0.0008 | 0.0177 | 255.6724 | 0.9623 | 3/3 |
| heterogeneous | 80 | 256 | shared_iso | 0.8235 | 0.0009 | 0.0279 | 255.7066 | 0.4658 | 3/3 |
| interaction | 80 | 256 | cv_selected | 0.7359 | 0.3255 | 0.3917 | 135.4460 | 1.1236 | 3/3 |
| interaction | 80 | 256 | independent | 0.7898 | 0.0001 | 0.0070 | 255.9618 | 6.8635 | 3/3 |
| interaction | 80 | 256 | map_0.5 | 0.7880 | 0.0012 | 0.0270 | 255.5406 | 0.6559 | 3/3 |
| interaction | 80 | 256 | map_1.0 | 0.7832 | 0.0011 | 0.0266 | 255.5355 | 1.0659 | 3/3 |
| interaction | 80 | 256 | map_2.0 | 0.7845 | 0.0011 | 0.0265 | 255.5231 | 1.5052 | 3/3 |
| interaction | 80 | 256 | shared_ard | 0.7433 | 0.3291 | 0.3936 | 133.9552 | 1.5449 | 3/3 |
| interaction | 80 | 256 | shared_iso | 0.7391 | 0.3255 | 0.3917 | 135.4712 | 1.0498 | 3/3 |

## Training-only selections

| Function | D | N | Seed | Selected | CV RMSE | CV fitting seconds |
| --- | --- | --- | --- | --- | --- | --- |
| additive | 4 | 128 | 1001 | shared_ard | 0.0540 | 0.130 |
| additive | 4 | 128 | 2003 | shared_ard | 0.0620 | 0.091 |
| additive | 4 | 128 | 3007 | shared_ard | 0.0647 | 0.088 |
| additive | 4 | 256 | 1001 | shared_ard | 0.0413 | 0.323 |
| additive | 4 | 256 | 2003 | shared_ard | 0.0460 | 0.326 |
| additive | 4 | 256 | 3007 | shared_ard | 0.0390 | 0.339 |
| additive | 20 | 128 | 1001 | shared_ard | 0.3995 | 0.635 |
| additive | 20 | 128 | 2003 | shared_ard | 0.3581 | 0.618 |
| additive | 20 | 128 | 3007 | shared_ard | 0.3327 | 0.672 |
| additive | 20 | 256 | 1001 | shared_ard | 0.1374 | 3.148 |
| additive | 20 | 256 | 2003 | shared_ard | 0.1745 | 2.954 |
| additive | 20 | 256 | 3007 | shared_ard | 0.1460 | 3.116 |
| additive | 80 | 128 | 1001 | shared_ard | 0.6788 | 3.431 |
| additive | 80 | 128 | 2003 | shared_ard | 0.7112 | 4.273 |
| additive | 80 | 128 | 3007 | shared_ard | 0.6830 | 4.950 |
| additive | 80 | 256 | 1001 | shared_ard | 0.5245 | 14.743 |
| additive | 80 | 256 | 2003 | shared_ard | 0.5831 | 17.250 |
| additive | 80 | 256 | 3007 | shared_ard | 0.5440 | 16.994 |
| heterogeneous | 80 | 128 | 1001 | shared_iso | 0.7988 | 3.430 |
| heterogeneous | 80 | 128 | 2003 | shared_ard | 0.9711 | 3.098 |
| heterogeneous | 80 | 128 | 3007 | shared_ard | 0.7167 | 3.140 |
| heterogeneous | 80 | 256 | 1001 | shared_ard | 0.6685 | 14.717 |
| heterogeneous | 80 | 256 | 2003 | shared_ard | 0.7448 | 16.520 |
| heterogeneous | 80 | 256 | 3007 | shared_ard | 0.6803 | 14.047 |
| interaction | 80 | 256 | 1001 | shared_iso | 0.7532 | 18.062 |
| interaction | 80 | 256 | 2003 | map_0.5 | 0.7466 | 15.928 |
| interaction | 80 | 256 | 3007 | shared_iso | 0.7166 | 17.236 |

Optimizer convergence: 648/648 fits.
All finite results are retained, including optimizer failures or iteration limits.
Refit timings exclude the training-only CV selection cost listed separately.
