# Dyadic Resolution and Emissions Holdout Sweep

Five folds hold out complete UTC days across both sites, with the configured buffer on both sides. Mole-fraction targets and residuals are not scored, although observed within-hour variability contributes to the fixed uncertainty weights. Boundary-condition sensitivities are not used.

## Main blocked-fold summary

Values are median [minimum, maximum] across folds.

| block width | algorithm | target K | actual K | training DFS fraction | holdout compression | pooled holdout compression | holdout DFS fraction | DP training gap |
| ---: | --- | ---: | --- | --- | --- | --- | --- | --- |
| 8 | dyadic greedy | 16 | 16.0 [16.0, 16.0] | 0.0795 [0.0673, 0.0865] | 0.1482 [0.0773, 0.1752] | 0.1165 | 0.0892 [0.0407, 0.1103] | 0.000e+00 [0.000e+00, 1.076e-01] |
| 8 | dyadic exact DP | 16 | 16.0 [16.0, 16.0] | 0.0808 [0.0779, 0.0865] | 0.1501 [0.0965, 0.2035] | 0.1683 | 0.0992 [0.0878, 0.1103] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 8 | dyadic greedy | 31 | 31.0 [31.0, 31.0] | 0.0946 [0.0917, 0.0983] | 0.1587 [0.1154, 0.2039] | 0.1748 | 0.1062 [0.0985, 0.1207] | 0.000e+00 [0.000e+00, 1.580e-03] |
| 8 | dyadic exact DP | 31 | 31.0 [31.0, 31.0] | 0.0948 [0.0917, 0.0983] | 0.1587 [0.1154, 0.2037] | 0.1747 | 0.1062 [0.0985, 0.1207] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 8 | dyadic greedy | 64 | 64.0 [64.0, 64.0] | 0.0995 [0.0931, 0.1034] | 0.1593 [0.1173, 0.2040] | 0.1754 | 0.1121 [0.0998, 0.1212] | 5.059e-04 [1.429e-04, 2.450e-03] |
| 8 | dyadic exact DP | 64 | 64.0 [64.0, 64.0] | 0.0996 [0.0931, 0.1035] | 0.1597 [0.1173, 0.2040] | 0.1752 | 0.1121 [0.0985, 0.1224] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 8 | dyadic greedy | 250 | 250.0 [250.0, 250.0] | 0.1003 [0.0932, 0.1045] | 0.1606 [0.1175, 0.2040] | 0.1756 | 0.1126 [0.1001, 0.1247] | 2.411e-05 [1.414e-05, 2.736e-05] |
| 8 | dyadic exact DP | 250 | 250.0 [250.0, 250.0] | 0.1003 [0.0932, 0.1045] | 0.1606 [0.1175, 0.2040] | 0.1756 | 0.1126 [0.1001, 0.1247] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 4 | dyadic greedy | 16 | 16.0 [16.0, 16.0] | 0.1172 [0.0855, 0.1377] | 0.0612 [0.0217, 0.2782] | 0.1053 | 0.0893 [0.0135, 0.2086] | 0.000e+00 [0.000e+00, 4.840e-01] |
| 4 | dyadic exact DP | 16 | 16.0 [16.0, 16.0] | 0.1280 [0.1172, 0.1399] | 0.0324 [0.0217, 0.0779] | 0.0415 | 0.0286 [0.0135, 0.1011] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 4 | dyadic greedy | 31 | 31.0 [31.0, 31.0] | 0.2073 [0.1878, 0.2292] | 0.3382 [0.1954, 0.4954] | 0.3778 | 0.2380 [0.1917, 0.2828] | 3.354e-03 [0.000e+00, 6.363e-02] |
| 4 | dyadic exact DP | 31 | 31.0 [31.0, 31.0] | 0.2145 [0.1961, 0.2292] | 0.3382 [0.1954, 0.4197] | 0.3485 | 0.2254 [0.1917, 0.2630] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 4 | dyadic greedy | 64 | 64.0 [64.0, 64.0] | 0.2413 [0.2356, 0.2554] | 0.3679 [0.2937, 0.4969] | 0.4076 | 0.2758 [0.2411, 0.2889] | 1.409e-03 [1.104e-03, 5.620e-03] |
| 4 | dyadic exact DP | 64 | 64.0 [64.0, 64.0] | 0.2414 [0.2358, 0.2555] | 0.3693 [0.2935, 0.4969] | 0.4079 | 0.2758 [0.2409, 0.2889] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 4 | axis-parallel (no mask) | 64 | 64.0 [64.0, 64.0] | 0.2389 [0.2367, 0.2567] | 0.3692 [0.2917, 0.4961] | 0.4073 | 0.2756 [0.2413, 0.2859] | - |
| 4 | quadtree (no mask) | 64 | 64.0 [61.0, 64.0] | 0.2203 [0.2102, 0.2364] | 0.3506 [0.2533, 0.4906] | 0.3918 | 0.2472 [0.2255, 0.2814] | - |
| 4 | dyadic greedy | 250 | 250.0 [250.0, 250.0] | 0.2478 [0.2427, 0.2621] | 0.3709 [0.2977, 0.5004] | 0.4112 | 0.2823 [0.2493, 0.2932] | - |
| 4 | axis-parallel (no mask) | 250 | 250.0 [250.0, 250.0] | 0.2479 [0.2427, 0.2621] | 0.3709 [0.2976, 0.5003] | 0.4112 | 0.2828 [0.2493, 0.2932] | - |
| 4 | quadtree (no mask) | 250 | 253.0 [250.0, 253.0] | 0.2271 [0.2195, 0.2422] | 0.3521 [0.2560, 0.4932] | 0.3941 | 0.2528 [0.2336, 0.2847] | - |
| 2 | dyadic greedy | 16 | 16.0 [16.0, 16.0] | 0.1357 [0.0829, 0.1474] | 0.0186 [0.0170, 0.0782] | 0.0380 | 0.0151 [0.0089, 0.1099] | 0.000e+00 [0.000e+00, 1.185e-01] |
| 2 | dyadic exact DP | 16 | 16.0 [16.0, 16.0] | 0.1357 [0.0971, 0.1474] | 0.0186 [0.0129, 0.0782] | 0.0363 | 0.0151 [0.0127, 0.1099] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 2 | dyadic greedy | 31 | 31.0 [31.0, 31.0] | 0.2540 [0.2017, 0.2640] | 0.1886 [0.0415, 0.3897] | 0.2563 | 0.1975 [0.0558, 0.2755] | 1.969e-01 [1.669e-01, 4.334e-01] |
| 2 | dyadic exact DP | 31 | 31.0 [31.0, 31.0] | 0.2787 [0.2462, 0.2990] | 0.0488 [0.0199, 0.1164] | 0.0594 | 0.0685 [0.0164, 0.1770] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 2 | dyadic greedy | 64 | 64.0 [64.0, 64.0] | 0.4368 [0.4318, 0.4565] | 0.5540 [0.3317, 0.5719] | 0.5309 | 0.4602 [0.2586, 0.4856] | 8.979e-02 [7.892e-02, 1.405e-01] |
| 2 | dyadic exact DP | 64 | 64.0 [64.0, 64.0] | 0.4502 [0.4464, 0.4663] | 0.5521 [0.3790, 0.5683] | 0.5319 | 0.4363 [0.3052, 0.4948] | 0.000e+00 [0.000e+00, 0.000e+00] |
| 2 | dyadic greedy | 250 | 250.0 [250.0, 250.0] | 0.5171 [0.5019, 0.5332] | 0.6006 [0.5760, 0.6744] | 0.6276 | 0.5246 [0.4854, 0.5427] | - |

## Coarsening ceiling

| block width | search shape | training all-leaf DFS fraction | unresolved training DFS | holdout all-leaf compression | pooled holdout compression | top cell share | top 10 share |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 8 | (37, 49) | 0.1003 [0.0932, 0.1045] | 8.2283 [6.2797, 9.6863] | 0.1606 [0.1175, 0.2040] | 0.1756 | 0.0467 [0.0382, 0.0711] | 0.3105 [0.2998, 0.3850] |
| 4 | (74, 98) | 0.2482 [0.2429, 0.2624] | 6.7548 [5.2720, 8.0450] | 0.3709 [0.2978, 0.5004] | 0.4113 | 0.0467 [0.0382, 0.0711] | 0.3105 [0.2998, 0.3850] |
| 2 | (147, 196) | 0.5227 [0.5089, 0.5386] | 4.2253 [3.3237, 5.0990] | 0.6053 [0.5801, 0.7094] | 0.6440 | 0.0467 [0.0382, 0.0711] | 0.3105 [0.2998, 0.3850] |

## Training-thinning phase sensitivity

Values are median [minimum, maximum] across wall-clock phase offsets; the hourly holdout is unchanged.

| algorithm | target K | completed phases | training DFS fraction | holdout compression | holdout DFS fraction |
| --- | ---: | ---: | --- | --- | --- |
| dyadic greedy | 64 | 6 | 0.3557 [0.3370, 0.3781] | 0.3693 [0.3679, 0.3697] | 0.2863 [0.2839, 0.2869] |
| dyadic exact DP | 64 | 6 | 0.3557 [0.3371, 0.3783] | 0.3686 [0.3676, 0.3693] | 0.2851 [0.2835, 0.2863] |
| dyadic greedy | 250 | 6 | 0.3592 [0.3410, 0.3822] | 0.3708 [0.3708, 0.3708] | 0.2889 [0.2889, 0.2889] |

## Interpretation boundaries

- Exact DP is the fixed-K oracle only for the additive projected dyadic objective. Each frontier is computed once per fold/width or thinning phase.
- Axis-parallel and quadtree candidates have no spatial mask and use coarsened training native-cell DFS as their construction field. Their construction objective differs from the dyadic objective.
- Training thinning tests phase sensitivity to fewer closely spaced rows. It does not simulate temporal correlation or replace a non-diagonal likelihood.
- 16 configured candidate rows were explicitly omitted by the fine-grid DP limit; their metric fields are blank in the candidate CSV.
- Configuration, fixture/source hashes, and timing scopes are recorded in `emissions_holdout_sweep_manifest.json` and the CSVs.
