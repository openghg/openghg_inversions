# Synthetic TAC/MHD Basis Diagnostics

## Scope

This is a controlled analytic Gaussian inversion, not a fit to the stored real observations. The real-data consistency gate fails because the frozen prior emissions plus boundary contribution leave residuals far larger than the supplied errors. Synthetic observations use the same emissions and boundary sensitivity matrices, so the baseline is explicit and internally consistent.

The search block width is 8 native cells along each spatial axis. That is grid coarsening; it is unrelated to the up-to-eightfold storage bound for a fully precomputed space-time multiscale Jacobian.

## Primary smooth-truth result

Assumed covariance: `observation_plus_5ppb`. Bases use training rows only.

| Basis | K effective | Train DFS | Holdout compression | Known-base emissions RMSE | Joint emissions RMSE | Boundary RMSE | Total RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dyadic_greedy | 31 | 0.25222 | 0.088467 | 6.5811 | 7.0621 | 42.045 | 48.777 |
| dyadic_sls_p50 | 31 | 0.25222 | 0.088467 | 6.5811 | 7.0621 | 42.045 | 48.777 |
| dyadic_sls_p10 | 31 | 0.25222 | 0.088467 | 6.5811 | 7.0621 | 42.045 | 48.777 |
| dyadic_exact_dp | 31 | 0.25223 | 0.088727 | 6.5801 | 7.0622 | 42.045 | 48.777 |
| existing_quadtree | 31 | 0.1708 | 0.07119 | 6.6923 | 7.0866 | 42.045 | 48.791 |
| native_no_reduction | 110758 | - | 1 | 5.4885 | 7.3057 | 42.045 | 49.075 |

## Interpretation boundaries

- Exact DP is the emissions-only, known-baseline Gaussian oracle; it is not jointly optimal for uncertain boundary coefficients.
- Quadtree uses its existing cellwise precision-weighted proxy. Post-construction DFS and RMSE are comparable, but construction objectives and partition dictionaries both differ.
- Compression quality is emissions-only and does not use the synthetic baseline.
- The holdout uses 8 boundary directions; 8 are absent from training. Those directions remain at their prior mean, so boundary and total RMSE diagnose baseline extrapolation rather than basis quality.
- Held-out posterior means predict retained emissions and boundary components. They do not conditionally predict the unresolved fine-grid aggregation residual.
- Covariance sensitivity rows are in the CSV; the percentile floor is recomputed from training rows only.
  It is 9.48 ppb for MHD and 4.21 ppb for TAC in this split, so it remains a diagnostic rather than a recommended error model.

Artifacts: `synthetic_basis_comparison.csv`, `synthetic_basis_comparison.png`, and `synthetic_basis_comparison_manifest.json`.
