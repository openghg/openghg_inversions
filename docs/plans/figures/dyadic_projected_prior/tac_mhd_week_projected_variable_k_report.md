# Projection-Consistent Dyadic SLS Demonstration

## Result

- Starting point: greedy exact-DFS dyadic initializer with K=24.
- Best-utility state: K=28 after 2000 evaluated local moves.
- DFS at initializer/best-utility state: 0.478183 to 0.479182.
- Native-grid no-reduction DFS: 3.56372.
- Best-utility state reaches 13.45% of the bound.

## Inputs and score

The displayed background is the precision-weighted magnitude of the
coarsened footprint-times-prior-flux columns. It is context only; the
search score uses the full Gaussian observation-space covariance.
Regional columns are summed RHIME columns. Regional prior variance is
`relative_prior_sd**2 / native_support`, and unresolved fine-grid
variation is included in the effective observation covariance.

The prior-weighted restriction makes the regional coefficient and
unresolved residual independent. Consequently the reduced signal plus
aggregation error equals the same native innovation covariance for
every partition, and the native-grid DFS is a valid upper bound.

## Artifacts

- Static comparison: `tac_mhd_week_projected_variable_k_summary.png`
- Animation: `tac_mhd_week_projected_variable_k.gif`
- Search trace: `tac_mhd_week_projected_variable_k_trace.csv`
- Machine-readable assumptions: `tac_mhd_week_projected_variable_k_manifest.json`

## Limitations

This is stochastic local-search optimization, not partition posterior
inference. It assumes independent Gaussian native relative-scaling
errors and uses a fixed diagonal observation covariance benchmark.
The production RHIME mismatch model is not changed or reproduced.

Manifest method: `projection-consistent variable-count stochastic local search`.
