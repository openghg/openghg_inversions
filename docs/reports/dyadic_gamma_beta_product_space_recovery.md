# Positive Gamma-Beta product-space recovery

## Scope

This report records the first non-enumerating positive product-space inversion.
It uses the projectively consistent grouped Gamma-Beta prior and a Gaussian
observation likelihood. It is an experimental implementation benchmark, not a
production RHIME inversion or a reproduction of Lunt et al. (2016).

The benchmark adapts the smallest synthetic recovery contract from branch
`codex/tdmcmc-numba-rewrite`. Forty training observations separately identify
two grid cells, and twenty independent rows are held out. The true scaling
field is `[0.5, 1.5]`; residual standard deviation is 0.05.

## Product-space representation

The maximum Gamma-Beta forest has fixed dimension throughout sampling:

- one mean-one Gamma root scaling, with variance 0.25;
- one permanent Beta split fraction, with concentration 2;
- one Bernoulli indicator selecting the split or unsplit frontier;
- exact normalized prior `p(P) = p(K) / N_K`, computed without enumerating
  partitions;
- local split/merge Metropolis-Hastings for the indicator;
- native PyMC NUTS for the permanent Gamma and Beta coordinates.

Inactive Beta coordinates retain their normalized prior. This is a natural
product-space pseudo-prior here because integrating descendants of an inactive
node leaves the parent Gamma-Beta model unchanged.

The static observation design has one column per possible forest node. A
vectorized ancestry mask activates only the selected frontier, so changing `P`
or `K` does not rebuild the PyMC graph or change the NUTS variable order.

## Matched comparison

The declared run used 1,000 tuning and 1,000 retained draws for each fit, one
chain, seed 20260719, and `target_accept=0.9`.

| Fit | Mean K | Split probability | Field RMSE | Holdout RMSE | Holdout log predictive density | Divergences |
|---|---:|---:|---:|---:|---:|---:|
| latent K/P | 2.000 | 1.000 | 0.00648 | 0.00648 | 30.56 | 0 |
| fixed true split | 2.000 | 1.000 | 0.00665 | 0.00665 | 30.57 | 0 |
| fixed underfit unsplit | 1.000 | 0.000 | 0.50004 | 0.50004 | -852.41 | 0 |

The latent inversion therefore matches the fixed true-partition inversion and
decisively beats the underfit fixed partition under the benchmark's declared
tolerances. Its posterior mean field is `[0.5023, 1.5089]`.

The retained partition acceptance rate is zero. The latent chain starts
unsplit, accepts the overwhelmingly favored split during tuning, and accepts no
subsequent merge. This is adequate for the recovery gate but is not evidence of
good traversal of a broad or multimodal partition posterior.

## Code and validation

Primary implementation:

- `openghg_inversions/basis/experimental/dyadic/gamma_beta_coordinates.py`
- `openghg_inversions/basis/experimental/dyadic/gamma_beta_partition.py`
- `openghg_inversions/basis/experimental/dyadic/gamma_beta_product_space.py`
- `openghg_inversions/basis/experimental/dyadic/pymc_gamma_beta_product_space.py`
- `examples/basis/dyadic_gamma_beta_product_space_recovery.py`

Focused validation covers vectorized/top-down prior parity, exact forest count
dynamic programming, prior normalization, reversible local moves, NumPy/PyMC
likelihood parity, fixed variable ownership, and the seeded recovery benchmark.

Run the declared benchmark with:

```bash
HOME=/tmp MPLCONFIGDIR=/tmp .venv/bin/python \
  examples/basis/dyadic_gamma_beta_product_space_recovery.py \
  --draws 1000 --tune 1000
```

## Limitations and next experiment

The current case is intentionally easy: observations identify each grid cell
separately, the posterior over `P` is nearly degenerate, and only one split is
possible. It demonstrates correctness of the positive fixed-shape machinery,
not realistic spatial mixing.

That next scale-up is now implemented in
`docs/reports/dyadic_gamma_beta_intem_product_space_recovery.md`. The remaining
work is replicated-chain assessment, K-prior sensitivity, group-wise posterior
diagnostics, and comparison with the direct RJMCMC rewrite.
