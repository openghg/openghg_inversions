# Experimental dyadic SLS provenance

This package extracts the reusable mechanics from private exploratory code so
that the hackathon demonstration can cite public, reviewable source. It remains
experimental and is not part of the supported basis-function API.

## Historical sources

- `~/Documents/basis_functions/basis_fn_ipython_hist_14aug.py`
  - `make_weights`: pre-sums sensitivities over dyadic tiles and evaluates the
    original quadratic design proxy.
  - `get_split`, `get_merge`, `apply_move`, and `dof_change`: local dyadic tree
    moves and cheap score differences.
  - `ApplyStepCondition`: temperature-based acceptance with a region-count
    penalty. This is stochastic local search, not posterior MCMC.
- `~/Documents/basis_functions/hist_17aug.py`
  - later Python and Numba bisection/search experiments.
- branch `codex/basis-prototype-examples`, commit `b6ce565`
  - the first cleaned executable scaffold for dyadic indexing, precomputed
    multiscale sums, and local search.

The implementation here intentionally replaces mutable indicator matrices and
notebook-global random state with immutable partition states, explicit random
generators, independently testable proposals, and complete diagnostic traces.

## Score terminology

The historical quadratic score for a tile `v` is proportional to

```text
sum_i precision[i] * (sum_{cell in v} G[i, cell])**2 / area[v]
```

It is retained as a fast **prototype quadratic design score**. It must not be
described as exact degrees of freedom for signal (DFS). The benchmark also
computes Gaussian DFS from an explicitly supplied prior covariance.

For a partition projection matrix `P` and fine-grid prior covariance `B`, the
Bocquet-consistent covariance is partition-dependent:

```text
B_partition = P @ B @ P.T
```

Using the same isotropic covariance for every partition is permitted only as a
clearly labelled proof-of-concept benchmark.

## Exact Gaussian projection sources

The exact projection implementation in `gaussian_projection.py` follows the
conditional-Gaussian construction documented in
`docs/reports/rhime_bocquet_reduced_gaussian.md` and checked against:

- Bocquet, Wu, and Chevallier, *Bayesian design of control space for optimal
  assimilation of observations. Part I: Consistent multiscale formalism*;
- `~/Documents/verification-games/scripts/run_controlled_aggregation_error_coarsening.py`
  at commit `53a896c794ae...`, which contains an earlier exact conditional
  reduction and direct-versus-projected comparison; and
- `~/Documents/verification-games/src/verification_games/linear_gaussian.py`
  at commit `a257cd3a57ad...`, itself derived from
  `openghg_inversions` `origin/analytic_model` commit `03948c68889b...`.

The local implementation is a fresh, dense NumPy oracle with explicit support
for nonzero prior means, non-diagonal prior and observation covariances,
overlapping restrictions, and fixed regional prolongations. It has no runtime
dependency on either private checkout.

## Spatial covariance source

The separable exponential covariance operator in `grid_covariance.py` is
adapted from
`~/Documents/verification-games/src/verification_games/grid_covariance.py`,
principally commits `338fa8825212...` and `28e7bd624948...`. The retained idea
is to apply a Kronecker-separable latitude/longitude kernel to native-grid
vectors or batched right-hand sides without materializing the full dense
native covariance. The local API and validation are independent and tested
against explicit small Kronecker matrices.

The historical sparse taxi-cab kernel from `origin/covariances` commit
`ac943b90e528...` is deliberately not ported. It has different covariance
semantics and is not required for the distance-based prior experiment.
