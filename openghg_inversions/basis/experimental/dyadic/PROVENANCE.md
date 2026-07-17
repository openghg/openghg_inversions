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

