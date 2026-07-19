# Latent Dyadic Product-Space Checkerboard Benchmark

## Purpose

This report records the first product-space inversion in this repository that
samples a non-enumerated latent partition (P) and its derived region count
(K), then compares it with fixed-partition inversions using the same data,
likelihood, prior, and Gaussian conditional model.

The benchmark is adapted from the local Lunt et al. (2016)-inspired test on
branch `codex/tdmcmc-numba-rewrite`. It is an implementation proof of concept,
not a reproduction of the paper or a production RHIME inversion.

## Model and data

- Native grid: (8\times8).
- Truth: sixteen regular (2\times2) regions with alternating scaling factors
  0.5 and 1.5.
- Sensitivities: 96 smooth, positive, footprint-like rows with amplitudes from
  50 to 130 ppb-like units.
- Training data: 64 rows.
- Holdout data: 32 independently generated sensitivity rows.
- Observation error: independent Gaussian, standard deviation 5 ppb.
- Prior-forward mean: scaling factor one on every grid location.
- Continuous prior: the current Gaussian root-and-contrast construction induced
  by independent unit-variance native-grid anomalies.
- Latent support: (8\le K\le28).

The partition prior is specified marginally rather than as a penalty per
partition:

\[
p(P)=\frac{p(K(P))}{N_{K(P)}},
\]

where (p(K)) is uniform on 8 through 28 and (N_K) is the exact number of
valid dyadic frontiers with K regions. The counts are computed with a tree
dynamic program; the (8\times8) partition catalogue is never materialized.

## Comparators

All fixed inversions use the exact conditional Gaussian posterior from the same
`GaussianProductSpaceTarget`.

1. **True fixed partition:** the sixteen (2\times2) truth regions. This is an
   oracle with more information than the latent model.
2. **Wrong fixed partition:** a predeclared K=16 dyadic frontier with misplaced
   boundaries. This isolates partition geometry while holding K fixed.
3. **Coarse fixed partition:** the K=8 starting frontier. This is a secondary
   underfit baseline that changes both K and P.
4. **Latent K/P:** local product-space split/merge Metropolis updates followed by
   exact conditional Gaussian refreshes. No partition catalogue is used.

The defensible target is to beat the misspecified fixed partition and approach,
but not systematically beat, the true-partition oracle.

## Reproduction

The recorded run used commit `2377b99` plus the benchmark working tree:

```bash
HOME=/tmp MPLCONFIGDIR=/tmp .venv/bin/python \
  examples/basis/dyadic_product_space_checkerboard.py \
  --draws 4000 --warmup 2000 \
  --minimum-regions 8 --maximum-regions 28 --seed 481
```

The latent sampler completed 6,000 transition cycles in 47.6 seconds on the
local development machine.

## Results

| Inversion | K | holdout RMSE (ppb) | holdout log density | grid RMSE | checkerboard contrast |
| --- | ---: | ---: | ---: | ---: | ---: |
| True fixed P | 16 | 2.168 | -104.633 | 0.059 | 0.944 |
| Wrong fixed P | 16 | 11.076 | -160.570 | 0.433 | 0.452 |
| Coarse fixed P | 8 | 13.741 | -191.587 | 0.528 | 0.000 |
| Latent K/P | mean 25.63, range 19--28 | 3.516 | -107.317 | 0.141 | 0.930 |

The latent chain visited 887 distinct retained partitions. Structural proposal
acceptance was 0.253 during warmup and 0.240 afterward.

The latent inversion clearly beats both predeclared fixed misspecifications. It
recovers most of the checkerboard contrast and has a holdout RMSE 68% lower than
the wrong K=16 geometry. Its holdout log score is 0.084 nat per observation
below the true-partition oracle. That misses the stricter provisional
non-inferiority threshold of -0.05 nat per observation, so this run should not
be described as matching the oracle.

## Interpretation and limitations

- The chain over-refines: posterior mean K is about 25.6 rather than the truth
  K=16. Replacing the original per-partition complexity penalty with a uniform
  marginal prior on K did not remove this behavior. It therefore reflects the
  current likelihood/prior model or finite-data realization, not accidental
  partition multiplicity alone.
- The latent result is based on one synthetic design and noise realization.
  Predictive superiority needs paired replication before becoming a scientific
  claim.
- Fixed and latent predictive scores currently use posterior draws. Analytic
  Gaussian or Rao-Blackwellized scores would remove small Monte Carlo error.
- The active Gaussian prior is a reference model, not the positive Gamma-Beta
  process prior intended for the non-Gaussian implementation.
- Local split/merge movement traverses hundreds of partitions, but formal
  convergence assessment requires multiple dispersed chains and split-level
  diagnostics.

## Next checks

1. Complete an exact (4\times4) InTEM inner/outer recovery benchmark. Its 677
   partitions permit exact model-averaged predictive calculations and direct
   sampler validation.
2. Run paired synthetic realizations with a predeclared truth, same-K wrong
   partition, and holdout score.
3. Add split-indicator and K convergence diagnostics for several starting
   frontiers.
4. Replace the exact Gaussian continuous update with the positive Gamma-Beta
   product-space target only after the structural chain and benchmark metrics
   are stable.
5. Compare this dyadic product-space path with the Voronoi RJMCMC implementation
   on `codex/tdmcmc-numba-rewrite` without attempting to unify them yet.
