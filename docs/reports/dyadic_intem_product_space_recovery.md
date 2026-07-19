# Exact InTEM Product-Space Recovery Benchmark

## Purpose

This benchmark is the small exact companion to the non-enumerating 8 by 8
checkerboard experiment. It uses a 4 by 4 variable inner rectangle inside the
packaged EUROPE InTEM map and retains seven residual InTEM-region coefficients
under every inner partition. All 677 valid inner dyadic partitions can be
enumerated, so fixed and latent posterior predictions are integrated
analytically without Monte Carlo error.

The scientific target is explicit: latent K/P should beat a predeclared wrong
fixed partition with the same K and should be predictively non-inferior to the
true-partition oracle.

## Setup

- 48 deterministic smooth sensitivity rows.
- Rows 0--31 condition the inversion; rows 32--47 are holdout only.
- A regular four-quadrant K=4 inner truth represented exactly by the permanent
  root-and-contrast coordinates.
- Seven always-active residual InTEM-region coefficients with a separate prior.
- Known independent Gaussian observation error with standard deviation 0.05.
- Shared Gaussian target, coefficient priors, prior-forward mean, and R for all
  fixed and latent calculations.
- Uniform marginal prior on K from 1 through 16 and uniform partitions
  conditional on K: `p(P) = p(K) / N_K`.

Comparators are the true K=4 partition, a predeclared wrong K=4 partition, and
an underfit K=2 partition. The latent model averages all 677 partitions using
their exact posterior probabilities.

## Reproduction

```bash
HOME=/tmp MPLCONFIGDIR=/tmp .venv/bin/python \
  examples/basis/dyadic_intem_product_space_recovery.py
```

## Results

| Inversion | expected K | holdout log density | noiseless holdout RMSE | field RMSE | outer RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| True fixed P | 4.000 | 24.540 | 0.0453 | 0.0461 | 0.1323 |
| Wrong fixed P | 4.000 | 11.969 | 0.0916 | 0.1628 | 0.1805 |
| Underfit fixed P | 2.000 | 2.168 | 0.1040 | 0.1744 | 0.2420 |
| Exact latent K/P mixture | 4.800 | 24.480 | 0.0442 | 0.0476 | 0.1287 |

The latent-minus-oracle log-score difference is -0.0037 nat per holdout row,
well inside the provisional -0.05 non-inferiority threshold. Model averaging
slightly improves the holdout posterior-mean RMSE and outer-coefficient RMSE
relative to conditioning on the true partition, while its field RMSE remains
very close to the oracle.

The true K=4 partition is the posterior MAP with probability 0.5681. Total
posterior mass on K=4 is 0.5682, followed by K=5 at 0.2132 and K=6 at 0.1276.
The wrong K=4 and underfit K=2 partitions have posterior probabilities
2.34e-7 and 3.18e-7, respectively.

## Interpretation

This case meets the requested proof-of-concept goal at the exact-inference
level: latent K/P beats an otherwise matched wrong fixed K/P inversion and
matches the true-partition oracle predictively. It also retains the relevant
inner/outer state separation.

It is intentionally easy and should not be over-interpreted:

- the true partition is strongly identified in one synthetic realization;
- the observation covariance is known and diagonal;
- continuous priors are Gaussian rather than the positive Gamma-Beta process;
- the latent result is currently an exact enumerated mixture, not yet the
  output of the non-enumerating local chain;
- repeated noise realizations and prior-predictive calibration remain open.

The immediate implementation check is to run both augmented and collapsed local
chains on this same target and compare sampled K/P frequencies and predictive
metrics with the exact mixture. That directly validates the scalable sampler
before replacing its Gaussian continuous update.
