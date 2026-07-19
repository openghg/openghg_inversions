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
- exact enumeration is available only because this inner grid is deliberately
  small; the larger-grid chain must rely on convergence diagnostics instead;
- repeated noise realizations and prior-predictive calibration remain open.

## Non-enumerating chain validation

Both reusable local chains were initialized at the wrong K=4 partition and run
for 2,000 warmup cycles plus 20,000 retained draws. Neither transition used the
partition catalogue; enumeration was used only to calculate exact diagnostics
after sampling.

| Diagnostic | exact | augmented product-space | collapsed Gaussian |
| --- | ---: | ---: | ---: |
| expected K | 4.7998 | 4.7536 | 4.8054 |
| truth-P probability | 0.5681 | 0.5761 | 0.5702 |
| K total variation | 0 | 0.0138 | 0.0048 |
| full-P total variation | 0 | 0.0419 | 0.0409 |
| holdout log density | 24.4801 | 24.4883 | 24.4895 |
| noiseless holdout RMSE | 0.04424 | 0.04425 | 0.04418 |
| structural acceptance | n/a | 0.1380 | 0.1714 |
| unique retained P | n/a | 161 | 228 |

The augmented split and merge acceptance rates were 0.102 and 0.214. The
collapsed rates were 0.127 and 0.265. Despite the deliberately wrong starting
geometry and moderate local acceptance, both chains recover exact K mass,
truth-partition probability, and predictive performance within the predeclared
tolerances.

This completes the analytic Gaussian proof of concept end to end: latent K/P
beats the matched wrong fixed basis, is non-inferior to the true-partition
oracle, and can be sampled without global partition enumeration.

## Native PyMC split-mask and NUTS result

The same target now has a non-enumerating native PyMC implementation. The
partition is a canonical 15-bit ancestry-closed split mask; the likelihood uses
one static 32 by 16 contrast design rather than a 677-partition catalogue.
Local split/merge MH updates the mask first and native PyMC NUTS then updates
all 16 permanent inner coordinates plus the seven outer coefficients.

Reproduce the declared chain with:

```bash
HOME=/tmp MPLCONFIGDIR=/tmp .venv/bin/python \
  examples/basis/dyadic_intem_product_space_recovery.py \
  --sampler pymc --draws 20000 --warmup 3000 \
  --sampler-seed 481 --target-accept 0.95
```

| Diagnostic | exact latent mixture | PyMC split-mask plus NUTS |
| --- | ---: | ---: |
| expected K | 4.7998 | 4.9277 |
| truth-P probability | 0.5681 | 0.5344 |
| holdout log density | 24.4801 | 24.4859 |
| noiseless holdout RMSE | 0.04424 | 0.04402 |
| field RMSE | 0.04756 | 0.04820 |
| outer RMSE | 0.12870 | 0.12816 |

The K total-variation distance is 0.0426 and the full-P distance is 0.0708.
Structural acceptance is 0.149. The run took 37.2 seconds for 3,000 tuning
and 20,000 retained compound draws and reported no NUTS divergences. Its
holdout mean beats the wrong fixed K=4 and underfit fixed K=2 inversions. Its
log score differs from the true-P fixed oracle by -0.0034 nat per holdout row,
inside the predeclared -0.05 non-inferiority threshold.

The machine-readable result also records bulk ESS and MCSE for K and the
truth-partition indicator, plus the minimum bulk ESS among permanent inner
coordinates. These do not replace replicated chains, but make autocorrelation
visible beside raw TV distances and acceptance rates.

An earlier evaluator incorrectly reshaped depth-first leaf order directly into
row-major grid order. A regression now scatters every leaf by its stored grid
coordinates. The corrected sampled inner means agree with the analytic
Gaussian means; the error was in geographic reconstruction, not NUTS or the
static contrast design.

This is still one easy synthetic realization and one chain. Repeated
prior-predictive realizations, multi-chain diagnostics, and sensitivity to the
shared NUTS metric remain appropriate validation. The next implementation step
is to reuse the validated structural representation in a separate positive
Gamma-Beta model rather than generalize the Gaussian contrast classes until
both concrete models exist.
