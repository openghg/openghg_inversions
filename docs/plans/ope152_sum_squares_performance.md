# OPE-152 sum-of-squares cache experiment

Measured on 21 October 2026 at research commit
`f0cbebc2c5dafc09de937f87ea1d16887b09a7ff`, based on PR #922 at `c61ef039`.
This experiment changes no production numerical code. It follows the
[cached fixed-OU investigation](ope152_cached_ou_performance.md); implementation
and acceptance remain tracked in
[OPE-152](https://linear.app/openghg-inversions/issue/OPE-152/validate-vg-sampling-and-hpc-operation-for-specialised-co-runners).

Sum-of-squares is a promising further optimization for the usual
`aggregation_error_rank=256` configuration. Every measured rank-256 candidate
refresh was faster than every current refresh, and all numerical comparisons
passed. Full rank was slower than PR #922's observation-space Cholesky route.
The condition-only guard tested here does not resolve every cancellation case;
these measurements do not establish readiness for production.

## Paired measurements

All methods used the same target and cached fixed coordinate transforms within
each rank configuration. The 4,557 observations, 12 sites, 268 state
coefficients, scientific settings, and immutable source input match the previous
investigation. Rank 256 retains the public low-rank decomposition's diagonal
tail; it is a different covariance configuration from full rank.

Median seconds per complete refresh:

| Factor width | Amplitude (ppm) | PR #922 cache | Raw sum-of-squares | Condition-checked sum-of-squares |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.30 | 0.410470 | 0.099253 | 0.131259 |
| 256 | 0.75 | 0.408510 | 0.093789 | 0.124362 |
| 256 | 1.20 | 0.410316 | 0.093443 | 0.094132 |
| 4557 | 0.30 | 2.510552 | 2.762363 | 3.034813 |
| 4557 | 0.75 | 2.522828 | 2.773022 | 3.061451 |
| 4557 | 1.20 | 2.534938 | 2.773418 | 3.088847 |

Rank-256 raw medians are 4.14–4.39 times faster than the current cache;
condition-checked medians are 3.13–4.36 times faster. Full-rank raw medians are
9–10% slower and condition-checked medians are 21–22% slower. These are cache
refresh measurements, not end-to-end sampler speedups.

There was one warmup per method and amplitude, followed by seven rank-256 or
three full-rank repetitions in rotating method order. Whitening, core formation,
factorization, coefficient formation, and any attempted work before fallback
are timed. Diagnostics and numerical comparisons are separate.
[All 90 timed samples](ope152_sum_squares_timings.csv) include execution position.

Rank-256 timings have substantial order effects: raw calls first in a repeat
round had medians of 0.164–0.179 s, while the other positions were around
0.093–0.097 s. Seven repetitions do not balance the three execution positions
equally. Thus the differences between raw and guarded medians cannot be
attributed to the guard itself. Its separate diagnostic condition estimate
took about 0.0006 s. Across amplitudes and candidates, even the slowest
rank-256 candidate call (0.197121 s) beat the fastest current call (0.358604 s).
A follow-up comparison should balance execution positions.

## Calculation and accuracy

For amplitude-whitened factor `W` and fixed input columns `B`, the experimental
[profiler](../../scripts/profile_ope152_sum_squares.py) computes

```text
K = I + W.T @ W
Z = solve(K, W.T @ B)
E = B - W @ Z
G = E.T @ E + Z.T @ Z
```

`G` supplies all constant, linear, and precision coefficients. Cholesky supplies
the complete determinant update. The guarded variant additionally uses DPOCON
and accepts reciprocal condition estimates greater than `sqrt(float64 epsilon)`;
otherwise it calls the current refresh, including the attempted work in timing.
All measured candidates took the sum-of-squares route. Rank-256 reciprocal
condition estimates were 7.04e-5, 1.28e-4, and 1.59e-4.

Both configurations passed all 342 recorded comparisons. These compare all
Gram/cache coefficients and the determinant against the current cache, plus
likelihood and state gradients at zero and three nonzero states. Nonzero states
are all ones and ones plus/minus a fixed seeded perturbation; their likelihood
and gradients also agree with direct evaluation. These checks do not sample
the posterior or exhaust the state/amplitude space.

At rank 256, candidate/current Gram differences were at most 1.68e-8, log
determinants agreed at stored precision, and candidate/direct likelihood and
state-gradient differences were at most 9.99e-9 and 1.94e-8. At full rank,
candidate/current Gram differences reached 3.03e-6 (within the relative
tolerance), while nonzero-state candidate/direct likelihood and gradient
differences were at most 3.91e-9 and 1.50e-8. The full-rank determinant difference
was at most 2.18e-11. Default comparison tolerances were `rtol=1e-8, atol=1e-6`,
and `rtol=1e-10, atol=1e-7` for determinants.

Independent tiny-matrix checks used 90-digit arithmetic across 13 cases,
including the original 1e10-scale cancellation regression and nonzero design
columns. They support the sum-of-squares identity and reproduce two distinct
limitations: duplicate dominant factor columns can damage the formed core,
and at extreme scale a well-conditioned core can still leave an inaccurate
computed residual `E`. DPOCON detects the former example but misses the latter.
Residual stationarity refinement improved some examples but did not repair an
incorrect determinant. A condition-only screen is therefore insufficient for
a general stability claim. The current QR/SVD route also has extreme-scale
limitations, documented in the earlier investigation.

The next experiment adds a per-column screen for residual rounding. Starting
from the standard matrix-product bound in
[Higham and Mary, section 4](https://tmary.perso.lip6.fr/doc/surveyMixed.pdf),
our conservative norm estimate is
`d_j = gamma_(r+1) * (norm(B_j) + norm(W, Frobenius) * norm(Z_j))`, using
`gamma_k = k*eps/(1-k*eps)` and `eps = float64 epsilon`. For
`a = max_j d_j/sqrt(norm(E_j)^2 + norm(Z_j)^2)`, the residual-rounding
contribution to each Gram entry is bounded by `2*a + a*a` times the product
of the corresponding column energies' square roots. The experimental screen
accepts this estimate only below `sqrt(eps)`, alongside the existing core
condition screen. It does not certify latent-solve error, determinant error,
or relative accuracy of small cross entries. Zero columns are handled
explicitly, and nonfinite estimates reject the candidate.

## Provenance and reproduction

SLURM job `19583422` completed with exit status zero in 7 minutes on
`bp1-compute056`, with four CPUs and 8 GiB allocated. Sampled peak RSS was
1,961,380 KiB. Both NumPy and SciPy used four OpenBLAS threads, matching the
prior experiment. The imported numerical source hashes match PR #922, and
the recorded working-tree diff was empty. Source input SHA256:
`7ab7cf733eba7b12dd1fea7a5c8812236eda1d579486b03e5e2d27feab8866bf`.

To reproduce the three-method results, select research commit `f0cbebc2`.
Use the same runtime, public input loader, scientific settings, and cluster
cache/thread setup documented in the earlier investigation. Within a SLURM
allocation:

```bash
python scripts/profile_ope152_sum_squares.py --prepared SOURCE --out NEW_RANK256_DIR --aggregation-error-rank 256 --repeats 7
python scripts/profile_ope152_sum_squares.py --prepared SOURCE --out NEW_FULLRANK_DIR --repeats 3
```

Full manifests, numerical oracle reports, source snapshots, coefficient
artifacts, logs, batch script, and callback ticket are retained under
`inversion_outputs/ope152-sos-20261021-r1/` in the experiment checkout. The
profiler's tiny dense-oracle self-check, public-loader smoke test, injected
nonfinite-result recovery, and focused Ruff checks passed before submission.
Independent review checked numerical results, provenance, and timing fairness.
The production test suite was not rerun because production code was unchanged.
