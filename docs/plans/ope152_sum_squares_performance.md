# OPE-152 sum-of-squares cache implementation and experiments

Measured on 21 October 2026, following PR #922 at `c61ef039` and the
[cached fixed-OU investigation](ope152_cached_ou_performance.md); implementation
and acceptance remain tracked in
[OPE-152](https://linear.app/openghg-inversions/issue/OPE-152/validate-vg-sampling-and-hpc-operation-for-specialised-co-runners).

The guarded implementation at `5fa4a2bb` materially speeds up cache refreshes
for the usual `aggregation_error_rank=256` configuration. A balanced comparison
against PR #922 measured a 4.73-fold median speedup with four BLAS threads and
6.28-fold with one, at amplitude 0.75 ppm. All numerical checks passed, and
180 relevant tests passed on each of Python 3.12 and 3.13. These are isolated
cache measurements on one network, not sampled-chain or general scaling results.

## Implementation validation

`FixedOuCachedSigmaTarget.refresh` now forms all quadratic coefficients together
as a sum of squares when `0 < rank < n_observations`. It accepts the result
only after the core-condition and residual-rounding screens described below.
Rejected candidates use the existing solver, preserving the dominant-factor
cancellation regressions. Zero-rank and full-width factors keep their existing
routes. The covariance, priors, amplitude evaluation/gradients, and cache
lifecycle are unchanged; no runtime setting or dependency was added.

Job `19585464` compared the actual frozen implementation with PR #922 on the
same 4,557-observation, 268-state, rank-256 target at amplitude 0.75 ppm. Each
thread block warmed both methods, then used twelve alternating AB/BA pairs,
giving each method six first and six second positions. Both NumPy and SciPy
BLAS pools were checked inside each thread-limit context. Timings include a
lightweight call counter; numerical checks and separate tracing are outside
the timer. Checks and manifest writes occur between calls symmetrically.

| BLAS threads | PR #922 median (s) | Sum-of-squares median (s) | Median speedup | Baseline range (s) | Candidate range (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.491182 | 0.103846 | 4.73 times | 0.484574–0.509917 | 0.075141–0.193805 |
| 1 | 0.406038 | 0.064621 | 6.28 times | 0.404624–0.408443 | 0.062747–0.066556 |

Every candidate call was faster than every baseline call within its thread
block. All 24 candidate calls avoided `_solve_transformed`; all 24 baseline
calls used it once. Four-thread median process CPU time fell from 1.955391 to
0.413861 seconds, and one-thread CPU time from 0.404399 to 0.064337 seconds.
The four allocated logical CPUs were two physical cores with simultaneous
multithreading, one core on each socket. Four-thread candidate timings remain
order-sensitive: first-position median 0.159225 s versus second-position
0.076196 s. Balanced ordering limits bias but does not explain this effect.
The one-thread block followed the four-thread block; these data do not establish
thread-scaling recommendations.

All cache coefficients and likelihood/state gradients at three nonzero states
passed comparison with the baseline and direct likelihood. The maximum
coefficient difference was 1.31e-8; candidate/direct likelihood and gradient
differences were at most 8.32e-9 and 5.14e-9. Diagnostic log determinants matched
exactly. The residual-rounding estimate was 3.17e-11 against a 1.49e-8 budget,
with core reciprocal condition estimate 1.28e-4. These screens limit known
failure modes; they do not provide a universal forward-error guarantee.

The preceding implementation run, job `19584931`, measured all three amplitudes
at rank 256 and full rank, taking all baseline repeats before all candidate
repeats. Its medians were:

| Factor width | Amplitude (ppm) | PR #922 (s) | Candidate (s) |
| ---: | ---: | ---: | ---: |
| 256 | 0.30 | 0.482663 | 0.144111 |
| 256 | 0.75 | 0.483055 | 0.495577 |
| 256 | 1.20 | 0.483113 | 0.145483 |
| 4557 | 0.30 | 2.429175 | 2.431741 |
| 4557 | 0.75 | 2.445658 | 2.562079 |
| 4557 | 1.20 | 2.393841 | 2.402150 |

At rank 256/amplitude 0.75, four consecutive candidate calls took 0.799–0.829 s
and four took 0.101–0.193 s. Separate diagnostic calls accepted sum-of-squares
at every amplitude, but that run did not count routes during timed calls.
Unchanged full-rank baseline/candidate paths also had outliers of 6.54/7.62 s.
The interleaved follow-up above did not reproduce these large delays; their
cause remains unresolved. All samples are retained in the
[implementation timing CSV](ope152_sum_squares_implementation_timings.csv),
including the anomalous first run. Empty CPU-time/route-count fields denote
measurements not taken in that run. No outliers were removed. Full-rank
numerical checks passed with identical baseline/candidate densities, gradients
and determinants; this implementation makes no full-rank algorithm change.

Job `19584933` ran the repository's Slurm tox runner with two CPUs, 8 GiB,
one numerical thread and two Dask workers. Both `py312-openghgCur` and
`py313-openghgCur` passed all 180 tests across `test_fixed_ou`,
`test_cached_sigma`, `test_fixed_ou_rhime`, `test_cached_sigma_sampling`,
`test_co2_cached_sigma` and `test_co2_o2_fixed_ou`. Coverage includes dense
oracles, determinant/gradient parity, both rejection screens, dominant factors,
input ownership, finite-overflow rejection, sampler cache lifecycle, and public
CO₂/linked-model integration. Focused Ruff, Mypy, public-loader smoke and
independent implementation/evidence reviews also passed.

| Job | Purpose | Elapsed | Allocated CPUs / GiB | Process peak RSS (KiB) | Slurm sampled MaxRSS (KiB) |
| ---: | --- | ---: | --- | ---: | ---: |
| 19584931 | Both-rank implementation comparison | 4m14s | 4 / 4 | 1,948,320 | 1,874,808 |
| 19585464 | Interleaved rank-256 follow-up | 1m01s | 4 / 4 | 723,416 | 645,572 |
| 19584933 | Python 3.12/3.13 tests | 17m30s | 2 / 8 | — | 6,780,024 |

All jobs completed with exit status zero. Benchmarks ran on `bp1-compute052`
with Python 3.13.7, NumPy 2.2.6 and SciPy 1.15.3; NumPy/SciPy OpenBLAS versions
were 0.3.29/0.3.28. The follow-up loaded the already-derived rank-256 public
artifact, so its setup memory excludes dense-covariance preparation.

Reproduce the first implementation run with
`scripts/profile_ope152_fixed_ou.py --baseline-revision c61ef039 --algorithms baseline current`,
using `--aggregation-error-rank 256 --repeats 8` or full rank with `--repeats 3`.
The [interleaved driver](../../scripts/profile_ope152_sum_squares_interleaved.py)
preserves the measured timing/check loop and pins numerical source hashes,
while allowing documentation-only commits after the frozen implementation.
Use the first run's saved low-rank artifact through the public loader:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
python scripts/profile_ope152_sum_squares_interleaved.py \
  --prepared PREPARED_RANK256 --out NEW_RESULTS_DIR \
  --expected-sha256 22313f482e0eb2143804f2a290cddc5f247ad293db11776977b75d631df06353
```

Run production-sized inputs on Slurm with the node-local cache setup from the
earlier investigation. A newly serialized artifact may have a different file
hash; validate its derivation and pass its actual SHA256. The measured driver
starts with four BLAS threads, then reduces to one. Starting this particular
runtime at one thread and raising it to four caused a native SciPy triangular
solve crash, independently reproduced without OGI; starting at four passed.
The driver rejects the untested startup configuration before timing.

Full manifests, frozen sources, the original driver, native-runtime reproducer,
logs and tickets are retained under
`inversion_outputs/ope152-sos-candidate-20261021-r1/` and
`inversion_outputs/ope152-sos-candidate-20261021-r2/`. No production inversion
was run. Acceptance rates, fallback frequency along a sampled chain and
end-to-end sampler throughput remain unmeasured.

## Initial research experiment

Research commit `f0cbebc2c5dafc09de937f87ea1d16887b09a7ff` changed no production
numerical code. Every rank-256 candidate refresh in that experiment was faster
than every current refresh, and all numerical comparisons passed. Full rank
was slower than PR #922's observation-space Cholesky route. The condition-only
guard did not resolve every cancellation case, motivating the second screen.

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

The follow-up experiment adds a per-column screen for residual rounding. Starting
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

## Balanced rank-256 follow-up

Job `19584440` measured research commit `9c626754` using twelve repetitions
per method and amplitude. Four Williams-balanced orders balance each method's
position and every within-round predecessor pair three times. The new method
uses both the core-condition and residual-rounding screens. All 144 calls and
228 numerical comparisons passed; all 36 calls of the new method used
sum-of-squares without falling back. Its coefficients were bitwise identical
to the raw candidate at every measured amplitude.

| Amplitude (ppm) | PR #922 cache (s) | Raw (s) | Core screen (s) | Both screens (s) | Speedup with both screens |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.381981 | 0.105423 | 0.125795 | 0.149470 | 2.56 times |
| 0.75 | 0.380209 | 0.098861 | 0.135498 | 0.134750 | 2.82 times |
| 1.20 | 0.368970 | 0.105766 | 0.151092 | 0.136570 | 2.70 times |

[All follow-up timings](ope152_sum_squares_guarded_timings.csv) retain the
144 samples and their positions. The slowest residual-screened call was
0.199604 s, below the fastest current call, 0.355674 s. Individual candidate
calls still vary substantially, so differences between candidate medians do
not isolate the cost of a screen. Separate diagnostic calls spent 0.004–0.017 s
in the residual screen and about 0.00056 s in the condition estimator.

Worst-column residual-rounding estimates were 4.12e-11, 3.17e-11 and 2.57e-11,
against the 1.49e-8 budget: a margin of 362–580 times. Column 47 was worst at
all three amplitudes. Core-condition estimates and numerical discrepancies
match the first rank-256 run. Independent review verified source/input/artifact
hashes, all numerical comparisons, the accepted routes, and execution ordering.

The job completed with exit status zero in 2m04s on `bp1-compute052`, using
four CPUs and 4 GiB; sampled peak RSS was 1,274,492 KiB. Artifacts are under
`inversion_outputs/ope152-sos-20261021-r2/`. Reproduce at commit `9c626754`
with `--aggregation-error-rank 256 --repeats 12` and the same source input.
Full rank was not repeated because the first run established that sum-of-squares
was slower there.

This supports preparing a guarded cache-only implementation for factors
narrower than the observation vector. It does not establish acceptance rates
or fallback frequency along a sampled chain, performance on other networks,
or a universal stability guarantee. Implementation validation is separate
from these frozen experiment results.

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
Those research-only jobs did not rerun the production test suite; the separate
implementation validation above covers the subsequent numerical-code change.
