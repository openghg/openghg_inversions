# OPE-152 cached fixed-OU performance evidence

Measured on 21 October 2026 for candidate `51b07d761eda7a4b9f2cadbb9816006dc7291df6`.
This is dated computational validation for the proposed change, not a posterior
sampling acceptance result or a new numerical contract. Tracking:
[OPE-152](https://linear.app/openghg-inversions/issue/OPE-152/validate-vg-sampling-and-hpc-operation-for-specialised-co-runners).

The usual verification-games configuration, `aggregation_error_rank=256`,
benefits from caching the fixed OU coordinate transforms. It retains the
existing QR/SVD solve and now refreshes faster than the original Woodbury
implementation. Full-width covariance factors additionally use a guarded
observation-space Cholesky solve. Neither optimization changes the covariance
within a configuration, the priors, or the amplitude likelihood/gradient path.

## Same-input results

The input has 4,557 observations at 12 sites, 264 retained flux coefficients,
four boundary-condition coefficients, and 12 inferred OU amplitudes. Every
refresh solves 269 fixed right-hand-side columns. The immutable source is
`prepared-inputs.nc`, SHA256
`7ab7cf733eba7b12dd1fea7a5c8812236eda1d579486b03e5e2d27feab8866bf`.

Full rank uses all 4,557 factor columns. The separate rank-256 experiment uses
the public `prepare_low_rank_aggregation_error` helper on that same source,
including its diagonal residual variance. Its diagonal preservation error is
zero; retained positive spectral fraction is 0.99330412065. The derived artifact
SHA256 is `79503b96447b2165f20fb232d5b690919eb92ba924876c32d0c83665ea0bbbc7`.
Rank 256 and full rank are different covariance configurations: numerical
parity is checked between algorithms within each configuration.

The table gives median seconds per refresh, with five repetitions for rank 256
and three for full rank. All algorithms in this table ran in the same SLURM
allocation. [Individual repeat timings](ope152_cached_ou_timings.csv) retain
the unrounded measurements and separate direct likelihood/gradient timings.

| Factor width | Site amplitude (ppm) | Original Woodbury | Stability-fix QR/SVD | Candidate | Speedup over stability fix |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.30 | 0.684882 | 0.905528 | 0.367494 | 2.46× |
| 256 | 0.75 | 0.652906 | 0.912567 | 0.357725 | 2.55× |
| 256 | 1.20 | 0.665622 | 0.908605 | 0.358197 | 2.54× |
| 4557 | 0.30 | 3.178103 | 41.962988 | 2.365833 | 17.74× |
| 4557 | 0.75 | 3.178781 | 42.119623 | 2.327025 | 18.10× |
| 4557 | 1.20 | 3.179837 | 42.511883 | 2.336820 | 18.19× |

The final rank-256 comparison measures a 32–40% regression from original
Woodbury to the stability-fix implementation. An earlier five-repeat run
measured 33–47%. The candidate is 1.83–1.86× faster than original Woodbury in
the final comparison. Full-rank costs must not be used as representative
timings for the usual rank-256 configuration.

The direct amplitude likelihood plus gradients remain unchanged: their final
median times at amplitudes 0.30/0.75/1.20 ppm were 0.07594/0.12154/0.07592 s for
rank 256 and 3.36087/3.38488/3.36110 s for full rank. These evaluations still
occur during amplitude sampling. Refresh speedup is not an end-to-end sampler
speedup estimate.

## Why the change helps

Let `T` be the fixed generalized OU mode transform, `B` contain the zero-state
residual and state-design columns, and `M` be the amplitude-dependent
covariance in mode coordinates. The prepared covariance satisfies
`C^-1 = T.T @ M^-1 @ T`, without requiring `T` to be orthogonal. Therefore
the cache coefficients can be formed as
`(T @ B).T @ solve(M, T @ B)`. The constructor stores `T @ B` once; refresh
needs neither a repeated forward transform nor an observation-space solution.
The solve result still carries the complete covariance log determinant.

In a separate instrumented refresh at 0.75 ppm, the rank-256 baseline spent
0.285 s transforming the fixed inputs and 0.272 s transforming the solution
back. QR cost 0.244 s and SVD 0.040 s. The candidate removes those two repeated
transforms and keeps QR/SVD (0.257/0.041 s in its diagnostic call).

For full-width factors, the baseline performs QR of a 4,557 × 4,826 matrix
and SVD of a 4,557 × 4,557 matrix. The same diagnostic call measured QR at
5.440 s, factor projection at 1.922 s, and SVD at 34.292 s. The candidate
instead forms `I + W @ W.T`, where `W` is the amplitude-whitened covariance
factor. Its measured Gram construction, Cholesky, condition estimate and RHS
solve took 1.012, 0.706, 0.281 and 0.282 s respectively. These line timings are
from separate diagnostic calls, not the uninstrumented repeats; inclusive
caller and helper timings must not be added together.

Cholesky success alone is insufficient: rounding can erase the identity term
when factor columns dominate it. The candidate uses LAPACK
[DPOCON](https://www.netlib.org/lapack/explore-html/d3/dfd/group__pocon_gaadb1e19663b71521d30b5b5bbe3091f8.html)
to estimate reciprocal condition from the Cholesky factor, accepting values
greater than `sqrt(float64 epsilon)` (about 1.49e-8). Failed Cholesky or a
failed conditioning screen retains the QR/SVD route. Full-rank estimates at
the three measured amplitudes were 1.08474e-7, 4.78455e-7 and 1.46358e-6, so
all used Cholesky. This cutoff targets roughly half the available digits; it
is a routing heuristic, not a universal forward-error guarantee. Near-cutoff
unit tests allow the resulting floating-point error, and the existing fallback
also retains its extreme-scale limitations.

## Numerical alternatives and history

The performance regression followed the cancellation fix in
[commit bbdfd60d](https://github.com/openghg/openghg_inversions/commit/bbdfd60d151eef854c3e0be6ff119f59f33d9ab2).
The direct likelihood uses a residual-plus-ridge sum of squares; the general
inverse action used by cache refresh was changed to QR/SVD. The
[interface review](https://github.com/openghg/openghg_inversions/pull/673#discussion_r4004269059)
requested a small solve result and independent oracle tests. The later
[performance review](https://github.com/openghg/openghg_inversions/pull/673#discussion_r4006638142)
removed duplicate Cholesky/QR/SVD work; it did not establish QR/SVD as the only
acceptable algorithm.

Woodbury is used by established libraries, including
[PyTorch's low-rank normal](https://github.com/pytorch/pytorch/blob/main/torch/distributions/lowrank_multivariate_normal.py)
and [GPyTorch's linear operators](https://github.com/cornellius-gp/linear_operator/blob/main/linear_operator/operators/low_rank_root_added_diag_linear_operator.py).
Those implementations still use subtractive expressions. Woodbury instability
for small diagonal noise and augmented QR alternatives are discussed in
[Eriksson et al., section 3.3](https://papers.nips.cc/paper/7919-scaling-gaussian-process-regression-with-derivatives.pdf).
Their existence does not make a direct restoration of the old solve stable.

A cache-only sum-of-squares alternative remains credible. With whitened input
columns `B`, compute `Z = solve(I + W.T @ W, W.T @ B)` and `E = B - W @ Z`.
The cache Gram matrix is exactly `E.T @ E + Z.T @ Z`. Small prototypes against
90-digit references passed the original 1e10-scale rank-one case with nonzero,
orthogonal and nearly aligned design columns: maximum two-row errors were
1.82e-12 in the quadratic, 2.27e-13 in linear coefficients and 2.84e-14 in
precision. Thus extreme 1e20-scale failures do not disqualify this approach
for the original regression.

However, a separate 1e10-scale duplicate-column case gave substantial errors
with unchecked Cholesky/Gram; its reciprocal condition of 5.46e-17 identifies
the problem. At 1e20 scale even a scalar, well-conditioned latent core can
leave large error in `E`. Implicit Householder QR of `[W; I]`, applying its
rotations to `[B; 0]` without forming Q, removes explicit subtraction and SVD
but was less accurate for mixed large/small columns in these probes. A guarded
cache-Gram implementation would need its own coefficient and determinant
validation. It is not part of this patch, which already recovers performance
while preserving the tested low-rank solver.

## Validation and resources

All algorithm comparisons passed direct/cached density and state-gradient
checks. For the candidate, maximum absolute discrepancies were 1.42e-8 and
2.25e-8 respectively at rank 256; 9.41e-9 and 6.86e-8 at full rank. The maximum
full-rank log-determinant difference from the baseline was 2.24e-11; rank-256
log determinants agreed at printed precision. These are numerical parity
checks at fixed states/amplitudes, not posterior comparisons.

The six relevant numerical, builder and sampler test files passed **155 tests
on each of Python 3.12 and 3.13**, using the repository Slurm tox runner. Tests
cover zero/low/full/overcomplete factors, vector/matrix RHS, conditioning
fallback, dominant-factor cancellation, independent dense oracles, borrowed
inputs and cache lifecycle. Focused Ruff and `git diff --check` also passed.
Independent code review found no actionable issues.

Benchmark job `19581302` completed successfully in 14m31s on `bp1-compute066`,
with four CPUs and 8 GiB allocated. NumPy 2.2.6 and SciPy 1.15.3 used four
OpenBLAS threads (0.3.29/0.3.28); PyMC 6.3.2 and PyTensor 3.3.2 used Python
3.13. The imported OGI source was the candidate checkout; installed package
metadata still reported 0.7.4. The numerical thread limit, not total node
capacity, governs these measurements.

| Process | Load (s) | Build including initial refresh (s) | Process elapsed (s) | Process peak RSS (KiB) |
| --- | ---: | ---: | ---: | ---: |
| Rank-256 comparison | 13.480021 | 4.333279 | 74.587412 | 1,272,916 |
| Full-rank comparison | 12.184584 | 24.235378 | 655.991749 | 3,003,312 |
| Full-rank candidate only | 12.568812 | 23.835592 | 114.895391 | 1,946,548 |

Rank-256 derivation and serialization took an additional 11.755490 s and
0.738156 s. Build times include the new one-time fixed transform; baseline
startup was not separately benchmarked. Process elapsed starts after interpreter
startup and initial imports; it includes deferred imports and profiling overhead.
RSS is `resource.ru_maxrss`. SLURM's sampled job maximum was
2,933,352 KiB, which can differ from process high-water accounting. The separate
candidate-only process measured full-rank medians of 2.402945, 2.340053 and
2.414702 s and isolates memory from baseline QR/SVD allocations.

Test job `19581304` completed in 11m24s on `bp1-compute049`, with two CPUs,
8 GiB allocated, and sampled peak RSS 4,990,252 KiB. Prior full-rank runs on
another node had slower baselines (56–60 s) and occasional long outliers.
Reported speedups use paired measurements from the final allocation;
absolute times remain hardware/load dependent. No production inversion was
launched, no campaign artifacts were modified, and no posterior convergence
or effective-sample-size claim follows from these profiles.

## Reproduction and evidence locations

The [profiler](../../scripts/profile_ope152_fixed_ou.py) loads through current
public prepared-input/model-builder interfaces. It pins baseline
`1fe38ad91a5f5c14f7d0c4736ddcaf66487776a6` and reproduces pre-`bbdfd60d`
Woodbury algebra as a benchmark-only reference. Baseline numerical code matches
the originally profiled `4a786430` implementation. Both baseline `solve` and
`refresh` are substituted; current refresh uses the optimized implementation.
Historical bookkeeping/timing fields are omitted, slightly favoring that
reference. All refreshes use the same constructed target within a rank run.

The scientific settings are fixed OU time scale 5 h, site amplitude priors
`HalfNormal(sigma=0.75 ppm)`, boundary prior
`TruncatedNormal(mu=1, sigma=0.01, lower=0)`, and no offset. The reference
state has all coefficients equal to one. Rank selection affects only the
explicitly requested separate covariance configuration.

Within a four-CPU/eight-GiB Slurm allocation, load Git, select this checkout on
`PYTHONPATH`, and set numerical thread limits to four. Use writable node-local
Numba/PyTensor caches as in the repository cluster guidance. Given the source
artifact path and an installed runtime, the commands are:

```bash
python scripts/profile_ope152_fixed_ou.py --self-check
python scripts/profile_ope152_fixed_ou.py --prepared INPUT --out RESULTS/rank256 --aggregation-error-rank 256 --repeats 5
python scripts/profile_ope152_fixed_ou.py --prepared INPUT --out RESULTS/comparison --repeats 3
python scripts/profile_ope152_fixed_ou.py --prepared INPUT --out RESULTS/current-only --algorithms current --repeats 3
```

The profiler checks the source hash and saves manifests, source snapshots,
working-tree patch, environment, repeat timings, line diagnostics and numerical
parity. Original artifacts remain in the task worktree under
`inversion_outputs/ope152-20261021-r2/`; earlier attempts remain in
`ope152-20261021-r1/` and `ope152-20261021-rank256-r1/`. The source observational
artifact is cluster-held and is not distributed with this repository.
