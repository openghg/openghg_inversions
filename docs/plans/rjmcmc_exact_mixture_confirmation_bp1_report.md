# BP1 report: exact-mixture independent-scramble confirmation

## Disposition

The root-only structure-preserving mixture passed the frozen 18-shard
independent-scramble confirmation:

```text
sampling candidate:
98780405664d069366d5a2c143f7ee38d9c6b305

reporting-only certifier fix:
3792ed031fa4f38f6ef00e312d7424f2e6adf978

run root:
/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_confirmation/98780405664d069366d5a2c143f7ee38d9c6b305

certifier recovery evidence:
/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_confirmation_certifier_recovery/3792ed031fa4f38f6ef00e312d7424f2e6adf978
```

All six tiny exact-quadrature oracle cases passed at each independent
scrambled-Sobol source seed 1,877, 4,099, and 8,317.  The frozen construction used
\(S=65{,}536\) source locations, \(M=256\) moment-preserving components, and
cluster seed 731.  No retuning occurred.

This licenses a PARIS rank/resource probe and then a separately gated
fixed-root conditional likelihood experiment.  It does not establish
multi-region accuracy, a PARIS posterior, partition- or \(K\)-invariant
evidence, or authority to update structural weights.

Nothing was written to `PARIS_inversions`.

## Validation and certifier recovery

The original confirmation C0 passed 47 focused tests and static checks.  All
18 C1 tasks completed and published valid immutable JSON plus exact completion
markers.  The first C2 invocation failed before publishing any decision
because the certifier attempted to read
`between_bank_log_evidence_range_nat` as a per-shard metric.  That statistic
is defined across independent source banks.

The failure was reporting-only:

- no sampler artifact was changed or regenerated;
- no decision, digest, or completion marker existed;
- the corrected certifier requires every per-shard metric and finite evidence;
- it computes source and compressed evidence ranges across all three seeds
  for each case; and
- the recovery marker records both the original artifact revision and the
  certifier revision.

The fresh recovery preflight passed:

```text
50 focused tests
Ruff format and checks
Pyright: 0 errors
Python 3.10.20
NumPy 2.2.6
SciPy 1.15.2
Pixi 0.69.0
```

The recovered decision is canonical and eligible:

```text
raw decision SHA-256:
1691ae25fe46fa3fc432d1e46f6f2943826051075dae60b879f92213d7143a73
```

## Scientific confirmation

The largest direct-source errors across all 18 shards were:

| Metric | Maximum |
|---|---:|
| Absolute log-evidence error | \(1.41\times10^{-5}\) nat |
| Posterior mean error | \(5.27\times10^{-5}\) reference SD |
| Posterior SD relative error | \(3.74\times10^{-4}\) |
| Scaled coordinate-gradient error | \(1.19\times10^{-2}\) |

The largest compressed-mixture errors were:

| Metric | Maximum |
|---|---:|
| Absolute log-evidence error | \(7.62\times10^{-4}\) nat |
| Posterior mean error | \(2.40\times10^{-3}\) reference SD |
| Posterior SD relative error | \(1.69\times10^{-2}\) |
| Scaled coordinate-gradient error | \(1.03\times10^{-2}\) |

The evidence-spread limit was 0.05 nat.  The largest observed spread across
the three independent source scrambles was:

```text
direct source:       1.64e-5 nat
compressed mixture:  2.44e-4 nat
```

The worst compressed spread occurred in the boundary-heavy four-cell case.
It remained about 205 times smaller than the frozen limit.

Compression preserved finite-bank moments to numerical precision:

```text
maximum mean closure error:       1.66e-15
maximum covariance closure error: 1.71e-13
```

The finite-bank compression KL upper bound ranged from
\(2.91\times10^{-7}\) to 0.1187 across cases and seeds.  It is an integrated
unit-root source-versus-compression bound, not a pointwise likelihood bound.

## Runtime and storage

Slurm array `18188962` completed all 18 tasks with exit code zero.  Task
elapsed times were 49--81 seconds.  Peak batch-step RSS was 310--368 MiB.

Across the confirmation artifacts:

```text
source build:       0.125--0.654 s
compression build: 32.9--52.1 s
source evaluation: 185--685 states/s
compressed eval:   5,100--7,135 states/s
compressed-versus-finite-bank speedup:
8.65--28.36x (median 20.44x)
```

Source numerical arrays occupied 0.50--1.50 MiB.  Compressed arrays occupied
12.1--52.2 KiB.  These tiny-case timings confirm that cached component
eigendecompositions and batched NumPy evaluation removed the original
per-component Cholesky bottleneck.  The comparison is against the finite
\(S=65{,}536\) direct bank on the tiny cases, not the continuous exact mixture
or a PARIS likelihood.

## Evidence inventories

The complete confirmation run inventory verifies at:

```text
report/evidence-sha256sums.txt
inventory SHA-256:
6ddb8b903355e392b17b8faec18acad8a6c328f326b45fac09beb125ba10ae84
```

The certifier-recovery preflight inventory verifies at:

```text
report/evidence-sha256sums.txt
inventory SHA-256:
3551bae059aae3ef077a4e7f78897fc59de489fee122cb24eacfecd24126e50e
```

## Next bounded gate

Use the authenticated frozen PARIS input to measure:

1. the exact analytic root residual spectrum;
2. cumulative variance ranks and explicit projection KL/TV coefficients;
3. persistent direct-bank and compressed-mixture storage by candidate
   non-Gaussian rank; and
4. the current balanced-Sobol construction's unavoidable temporary-memory
   floor.

Do not construct the full PARIS Sobol bank until that probe has a committed
protocol and shows an acceptable resource design.  Keep one explicit native
concentration independent of partition and \(K\) for any subsequent
scientific model; historical concentrations 100 and 500 may be measured as
reference scalings but are not cross-\(K\) structural priors.
