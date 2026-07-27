# BP1 report: PARIS exact-mixture root spectrum and resources

## Disposition

The authenticated PARIS root-spectrum/resource probe passed:

```text
candidate:
fb119c47a3ee09a592b759132f961c4b283119d3

Slurm array:
18189056

run root:
/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_paris_probe/fb119c47a3ee09a592b759132f961c4b283119d3
```

The calculation establishes that the exact Gamma--Dirichlet allocation-error
covariance has a concentrated observation-space spectrum.  It also shows that
the current all-at-once balanced-Sobol source builder is not an acceptable
PARIS construction: its known simultaneous temporary-array floor is
22.48 GiB before active-tree, inverse-Beta, clustering, input, and numerical
library workspaces.

This result licenses a memory-bounded source-bank implementation and its own
replay tests.  It does not license a PARIS conditional posterior, a particular
non-Gaussian mixture rank, or structural inference over partitions or \(K\).
Nothing was written to `PARIS_inversions`.

## Frozen identity and preflight

The two tasks authenticated the same frozen input before and after loading:

```text
input:
/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc

input ID/schema:
paris-may-2014-gamma-beta-native-v1

whole-file SHA-256:
24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044

shape:
1,382 observations x 23,424 native cells
six fixed outer-region columns
```

The source converts the unit-scaling response to response per unit physical
native-cell mass using

\[
H_i = \frac{F_i}{u_i},
\qquad
u_i =
\frac{\widetilde u_i}{\sum_j\widetilde u_j}.
\]

For concentration \(\eta\), the probe sets
\(\alpha_i=\eta u_i\) and constructs the exact unit-root whitened covariance

\[
S_0 =
A\frac{\operatorname{diag}(u)-uu^\mathsf T}{\eta+1}A^\mathsf T,
\qquad
A =
\operatorname{diag}(\sigma)^{-1}
\left(H-Hu\mathbf1^\mathsf T\right).
\]

This is the covariance of the exact continuous Gaussian location mixture,
not a Gaussian assumption about that mixture.  The detailed probability
contract and the distinction between projection, Gaussian complement, and
mixture compression are in
[`rjmcmc_exact_mixture_compression.md`](rjmcmc_exact_mixture_compression.md).
The underlying derivations are cross-referenced there to
`inversions-knowledge/docs/derivations/non-gaussian-aggregation-error-by-marginalization.md`
and `inversions-knowledge/docs/source-notes/aggregation-error-and-priors.md`.

The BP1 preflight passed:

```text
62 focused tests
Ruff formatting and checks
Pyright: 0 errors
Python 3.10.20
NumPy 2.2.6
SciPy 1.15.2
xarray 2025.6.1
Pixi 0.69.0
```

The one emitted warning is the already-known binary-extension import warning
from the pinned environment; it did not alter any numerical or static gate.

## Spectrum

Both historical reference concentrations produced numerical rank 1,381 out
of the algebraic ceiling 1,382.  Their cumulative ranks agree exactly:

| Retained analytic trace | Required rank |
|---:|---:|
| 90% | 43 |
| 95% | 74 |
| 99% | 177 |
| 99.9% | 398 |
| 99.99% | 665 |
| 99.999% | 945 |

The normalized spectral shape is concentrated:

```text
stable rank:                    3.1723
spectral-entropy effective rank: 27.5571
```

Those two summaries do not make the mixture rank three or 28.  In particular,
retaining a covariance direction is not the same as proving that its
non-Gaussian location distribution must be represented explicitly.

The diagnostic ranks retain:

| Rank | Analytic trace retained |
|---:|---:|
| 16 | 75.3262% |
| 32 | 86.3458% |
| 64 | 93.9058% |
| 128 | 98.0593% |
| 256 | 99.6072% |
| 512 | 99.9640% |
| 1,024 | 99.99949% |

The intended hybrid does not discard all directions outside a chosen
non-Gaussian rank \(q\).  It represents the leading \(q\) directions with the
compressed location mixture, uses the analytic Gaussian moment closure for
the retained complement, and omits only a separately declared small
projection tail.  Consequently, the table is a resource diagnostic rather
than a post-hoc rank selection.

## Cross-concentration audit

The two historical concentrations are scaling diagnostics:

| \(\eta\) | Total variance per squared root mass | Numerical tolerance |
|---:|---:|---:|
| 100 | 33,395.7824692 | \(5.0301\times10^{-6}\) |
| 500 | 6,732.48309259 | \(1.0141\times10^{-6}\) |

The full retained spectra contain 1,381 values each.  After multiplication by
\(\eta+1\), their maximum absolute disagreement is
\(6.98\times10^{-10}\), with maximum relative disagreement
\(1.85\times10^{-9}\).  This confirms the required
\((\eta+1)^{-1}\) scaling to floating-point precision.

The physical-mass prior-mean closure passed identically in both tasks:

```text
maximum absolute error: 5.68e-14
RMSE:                   8.24e-15
tolerance:              4.79e-7
```

Neither artifact used observations, a partition, or \(K\) to select its
spectrum.  Both explicitly record that no posterior, production output,
protected-catalogue access, or structural inference occurred.

Concentrations 100 and 500 must not be compared as if they were the same
native prior.  A scientific partition-invariance experiment needs one common
native alpha field, with its concentration chosen independently of partition
and \(K\).

## Resource audit

The current \(S=65{,}536\) balanced-Sobol implementation materializes:

| Array/resource | Bytes | GiB |
|---|---:|---:|
| All native shares | 12,280,922,112 | 11.44 |
| Largest joint Sobol uniform block | 11,115,429,888 | 10.35 |
| Known simultaneous lower bound | 24,136,058,968 | 22.48 |
| Full-rank projected source, persistent | 739,706,968 | 0.689 |

The 22.48-GiB figure excludes active balanced-tree masses, SciPy inverse-Beta
temporaries, clustering arrays, the loaded NetCDF and physical-mass operator,
and Python/BLAS/LAPACK workspaces.  It is an implementation floor for the
current builder, not a mathematical memory lower bound for exact-mixture
methods.

Persistent projected-source storage is much smaller when only leading
non-Gaussian coordinates are kept:

| Mixture rank \(q\) | Projected source | Compressed artifact, \(M=256\) |
|---:|---:|---:|
| 16 | 8.55 MiB | 15.66 MiB |
| 32 | 16.72 MiB | 18.72 MiB |
| 64 | 33.05 MiB | 30.85 MiB |
| 128 | 65.73 MiB | 79.10 MiB |
| 256 | 131.08 MiB | 271.60 MiB |
| 512 | 261.78 MiB | 1.016 GiB |
| 1,024 | 523.18 MiB | 4.018 GiB |

At high \(q\), full component covariances dominate the compressed artifact.
This supports testing modest non-Gaussian ranks with an analytic Gaussian
complement, but does not by itself choose \(q\).

## Runtime and immutable evidence

Both Slurm tasks completed with exit code zero:

| Task | \(\eta\) | Elapsed | Total CPU | Peak batch RSS |
|---|---:|---:|---:|---:|
| `18189056_0` | 100 | 44 s | 9.798 s | 1.835 GiB |
| `18189056_1` | 500 | 44 s | 9.833 s | 1.833 GiB |

The canonical output artifacts are:

```text
probe/concentration-100.json
SHA-256:
a739fc5d2fcc31928b497cd5ee559f013d064b84ac5ad0c59f8780bd6f742206

probe/concentration-500.json
SHA-256:
7e5cdf842c9473f94e0a53ddd6b1bde08dcc269485b5c58f0716b74e473440f4
```

All preflight, probe, marker, and log entries verify against:

```text
report/evidence-sha256sums.txt

inventory SHA-256:
b5b86499259b3310394650e09d587cb4d055f30eedfc073c729f928ba0721737
```

## Next bounded implementation

Implement a source-bank constructor whose peak memory scales with a declared
sample chunk and projected mixture rank, rather than
\(S\times N_{\rm native}\).  It must:

1. preserve the frozen scrambled-Sobol coordinate catalogue and source seed;
2. record the chunking method in the construction identity;
3. establish allocation and projected-residual parity against the existing
   all-at-once builder on small and moderate cases;
4. retain create-only serialization, exact replay, and permutation tests;
5. report peak RSS on the frozen PARIS input before clustering; and
6. keep the concentration and non-Gaussian rank as explicit protocol inputs,
   not data-tuned choices.

Only after that builder passes should a PARIS source bank, compressed
likelihood, and fixed-root posterior screen be submitted.
