# BP1 test plan: PARIS exact-mixture root-spectrum resource probe

## Purpose

This bounded real-input probe follows the successful all-18-shard
confirmation in
[`rjmcmc_exact_mixture_confirmation_bp1_report.md`](rjmcmc_exact_mixture_confirmation_bp1_report.md).
It measures the analytic Gamma--Dirichlet root residual spectrum and
deterministic array-resource requirements on the frozen May 2014 PARIS input.

The probe does not:

- construct a PARIS Sobol allocation bank;
- cluster PARIS source locations;
- fit or sample a posterior;
- use the observed residual to choose a basis or rank;
- compare partitions or \(K\); or
- write to `PARIS_inversions`.

## Frozen input and model references

```text
input:
/group/chem/acrg/brendan_for_codex/rjmcmc_gamma_beta_hpc/dd687b92abb86ce0080a1c8a713f3eb9a57df3aa/input/paris_may_2014_gamma_beta_native.nc

input ID:
paris-may-2014-gamma-beta-native-v1

whole-file SHA-256:
24da69cab978051608313901b1c958200e0ad885a0a349bfa4fa1f9a0aaad044

shape:
1,382 observations x 183 latitude x 128 longitude

ordered outer labels:
intem_label_0,...,intem_label_5
```

The adapter converts unit-scaling response \(F_i\) to response per unit
physical native-cell mass:

\[
u_i=\frac{\widetilde u_i}{\sum_j\widetilde u_j},
\qquad
H_i=\frac{F_i}{u_i}.
\]

For one explicit global additive concentration \(\eta\),
\(\alpha_i=\eta u_i\).  With diagonal observation errors \(\sigma\), the
unit-root whitened covariance is

\[
S_0 =
A\frac{\operatorname{diag}(u)-uu^\mathsf T}{\eta+1}A^\mathsf T,
\qquad
A=\operatorname{diag}(\sigma)^{-1}
(H-Hu\mathbf1^\mathsf T).
\]

The spectrum is observation-blind.  At root mass \(T\), it scales as
\(T^2S_0\).

## Concentration matrix

Run two historical reference concentrations:

```text
100
500
```

These reproduce the allocation concentration scales used by the earlier
fixed-\(K=50\) and fixed-\(K=250\) experiments.  They are resource/scaling
diagnostics only.  Using \(\eta=2K\) defines different native priors across
\(K\) and cannot support representation-invariant structural inference.
Any subsequent scientific conditional-likelihood experiment must predeclare
one common native concentration independently of partition and \(K\).

Mathematically the two spectra differ by the scalar
\((100+1)/(500+1)\), so eigenvectors and exact explained-variance ranks are
unchanged.  Both are constructed because the implementation's
floating-point numerical-rank tolerance is part of the measured contract.

## Candidate identity

Use a fresh detached full-SHA worktree from:

```text
branch: codex/rjmcmc-exact-mixture-compression
```

The candidate SHA must contain the probe, focused tests, this plan, and both
launch assets.  Create:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_exact_mixture_paris_probe/<full-SHA>
```

Any source change requires a new commit, push, worktree, and run root.

## P0: preflight

Run the committed preflight on the quiet login node.  It must:

1. verify the clean full-SHA worktree apart from the canonical `.pixi` link;
2. authenticate the frozen input before any numerical work;
3. run focused exact-mixture, confirmation, and probe tests;
4. run focused Ruff formatting/checks and Pyright;
5. record Python, NumPy, SciPy, xarray, and Pixi versions; and
6. publish `PREFLIGHT_COMPLETE.txt` last.

Do not run the repository-wide tox matrix.

## P1: two-task Slurm matrix

Submit the committed array:

```text
array: 0-1
concentrations: 100, 500
account: chem007981
CPUs per task: 1
memory per task: 16 GiB
wall time: 30 minutes
BLAS/OpenMP/NumExpr threads: 1
```

Each task uses the existing public adapter, full-tiling bridge,
`RootResidualSpectrum`, and SciPy `eigh`.  It publishes:

```text
probe/concentration-<eta>.json
markers/concentration-<eta>.complete
logs/concentration-<eta>.<job-id>.log
```

The JSON must be canonical and create-only.  `/usr/bin/time -v` and `sacct`
provide process and Slurm resource evidence; `sacct MaxRSS` is authoritative.

## P2: interpretation gate

Authenticate both artifacts and record:

- physical-mass prior-mean closure;
- numerical rank and tolerance;
- full eigenvalue identities;
- ranks at 90%, 95%, 99%, 99.9%, 99.99%, and 99.999% trace;
- omission KL/TV coefficients at ranks 16--1,024 and full rank;
- full-rank current source-artifact storage;
- candidate \(q\)-rank projected-source and compressed storage;
- the current Sobol shares array;
- the current largest Sobol uniform block; and
- their simultaneous lower bound before active-tree/library workspaces.

Both artifacts must pass input/profile/closure/side-effect checks.  Their
explained-variance ranks must agree.  After multiplying eigenvalues by
\(\eta+1\), overlapping numerical spectra must agree within a strict
floating-point tolerance.  A mismatch is a software/identity failure, not a
scientific tuning opportunity.

## Decision boundary

This probe may authorize design work for a memory-bounded PARIS source-bank
builder.  It does not itself authorize the bank, compression, likelihood,
posterior, or structural inference.

In particular:

- trace rank controls mean-square whitened residual, not non-Gaussianity;
- the KL/TV omission coefficients are distributional, not pointwise
  likelihood/evidence bounds;
- keeping remaining directions as a Gaussian complement avoids literal
  omission but remains a shape approximation;
- current Sobol temporary-memory estimates are implementation facts, not
  mathematical lower bounds on all possible exact-mixture algorithms; and
- one coherent concentration must be chosen before a scientific posterior
  experiment.

## Required report

Record candidate identity, paths, input digest, P0 checks, job ID and terminal
states, elapsed/CPU/MaxRSS, both canonical artifacts, spectrum/rank tables,
resource estimates, cross-concentration scaling audit, SHA-256 inventory, and
explicit confirmation that no protected catalogue or `PARIS_inversions`
output was touched.
