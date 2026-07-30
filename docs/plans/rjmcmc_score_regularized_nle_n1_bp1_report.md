# Score-Regularized Projected-Marginal NLE: BP1 N1 Result

## Decision

The score-regularized root-only neural likelihood stopped at N1.  The
authoritative merger authenticated all 24 declared task bundles but found no
passing task at any training size.  It therefore published no common
all-six-case, all-larger two-size lock.

The certificate's terminal decision is:

```text
no common all-six-case all-larger passing suffix of length at least two;
terminal N1 architecture stop
```

N2 confirmation, the PARIS engineering/development stages, and synthetic
calibration were not run.  The protected catalogue remained sealed, and no
output was written to `PARIS_inversions`.

## What Was Tested

This was an observation-blind synthetic approximation experiment for the
marginal distribution induced by applying a fixed linear projection to one
native Gamma--Dirichlet root model and adding Gaussian measurement noise.  It
did not use the Lunt-like CH4 observations.

Here, “truth” means the exact tiny-model likelihood, posterior, evidence, and
fixed-observation retained-mass derivative evaluated by the declared
two-cell and four-cell synthetic oracles.  It is approximation truth for
these small probability models, not atmospheric flux truth and not
observation-space ground truth.

The six public cases combined near-Gaussian, skewed, and boundary-heavy
allocation regimes with two-cell and four-cell native models.  Their retained
projection dimensions were respectively \(q=1\) and \(q=3\).  Each case was
trained at

\[
S\in\{4096,16384,65536,262144\},
\]

using exact nested simulation prefixes, two frozen initializations, an
independent model-selection domain, and an independent development reporting
domain.

The learned density was a normalized conditional rational-quadratic-spline
FlowJAX model.  Its objective combined negative log likelihood with exact
simulator-derived supervision of the raw-log-root-mass score.  The recovered
implementation used a scalar forward JVP for that score followed by the outer
reverse parameter gradient.  Direct tests established that this is the same
derivative and composite objective gradient as the original
reverse-over-reverse formulation; it changes the compiler graph, not the
scientific model.

The Gaussian term outside the learned leading coordinates remained a
moment-closure approximation.  It was not treated as an exact statement about
the native projected marginal.

## Thresholds In Plain Language

The frozen scientific thresholds were:

| Quantity | Required error |
|---|---:|
| Prior-weighted median absolute conditional log likelihood | at most 0.05 nat |
| Posterior-weighted 99th-percentile absolute conditional log likelihood | at most 0.20 nat |
| Absolute log evidence | at most 0.05 nat |
| Scaled retained-mass derivative | at most 0.05 |
| Posterior mean | at most 0.05 reference SD |
| Posterior SD | at most 2% relative error |
| Posterior interval endpoints | at most 0.05 reference SD |

A 0.05-nat log-density or log-evidence error corresponds to a multiplicative
factor of about \(e^{0.05}=1.051\), while 0.20 nat corresponds to about
\(e^{0.20}=1.221\).  These are roughly 5% and 22% density/evidence factors,
not percentage errors in flux.  The median and 99th-percentile likelihood
metrics use different prior and posterior weightings, so their numerical
ordering need not follow that of two percentiles from one sample.

A task also required a valid fit, finite scores, exact artifact replay, and
independent model-selection-versus-reporting NLL agreement.  The size lock
required every case to pass at two consecutive sizes through the top of the
ladder.

## What Happened

The initial mixed reverse-mode derivative graph exhausted LLVM executable
section memory before fitting.  Doubling requested memory and serializing XLA
CPU code generation did not resolve that technical failure.  The exact
forward-JVP refactor did: both \(q=1\) and \(q=3\) compile canaries passed,
and all 24 scientific fits then completed on ordinary shared nodes.

All 24 fits were valid, all 24 score checks were finite, and all 24 serialized
artifacts replayed byte-for-byte.  Only one task passed the independent
generalization check, and none passed the combined scientific thresholds.
The failures therefore reflect approximation accuracy, not a missing file,
failed optimizer, non-finite density, serialization problem, or merger
technicality.

The table below shows the four most direct likelihood/derivative diagnostics.
`NG`, `SK`, and `BH` denote near-Gaussian, skewed, and boundary-heavy cases;
`2` and `4` denote the native cell count.  Likelihood and evidence errors are
in nat.  Every row also failed the overall scientific decision after the
posterior diagnostics were included.

| \(S\) | Case | Median | p99 | Evidence | Gradient |
|---:|---|---:|---:|---:|---:|
| 4,096 | NG-2 | 0.073 | 0.648 | 0.045 | 3.776 |
| 4,096 | NG-4 | 0.570 | 0.644 | 0.559 | 0.749 |
| 4,096 | SK-2 | 0.906 | 1.029 | 0.375 | 3.295 |
| 4,096 | SK-4 | 0.420 | 0.611 | 0.430 | 0.553 |
| 4,096 | BH-2 | 33.557 | 0.234 | 0.983 | 0.721 |
| 4,096 | BH-4 | 1.998 | 0.610 | 0.613 | 0.567 |
| 16,384 | NG-2 | 0.018 | 0.328 | 0.138 | 2.825 |
| 16,384 | NG-4 | 0.385 | 0.449 | 0.396 | 0.373 |
| 16,384 | SK-2 | 0.944 | 3.613 | 0.230 | 4.320 |
| 16,384 | SK-4 | 0.546 | 0.713 | 0.553 | 0.619 |
| 16,384 | BH-2 | 33.514 | 0.601 | 3.130 | 0.505 |
| 16,384 | BH-4 | 4.160 | 4.160 | 3.996 | 1.776 |
| 65,536 | NG-2 | 0.206 | 0.261 | 0.192 | 1.945 |
| 65,536 | NG-4 | 0.318 | 0.437 | 0.314 | 0.723 |
| 65,536 | SK-2 | 0.835 | 1.756 | 0.808 | 1.501 |
| 65,536 | SK-4 | 0.586 | 1.707 | 0.357 | 3.005 |
| 65,536 | BH-2 | 33.698 | 0.604 | 4.438 | 0.596 |
| 65,536 | BH-4 | 4.673 | 4.673 | 4.125 | 2.976 |
| 262,144 | NG-2 | 0.256 | 0.636 | 0.291 | 1.860 |
| 262,144 | NG-4 | 2.463 | 2.662 | 2.463 | 0.703 |
| 262,144 | SK-2 | 1.007 | 2.233 | 0.396 | 3.242 |
| 262,144 | SK-4 | 0.579 | 1.967 | 0.204 | 2.545 |
| 262,144 | BH-2 | 41.235 | 0.263 | 1.012 | 0.678 |
| 262,144 | BH-4 | 5.256 | 5.256 | 3.753 | 8.584 |

There was no monotone convergence toward the thresholds.  At the largest
size, every case still exceeded the median likelihood, evidence, and
retained-mass derivative limits.  More simulations alone are therefore not
supported as a recovery.

## Source And Artifact Identity

The authoritative execution identity was:

```text
branch:
  codex/rjmcmc-score-regularized-nle
source revision:
  475d5db6026a8472fd3c44eac2e0d2369686c78b
detached source:
  /group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle/
  source-475d5db6026a8472fd3c44eac2e0d2369686c78b
run root:
  /group/chem/acrg/brendan_for_codex/rjmcmc_score_regularized_nle/
  run-475d5db6026a8472fd3c44eac2e0d2369686c78b
driver SHA-256:
  f9ff6221ca8ce92506225dd50d0b4073a1693dbc5894b965a3aaeb1b6f555aba
protocol SHA-256:
  ec40ba6c1f73f511d7f5766310a0889a1bea2d4c742c075f9d8ad771dcd239af
```

The authoritative merger outputs were:

| Evidence | SHA-256 |
|---|---|
| `development-merge/development-certificate.json` | `794d4ea42e2b7642c83974168f51c7310a7a7885a7f74d46f2fd5d6234b0db0c` |
| `development-merge/MERGE_COMPLETE.json` | `a40fdbc1ebdee480bfa2d68620a7051bd6995e216843f9be5fd78d38f9b0ee6a` |
| `logs/development-merge/N1_merge.18215095.log` | `a40fdbc1ebdee480bfa2d68620a7051bd6995e216843f9be5fd78d38f9b0ee6a` |

The certificate has payload hash
`153a468d9cf0e1e9a13ad0a983e4557f0bb1c6f0729d4081302cf9f40572237b`.
It records the artifact, report-payload, report-file, and final-marker hashes
for every task.  The companion
`rjmcmc_score_regularized_nle_n1_bp1_manifest.sha256` records the complete
file-byte SHA-256 manifest relative to the run root, including N0, canaries,
all 24 task triplets, all task logs, the certificate, and the final marker.
The manifest file itself has SHA-256
`062db670301b4d7e2fa271d3adaa0dffcc89f2c3a0a22076a2245dc04be4c8be`.

## Slurm And Resource Evidence

Every repeated scientific tier ran as one six-task array on ordinary shared
nodes.  The merger was a distinct serial job because it authenticated and
reduced the completed matrix rather than fitting homogeneous candidates.

| Stage | Primary job | Callback | Wakeup ticket |
|---|---:|---:|---|
| N0 | `18214772` | `18214773` | `sw-20260729T230100Z-fe11d9a8d215` |
| Compile canary | `18214827_[0-1]` | `18214828` | `sw-20260729T231322Z-3ecf4abe0fdf` |
| \(S=4096\) | `18214844_[0-5]` | `18214845` | `sw-20260729T231857Z-df2fcdaeede4` |
| \(S=16384\) | `18214933_[0-5]` | `18214934` | `sw-20260729T233916Z-005741040441` |
| \(S=65536\) | `18214966_[0-5]` | `18214967` | `sw-20260730T001150Z-2d634391069f` |
| \(S=262144\) | `18215006_[0-5]` | `18215007` | `sw-20260730T012749Z-c90189b427ed` |
| N1 merger | `18215095` | `18215096` | `sw-20260730T034013Z-7328c2dc1fa1` |

The table below distinguishes requested memory from measured peak resident
memory per task.  Walltime requests were ceilings and did not alter the
scientific computation.

| Stage | Request per task | Elapsed range | Peak RSS range |
|---|---:|---:|---:|
| N0 | 8 GB | 6:15 | 4.19 GiB |
| Compile canary | 8 GB | 1:23 | 0.48–0.49 GiB |
| \(S=4096\) | 8 GB | 11:22–15:38 | 4.51–4.84 GiB |
| \(S=16384\) | 8 GB | 14:48–27:56 | 4.53–4.79 GiB |
| \(S=65536\) | 8 GB | 21:50–1:09:51 | 4.60–4.74 GiB |
| \(S=262144\) | 8 GB | 41:28–2:09:05 | 4.54–4.78 GiB |
| N1 merger | 2 GB | 0:20 | 0.39 GiB |

## Interpretation

The forward-JVP recovery successfully made the declared score-regularized
flow trainable on BP1.  It did not make the learned marginal accurate enough.
Increasing \(S\) by a factor of 64 did not produce a common passing suffix,
and the largest fits retained substantial likelihood, evidence, posterior,
and mass-derivative errors.

This result is terminal for the frozen score-regularized root-flow
architecture.  It does not show that the exact native marginal depends on
the basis partition or \(K\), and the approximate evidence errors must not be
used as structural weights.  It also does not disprove all possible marginal
representations.  A different native model, multi-root factorization,
conditional row construction, or learned architecture would be a new
scientific experiment requiring an independently predeclared plan; none is
authorized by this result.

The preceding GMM, finite projected-bank, likelihood-only FlowJAX,
`sbi`/`nflows`, and score-regularized NLE results together show that the
current root-only approximation route has been tested beyond a simple lack of
mixture components, simulations, or neural-network availability.

## Follow-Up

No further job is permitted under this plan.  N2--N6 remain closed.  Retain
the run root and its hashes as the terminal evidence bundle.  If work
continues, begin from a new scientific question and frozen model plan rather
than tuning this development matrix or using its evidence errors to select a
basis.
