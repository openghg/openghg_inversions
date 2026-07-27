# BP1 Native-Quadrature Marginal Report

## Outcome

The support-aware native-quadrature architecture passed G0, the complete
24-task G1 matrix, the common-lock merger, all 18 G2 confirmation shards, and
the G2 certifier.  G1 locked the smallest declared order, `O=24`, because all
six cases passed at every declared order `24, 32, 40, 48`.  G2 published
`all_tasks_pass=true` and `holdout_eligible=true`.

This is the first tested architecture in this sequence to provide both:

- a normalized conditional density for the marginal induced by the fixed
  linear basis map; and
- a forward simulator from exactly the same finite weighted marginal.

The result is a deterministic non-RJ approximation to the common native
Gamma--Dirichlet model after conditioning on retained root mass and
integrating the hidden allocation.  It is not a posterior estimator and does
not make evidence differences into basis weights.

## Identities

| Item | Identity |
| --- | --- |
| Branch | `codex/rjmcmc-marginal-support-convolution` |
| Scientific source commit | `b496fb6b0b8595039e131d1700c3b71b6ac1f9b0` |
| Detached source | `/group/chem/acrg/brendan_for_codex/openghg_inversions-worktrees/rjmcmc-marginal-native-quadrature-b496fb6b0b8595039e131d1700c3b71b6ac1f9b0` |
| Run root | `/group/chem/acrg/brendan_for_codex/rjmcmc_marginal_native_quadrature/b496fb6b0b8595039e131d1700c3b71b6ac1f9b0` |
| Driver SHA-256 | `a2f850d75a61fd90182e978574c7774afb7160a44e0ce44ed4d1109368ee4899` |
| Protocol SHA-256 | `d11ba4b37c973772a0ad2bc6caff1e29c56deac0ef6d94fd348a25f468c9b49c` |
| Runtime | Pixi `0.69.0`; Python `3.10.20`; NumPy `2.2.6`; SciPy `1.15.2` |
| G0 marker SHA-256 | `756b695e5e9bcdb841e4b92d03f940b3a8d3a20d97322c93ade5493414012487` |
| G1 certificate SHA-256 | `d6ce7e448886c827a77d6085c27058fd15e05558ee7edaed1f1f307c789fb0ed` |
| G1 lock-file SHA-256 | `d80c57861c00470c381cb1d29330c715957da69666d52a76e53f2d24b00e318f` |
| G2 certificate SHA-256 | `8278fd2d4dbee869c65eefcf2a8af8ff7291dccc0ad23b79abef025cfed4ea35` |

The detached source owns its `.pixi` environment and remained clean at the
scientific source commit.  The run contains 165 files occupying 23 MiB; the
complete relative-path manifest is
`rjmcmc_marginal_native_quadrature_bp1_manifest.sha256`.

## What was tested

For two native cells, an `O`-point normalized Gauss--Jacobi rule integrates
the exact conditional Beta allocation.  For four native cells, three native
Beta coordinates form an `O**3` positive-weight tensor rule.  The allocation
pushforward is analytically convolved with Gaussian measurement noise in the
complete error-whitened residual image.

The candidate density is normalized for every admissible retained mass.
Simulation chooses one authenticated quadrature component and adds the same
Gaussian kernel used in the density.  The simulator therefore targets the
finite quadrature marginal, not a separately fitted model.  Its exact
`O -> infinity` limit is the common conditional native model and is invariant
to computational chart, partition, and component count.

## G0

The committed local preflight passed:

- 24 focused experimental tests;
- Ruff;
- focused Pyright with zero errors;
- shell syntax for every committed launcher;
- exact source, driver, protocol, Python, NumPy, and SciPy identities;
- bounded `O=8` smoke construction;
- canonical same-process and separate-process replay; and
- deterministic density, gradient, component-index, and simulator replay.

Cross-node Slurm job `18189081` passed on `bp1-compute071` with exit `0:0`.
Its binary64 likelihood, gradient, selected components, and simulator draws
matched the local replay exactly.

Before G0, an intentionally non-authoritative `uv` test invocation used
different runtime versions and failed two protocol-digest assertions.  This
failure was retained in the work record.  Repeating the identical 24 tests in
the source-pinned Pixi environment passed 24/24; only that locked result was
admitted to G0.

The earlier FlowJAX and sbi-NSF attempts added and pinned their requested
dependencies, including FlowJAX, sbi, nflows, and PyTorch.  Those dependency
changes are preserved.  This successful architecture needed no additional
package beyond the already locked NumPy/SciPy stack; that was an engineering
outcome, not a restriction against adding dependencies.

## G1 development result

Slurm array `18189082` comprised jobs `18189082` through `18189105`.  Every
one of the 24 tasks completed with exit `0:0`, published artifact, report, and
completion marker in order, and passed every unchanged scientific check.
Merger job `18189110` completed with exit `0:0` and locked `O=24`.

All four orders passed for all six cases:

| Case | O24 | O32 | O40 | O48 |
| --- | ---: | ---: | ---: | ---: |
| near-Gaussian, two cell | pass | pass | pass | pass |
| near-Gaussian, four cell | pass | pass | pass | pass |
| skewed, two cell | pass | pass | pass | pass |
| skewed, four cell | pass | pass | pass | pass |
| boundary-heavy, two cell | pass | pass | pass | pass |
| boundary-heavy, four cell | pass | pass | pass | pass |

The locked `O=24` metrics below are errors relative to the source-pinned C1
truth likelihood.  “Gradient” is the maximum scaled coordinate-gradient
error.

| Case | Components | p99 log-likelihood error (nat) | Gradient | Absolute log-evidence error (nat) |
| --- | ---: | ---: | ---: | ---: |
| near-Gaussian, two cell | 24 | `2.22e-16` | `1.23e-12` | `4.44e-16` |
| near-Gaussian, four cell | 13,824 | `5.33e-15` | `4.75e-12` | `4.88e-15` |
| skewed, two cell | 24 | `6.88e-14` | `4.08e-09` | `9.08e-12` |
| skewed, four cell | 13,824 | `9.08e-15` | `4.22e-12` | `4.44e-16` |
| boundary-heavy, two cell | 24 | `3.17e-02` | `1.85e-02` | `9.20e-09` |
| boundary-heavy, four cell | 13,824 | `1.26e-12` | `1.13e-02` | `2.32e-08` |

Every posterior-mean, posterior-SD, interval-endpoint, likelihood,
gradient, and evidence gate passed.  Exact Dirichlet residual moment audits
also passed.

## G2 simulator confirmation

Slurm array `18189113` comprised jobs `18189113` through `18189130`.  All 18
shards completed with exit `0:0`.  Each case used the locked `O=24` artifact
at seeds `1877`, `4099`, and `8317`, drawing 131,072 observations per shard.
Certifier job `18189135` completed with exit `0:0`.

The table reports the worst result across the three confirmation seeds.

| Case | Max mean error (MCSE) | Max covariance error (MCSE) | Minimum frequency-test p | Finite density audit |
| --- | ---: | ---: | ---: | --- |
| near-Gaussian, two cell | 2.398 | 2.431 | 0.0951 | pass |
| near-Gaussian, four cell | 2.043 | 2.362 | 0.1322 | pass |
| skewed, two cell | 0.966 | 1.816 | 0.1347 | pass |
| skewed, four cell | 2.170 | 1.934 | 0.0350 | pass |
| boundary-heavy, two cell | 2.140 | 2.578 | 0.5289 | pass |
| boundary-heavy, four cell | 1.878 | 2.079 | 0.0855 | pass |

All errors were below the predeclared five-MCSE limit.  Every 256-draw
density audit was finite.  All canonical grouped component-frequency tests
passed the `p >= 1e-6` gate with minimum expected counts above 512 except the
two-cell rules, whose minimum expected counts were also safely above 20.
Repeated scientific metrics were seed-invariant, and the between-seed
log-evidence range was exactly zero for every case.

## Computational-chart leakage diagnostic

The published four-cell artifacts use the frozen column-first native chart.
The row-first same-order audit is diagnostic only.  It is nearly identical
for the near-Gaussian case, but finite-order chart leakage is material in
harder cases:

| Case | Order | Maximum row/column log-likelihood difference (nat) | Maximum mass-gradient difference |
| --- | ---: | ---: | ---: |
| skewed, four cell | 24 | 0.819 | `3.02e-12` |
| skewed, four cell | 48 | 0.118 | `2.31e-14` |
| boundary-heavy, four cell | 24 | 3,903.6 | 1.694 |
| boundary-heavy, four cell | 48 | 835.1 | 0.0131 |

This does not alter the G1 decision: the audit was predeclared as a finite
diagnostic, while the published artifacts passed the independently scored
truth gates at every order.  It does qualify interpretation.  The
boundary-heavy tensor rule is not uniformly chart-converged over every
diagnostic state at the locked finite order.  These discrepancies are
numerical leakage diagnostics only.  They are not structural information,
model probabilities, or data-dependent basis weights.

## Decision and protected status

The experiment achieved its stated G1 and G2 goals.  The locked
native-quadrature density and simulator are a validated finite approximation
for the BP1 root cases under the declared truth and thresholds.

G2 made the source-pinned artifacts holdout-eligible, but no G3 action was
taken.  The protected catalogue remains sealed.  No ladder extension,
threshold change, learned correction, component escalation, conditional row
model, or evidence-derived basis weighting was introduced.
