# BP1 Direct Marginal NLE Flow: Terminal G1 Result

## What was tested

This experiment tested a normalized neural-likelihood approximation to the
observation marginal induced by a fixed linear basis map.  It was the first
independent NLE escalation after the terminal root-GMM result.

The fitted artifact models the complete projected noisy residual with a
conditional FlowJAX triangular spline flow.  Exact conditional covariance
whitening and the Gaussian residual-image complement remain outside the
learner.  The same authenticated artifact provides both `log_likelihood` and
forward observation simulation.

The immutable scientific source and protocol were:

- source Git SHA:
  `35201d9cbf2410b873055a325e14fd929fb211f6`;
- driver SHA-256:
  `0cdbe7950fd62a59725e206385408af75bdd667a63c0a315076fa1f25e832d71`;
- protocol SHA-256:
  `b4c548bcb9b83dcd2837a1a5ae88f716b3cf61a32c5acedd77ef75e1f5efcaf2`;
- detached source:
  `/group/chem/acrg/brendan_for_codex/openghg_inversions-worktrees/rjmcmc-marginal-nle-flow-35201d9cbf2410b873055a325e14fd929fb211f6`;
- run root:
  `/group/chem/acrg/brendan_for_codex/rjmcmc_marginal_nle_flow/35201d9cbf2410b873055a325e14fd929fb211f6`; and
- pinned runtime: Python 3.10.20, NumPy 2.2.6, SciPy 1.15.2,
  JAX/JAXlib 0.6.2 in float64 CPU mode, FlowJAX 17.2.1,
  Equinox 0.13.8, Optax 0.2.8, and Paramax 0.0.5.

The frozen matrix contained six root cases and training sizes 4,096, 16,384,
65,536, and 262,144, for 24 G1 shards.  Each shard used two deterministic
initializations, an independent 65,536-draw validation domain, and an
independent 131,072-draw development test domain.

## Terminology and truth

The **native truth model** is the source-pinned additive independent-Gamma
cell model.  Conditional on retained mass, native fractions have the exact
within-region Dirichlet distribution; source-pinned Gaussian measurement
noise is then added.

The **truth likelihood** is the deterministic C1 quadrature marginal over
native allocations.  It is an observation-space comparator, not a fitted
density.

The **learned marginal simulator** draws from the same normalized flow
artifact used for likelihood scoring.  Passing artifact replay therefore
shows internal density/simulator consistency, not agreement with native truth.

Approximate evidence differences are leakage diagnostics only.  They were not
used as basis weights, flow inputs, architecture choices, or training weights.

## What happened

G0 passed:

- 17 focused experimental tests;
- Ruff;
- focused Pyright;
- runtime and protocol identity;
- a five-epoch operational smoke;
- separate-process artifact replay; and
- completion-marker-last publication.

Slurm array `18188441` completed all 24 G1 shards with zero scheduler exit
codes.  All 24 fitted artifacts were finite and replayed from authenticated
bytes.

The first merger attempt, job `18188466`, failed before inspecting artifacts
because `git` was absent from the compute-node `PATH`.  Its scheduler error was
preserved.  Wrapper-only commit
`6e02bfdcaf10e0b171045101b049589d8fff7c38` added the same pinned Git module
load already used by the arrays.  Merger job `18188535` then authenticated all
24 untouched artifacts and completed.

The merger published no common lock.  Every case failed every training size,
so there was no common all-six-case, all-larger passing suffix of length two.
G2 and G3 were not submitted, and the protected catalogue remained sealed.

## Key results

All 24 fitting stages passed, but only 11 of 24 selected artifacts passed the
independent validation-versus-test NLL gate.  No artifact passed all
scientific gates.  Most decisively, the scaled coordinate-gradient gate passed
0 of 24 artifacts.

This table summarizes the four training sizes for each case.  “Gen pass”
lists sizes that passed the independent density-generalization gate; “science
pass” lists sizes that passed every unchanged C1 likelihood, gradient,
evidence, and posterior-summary threshold.

| Case | Gen pass sizes | Science pass sizes | Decision |
|---|---:|---:|---|
| near-Gaussian, two-cell | 4,096; 16,384; 65,536; 262,144 | none | gradient failed all sizes |
| near-Gaussian, four-cell | 4,096; 16,384; 65,536; 262,144 | none | likelihood/gradient failures |
| skewed, two-cell | none | none | generalization and science failures |
| skewed, four-cell | 16,384; 262,144 | none | non-monotone generalization; science failures |
| boundary-heavy, two-cell | none | none | large likelihood, gradient, and evidence errors |
| boundary-heavy, four-cell | 4,096 | none | large likelihood, gradient, and evidence errors |

The near-Gaussian two-cell case came closest.  At 65,536 training draws its
median absolute conditional log-likelihood error was 0.00420 nat, posterior
weighted p99 error was 0.0268 nat, and absolute log-evidence error was
0.00376 nat.  Its scaled coordinate-gradient error was nevertheless 0.0891,
above the frozen 0.05 threshold.  All four sizes in that case failed only the
gradient gate.

Across all 24 tasks, the number passing each threshold was:

| Gate | Passing artifacts / 24 |
|---|---:|
| median absolute conditional log-likelihood error | 4 |
| posterior-weighted p99 log-likelihood error | 8 |
| scaled coordinate-gradient error | 0 |
| absolute log-evidence error | 6 |
| posterior mean error | 12 |
| posterior SD relative error | 7 |
| interval endpoint error | 17 |

For the boundary-heavy two-cell cases, prior-weighted median absolute
log-likelihood errors ranged from 20.6 to 29.4 nat.  For boundary-heavy
four-cell cases, the median errors ranged from 1.74 to 4.29 nat.  Increasing
training size did not produce a monotone passing trend.

## Artifacts and checksums

The G1 development directory contains exactly 72 files: one flow, one
authenticated report, and one completion marker for each of 24 tasks.  Their
combined size is 630,503 bytes.

The decision artifacts are:

- full preserved-run checksum manifest: `checksums.sha256`, containing 110
  relative-path entries, raw SHA-256
  `3385fa753f14e1fca502a5f5b9b12301b5134147bd875a67edaf35823892779d`;
- development certificate raw SHA-256:
  `dbff88a3af72e4fae087f00f38518850d391565f3951663be94f812e84e357ae`;
- development certificate envelope SHA-256:
  `8e936ec7820696ac4476a530ff31ab220c8aebc15130db11109a08d2e777ddc2`;
- merger completion marker raw SHA-256:
  `a8f98f7d5e0eb2cd571a8932798e70120f40360e1367080b50ea3f8aaf78b8e6`;
- G0 completion marker raw SHA-256:
  `f97fba2d825945117a72d14eb66d97d604917bd195290805374be54d8ff69891`;
- G0 log raw SHA-256:
  `2e0458b01557d67bc1ed5475ea81087c17a0035c69aa3a4b94e1d488af82f1f2`;
- failed first-merger scheduler error SHA-256:
  `b59ad40a6428faea63a93548d096dfc8cbac954309893ac005e0ba12fcf43434`;
  and
- successful merger job log SHA-256:
  `a8f98f7d5e0eb2cd571a8932798e70120f40360e1367080b50ea3f8aaf78b8e6`.

## Interpretation

The experiment establishes that the implemented artifact is normalized,
sampleable, authenticated, and operationally reproducible.  It does not
establish an accurate marginal likelihood under the frozen BP1 gates.

Validation/test NLL agreement was not sufficient for likelihood-gradient or
evidence accuracy.  Even the easiest one-dimensional near-Gaussian case
showed smooth-value accuracy without adequate mass-coordinate derivatives.
The boundary-heavy distributions remained poorly represented, and larger
training banks did not cure the error.  These are approximation failures, not
information about the basis or the common native model.

This result is terminal for the declared direct-noisy FlowJAX triangular
spline architecture.  Adding layers, knots, epochs, training sizes, or relaxed
thresholds inside this run would invalidate the predeclaration.

## Follow-up

Any further work must be a new independently predeclared NLE architecture with
fresh domains and run roots.  Development evidence suggests that the next
design should address both hard-boundary representation and derivative
quality, rather than merely increasing simulation count.  Candidate designs
include a support-aware flow for the noise-free native pushforward followed by
a fixed normalized Gaussian convolution, or an established conditional NSF
stack with derivative/replay validation declared before G1.

The protected catalogue remains closed.  No G2 confirmation or G3 protected
action is authorized by this result.
