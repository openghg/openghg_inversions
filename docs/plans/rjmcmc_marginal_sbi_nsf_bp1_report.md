# BP1 Marginal sbi-NSF NLE: Terminal G1 Result

## Decision

The independently predeclared sbi neural-spline likelihood did not publish a
G1 common lock.  The pure merger authenticated all 24 development tasks, but
every task failed at least one frozen scientific gate.  There is no common
all-six-case, all-larger passing suffix of length two.

This is a hard stop for the declared noisy-residual sbi-NSF architecture.
G2 and G3 were not submitted, and the protected catalogue remains sealed.
Nothing in this result changes the basis map or the common native model.

## What was tested

This experiment tested a normalized, sampleable neural likelihood for the
observation marginal induced by a fixed linear basis map.  It was independent
of both the terminal root-GMM ladder and the failed direct FlowJAX likelihood.

For a fixed retained mass, the implementation:

- removes the exact conditional observation mean;
- projects the residual into the complete aggregation-residual image;
- whitens that image by its exact conditional covariance;
- fits an autoregressive rational-quadratic neural spline to the remaining
  conditional density;
- retains the exact Gaussian residual-image complement outside the learner;
  and
- uses one authenticated artifact for both normalized density evaluation and
  forward observation simulation.

The run deliberately added and pinned an established likelihood-learning
stack rather than restricting the experiment to the packages already present:
`sbi 0.26.1`, `nflows 0.14`, and CPU PyTorch `2.10.0`.  This dependency
addition was isolated in the opt-in `nle` Pixi solve group.

The immutable scientific source and protocol were:

- source Git SHA:
  `73bbe3a72fd7ee59de5a34b4eb698d5af112f73d`;
- driver SHA-256:
  `ee37b78588e39875d5d204c7c3015e23d70a073e905f8db8e6f8a0be8a8fef2a`;
- protocol SHA-256:
  `ef8441560ac107e377cebe7785259bff0ff288d5e84ea013908f5aa52c752f27`;
- detached source:
  `/group/chem/acrg/brendan_for_codex/openghg_inversions-worktrees/rjmcmc-marginal-sbi-nsf-73bbe3a72fd7ee59de5a34b4eb698d5af112f73d`;
- run root:
  `/group/chem/acrg/brendan_for_codex/rjmcmc_marginal_sbi_nsf/73bbe3a72fd7ee59de5a34b4eb698d5af112f73d`; and
- pinned runtime: Python 3.10.20, NumPy 2.2.6, SciPy 1.15.2,
  CPU-only PyTorch 2.10.0 in float64 mode, sbi 0.26.1, and nflows 0.14.

The architecture used eight autoregressive NSF transforms, 16 spline bins,
128 hidden features, two deterministic initializations, and exact analytic
conditioner standardization.  The 24-task G1 matrix comprised six root cases
and training sizes 4,096, 16,384, 65,536, and 262,144.  Optimizer validation,
model-selection validation, and development test domains were separate.

## Terminology and truth

The **native truth model** is the source-pinned additive independent-Gamma
cell model.  Conditional on retained mass, native fractions have the exact
within-region Dirichlet distribution; source-pinned Gaussian measurement
noise is then added.

The **basis map** is the fixed linear aggregation from native cell values to
retained masses.  It was not selected from realized observations or any
approximate evidence result.

The **truth likelihood** is the deterministic C1 quadrature marginal over
native allocations.  It is the comparator for likelihood, retained-mass
gradient, evidence, and posterior-summary scores.  It is not fitted.

The **learned marginal simulator** draws from the same normalized
conditional NSF artifact used for likelihood scoring.  Artifact replay and
density/sampler consistency establish that the fitted object is internally
coherent; they do not establish agreement with native truth.

Approximate evidence differences are leakage diagnostics only.  They were not
used as basis weights, training weights, conditioner inputs, model-selection
criteria, or architecture choices.

## Operational record

The first G0 attempt used implementation source
`784d59f7fbe6dbb53cf733baa39489fbbdb4b49a` and run root
`/group/chem/acrg/brendan_for_codex/rjmcmc_marginal_sbi_nsf/784d59f7fbe6dbb53cf733baa39489fbbdb4b49a`.
It failed before tests or smoke evaluation because a committed shell wrapper
misquoted the Python runtime-version probe.  The only artifact is the
preserved preflight log, SHA-256
`03d979991a30f822cfa2b9b233f4d17f56a56fa296ce573b93ee73ccdb2445b3`.
No completion marker was published and that run root was not reused.

Wrapper-only commit
`73bbe3a72fd7ee59de5a34b4eb698d5af112f73d` fixed the version probe.  A new
detached full-SHA source and fresh run root were then used for all scientific
work.

G0 passed:

- 15 focused experimental tests;
- Ruff;
- focused Pyright with zero errors or warnings;
- runtime, source, driver, and protocol identity checks;
- an operational fitted smoke;
- same-process and separate-process authenticated replay;
- sample/log-probability agreement to
  `1.1546319456101628e-14`;
- analytic retained-mass gradient replay; and
- cross-node replay job `18188702` on `bp1-compute051`, which completed in
  58 seconds with exit `0:0`.

The G0 completion marker was published last.

Slurm array `18188705` ran the complete 24-task G1 matrix.  All 24 array tasks
completed with exit `0:0`; elapsed times ranged from 3 minutes 26 seconds to
1 hour 10 minutes 15 seconds, and peak batch-step resident memory ranged from
580,724 KiB to 1,132,720 KiB.  The development directory contains exactly 72
files: 24 authenticated NSF artifacts, 24 reports, and 24 completion markers.
Their combined size is 76,925,316 bytes.

Pure merger job `18189004` ran on `bp1-compute070`, completed in 39 seconds
with exit `0:0`, authenticated all 24 tasks, and published its completion
marker last.  It published no `common-lock.json`.

The detailed Slurm ledger is
`rjmcmc_marginal_sbi_nsf_bp1_slurm.csv`.

## G1 results

All 24 fits produced finite selected artifacts, and all 24 authenticated
artifacts replayed successfully.  Eleven selected artifacts passed the
independent model-selection-validation versus development-test NLL gate.
Zero artifacts passed every scientific gate.

The merger recorded the following per-case result:

| Case | Generalization-pass sizes | Scientific-pass sizes | Main result |
|---|---|---|---|
| near-Gaussian, two-cell | 4,096; 16,384; 65,536; 262,144 | none | gradient failed every size |
| near-Gaussian, four-cell | 4,096; 16,384; 65,536; 262,144 | none | gradient failed every size |
| skewed, two-cell | none | none | generalization and scientific gates failed |
| skewed, four-cell | none | none | generalization and scientific gates failed |
| boundary-heavy, two-cell | none | none | likelihood, gradient, evidence, and spread errors |
| boundary-heavy, four-cell | 4,096; 16,384; 65,536 | none | large likelihood, gradient, evidence, and posterior errors |

Across all 24 tasks, the number passing each frozen threshold was:

| Gate | Passing artifacts / 24 |
|---|---:|
| prior-weighted median absolute conditional log-likelihood error | 6 |
| posterior-weighted p99 absolute conditional log-likelihood error | 8 |
| scaled retained-mass coordinate-gradient error | 0 |
| absolute log-evidence error | 6 |
| posterior mean error | 13 |
| posterior SD relative error | 9 |
| interval endpoint error | 15 |

The smallest scaled coordinate-gradient error was `0.3464168155916313`,
from the skewed four-cell artifact trained with 262,144 draws.  It remained
well above the frozen `0.05` threshold and also failed the density
generalization, likelihood, evidence, posterior-SD, and interval gates.

The near-Gaussian likelihood values were often close while their derivatives
were not.  For example, the near-Gaussian four-cell artifact at 4,096 draws
had median absolute likelihood error `0.01567000035440691` nat and absolute
log-evidence error `0.002363826015144843` nat, but scaled gradient error
`1.4902423004576764`.

Boundary-heavy failures were not cured by more simulations.  The two-cell
prior-weighted median likelihood error was `36.77839885069645` nat at every
size.  Four-cell median errors ranged from `1.0818644765859418` to
`3.24192453633772` nat.  The 262,144-draw four-cell artifact improved some
value metrics but still failed every scientific check.

## Gate decision

The authenticated development certificate records:

```text
complete_matrix: true
authenticated_task_count: 24
lock_published: false
locked_sample_count: null
terminal_reason:
  no common all-six-case all-larger passing suffix of length at least two
```

This is the first hard scientific gate for this run.  No G2 confirmation
shards were submitted.  G3 remains forbidden, and the protected catalogue was
never opened.

## Artifacts and checksums

The principal decision artifacts are:

- development certificate raw SHA-256:
  `dc3a3f7af8b05090676ac37a20d673e91043628a7dd963679d3ff341f6ea8ab3`;
- development certificate envelope SHA-256:
  `7213ccb02d58a7d1d9cdc7a914a46130a51a62f54adabaeaf3bd7c74bdf70587`;
- merger completion marker raw SHA-256:
  `db3101aa5c9a42dac4d8d31990fccdcde0095b323faa11c3c1e44c9ad0e55418`;
- G0 completion marker raw SHA-256:
  `88aa2ee19ed1a2d67147fbcb95ddf19c989cd17adccc80803a1e98aab3fad491`;
- cross-node replay record raw SHA-256:
  `85c44d243e354b79fd9951e19135338492933dcf5acc48a80f8c0097264861da`;
- G0 log raw SHA-256:
  `3dd06dc2089399a97fe54b2da651fd37c2028466435a45b9a04ccb2dd10a5ca3`;
  and
- fitted G0 smoke artifact SHA-256:
  `cc54f92c67437743c666cf51a49ef18f4900b8d7c504a52ea5985781856eea72`.

The full preserved-run checksum manifest is `checksums.sha256`.  It contains
111 relative-path entries, passed `sha256sum -c`, and has raw SHA-256
`f9415637849b79e787985deaa45af69a789b8d8fe41e203b64e1bf2bd92598a1`.
The report copy authenticated by that manifest is the pre-manifest archival
copy; this repository copy adds only the manifest count and digest.

## Interpretation and next architecture

The experiment establishes that this implementation creates normalized,
sampleable, authenticated, differentiable NSF artifacts and can train and
replay them reproducibly across BP1 nodes.  It does not establish an accurate
marginal likelihood under the frozen gates.

Increasing training size did not lead to a passing suffix.  Validation/test
NLL agreement also did not imply accurate retained-mass derivatives.
Consequently, more transforms, bins, hidden features, epochs, training draws,
or threshold changes inside this run would be post-result tuning and are not
permitted.

The shared failure pattern of the direct noisy FlowJAX likelihood and this
independent noisy sbi-NSF points to a more specific next test: represent the
bounded, noise-free native pushforward in support-aware coordinates, then
apply the fixed Gaussian measurement convolution outside the learned object.
That design must be independently predeclared with fresh domains and a fresh
run root.  Any approximate evidence differences remain diagnostics only and
must never become data-dependent basis weights.
