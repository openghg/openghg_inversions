# Sixteen-component conditional residual GMM BP1 result

## What was tested

This report records the single predeclared sixteen-component escalation of the
root-only conditional residual-image Gaussian-mixture approximation.

```text
branch: codex/rjmcmc-aggregation-conditional-likelihood
candidate revision: 625dc3b26dcad646ee144eea2c5fdc507851cdfa
architecture stage: sixteen-component-underfit-escalation-v1
component count: 16
development protocol SHA-256:
71352ed31c8b90c093a7d50ef7e8fb64bccce84e5521bf1134932f509b4cedc3
detached source:
/group/chem/acrg/brendan_for_codex/rjmcmc_gmm_worker_625dc3b26dca
run root:
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_residual_gmm/625dc3b26dcad646ee144eea2c5fdc507851cdfa
G1 Slurm array: 18187541
```

The source was resolved from the full `origin` branch head. The preserved
eight-component run at `3c91beea7836a9996d2850aadbc6892d2ed0d46a`
was not modified and none of its artifacts was reused.

G0 used the committed preflight script. G1 submitted the committed complete
six-case by four-size matrix, using development seed 731 and training sizes
4,096, 16,384, 65,536, and 262,144 whole draws.

## Terminology and truth

The truth for this tiny-oracle experiment is the frozen exact
conditional-likelihood construction and quadrature catalogue inherited from
C1. It supplies exact likelihood values, coordinate gradients, evidence, and
posterior summaries for the six declared root contexts. The scoring domain is
the committed development validation/test and quadrature views; the sealed
protected density holdout is not part of this report and was not opened.

The learned object is a normalized, non-RJ marginal-likelihood approximation
for one common native model. The exact limit is invariant to computational
partition and \(K\). Evidence error is therefore an approximation-leakage
diagnostic and is not structural information. Every published artifact
retained `structural_inference_licensed=false`.

## What happened

G0 passed:

- 71 focused experimental tests;
- Ruff;
- focused Pyright;
- the pinned Python 3.10.20, NumPy 2.2.6, and SciPy 1.15.2 runtime; and
- the committed bounded smoke screen.

All 24 G1 array elements reached terminal Slurm states. Twenty completed with
exit code zero and published one canonical JSON artifact followed by one
completion marker. All four `skewed__two_cell__root` elements failed with exit
code 1 because all three deterministic EM initializations failed. They
published neither artifacts nor completion markers, and their four traceback
logs were preserved.

The committed merger requires exactly 24 artifacts and 24 markers. It was
therefore withheld rather than invoked on an incomplete matrix. No common lock
or completion marker was published. G2 and G3 were withheld, and the protected
catalogue remained sealed.

## Key results

The table shows the authoritative per-size G1 outcome. “Pass” means the
published shard passed both fitting/generalization and scientific-model gates;
“science fail” means a numerically valid artifact failed one or more unchanged
scientific gates; “EM fail” means no artifact was valid because all three
frozen initializations failed.

| Case | 4,096 | 16,384 | 65,536 | 262,144 |
|---|---:|---:|---:|---:|
| near-Gaussian, two-cell root | Pass | Pass | Pass | Pass |
| near-Gaussian, four-cell root | Pass | Pass | Pass | Pass |
| skewed, two-cell root | EM fail | EM fail | EM fail | EM fail |
| skewed, four-cell root | Pass | Pass | Pass | Pass |
| boundary-heavy, two-cell root | Science fail | Science fail | Science fail | Science fail |
| boundary-heavy, four-cell root | Science fail | Science fail | Science fail | Science fail |

The worst decision-driving diagnostic is the complete four-size numerical
failure of the skewed two-cell case. Among the 20 valid artifacts, the largest
gated conditional-likelihood error was the boundary-heavy two-cell
prior-weighted median absolute error of 11.743 nat at 4,096 draws, against the
0.05-nat limit. At 262,144 draws:

- boundary-heavy two-cell had 0.0297-nat evidence error, 11.737-nat weighted
  median likelihood error, and 3.44% posterior-SD relative error; and
- boundary-heavy four-cell had 1.588-nat evidence error, 1.596-nat weighted
  median and p99 likelihood error, 0.344 scaled-gradient error, 0.120 posterior
  mean error in exact-posterior SD units, and 64.2% posterior-SD relative
  error.

The largest unweighted full-grid diagnostic was 1,955.478 nat for the
boundary-heavy four-cell 262,144-draw artifact. It is reported as a tail
diagnostic, not substituted for a weighted scientific gate.

## Provenance and inventories

The machine-readable report assets are:

- `rjmcmc_conditional_residual_gmm_16_component_bp1_report_assets/summary.json`;
- `rjmcmc_conditional_residual_gmm_16_component_bp1_report_assets/slurm.csv`;
- `rjmcmc_conditional_residual_gmm_16_component_bp1_report_assets/artifact_status.csv`;
- `rjmcmc_conditional_residual_gmm_16_component_bp1_report_assets/evidence_sha256s.txt`;
  and
- `rjmcmc_conditional_residual_gmm_16_component_bp1_report_assets/report_sha256s.txt`.

The same report bundle is copied beneath the immutable run root. The evidence
inventory covers all preflight files, 20 development artifacts, 20 completion
markers, and 24 development logs. It was verified with `sha256sum -c`.
`report_sha256s.txt` covers every report-bundle file except itself and was
also verified with `sha256sum -c`. The lock, confirmation, certificate, and
protected directories contain no files.

The source worktree remained clean apart from the authenticated `.pixi` link.
The sibling `inversions-knowledge` repository was clean at
`e77d20cffe7ee0298d9106065c962d24198dabdc`.

## Interpretation and terminal decision

This is a G1 hard stop before merger, not a missing scientific lock produced by
the merger. The sixteen-component escalation is numerically invalid for one
entire required case and still misses the unchanged scientific gates in both
boundary-heavy cases. Consequently no all-six-case two-size passing suffix can
exist.

The root-GMM architecture is terminal under the predeclared protocol. Do not:

- add mixture components;
- extend the training-size ladder;
- change thresholds or initialization rules;
- add a flow or conditional row model;
- introduce `sbi`, PyMC, or a Torch bridge as a continuation of this phase; or
- open the protected catalogue.

The result does not license data-dependent basis or \(K\) weights. The Gaussian
closure remains a separately declared approximation that may be used inside
RJ, but this learned-density track is closed.

## Withheld stages

- G1 merger: withheld because 24 artifacts and markers were not present.
- G2 confirmation: forbidden because no common lock exists.
- G3 protected certification: forbidden because no passing, holdout-eligible
  G2 certificate exists.
- PARIS work: not run; nothing was written to `PARIS_inversions`.
