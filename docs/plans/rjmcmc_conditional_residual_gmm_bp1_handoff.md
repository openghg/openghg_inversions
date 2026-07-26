# Conditional residual GMM BP1 handoff

## Objective and model boundary

Continue the non-RJ learned residual-likelihood experiment on
`codex/rjmcmc-aggregation-conditional-likelihood`. The model approximates the
normalized marginal likelihood obtained after unresolved native-cell
allocations are integrated out for one fixed computational partition.

This is distinct from the Gaussian closure, which remains a potentially useful
explicit approximation inside RJ. Under one common proper native model, the
exact learned/marginal likelihood is invariant to the computational partition
and \(K\). Approximate evidence differences are leakage diagnostics and must
not become posterior structural weights. Every output must retain
`structural_inference_licensed=false`.

Do not add `sbi`, Torch, a PyMC bridge, a conditional row network, or a flow in
this phase. The current task is only the single predeclared
sixteen-component root-GMM escalation.

## Established facts

The repaired eight-component source was:

```text
branch: codex/rjmcmc-aggregation-conditional-likelihood
revision: 3c91beea7836a9996d2850aadbc6892d2ed0d46a
G1 Slurm array: 18187304
run root: /group/chem/acrg/brendan_for_codex/rjmcmc_conditional_residual_gmm/3c91beea7836a9996d2850aadbc6892d2ed0d46a
```

- G0 passed 71 focused tests, Ruff, Pyright, the pinned NumPy 2.2.6/SciPy
  1.15.2 runtime, and smoke.
- All 24 G1 tasks completed with exit code zero and published exactly 24
  canonical artifacts and 24 completion markers.
- The merger replayed all artifacts across the distinct compute nodes. The
  residual-context portability and scoped replay fixes therefore passed.
- The merger then made a genuine scientific hard stop before lock publication:
  no common six-case, two-size passing suffix exists.
- Both near-Gaussian cases and both skewed cases passed every size.
- Both boundary-heavy cases failed scientific gates at every size, although
  every fit was numerically valid and every independent
  validation-versus-test generalization gate passed.
- At 262,144 training draws, boundary-heavy evidence error was 0.265 nat for
  the two-cell case and 2.062 nat for the four-cell case.
- G2 and G3 did not run. The protected catalogue was not opened.
- Nothing was written to `PARIS_inversions`.

This stable, sample-size-insensitive boundary failure is the predeclared
justification for one sixteen-component escalation. It is an inference that
the eight-component mixture underfits; the recorded gate failures are facts.

## Sixteen-component candidate

The branch now freezes:

```text
architecture stage: sixteen-component-underfit-escalation-v1
component count: 16
development protocol SHA-256:
71352ed31c8b90c093a7d50ef7e8fb64bccce84e5521bf1134932f509b4cedc3
```

The GMM scientific protocol name and sealed protected-catalogue commitment do
not change. Component count and architecture stage are included in the
development-protocol digest, so eight- and sixteen-component artifacts cannot
be mixed. The pushed branch head after this handoff document is the
authoritative candidate SHA; resolve it from `origin`, do not guess it from
this document.

Local validation before handoff:

```text
71 focused tests passed
Ruff passed
Ruff format check passed
Pyright passed
git diff --check passed
```

## Required BP1 sequence

Use a fresh detached full-SHA worktree and full-SHA run root. Preserve the
`3c91beea...` run unchanged and never reuse one of its shards.

```bash
repository=/group/chem/acrg/brendan_for_codex/openghg_inversions
branch=codex/rjmcmc-aggregation-conditional-likelihood
git -C "${repository}" fetch origin "${branch}"
revision="$(git -C "${repository}" rev-parse "origin/${branch}")"
short_revision="${revision:0:12}"
source_root="/group/chem/acrg/brendan_for_codex/rjmcmc_gmm_worker_${short_revision}"
run_root="/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_residual_gmm/${revision}"
git -C "${repository}" worktree add --detach "${source_root}" "${revision}"
ln -s /group/chem/acrg/brendan_for_codex/openghg_inversions/.pixi \
  "${source_root}/.pixi"
mkdir -p \
  "${run_root}/preflight" \
  "${run_root}/development" \
  "${run_root}/confirmation" \
  "${run_root}/lock" \
  "${run_root}/certificate" \
  "${run_root}/protected" \
  "${run_root}/markers/development" \
  "${run_root}/markers/confirmation" \
  "${run_root}/logs/development" \
  "${run_root}/logs/confirmation"
export GMM_SOURCE="${source_root}"
export GMM_RUN_ROOT="${run_root}"
export GMM_REVISION="${revision}"
```

Run G0:

```bash
bash \
  "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_preflight.sh"
```

Stop unless `preflight/PREFLIGHT_COMPLETE.txt` exists and the worktree is
clean apart from the authenticated `.pixi` link.

Submit G1:

```bash
development_job="$(
  sbatch --parsable --export=ALL \
    "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_development_array.sbatch"
)"
echo "${development_job}"
```

Require all 24 Slurm elements to complete with exit code zero, exactly 24 JSON
artifacts, and exactly 24 completion markers. Then run:

```bash
bash \
  "${GMM_SOURCE}/docs/plans/rjmcmc_conditional_allocation_assets/run_gmm_c4b_merge_development.sh"
```

The merger is a hard gate:

- If it publishes no common lock, preserve all artifacts, write a readable
  report and checksum inventory beneath the run root, commit/push the status,
  and stop. This is terminal for the root-GMM architecture; do not add more
  components, a flow, or the conditional row model.
- If it publishes a lock, submit the 18-task G2 confirmation array and follow
  the exact commands in
  `rjmcmc_conditional_allocation_likelihood_hpc_test_plan.md`.
- Run the G2 certifier only after all 18 artifacts and markers are complete.
- G3 is permitted only after a passing, holdout-eligible G2 certificate.
  Authenticate all source/certificate/artifact inputs before transferring or
  opening the sealed catalogue. A protected failure is terminal and may not
  cause retuning.

Use only the experimental tests in the committed preflight script. Preserve
failed artifacts. Publish completion markers last. Do not modify scientific
source in the detached candidate worktree and do not write to
`PARIS_inversions`.

## Open questions after this phase

These are deliberately deferred until the sixteen-component root result:

1. Whether a two-region conditional MDN is justified. It is allowed only
   after complete root G2/G3 certification.
2. Whether PARIS residual-image rank permits the current dense
   full-covariance representation. A read-only spectrum/resource probe must
   precede real-data training; the current scalar canonical-basis builder is a
   tiny-oracle implementation, not a PARIS-scale algorithm.
3. Whether a future learned artifact should be exported into native PyTensor.
   `sbi`'s PyMC sampler does not insert an arbitrary Torch likelihood into an
   existing PyMC graph. The preferred bridge is a native PyTensor
   implementation of accepted plain float64 arrays.
4. How to estimate real-data absolute evidence with independently audited
   Monte Carlo uncertainty materially below the 0.05-nat gate.

## Copy-paste prompt for the BP1 agent

```text
Continue the conditional residual-likelihood experiment in
/group/chem/acrg/brendan_for_codex/openghg_inversions and the sibling
inversions-knowledge repository.

First read:
- docs/plans/rjmcmc_conditional_residual_gmm_bp1_handoff.md
- docs/plans/rjmcmc_conditional_allocation_likelihood_hpc_test_plan.md
- docs/plans/rjmcmc_bp1_handover.md
- the relevant learned-marginal-model notes in ../inversions-knowledge

Fetch origin/codex/rjmcmc-aggregation-conditional-likelihood and resolve the
exact full branch-head SHA. Confirm the worktree is clean. The prior
eight-component run at
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_residual_gmm/3c91beea7836a9996d2850aadbc6892d2ed0d46a
is preserved evidence: do not modify it or reuse its shards. It passed
cross-node replay but failed the scientific common-suffix gate because both
boundary-heavy cases failed at every size. G2/G3 never ran and the protected
catalogue remains sealed.

The branch now contains the single predeclared 16-component underfit
escalation. Create a fresh detached full-SHA worktree and full-SHA run root.
Run G0, then the complete 24-task G1 matrix, then the merger, strictly in
order. Stop at the first hard gate. Do not tune thresholds, add components,
add a flow, start a conditional row model, or use sbi/PyMC if the
16-component merger fails. If a common lock is published, continue through
G2 confirmation and only then the independently sealed G3 protected
certification exactly as specified.

Keep the scientific roles explicit: Gaussian closure may be used separately
as an approximation inside RJ; this learned-density track is a non-RJ
marginal likelihood for a common native model. Its exact limit is invariant
to partition and K, so no result licenses data-dependent structural weights.

Use only experimental tests, Ruff, focused Pyright, and committed scripts.
Preserve every artifact and failure; publish markers last; record exact Git
SHAs, job IDs, run paths, pass/fail gates, and checksum inventories. Commit
and push code/plans/reports in small coherent increments. Write nothing to
PARIS_inversions.
```
