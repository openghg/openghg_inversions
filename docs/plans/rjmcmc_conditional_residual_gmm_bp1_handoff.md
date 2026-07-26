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

## Terminal sixteen-component outcome

The escalation ran from the exact pushed candidate
`625dc3b26dcad646ee144eea2c5fdc507851cdfa` in the fresh detached worktree
`/group/chem/acrg/brendan_for_codex/rjmcmc_gmm_worker_625dc3b26dca`.
Its fresh run root is:

```text
/group/chem/acrg/brendan_for_codex/rjmcmc_conditional_residual_gmm/625dc3b26dcad646ee144eea2c5fdc507851cdfa
```

G0 passed 71 focused experimental tests, Ruff, focused Pyright, the pinned
NumPy 2.2.6/SciPy 1.15.2 runtime, and smoke. G1 array `18187541` submitted all
24 frozen shards. Twenty completed and published artifacts and markers. All
four sizes of `skewed__two_cell__root` failed because all three deterministic
EM initializations failed, so they published no artifacts or markers.

The 20 valid artifacts also showed that both near-Gaussian cases and the
skewed four-cell case passed every size, while both boundary-heavy cases
failed scientific gates at every size. The exact 24-artifact merger
precondition was false, so the merger was not run and no lock exists. G2 and
G3 were withheld and the protected catalogue remained sealed.

This is the first hard gate and is terminal for the root-GMM architecture.
Do not rerun, add components, extend the ladder, change thresholds, add a flow
or conditional row model, introduce `sbi`/PyMC, or open the catalogue. See
[`rjmcmc_conditional_residual_gmm_16_component_bp1_report.md`](rjmcmc_conditional_residual_gmm_16_component_bp1_report.md)
for the complete report, Slurm inventory, and checksums.

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

## Sixteen-component candidate identity

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
be mixed. The authoritative candidate SHA resolved from `origin` was
`625dc3b26dcad646ee144eea2c5fdc507851cdfa`.

Local validation before handoff:

```text
71 focused tests passed
Ruff passed
Ruff format check passed
Pyright passed
git diff --check passed
```

## Executed BP1 sequence

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

## Closed phase

The sixteen-component result closes this bounded learned-density phase.
Questions about a conditional row model, a flow, PARIS-scale dense deployment,
or a PyTensor/PyMC bridge were conditional on complete root G2/G3
certification and are not activated. A future project may revisit a different
scientific model only under a new, independently reviewed protocol; it must not
be presented as a continuation or rescue of this root-GMM experiment.

Preserve both full-SHA run roots, the terminal report, and the sealed protected
catalogue. No further BP1 launch is authorized by this handoff.
