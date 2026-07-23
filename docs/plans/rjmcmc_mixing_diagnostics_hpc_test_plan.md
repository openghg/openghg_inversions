# RJMCMC mixing diagnostics: HPC test plan

## Purpose

This plan validates the proposal-level mixing diagnostics added to the
experimental moving-Voronoi sampler and then uses them to distinguish:

- slow nearest-neighbour diffusion in the region count;
- repeated local reversals;
- poorly informed structural churn;
- the cost of rebuilding affected sensitivity columns;
- a differentiated-region deletion barrier; and
- a barrier introduced by the observation likelihood.

The first stage is an instrumentation experiment, not a new sampler or a new
scientific inversion. It uses the existing corrected transition kernel and
must reproduce its random trajectory exactly.

## Likelihood powers and parallel tempering

The powered target

\[
\pi_\beta(s) \propto p(s)L(y\mid s)^\beta
\]

is the same family normally used for the replicas in likelihood-tempered
parallel tempering. The distinction here is operational:

- the diagnostic runs are independent fixed-\(\beta\) chains;
- no state swaps or temperature-walk process are introduced; and
- only the likelihood is powered, while the prior remains unchanged.

Thus \(\beta=0\) is a prior-only chain and \(\beta=1\) is the ordinary
posterior chain. These endpoint runs can diagnose whether switching on the
likelihood causes the mobility loss without committing to parallel tempering.
The archived Brazil experiment accepted no swaps, so a new tempering
implementation is not a priority unless the fixed-power comparison shows a
clear likelihood barrier.

## Diagnostic contract

Diagnostics are opt-in and transition-indexed. They are recorded only for
structural attempts: insertion, deletion, and global or local nucleus moves.
They are not part of the retained-state trace, transition kernel, target,
checkpoint compatibility contract, or random-number state.

For each structural candidate, retain:

- global atomic-transition number, move, validity, and acceptance;
- source, candidate, and resulting region counts;
- candidate changes in likelihood, every prior component, and the complete
  target;
- forward and reverse proposal densities, Jacobian, and untruncated
  Metropolis--Hastings ratio;
- removed and added nucleus identities where applicable;
- native cells whose owning nucleus identity changes;
- candidate design columns whose membership must be rebuilt;
- Euclidean and observation-error-standardized prediction changes;
- standardized sensitivity of the event region; and
- a coefficient contrast for the affected region.

Rejected candidates retain their proposed changes. A realized change is zero
for a rejection. Owner changes must be calculated from nucleus identity rather
than canonical label position, because inserting a low-index nucleus can
renumber otherwise unchanged labels.

The observation-error-standardized norm is

\[
\left[
\sum_i
\left(
\frac{\hat y'_i-\hat y_i}{s_i}
\right)^2
\right]^{1/2}.
\]

It is a fully whitened norm for the current fixed diagonal error model. Under
the latent-OU mismatch model it is only measurement-error-standardized; it
must not be described as \(C^{-1}\)-whitened.

Immediate reversals and intervals are derived after concatenating all
segments. `derive_region_lineage_intervals` measures region lineage tenure:
initial regions are left-censored, active regions at the end are
right-censored, and an accepted location move transfers rather than terminates
a lineage. `derive_nucleus_residence_intervals` instead measures residence at
a particular grid-cell nucleus and therefore closes and reopens on a move.

### Run-driver integration

For the initial segment, set
`SamplerConfig.collect_structural_diagnostics=True` and supply a
`StructuralDiagnosticsProvenance` containing the stable chain ID and
`problem_fingerprint(problem)`. For every continuation, pass both
`collect_structural_diagnostics=True` and the same provenance explicitly to
`continue_sample`; these settings are intentionally not inherited from the
checkpoint because they do not define the Markov kernel.

Minimal driver wiring:

```python
from dataclasses import replace

from openghg_inversions.experimental.rjmcmc import (
    StructuralDiagnosticsProvenance,
    continue_sample,
    problem_fingerprint,
    sample,
)

fingerprint = problem_fingerprint(problem)
diagnostic_provenance = StructuralDiagnosticsProvenance(
    chain_id=f"{run_profile}:chain-{chain_index}",
    problem_fingerprint=fingerprint,
)
diagnostic_config = replace(
    base_config,
    collect_structural_diagnostics=True,
    structural_diagnostics_provenance=diagnostic_provenance,
)
first_result = sample(problem, initial_state, diagnostic_config, retention=retention)
next_result = continue_sample(
    problem,
    first_result.checkpoint,
    iterations=segment_transitions,
    collect_structural_diagnostics=True,
    structural_diagnostics_provenance=diagnostic_provenance,
)
```

Persist `result.trace` with the existing retained-state converter and persist
`result.structural_diagnostics` separately with
`structural_diagnostics_to_dataset`. Use a segment-specific file name and add
the segment, code revision, and run profile as dataset metadata. The chain ID
and problem fingerprint are reserved attributes written from the diagnostic
object's intrinsic provenance. The dataset's `structural_transition`
coordinate is global.

Before calculating reversals or residence intervals, load all segments and
restore each object with `structural_diagnostics_from_dataset`, passing the
expected identity as
`required_metadata={"chain_id": chain_id, "problem_fingerprint": fingerprint}`;
then use `concatenate_structural_diagnostics` on those objects. Do not call
`xarray.concat` directly: the variable-length initial and final nucleus sets
belong to each segment and would otherwise be broadcast across proposal rows.
The object-level helper requires identical intrinsic chain/problem provenance,
validates that the previous segment's final nucleus set equals the next
segment's initial set, and checks that the complete atomic transition bounds
are adjacent; it therefore rejects different chains even if their boundary
states happen to match.

## Stage A: local and CI validation

Before an HPC run, the focused suite must establish:

1. diagnostics disabled and enabled give identical attempted moves,
   acceptances, retained states, final cached state, and PCG64 checkpoint;
2. valid accepted, valid rejected, and invalid boundary attempts remain
   distinct;
3. target-component deltas sum to the complete target delta;
4. canonical nucleus reordering does not create false owner changes;
5. prediction and design norms agree with direct calculations on a small
   problem;
6. diagnostic segments concatenate in global-transition order and agree with
   one uninterrupted segment;
7. direction reversals, exact endpoint reversals, edge flows, and censored
   residence intervals are reconstructed correctly; and
8. NumPy and Numba diagnostic values agree to the same tolerances as their
   cached states.

This validation should also confirm that ordinary runs allocate no
proposal-diagnostic arrays when collection is disabled.

## Stage B: paired HPC instrumentation benchmark

Use the frozen PARIS May 2014 input, the correct arithmetic-mean-one and
arithmetic-SD-one coefficient prior, the fixed diagonal error model, and the
current 14-slot opportunity-matched schedule.

Run paired jobs with identical input, initial state, seed, backend, and
transition count:

- diagnostics disabled;
- diagnostics enabled.

Use at least one \(k=50\) and one \(k=250\) start. A first useful duration is
140,000 atomic transitions per chain, or 10,000 complete cycles. Run the two
members of each pair on the same node class, preferably sequentially or with
exclusive equivalent resources, and warm the Numba cache before timing.

Required acceptance criteria:

- exact equality of sampler traces and final checkpoints for each pair;
- equal `problem_fingerprint(problem)`, `checkpoint.kernel_settings`, and
  `checkpoint.schedule_id` (plus equal run manifests when the driver uses
  them);
- no material increase in peak RSS relative to the allocation;
- report median and per-chain atomic transitions per second; and
- report diagnostic overhead with uncertainty across repeated segments.

An overhead below 10% is a useful target, not a correctness threshold. If
overhead exceeds 15%, profile owner comparison and standardized prediction
norms separately before changing the sampler. Record filesystem writing time
separately from transition time.

## Stage C: full-likelihood profiling

Start with four chains and 40,000 cycles each, retaining the established
low/high initial region counts and complete-cycle checkpoint boundaries.
Extend to the production 120,000-cycle budget if first-passage behavior remains
the primary question: the optimistic random-walk calculation already puts a
200-region traversal near 57,000 cycles.

Save the ordinary retained-state dataset, exact restart checkpoint, structural
diagnostics, wall time, CPU time, peak RSS, host, code revision, input
fingerprint, and full sampler configuration for every segment. Concatenate
structural diagnostics by their global transition coordinate before computing
cross-boundary reversals or residence intervals.

For every chain and checkpoint report:

- attempts, valid proposals, acceptances, and realized flow for every edge
  \(k\leftrightarrow k+1\);
- mean-squared displacement of \(k\) by lag;
- first-passage and low-to-high-to-low round-trip counts;
- invalid-proposal, valid-rejection, and accepted-event fractions;
- opposite-direction and exact-endpoint immediate reversal fractions;
- region residence distributions, with censoring reported explicitly;
- owner-changed cells and affected-column counts by move, acceptance, and
  \(k\) band;
- prediction-change and event-sensitivity distributions;
- likelihood, prior-component, proposal-ratio, and total-target contributions
  by move and outcome;
- deletion acceptance against region age, sensitivity, and coefficient
  contrast;
- accepted standardized prediction path length per wall time; and
- `R-hat` and ESS for \(k\), predictions, native-grid flux summaries, country
  totals, outer coefficients, and mismatch parameters where present.

Interpretation:

- balanced aggregate up/down acceptance is expected and is not evidence of
  traversal;
- broad edge flow with few passages supports ordinary local diffusion;
- concentrated low-flow edges support a structural-prior or likelihood
  barrier;
- frequent exact undo supports interleaving ordinary coefficient
  rejuvenation;
- declining deletion acceptance with age or contrast supports the
  differentiated-region ratchet;
- high acceptance with tiny standardized prediction movement identifies
  weakly informed churn; and
- runtime following changed ownership or affected-column count confirms the
  remaining aggregation bottleneck.

## Stage D: fixed-power endpoint comparison

Likelihood-power support is a separate point-2 change and is not required for
the instrumentation commit. Once it is available, compare \(\beta=0\) and
\(\beta=1\) first, using the same support, starts, proposal settings,
opportunity counts, and diagnostic summaries.

- Poor mobility already at \(\beta=0\) argues for changing the proposal graph,
  compound rejuvenation, or the state space.
- Good prior mobility but a large deterioration at \(\beta=1\) implicates
  likelihood-induced restriction. Inspect powered edge flows, round trips,
  acceptance, and proposal effectiveness before calling it an energy barrier
  or expecting parallel tempering to help. One intermediate power can then
  locate where the restriction develops.
- Similar endpoint behavior provides little reason to build parallel
  tempering.

Do not require powered chains to share a realized random trajectory: their
acceptance decisions legitimately diverge as soon as \(\beta\) differs.

## Next restricted gamma--beta baseline

After the Voronoi profile, implement a minimal active-only RJMCMC on one
canonical, fixed-direction dyadic supertree. This is deliberately a restricted
tree-frontier model, not yet the proposed full-tiling model:

- the split orientation and hierarchy are fixed in advance;
- active leaves form an exact frontier of that one tree;
- the structural prior and its induced leaf-count prior are explicit;
- positive mass is represented by a root total and active split fractions;
- local split and sibling-merge moves are the only structural moves initially;
  and
- the same generic flow, reversal, residence, target-delta, sensitivity, and
  prediction diagnostics are reused.

Validate the sampler first on an exhaustively enumerated tiny tree and compare
its stationary probabilities with the existing fixed-tree product-space
oracle. Then run a prior-only mobility benchmark and a small synthetic
likelihood. This establishes how slowly the local split/merge graph mixes
before adding alternative split directions, full-tiling multiplicity,
dimension-preserving edge moves, informed selection, Numba, or production
NAME data.

The baseline should be judged as a diagnostic of local tree moves, not as
evidence that one preselected hierarchy adequately represents partition
uncertainty.
