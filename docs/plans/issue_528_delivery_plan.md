# Issue 528 Delivery Plan

Date: 2026-08-11

Status: proposed delivery sequence for review; focused ownership issues
[#573](https://github.com/openghg/openghg_inversions/issues/573)--
[#576](https://github.com/openghg/openghg_inversions/issues/576) were opened
on 2026-08-10. PR ownership remains a project-owner decision.

## Purpose

This plan turns [#528](https://github.com/openghg/openghg_inversions/issues/528)
from an umbrella design issue into a sequence of independently reviewable
changes. It covers:

- the semantic-model ADR and scientist-facing model description;
- the coherent native-to-reduced prior and likelihood transformation;
- correlated LogNormal states and covariance-aware observation models;
- durable identities, reconstruction, and outputs;
- immediate correctness defects found during the 2026-08 architecture review;
- pressure tests from `verification-games`, inner/outer models, and linked
  CO2/O2 models.

The plan deliberately does not use an agent tracker and does not assign work to
named people. Each slice should have one implementation owner and a reviewer
who did not author the slice.

## Decisions To Preserve

### Preserve the concrete one-sector model

The familiar one-sector RHIME route remains valuable as:

- a readable concrete implementation of the simplest scientific model;
- a regression and parity oracle for semantic normalization and compilation;
- a documentation example that a scientist can compare directly with the
  model equations.

Removing the concrete route or the binary standard/multisector orchestration is
not an immediate objective. Once the semantic route is mature, the concrete
route may remain as executable reference code or become documentation-only.
That later choice must be based on maintenance cost and parity coverage, not on
an abstract desire to remove duplication.

### Treat the native model as the scientific prior

The current convenient model places the same configured prior family and
usually the same scalar parameters on every basis coefficient. Consequently,
changing the number, size, or shape of basis regions changes the implied prior
on physical flux. Small and large regions receive the same coefficient prior,
and aggregate uncertainty does not follow from a common probability model.

Because users are free to choose basis regions, the more coherent
interpretation is that a probability model exists on a declared native grid.
The retained basis state, effective forward operator, unresolved observation
covariance, and reconstruction products are all derived from that one native
model. Basis choice then changes computational allocation and the retained
coordinates, rather than silently selecting a new scientific prior.

This is the central scientific invariant for #493 and #566.

Gaussian and LogNormal native priors use the same labelled arithmetic-moment
reduction contract: project the native mean and covariance and derive the
retained moments, effective forward operator, and unresolved second moments
together. For the Gaussian family the affine conditional reduction is exact.
For the LogNormal family, the affine conditional map and constant unresolved
covariance are linear-Bayes products derived from the first two moments;
fitting the retained LogNormal law and closing the conditional residual
distribution are further declared approximations. The native probability
family and every approximation or closure therefore remain part of the
identity-bound reduction record.

### Make reporting boundaries a basis-design constraint

The simplest reliable scientific guidance is:

> Do not allow basis regions to cross countries or other reporting regions for
> which posterior totals will be required.

The basis machinery can enforce this policy. It makes the desired reporting
functionals lie in the retained span and avoids presenting deterministic
prolongation as exact native posterior reconstruction.

Crossing a reporting boundary is still permitted for advanced use, but then
the preparation request must explicitly include the reporting functional and
retain the required functional-state, functional-observation, and functional
variance products. Output code must say whether a result is retained exactly,
reconstructed with unresolved uncertainty, or only approximated.

### Describe models mathematically before compiling them

The target level of inspection is a scientist-facing model card containing:

- labelled state blocks and their arithmetic prior moments;
- named forward-model terms and a complete mean equation for each observation
  model;
- named covariance components and their scientific provenance;
- retained, fixed, structurally absent, and coherently marginalized states;
- requested physical output functionals and reconstruction status;
- exact, moment-closed, low-rank-approximated, and numerical choices;
- a separate compilation manifest mapping serialized semantic IDs to PyMC and
  output names.

PyMC variable names, sanitized suffixes, and builder strategy are compilation
details, not the description of the scientific model.

## Status Vocabulary

The following terms should be used consistently in issues, documentation, and
progress reports.

| Term | Meaning |
|---|---|
| **Implemented** | Present on `devel` with tests. Limitations may remain, but the stated capability exists. |
| **In flight** | An open PR contains a reviewable implementation slice. |
| **Planned** | An open issue owns the work and has acceptance criteria; absence from `devel` is not an unplanned design omission. |
| **Implied but unscheduled** | Mentioned by an umbrella or adjacent issue, but not isolated into a reviewable deliverable. |
| **Identified gap** | Required behaviour has no adequate issue owner or acceptance criteria yet. |
| **Deferred pressure test** | Deliberately later work that must shape the design now but should not block the first semantic-model slice. |

## Current Design Model

The design should retain six distinct layers:

```text
scientific specification
    native states, priors, forward-model terms, observation models, covariance sources, outputs
        -> canonical prepared inputs
           observations, native products, labels, units, provenance
               -> bound mathematical model
                  labelled operators, moments, reductions, equations, exactness ledger
                      -> derived numerical views/artifacts
                         dense/LRPD/operator representations, derived identities, diagnostics
                             -> backend realization
                                PyMC initially; analytic Gaussian as an independent oracle
                                    -> compilation and output manifest
                                       trace names, reconstruction recipes, product adapters
```

`RhimePreparedInputs` remains the canonical gathered-data boundary. A semantic
model refers to its labelled data; it should not duplicate acquisition or use
the shape of `H` to infer scientific identities.

The current private `_FluxPlan` is a useful normalized flux compilation plan,
not the semantic model itself. It can be one lowering target while #528 is
developed.

## Facts, Decisions, And Inferences

### Facts from current code and issue scope

- Current scalar and array RHIME priors broadcast a distribution over basis
  coefficients; they do not derive a joint prior from a correlated native
  covariance.
- Independent-cell prior-width projection and calibration are implemented, but
  the correlated native covariance action and coherent reduction are not.
- Dense and low-rank-plus-diagonal observation covariance can be consumed by
  the likelihood, but OGI does not yet produce the coherent covariance or its
  approximation from a native covariance ledger.
- Standard and multisector declarations normalize to a private flux plan, but
  current sector/source specifications still encode convenient one-to-one
  relationships.
- Merged [PR #571](https://github.com/openghg/openghg_inversions/pull/571)
  provides the backend-neutral correlated-LogNormal arithmetic-moment,
  whitening, label, and serialization foundation for #565. Built-in
  `RhimeModelSpec` routing is explicitly left for a later slice.
- #493, #565 through #570, and #572 contain explicit implementation or
  documentation scopes. They are planned work, not discoveries made by this
  review.
- #414, #415, #456, #511, and #411 through #413 already own important
  component, persistence, grouped-state, product, and linked-observation-model
  work.

### Decisions already implied by the work

- Native covariance projection and coherent reduction are separate contracts:
  #493 produces labelled product blocks; #566 performs the solve-based
  reduction and produces one coherent artifact.
- The coherent artifact owns the identity-bound mathematical covariance or
  covariance action. Dense, diagonal, LRPD, block, and matrix-free forms are
  downstream numerical views with separate derived identities and diagnostics;
  an LRPD view makes repeated likelihood evaluation and sampling feasible.
- Fixed-value state activity and coherent marginalization remain different
  operations.
- Flux component is the canonical physical/reporting identity. The existing
  `sector`, one-sector, and multisector vocabulary remains a compatibility and
  group-facing name; “multisector” means multiple flux components and does not
  imply EDGAR-style inventory sectors.
- A fixed state-to-term coupling records a labelled transform such as an
  oxidative ratio, sign, and unit conversion. If the coupling is uncertain,
  it is represented by a state block and the resulting bilinear or nonlinear
  forward-model dependence is explicit.
- Arithmetic moments are the public scientific contract for positive state
  coefficients. Latent Gaussian moments and whitening are realization details
  with recorded provenance.
- Aggregation covariance, temporal mismatch, transport uncertainty, and other
  discrepancy sources are distinct scientific components even when they share
  a dense or LRPD numerical representation.
- Adding those components assumes their residuals have zero cross-covariance.
  Dependent mechanisms require a joint component or explicit cross terms; the
  DUBFI operator--aggregation interaction is the immediate pressure test.
- Approximation policy must be explicit. Exact Gaussian reduction, LogNormal
  moment closure, and low-rank truncation must not be reported as the same kind
  of approximation.

### Design inference to validate in the ADR

The semantic core should be a small relational linear model, not a general
probabilistic-programming graph. The minimum relations are flux components,
native-state models, retained state blocks, reductions, forward-model terms,
fixed state-to-term couplings, named observation-model means, covariance
components, observation models, output functionals, and a compilation
manifest.

This inference should be pressure-tested against grouped source inputs,
inner/outer bases, and shared CO2/O2 states before the representation is made a
public extension API.

## Planned Versus Missing

| Capability or defect | Status | Existing owner | Delivery decision |
|---|---|---|---|
| Independent-cell basis prior-width projection | Implemented | #521 | Keep as a documented special case and diagnostic; do not present it as coherent correlated reduction. |
| Dense, diagonal, and LRPD likelihood consumption | Implemented | #564 | Generalize terminology through a covariance-component ledger; retain existing API compatibility. |
| Correlated gathered LogNormal state foundation | Merged foundation | #565 / PR #571 | Use the merged arithmetic-moment, whitening, label, and serialization contract. |
| Built-in model-spec routing for one gathered correlated state | Planned | Remaining #565 | Separate PR after #571; preserve per-sector compatibility views without creating independent states. |
| Native covariance action and labelled product-block projection | In flight | #493 | Implement before #566; start with the separable exponential operator and small dense oracles. |
| Coherent native-to-reduced preparation | Planned | #566 | Consume #493 products; return one identity-bound result containing prior, forward, residual, and reconstruction products. |
| Fixed aggregation covariance plus site OU mismatch | Planned | #567 | Implement after covariance-component semantics are agreed; do not hard-code experiment-specific time scales. |
| Cached conditional sampler lifecycle | Planned | #568 | Defer until #567 has a correct normalized likelihood and profiling shows the need; keep it outside the scientific model. |
| Conditional held-out prediction | Planned | #569 | Depends on #567 covariance descriptors and #415 persistence. |
| Covariance-safe derived output routing | Planned | #570 | Implement incrementally after a manifest exists; do not block safe state/flux outputs on unsupported concentration products. |
| User-facing coherent-reduction examples | Planned | #572 | Write concepts now; executable examples follow stable #493/#566 APIs. |
| Backend-neutral component preparation and reconstruction | Planned | #414 | Reuse as the transform/reconstruction seam; avoid a second component framework. |
| Reproducible run bundle | Planned | #415 | Persist semantic/model-card, prepared-reduction, covariance, and manifest versions. |
| Grouped inner/outer state layouts | Planned | #456 and #407-#410 | Treat as a required semantic pressure test, not a prerequisite for the first implementation. |
| Per-sector PARIS concentration product | Planned | #511 | Consume stable product-neutral identities; keep product schema decisions out of the semantic core. |
| Shared-state CO2/O2 observation models | Planned | #411-#413 | Required ADR pressure test; implement after observation-model/state/term relations are stable. |
| Validate covariance second-axis labels, ordering, and uniqueness | Planned | [#573](https://github.com/openghg/openghg_inversions/issues/573) | Small correctness PR; do not hide it within #493 because current dense covariance is already exposed. |
| Define positive-diagonal semantics for standalone and composed LRPD covariance | Planned | [#573](https://github.com/openghg/openghg_inversions/issues/573), adjacent to #564/#567 | Isolate contract and regression tests before #567. A zero residual tail may be valid only when another declared component makes the complete covariance proper. |
| Make `output_format="none"` skip gathered-state diagnostics and reconstruction | Planned | [#574](https://github.com/openghg/openghg_inversions/issues/574), related to #570 | Small correctness PR, independent of the broader output redesign. |
| Let complete custom builders bypass built-in sector/source preflight | Planned | [#574](https://github.com/openghg/openghg_inversions/issues/574) | Small runner-contract PR. Built-in builders retain their existing validation. |
| Typed, versioned output/compilation manifest | Planned | [#575](https://github.com/openghg/openghg_inversions/issues/575), with consumers in #414/#415/#570 | First records may remain private while the pressure tests stabilize their identities. |
| Scientist-facing model-card schema and renderer | Planned | [#575](https://github.com/openghg/openghg_inversions/issues/575) | Primary #528 scientific deliverable; first artifact can be documentation/data-model only. |
| Analytic-Gaussian realization of the bound model | Planned | [#576](https://github.com/openghg/openghg_inversions/issues/576), consuming #493/#566/#575 | Use as an independent mathematical and PyMC parity oracle, not the production scaling path. |
| DUBFI-like state-dependent transport covariance | Deferred pressure test | No OGI implementation issue identified | Record requirements in #528 before #567 settles the covariance interface; open implementation work only after the next verification experiment defines the intended model. |

## Ownership And Dependency Map

```text
#528 ADR and vocabulary
  |
  +--> #573 covariance validation/LRPD positivity (independent)
  +--> #574 builder/no-output runner boundaries (independent)
  +--> #575 model card and typed manifest
  |
  +--> PR #571 merged correlated-state foundation
  |      `--> #565 built-in gathered-state routing
  |
  +--> #493 native covariance action and product blocks
  |      `--> #566 coherent reduction
  |              +--> #576 analytic Gaussian realization
  |              +--> #572 executable documentation
  |              +--> #570 covariance-safe outputs
  |              `--> #414 reconstruction products
  |
  +--> covariance-component semantics
  |      `--> #567 fixed aggregation + site OU
  |              +--> #568 cached conditional sampler, if justified
  |              `--> #569 conditional held-out prediction
  |
  +--> #575 typed/versioned compilation-output manifest
         +--> #414 reconstruction facade
         +--> #415 run bundle
         +--> #570 output routing
         +--> #511 PARIS concentration design
         `--> #413 tracer-aware outputs

Pressure tests, not initial blockers:
  #456 / #407-#410 inner/outer grouped layouts
  #411-#413 shared-state linked observation models
  DUBFI-like uncertain-operator marginalization
```

The dependency arrows describe contract dependencies, not a requirement that
every parent issue be closed first. For example, active #493 covariance-action
work can proceed before #565 built-in routing, and #528 documentation can
proceed without every runtime implementation.

## The Next Two Days

The aim is significant, reviewable progress without committing to a hurried
framework implementation.

### Day 0: 2026-08-10 — agree the model and protect current behaviour

1. **Finish the #528 ADR and overview document.**
   - Record the relational vocabulary, identity/cardinality rules, equations,
     exactness ledger, model-card example, and migration constraints.
   - Include the current equal-per-basis-region prior as the motivating
     counterexample.
   - Include one-sector, gathered multisource, inner/outer, linked CO2/O2, and
     state-dependent covariance pressure tests.

2. **Freeze the concrete one-sector parity oracle.**
   - Name the existing route as reference behaviour.
   - Specify parity checks before any compiler or semantic-model refactor.
   - Do not remove the route or change user-visible names in the two-day window.

3. **Use the foundation merged in PR #571.**
   - Keep its arithmetic-to-latent moment equations, label validation,
     state-treatment semantics, and serialization as the bounded foundation.
   - Leave built-in model-spec routing to the remaining #565 slice rather than
     expanding the merged contract retrospectively.

4. **Open or prepare the four small correctness slices.**
   - Covariance second-axis alignment and uniqueness.
   - LRPD complete-covariance positivity semantics.
   - `output_format="none"` early exit for custom/gathered states.
   - Custom-builder bypass of built-in sector/source preflight.
   - Scheduling complete: the covariance pair is #573 and the runner pair is
     #574.

5. **Decide explicit ownership for model card and typed manifest.**
   - Scheduling complete: #575 owns the minimal model card and typed manifest;
     #576 owns the analytic-Gaussian realization used as an independent oracle.

### Day 1: 2026-08-11 — land foundations in small slices

1. Merge or make review-ready the four correctness PRs. Keep them separate from
   semantic-model refactoring so they can be reviewed and backported easily.
2. Start the next #565 slice: bind the prepared correlated prior to the built-in
   model spec, construct one gathered public state, expose labelled per-sector
   selectors, and cover standard-runner save/load.
3. Start #493 slice A: labelled separable covariance action with no class
   blocking and explicit-dense equality tests. Do not add coherent solves yet.
4. Add a frozen synthetic Gaussian theorem fixture that later #566 code must
   satisfy for two different basis choices.
5. Turn the ADR's worked example into a hand-authored model card. It need not be
   generated by code yet; it is the acceptance target for the future renderer.

### Day 2: 2026-08-12 — demonstrate the route through the architecture

1. Make #493 slice A review-ready; if it is already stable, begin slice B for
   class blocking and labelled product blocks.
2. Make the #565 built-in routing slice reviewable, or document the precise
   blocker if it depends on a term-selection contract.
3. Draft #566's pure result contract and reference identities against the
   frozen Gaussian fixture. Avoid binding it prematurely to `RhimeModelSpec`.
4. Update the high-level overview with actual merge/PR status using the status
   vocabulary above.
5. Record what remains planned rather than describing all absent functionality
   as missing.

### Two-day success criteria

- The ADR and overview are reviewable and use stable vocabulary.
- The concrete one-sector route has an explicit parity role.
- PR #571 is merged as a bounded correlated-state foundation.
- The four correctness gaps have explicit owners and at least focused tests or
  review-ready PRs.
- A first #493 covariance-action slice is open or review-ready.
- The #565 built-in integration slice is started with bounded scope.
- A golden Gaussian fixture and a representative scientist-facing model card
  exist as shared acceptance targets.

Completion of all #493/#566 mathematics, the public component API, or DUBFI-like
model error is not a two-day success criterion.

## Independently Reviewable Implementation Slices

### C1 — Dense covariance coordinate safety

**Owner:** #573.

**Scope:** validate both covariance axes against canonical `nmeasure` identity;
require uniqueness; either align explicitly by labels or reject reordered data;
retain labels until after validation.

**Acceptance tests:**

- matching site/time labels pass;
- reordered labels are deliberately aligned or rejected according to the
  documented policy;
- duplicate labels fail;
- same-shaped but different labels fail;
- dense and LRPD entry points use the same identity policy.

### C2 — LRPD positivity and zero-tail semantics

**Owner:** #573, adjacent to #564 and #567.

**Scope:** distinguish a standalone fixed LRPD covariance from a residual
component that will be combined with sampled or fixed positive variance. Test
the complete covariance actually used by the normalized likelihood.

**Acceptance tests:**

- standalone LRPD with an improper complete covariance fails clearly;
- zero residual-tail entries are accepted only when the composed covariance is
  proper on the declared support;
- no whitening path divides by zero;
- dense and LRPD log probabilities agree for the smallest valid composed case;
- the error message identifies the offending component and coordinate.

### C3 — No-output means no reconstruction

**Owner:** #574, related to #570.

**Scope:** return before built-in multisector diagnostics when
`output_format="none"`; sampling and trace production remain available to the
caller, but no conventional `x_<suffix>` variables are assumed.

**Acceptance tests:**

- a custom gathered-state builder with no `x_<suffix>` variables completes
  when output is disabled;
- no diagnostic or product writer is called;
- current basic, PARIS, legacy, and `inv_out` routing is unchanged.

### C4 — Complete-builder preflight boundary

**Owner:** #574.

**Scope:** built-in builders retain sector/source/basis preflight. A complete
custom builder receives canonical prepared inputs without first being forced
through the built-in one-source-per-sector contract. The builder must declare
or validate its own supported semantic requirements.

**Acceptance tests:**

- grouped sources and a gathered state can reach a complete custom builder;
- built-in builders still reject malformed or ambiguous sector/source maps;
- failure attribution distinguishes runner input failure from custom-builder
  contract failure;
- existing custom builder behaviour remains compatible.

### S1 — Correlated state foundation

**Owner:** merged PR #571 under #565.

**Status and scope:** merged backend-neutral arithmetic moments, latent
conversion/whitening, labels, state-treatment metadata, and serialization.

**Acceptance tests:** use #565's current analytic and Monte Carlo moment tests,
label reorder/rejection tests, fixed-versus-marginalized tests, and save/load
coverage.

### S2 — Built-in gathered correlated state

**Owner:** remaining #565.

**Scope:** a prepared correlated-prior reference in the model spec; one public
gathered state; explicit labelled term selectors; compatibility deterministics
only where required; variable roles and provenance through standard output.

**Acceptance tests:**

- one gathered state feeds multiple source/sector terms;
- cross-source covariance is retained;
- selectors preserve non-lexical source order and ragged labels;
- save/load retains the public state and moment provenance;
- diagonal covariance agrees with current independent-state behaviour;
- concrete one-sector parity remains exact within numerical tolerance.

### P1 — Native covariance action

**Owner:** #493 slice A.

**Scope:** labelled separable exponential covariance action on one or multiple
right-hand sides; no class blocking or source correlation in the first PR.

**Acceptance tests:** explicit dense Kronecker equality on small grids, labelled
dimension/order validation, configurable length scales and amplitude, and a
test that the implementation path does not materialize a native `N x N`
matrix.

### P2 — Class/source blocks and product projections

**Owner:** #493 slices B and C; split if review size grows.

**Scope:** class blocking, independent source blocks with source-specific
amplitudes, gathered/ragged ordering, and labelled `Pi B Pi.T`, `H B Pi.T`,
`H B H.T`, and optional `Q` products.

**Acceptance tests:** explicit dense equality; zero cross-class covariance;
non-lexical source ordering; arbitrary reporting functional blocks; stable
content identity and serialization.

### R1 — Pure coherent reduction

**Owner:** #566 slice A.

**Scope:** consume labelled product blocks; solve for the induced state
covariance, effective observation operator, centering, and unresolved
observation covariance without explicit inverses. Return one immutable,
identity-bound result.

**Acceptance tests:**

- `A + H_alpha C_alpha H_alpha.T = H B H.T`;
- centred forward equality;
- native and reduced Gaussian prior-predictive equality;
- two basis choices produce the correct projected posteriors and equal model
  evidence in the exact synthetic case;
- redundant retained coordinates fail validation before solving; no
  pseudoinverse semantics are selected implicitly;
- label reorder/rejection and solve-tolerance failures are explicit.

### R2 — Approximation, reconstruction, and preparation adapters

**Owner:** #566 slices B and C, with #414/#415 coordination.

**Scope:** construct a diagonal-preserving LRPD derived numerical view, with
its own identity and diagnostics, from the exact labelled covariance or
covariance action produced by R1; add arbitrary functional reconstruction
products, serialization, and adapters to
`RhimePreparedInputs`, #565, and the observation-covariance API.
The durable result retains \(B_\perp\), a conditional-covariance action, or
sufficient product access for a future operator-anomaly calculation of
\(\mathcal K_H(B_\perp)\); it must not make a single fixed observation matrix
the only surviving mathematical content.

**Acceptance tests:**

- declared covariance diagonal is preserved;
- approximation rank and discarded-error metrics are serialized;
- requested country functionals match explicit dense calculations;
- crossing reporting boundaries cannot silently use deterministic
  prolongation;
- a saved result retains one content identity across prior, forward,
  unresolved covariance, and reconstruction products.

### G1 — Analytic-Gaussian realization

**Owner:** #576, consuming #493, #566, and #575.

**Scope:** realize the labelled linear-Gaussian subset without PyMC; provide
normalized prior-predictive, posterior, and quantity-of-interest calculations
as an independent mathematical oracle. Dense algebra is acceptable because
this is a correctness reference rather than the production scaling path.

**Acceptance tests:** two bases give corresponding pushforwards of the same
native posterior; nonzero means verify centring; aligned and deliberately
non-aligned output functionals verify the reconstruction boundary; analytic
and PyMC realizations agree on one small bound-model fixture; labels and one
shared provenance identity prevent incoherent blocks from being mixed.

### M1 — Minimal semantic model and model card

**Owner:** #575, a focused deliverable under #528.

**Scope:** backend-neutral value objects or a versioned serialized schema for
state blocks, forward-model terms, named observation-model means, covariance components, reductions,
outputs, and approximation status. Render a scientist-facing model card. Do
not introduce a registry or class hierarchy until two independent extensions
need it.

**Acceptance tests:**

- the simple one-sector RHIME model is representable without PyMC names;
- one state can contribute to two terms or observation models;
- one term can select a labelled subset of a gathered state;
- input source, physical component, state, trace variable, and product name are
  demonstrably distinct identities;
- validation rejects dangling IDs, incompatible dimensions/units, and unnamed
  observation-model means;
- model-card output contains equations, arithmetic moments, covariance ledger,
  state treatment, outputs, and exactness labels;
- schema/version round trip is stable.

### M2 — Typed compilation and output manifest

**Owner:** #575, consumed by #414, #415, #570, and #413.

**Scope:** versioned mapping from serialized semantic IDs to concrete trace/data
names, public effective states, reconstruction recipes, product-neutral
coordinates, and product-specific adapters. Retain legacy suffix fallback only
for old artifacts.

**Acceptance tests:**

- gathered state selectors and public effective state are recorded;
- semantic identities survive changed PyMC variable names;
- outputs can reconstruct from the manifest without suffix parsing;
- old suffix-based artifacts remain readable;
- manifest and model-card versions are included in the run bundle.

### O1 — Covariance-component ledger and site OU

**Owner:** semantic portion under #528; implementation under #567.

**Scope:** name scientific covariance sources independently of their dense,
diagonal, block, or LRPD representation; compose fixed coherent aggregation
covariance with site OU mismatch and observation variance.

**Acceptance tests:** retain all #567 dense/LRPD, limiting-case, gradient,
predictive, reordered-label, and provenance criteria. Additionally ensure the
model card names each component and reports which parameters are fixed or
sampled.

### O2 — Conditional prediction and optional custom sampling

**Owner:** #569 and, only when justified, #568.

**Order:** implement saved-artifact covariance descriptors and correct
conditioning first. Add cached sampler lifecycle only after profiling identifies
it as the highest-value route and stale-cache safety can be tested.

## Parity Oracle

Before semantic compilation becomes the default for any current model, compare
it with the concrete one-sector route on a frozen small fixture.

Required parity checks are:

- state coordinate and prior arithmetic moments;
- modelled observation expression and prior predictive moments;
- normalized pointwise log probability at fixed parameter values;
- public effective state and deterministic `mu`;
- variable roles and supported modern outputs;
- saved/reloaded coordinates and metadata;
- posterior summaries within Monte Carlo tolerance for one short integration
  test.

The analytic Gaussian realization should become a second, mathematically
independent oracle for Gaussian cases. It is especially valuable for #566,
where comparing two PyMC graphs could reproduce the same implementation error.

## Documentation Deliverables

### ADR under #528

The ADR should decide:

1. semantic identities and their cardinalities;
2. prepared-input, bound-math, numerical-artifact, and backend boundaries;
3. state treatment and reduction semantics;
4. observation-model mean and covariance composition;
5. exactness/closure/approximation vocabulary;
6. output functional and manifest contracts;
7. migration and one-sector parity policy.

It should defer concrete public class names, registries, and a fully general
nonlinear graph API.

### High-level overview / slide-source document

The companion overview should cover:

- why equal priors on arbitrary basis regions are scientifically unstable;
- how native covariance produces realistic regional uncertainty without huge
  cellwise variance;
- coherent reduction and unresolved covariance;
- LogNormal moment conversion, closure, and whitening;
- LRPD numerical representation and validation;
- empirical need for temporal observation correlation;
- current status using the vocabulary in this plan;
- the semantic model and model-card target;
- DUBFI-like uncertain-operator marginalization as the next design pressure.

### User documentation under #572

Concept prose and notation can start immediately. Executable examples should be
merged alongside stable #493/#566 interfaces to avoid documenting speculative
function names.

## DUBFI-Like Model Error As The Next Pressure Test

The next model-error work is another marginalization problem. An uncertain
affine observation operator produces a state-dependent covariance after the
operator perturbation is marginalized. That creates requirements not covered
by a fixed aggregation covariance or a fixed-tau OU component:

- a covariance component may depend on the retained physical state;
- the likelihood must remain normalized, including the log-determinant term;
- the deterministic mean operator and ensemble-anomaly covariance have
  different scientific roles;
- grouping or localization choices and their provenance must be explicit;
- fixed, sampled, and state-dependent covariance components must compose;
- dense, low-rank, latent-factor, and cached representations are numerical
  strategies for the same declared component;
- posterior prediction needs the same state-dependent covariance law;
- joint-observation covariance and cross-observation blocks may eventually be
  needed.

The likelihood convention is also scientific. `determinant_weight=1` denotes
the normalized Gaussian uncertain-operator model; zero denotes an
unnormalized Bruch-style quadratic objective; and fractional values are
calibration or sensitivity objectives. They must be separate model-card
choices, not hidden numerical switches.

The #528 ADR and covariance-component contract should be able to describe such
a component, but the first OGI implementation should not be invented before
the verification experiment fixes the intended scientific model. In
particular, #567's site OU mismatch should not become an accidental generic
transport-error abstraction.

The follow-up must also preserve the interaction with coherent aggregation.
Let \(\mu_{x\mid\alpha}:=m+U_*(\alpha-\Pi m)\),
\(x\mid\alpha=\mu_{x\mid\alpha}+u\), and
\(u\sim\mathcal N(0,B_\perp)\). Also let
\(\bar H:=\mathbb E(H\mid\alpha)\),
\(\Delta H:=H-\bar H\), and
\(\mathcal K_H(S)=\mathbb E(\Delta H S\Delta H^{\mathsf T}\mid\alpha)\).
When \(u\) and \(\Delta H\) are conditionally independent given \(\alpha\),
the observation covariance contains

$$
D_{\mathrm{obs}}
+\bar H B_\perp\bar H^{\mathsf T}
+\mathcal K_H(B_\perp)
+\mathcal K_H(\mu_{x\mid\alpha}\mu_{x\mid\alpha}^{\mathsf T}).
$$

The \(\mathcal K_H(B_\perp)\) interaction is neither a second aggregation
component nor a temporal term, and it must not be counted twice. Because the
operator acts bilinearly on unresolved random flux, the joint Gaussian
likelihood is normally a declared moment closure. Issue #566 should therefore
retain \(B_\perp\), or sufficient product-space information for this future
calculation, rather than making one fixed observation covariance the only
durable reduction artifact.

Relevant curated background lives in:

- `inversions-knowledge/docs/topics/ensemble-transport-uncertainty-and-dubfi.md`;
- `inversions-knowledge/docs/derivations/uncertain-affine-operator-marginalization.md`;
- `inversions-knowledge/docs/topics/rhime-pollution-event-scaling.md`.

## Open Questions Requiring Explicit Decisions

1. Is the first semantic representation a versioned data schema, Python value
   objects, or both? Prefer the smallest form that can render and serialize the
   model card.
2. Which unit system is authoritative at the native-state, arithmetic-state,
   observation, and output-functional boundaries?
3. Is label reordering ever automatic for covariance inputs, or must all
   covariance products be in canonical order? Whatever is chosen must be
   uniform across dense and LRPD paths.
4. Which exactness labels are stable enough for serialization: for example,
   `exact_gaussian_marginal`, `arithmetic_moment_closure`,
   `diagonal_preserving_lrpd`, and `backend_numerical`?
5. How should a model card display a state-dependent covariance without
   confusing a scientific covariance component with its cached numerical
   realization?
6. Which reporting-region constraints should basis generation enforce by
   default, and which should be warnings?
7. At what point is the semantic extension API public? Recommendation: after
   the current one-sector model, inner/outer layout, and linked CO2/O2 model
   have all exercised the internal relation model.

## Review And Merge Discipline

- One scientific or architectural contract per PR.
- Include equations and labelled dimensions in every covariance/reduction PR.
- Prefer pure NumPy/xarray tests before PyMC integration tests.
- Require an explicit dense oracle for every structured numerical route.
- Require save/load tests whenever labels, provenance, model cards, or manifests
  are added.
- Record exactness and approximation status in test names and public metadata.
- Run focused tests and lint while iterating; run the repository's full default
  tox workflow exactly once when each branch is ready for a draft PR, following
  `AGENTS.md`.
- Do not combine removal of legacy/concrete paths with introduction of the new
  semantic path.
- Do not merge an output change that requires inference from a new sanitized
  suffix; extend the manifest instead.

## Completion Criteria For Issue 528

#528 can close when:

- the ADR and vocabulary are accepted;
- current one-sector and multisector models normalize through the declared
  relations or have an explicit documented compatibility boundary;
- a scientist-facing model card can describe the implemented model before
  compilation;
- state, source, component, forward-model term, observation-model, and output
  identities are distinct;
- state treatment and exactness/approximation records are serializable;
- the typed/versioned compilation-output manifest is consumed by modern output
  and persistence paths;
- one-state/multiple-term, ragged gathered state, inner/outer layout, and linked
  observation-model fixtures have pressure-tested the representation;
- the concrete one-sector route passes the parity suite;
- extension guidance explains how to add a prior/state component, forward term,
  covariance component, observation model, and output view without requiring
  users to reproduce the complete PyMC runner.

Closing #528 does not require closing every scientific implementation issue in
this plan. It requires that those implementations have a coherent, tested place
to attach without conflating scientific identities or backend names.
