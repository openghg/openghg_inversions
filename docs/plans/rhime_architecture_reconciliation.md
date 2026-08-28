# RHIME architecture reconciliation and delivery proposal

Status: proposed one-time reconciliation, not a third architecture roadmap

Evidence date: 2026-08-28

Local repository snapshot: `origin/devel` at `da008974`

Tracker evidence updated through 13:47 UTC, including merged PR #660

## Purpose and authority

This note verifies the attached architecture review against the repository,
GitHub, and Linear, then turns the verified remainder into bounded delivery
work. It does not supersede the two canonical plans:

- [`run_rhime` readability and modifiability](run_rhime_readability_and_modifiability.md);
- [RHIME model-family expansion](rhime_model_family_expansion.md).

The longer-lived destination, coding guardrails, programme outcomes, and
milestone model are defined in the [RHIME programme architecture and outcome
roadmap](rhime_programme_architecture_and_outcome_roadmap.md).

Once the proposed tracker changes and sequencing are accepted, update those
two plans and archive this note. Linear remains authoritative for priority,
ownership, dependencies, and status. GitHub remains authoritative for public
technical scope, review, acceptance evidence, and closing pull requests.

## Executive conclusion

The attached review reaches the right architectural conclusion: retain the
reusable labelled-data and PyMC components, make concrete procedural recipes
the primary scientific surface, and do not replace the rejected semantic
compiler with another universal framework.

The main correction is one of status and scope. The project is no longer
deciding whether to adopt recipe-first RHIME. Much of that transition has
already landed. The immediate work is to:

1. reconcile trackers and the canonical roadmaps with merged work;
2. make derived products and diagnostics scientifically safe;
3. finish concrete CO2, Verification Games, and user-adoption gates;
4. settle the remaining package, configuration, validation, and persistence
   decisions narrowly; and
5. use nested-domain RHIME as the next complete model-family acceptance test.

The work should be managed as three independent release gates:

- **scientific-output safety** — all-chain products, automatic diagnostics,
  and an explicit validity policy;
- **readable model integration** — recipe ownership, typed configuration,
  concrete model-family delivery, and truthful outputs; and
- **reproducible adoption** — real-run evidence, tutorials, replay, user task
  tests, and only then legacy retirement.

## Verified corrections to the attached review

| Review statement | Current evidence | Required correction |
| --- | --- | --- |
| Finish PR #659 as the next likelihood step. | [PR #659](https://github.com/openghg/openghg_inversions/pull/659) merged as `da008974` after the review was written. It added typed `PollutionEventSettings`, `AdditiveSigmaSettings`, and `FixedErrorSettings`, and removed both public `likelihoods.py` modules. | Treat #659 as the delivered baseline. Reconcile #611/#639 against it and use it for the in-flight fixed-OU work. |
| W2b and W4-W6 are future roadmap stages. | W2b, W4, W4b, W5, and W6 have merged delivery in PRs #598, #601, #605, #606, and #623. Their Linear items are Done. | Correct the canonical roadmap and #587 checklist before opening another architecture programme. Record any acceptance gaps separately from implementation status. |
| A small documented validation policy is still missing. | [`validation_and_xarray.rst`](../development/validation_and_xarray.rst) already defines ownership, exact alignment, eager boundaries, units, and testing policy. | Treat #665/OPE-127 as conformance and remaining ownership work, not as creation of the first policy. Keep OPE-84 as a non-blocking library pilot. |
| CO2 and linked CO2/O2 are future pressure tests. | OPE-74/75/77/119 are Done; PRs #617, #632, and #646 are merged. The repository contains concrete `rhime/co2` preparation, model, and replay runners. | Publish a recipe-maturity matrix. Finish outputs/configuration/acceptance and real-run cutover rather than re-designing the graphs. |
| The `rhime`/`models` distinction is wholly unsettled. | `models` contains reusable PyMC mechanics, has no `models -> rhime` imports, and concrete recipes live under `rhime`. Tests enforce recipe ownership. | #663/OPE-128 should decide residual ownership, exact import exceptions, and migrations. Do not reopen the settled dependency direction or require a package rename. |
| No `.values` or `np.asarray` is allowed before materialisation. | Canonical numerical guidance allows inspection of normally eager indexed dimension coordinates. It prohibits hidden payload coercion or execution. | State the rule precisely: no scientific data-payload coercion before its named boundary; eager indexed-label inspection is allowed. |
| Reframe #416 as active legacy-removal work. | [#416](https://github.com/openghg/openghg_inversions/issues/416) is closed. [#587](https://github.com/openghg/openghg_inversions/issues/587) revised its assumptions and owns the live retirement gate. | Do not revive #416. Open a new bounded residual-removal issue only after the current gates pass. |
| The open-item counts describe current repository state. | The review's scoped count became stale as #659 merged. Queries scoped to items created in 2026 and unscoped full-repository queries produce different totals. | Every published count must include query, scope, and timestamp; counts are not planning inputs. |

Two additional qualifications are important:

- PR #529 merged before its compiler path was superseded and removed. Describe
  it as **merged then superseded**, not as an unmerged rejected PR.
- The #637 experiment strongly implicates pollution-event feedback for the
  tested HFC-134a cases. It does not establish a universal causal theorem for
  every species, duration, or configuration.

## Current maturity by recipe

| Recipe | Current maturity | Remaining production claim |
| --- | --- | --- |
| Standard RHIME | Full acquisition-to-output recipe with visible orchestration and typed likelihood settings. | All-chain derived products, automatic diagnostics, tutorials/user acceptance, and legacy-retirement evidence. |
| Multisector RHIME | Full acquisition-to-output recipe with a separate visible runner. | Same safety/adoption gates; resolve outstanding sector/output issues independently. |
| CO2 only | Concrete prepared-input/model/sample seam is merged. | Complete acquisition/configuration/result/output contract, grouped reporting evidence, units, diagnostics, and scientist acceptance. |
| CO2/O2 | Concrete linked preparation, graph, and advanced replay seam are merged. | Unit-safe heterogeneous-channel contract, configuration, tracer-aware outputs, diagnostics, and real-run acceptance. |
| Nested domain | No production implementation on `devel`; #359 and #600 remain divergent reference branches. | Execute #407 -> #408 -> #409 under #666. |

Calling the current CO2 seams “production recipes” without this qualification
would overstate their acquisition, configuration, result, and output support.

## Scientific-output safety is the first release gate

The raw `InferenceData` and serialized inversion output retain all chains. The
defect is the derived-product boundary: `convert_idata_to_dataset` currently
uses `isel(chain=0)`, and modern basic/PARIS products consume that conversion.
The explicit legacy adapter also selects chain zero for compatibility.

Sampling currently logs divergences and sampler mechanics, while R-hat and ESS
are only available through an explicitly requested diagnostic. A standard
result does not automatically retain a machine-readable convergence summary.

[GitHub #645](https://github.com/openghg/openghg_inversions/issues/645) is the
current modelling-workflow umbrella. Its children separate prior-only work
(#655), diagnostics (#656), all-chain outputs (#657), and the in-memory Fluxie
adapter (#658). Preserve that hierarchy rather than making one architecture
issue own every modelling-workflow concern.

The safety work therefore has three distinct owners:

1. [#657](https://github.com/openghg/openghg_inversions/issues/657) — preserve
   all chains in derived concentration, flux, country, basic, and PARIS paths;
2. [#656](https://github.com/openghg/openghg_inversions/issues/656) — calculate
   and retain chain count, draws, divergences, R-hat, bulk ESS, and tail ESS;
3. a focused decision arising from
   [#637](https://github.com/openghg/openghg_inversions/issues/637) — define
   when products warn, fail closed, or are scientifically unsupported.

#656 and #657 can proceed in parallel. Neither should silently acquire the
policy decision: #656 explicitly does not define universal hard thresholds.

### Safety acceptance criteria

- A two-chain fixture with deliberately different chain values changes every
  supported modern derived product when either chain changes.
- Ordinary modern paths contain no implicit first-chain selection.
- Reductions state whether they preserve `chain`/`draw`, stack them into a
  sample axis, or reduce both dimensions.
- Explicit legacy compatibility either documents first-chain behaviour or
  receives an approved multi-chain mapping; it never determines modern output
  semantics.
- Diagnostics are machine-readable and travel with the result and persisted
  artifact.
- One-chain R-hat and ESS are recorded as **not assessable**, never passed.
- Converged, non-converged, divergent, and one-chain fixtures exercise the
  approved policy before scientific products are written.
- The #637 evidence is retained in a merged artifact or a durable linked
  archive with revision, configuration, outputs, and interpretation.

## Linear work omitted from the GitHub-derived review

The following items must be included in delivery planning even when they do
not need a new GitHub issue immediately.

### Model-data ownership and recoverable output

The dependency chain is:

```text
OPE-105 model-owned alignment assembly ─┐
                                       ├──> OPE-107 postprocessing ownership
OPE-106 serialization investigation ───┘
                                              └──> OPE-125 compact,
                                                   failure-safe output
```

OPE-125 records a concrete failure and cost, not speculative optimization:

- one full output was 3.906 GB and took 93.2 minutes to write;
- a one-month artifact was 721.8 MB, with 647.9 MB in `inv_inputs`; and
- the large artifact was written before the useful compressed trace, so an
  output failure could prevent recovery of the trace.

This work is separate from #656/#657. The smaller chain and diagnostic fixes
must not wait for the broader storage redesign.

### CO2 production and Verification Games cutover

```text
OPE-124 / PR #659 (Done) ───> OPE-22 / PR #648 fixed OU
OPE-114 inferred site sigma ───> OPE-115 cached-sigma sampler
OPE-116 scalar/global sigma

{OPE-22, OPE-115, OPE-116} ───> OPE-91 VG production cutover
                                      └──> OPE-99 old VG implementation retirement
```

OPE-76 remains In Progress despite merged PR #635 because cross-repository
verification evidence is still outstanding. OPE-78, OPE-79, and OPE-86 remain
active family work. OPE-118 is a deferred linked-ratio configuration option
unless a current production case promotes it.

### Verification Games operating model

```text
OPE-96 versioned run contract ───> OPE-98 tracker pilot ─┐
                                                        ├──> OPE-100 queryability
OPE-97 result/artifact catalog ─────────────────────────┘
                                                               └──> OPE-101/102/103 cleanup
```

This cross-repository track defines reproducible run identity, artifact
references, queryable evidence, and the order in which old code/data may be
retired. It is an adoption and retirement dependency, not an implementation
dependency for the ordinary recipe spine.

### Executable adoption evidence

OPE-108 and open PR #626 provide executable standard and multisector tutorials.
Tutorial-data preparation OPE-109-111 is Done; OPE-112/113 is In Review.
This is current delivery work and should feed OPE-49's user gate.

The current W2b fixture proves the public orchestration surface inside this
repository, but it monkeypatches package stages. It does not yet prove a fresh
cookiecutter-generated external package, dependency installation, real
execution, or external CI. Either run the stated external acceptance or
explicitly narrow the canonical requirement.

## Tracker reconciliation

Phase 0 is a short, read-only-first administration pass. No implementation
issue should be widened during this pass.

### Missing canonical Linear records

- GitHub #637, #645, #656, and #657 have no Linear mirrors even though Linear
  is supposed to own priority and dependencies.
- GitHub #661 has no Linear parent, while OPE-126, OPE-127, and OPE-128 act as
  its children.
- OPE-126/127/128 currently have no priority or owner despite being described
  as the near-term architecture agenda.

Create the missing canonical relationships, then record one owner and priority
for each safety or architecture decision. Do not duplicate the complete issue
body in both systems.

### Recommended GitHub close/rewrite actions

| Item | Action | Evidence required in the closing/rewrite note |
| --- | --- | --- |
| #370 | Close as superseded by #587/#663. | Link the extracted builder, current concrete recipes, and remaining ownership issue. |
| #563 | Close as implemented. | Link `registered_model()`, its public export, merged #616/#633, documentation, and coordinate-restoration tests. |
| #575 | Close as not planned in its current #528 manifest/model-card form. | Preserve only a separately justified small human-readable resolved-model/provenance need. |
| #576 | Rewrite under #566/shared scientific foundations. | Retain the dense analytic Gaussian oracle; remove dependencies on #528/#575 and compiler terminology. |
| #622 | Close as trigger-only reference work. | Copy its “revisit when another production consumer exists” condition into the closure. |
| #611 | Narrow to the still-hidden distinction between the two pollution-event variants. | Reference current peer components and typed dispatch. |
| #639 | Reconcile against merged #641/#659. | Move the remaining minimum-floor/default decision to #664; close or rewrite only the residual scientific contract. |
| #359/#600 | Mark reference-only and superseded by #666/#407-409; close after evidence extraction. | Record configs, fixtures, assets/licences, diagnostics, scientific checks, and whether real-data evidence is trustworthy. |
| #392/#410 | Audit against #626/#587 before consolidation. | Map each acceptance criterion; do not close by title similarity alone. |

Do not reactivate closed #416. Open a new residual-removal issue after the
retirement gates pass.

### Linear cleanup actions

- Make OPE-21 an explicit coordination/reference umbrella over the concrete
  OPE-55 replay work, or close it after all distinct children are represented.
- Resolve OPE-30 versus the concrete WUR-scoped OPE-88 audit.
- Move any still-useful OPE-37 documentation to OPE-73/OPE-79, then close the
  stale semantic-pressure-test wording.
- Reparent OPE-36 under the current typed-configuration programme if work
  remains.
- Audit OPE-80 for completion: `_rhime_flux.py` and compiler plans are gone,
  reusable operations live in `_flux.py`, and recipes own state decisions.
- Keep OPE-104 as a research trigger unless a concrete component-return API
  change is ready.
- Do not mark OPE-76 complete solely because its repository PR merged; its
  Verification Games evidence is part of its acceptance.

### Phase 0 exit criteria

- The two canonical repository plans and their Linear delivery documents show
  the same completed and remaining stages.
- No active item treats #528 or the compiler as a production parent.
- Every active urgent/high item has one owner, priority, dependencies, output,
  and acceptance test.
- Every top-level Linear project has a real lead and status.
- Counts in status reports are generated from dated, scoped queries.

## Actionable delivery plan

The phases below are gates, not estimates. In-flight, independent pull requests
may proceed while Phase 0 is reconciled.

### Phase 1 — make results scientifically safe

**Work**

- Implement #657 and #656 as independent focused changes.
- Create and approve the #637 output-validity/fail-closed decision.
- Define a bounded PEFO replacement or validation task using additive-sigma and
  fixed/OU evidence; do not infer universal replacement from one experiment.
- Add these gates to #587 and both canonical plans.

**Exit**

- All safety acceptance criteria above pass.
- Modern production claims and legacy retirement are blocked explicitly on the
  safety gate.

### Phase 2 — finish current delivery and adoption evidence

**Work**

- Review and finish PR #648/OPE-22 and PR #626/OPE-108.
- Record merged PR #660's sparse-site minimum-error correction and its
  resulting scientific/default policy in the #664 ledger.
- Complete OPE-112/113 tutorial-data review.
- Decide whether W2b means a real fresh external package; execute that test or
  narrow the requirement explicitly.
- Record the scientist cohort, exact tasks, rubric, and evidence location for
  OPE-49.

**Exit**

- Standard and multisector tutorials run from clean documented prerequisites.
- A representative scientist can locate and change one prior or likelihood,
  run acquisition through output, and explain the result.
- #587 shows only genuinely remaining acceptance work.

### Phase 3 — settle residual ownership, configuration, and persistence

These decisions can be prepared in parallel, but implementation dependencies
are explicit.

**3a. OPE-128 / #663 — narrow package ownership**

- Treat `rhime -> models` and recipe-first ownership as settled.
- Decide residual owners for `_model_building.py`, compatibility aliases from
  modern preparation into `hbmcmc`, CO2 configuration/outputs, and stale public
  helpers such as `add_inferpymc_likelihood_component`.
- Publish current/target trees, allowed import edges, exact compatibility
  exceptions, and a public import migration table.
- Split moves into small issues; do not rename `models` without a demonstrated
  consumer benefit.

**3b. #661 + OPE-126/127 — configuration, defaults, validation**

- Mark #661 Phase 0 (#659) complete.
- Treat standard/multisector typed likelihood alternatives as delivered.
- Inventory the remaining recipe-level settings for standard, multisector,
  CO2, and CO2/O2.
- Classify every default as retained, explicit-required, or legacy-only, with
  units, scientific owner, and resolved provenance.
- Resolve the direct-Python versus shipped-template mismatch-model behaviour.
- Route Python and external values through one resolver; add a section-
  preserving external format only after the in-memory pressure tests pass.
- Apply the existing xarray validation policy at owning boundaries. Run OPE-84
  only as a measured pilot; add no dependency unless it deletes materially more
  code than its adapter costs.

**3c. OPE-105/106 -> OPE-107 -> OPE-125 — durable model data**

- Decide which values belong to reusable prepared inputs, model-owned assembly,
  `InferenceData.constant_data`, replay bundles, and product-specific output.
- Remove accidental duplication and raw high-volume inputs from ordinary
  durable outputs without losing reproducibility.
- Write the useful trace safely before or atomically with optional large
  products.
- Re-run the OPE-125 size/time profile and verify explicit size, time, and
  recovery targets.

**3d. OPE-55 — replay**

- Keep reusable prepared caches separate from model-bound replay bundles.
- Prove no-object-store replay, reuse by two compatible model specifications,
  early rejection of an incompatible specification, and a truthful custom-
  builder opt-out.

**Phase 3 exit**

- Each pipeline stage has one accepted owner and enforceable import direction.
- Equivalent Python and external settings resolve identically.
- Irrelevant settings fail before retrieval; resolved settings/defaults are
  retained in provenance.
- Modern output is compact and recoverable, and replay does not confuse cached
  scientific data with a bound executable model.

### Phase 4 — complete concrete model-family proof

**CO2 and Verification Games**

- Finish OPE-22, OPE-114 -> OPE-115, and OPE-116, then complete OPE-91 cutover.
- Complete OPE-76 cross-repository evidence.
- Finish OPE-86 unit-safe linked-channel reduction and OPE-79 outputs/config/
  acceptance.
- Promote shared components through OPE-78 only where two production recipes
  now demonstrate the same meaning.
- Keep OPE-118 deferred unless a production linked-ratio configuration requires
  it.

**Nested domain**

Execute the existing dependency chain:

```text
#407 / OPE-71 preparation
    -> #408 / OPE-72 readable graph and runner
    -> #409 / OPE-73 outputs, config, docs, and scientist acceptance
```

Before implementation, preserve or reject the reference evidence from #359 and
#600, including asset provenance and licence.

**Phase 4 exit**

- CO2-family production claims are supported by real Verification Games
  evidence, unit-safe labelled products, all-chain outputs, diagnostics, and a
  scientist walkthrough.
- Nested-domain support proves complementary masks/no double counting,
  one-to-one timestamp alignment, explicit outer/inner bases and priors, the
  visible `H_outer @ x_outer + H_inner @ x_inner` equation, native-grid outputs,
  a controlled end-to-end test, and a scientist modification exercise.

### Phase 5 — operating model and legacy retirement

**Work**

- Finish OPE-96 -> OPE-98, with OPE-97 in parallel, then OPE-100.
- Proceed to OPE-99 and OPE-101-103 cleanup only after parity and queryability
  exist.
- Complete OPE-66/69/70 and OPE-49's user gate.
- Decide genuine legacy product parity and migration for real INI/SLURM users.
- Open a new, bounded residual-removal issue with a deprecation window and
  release notes; do not revive #416.

**Exit**

- A run can be traced from configuration and code/data identity through jobs,
  diagnostics, evaluations, artifacts, and report.
- Real users have a documented, observed migration path.
- No modern route silently downgrades to legacy execution or first-chain
  output.
- Legacy removal is supported by safety, parity, adoption, and reproducibility
  evidence rather than the existence of a modern function name.

## Decisions that must become more specific

The attached review should not be treated as executable until these questions
have named answers:

1. **Output validity:** Which metric thresholds or model-specific rules warn,
   fail, or mark a product unsupported? Who approves them?
2. **Chain handling:** Which outputs preserve chain/draw, which stack samples,
   and which report summary statistics?
3. **Production recipe:** Does the label require acquisition, external config,
   outputs, diagnostics, provenance, examples, and scientist acceptance, or is
   a prepared-input replay seam sufficient?
4. **Package ownership:** What exact modules may import `hbmcmc` for
   compatibility, and when do those exceptions expire?
5. **Configuration:** What is the canonical in-memory type for each real recipe,
   which defaults are scientifically valid, and which external format is
   supported first?
6. **Scientist acceptance:** Which users perform which tasks, on which revision,
   with what rubric, and where is the evidence stored?
7. **Nested reference evidence:** Which #600 run, configuration, data identities,
   outputs, diagnostics, and tolerances are trustworthy enough to retain?
8. **Legacy retirement:** Which real configurations and products require parity,
   what observed migration is sufficient, and what is the release window?
9. **Gamma-Dirichlet note:** The attachment refers to an uploaded aggregation-
   error note that is absent from the attachment directory and repository.
   Link the source or remove that pressure-test claim.

## Architecture definition of done

The architecture is complete when:

- the scientific equation and execution sequence remain readable in one recipe;
- replacing a prior or complete likelihood does not require editing generic
  orchestration;
- Python and external configuration resolve through one recipe-owned boundary;
- xarray labels survive to an explicit backend boundary, with no hidden payload
  computation or coercion;
- every pipeline stage and compatibility exception has one owner;
- modern derived products use all chains and automatically retain diagnostics;
- unsupported products fail specifically before expensive or irreversible
  work;
- each claimed production recipe meets its documented maturity contract;
- real-run evidence is reproducible and queryable; and
- legacy retirement is approved from observed safety, parity, and user
  migration evidence.

This is consolidation work. No new compiler, registry, universal component
protocol, generic pipeline, or package-wide validation framework is required.
