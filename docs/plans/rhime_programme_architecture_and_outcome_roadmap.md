# RHIME programme architecture and outcome roadmap

Status: proposed programme-level north star and milestone model

Evidence date: 2026-08-28

Local repository snapshot reviewed: `origin/devel` at `da008974`

Tracker evidence updated through 13:47 UTC, including merged PR #660

Tracker state may post-date the reviewed code snapshot; in particular, PR #660
merged after `da008974` and is recorded here as tracker evidence, not as code
inspected at that revision.

## Purpose and authority

This document explains what RHIME is becoming, how the pieces fit together,
and how to organise the remaining work around observable outcomes. It is
intended to stand on its own as the programme-level roadmap.

The detailed engineering rules remain normative in:

- [`Developing RHIME models`](../development/rhime_model_development.rst);
- [`Validation and labelled-array patterns`](../development/validation_and_xarray.rst);
- [`Numerical data ownership and execution boundaries`](numerical_data_ownership_and_execution_boundaries.md).

The approved [`run_rhime` readability and modifiability
plan](run_rhime_readability_and_modifiability.md) and the [model-family
expansion plan](rhime_model_family_expansion.md) remain the detailed delivery
plans. They should be updated to distinguish work which has merged from
acceptance evidence which is still missing.

The [architecture reconciliation](rhime_architecture_reconciliation.md) is the
one-time operational companion to this roadmap. It owns the dated issue-level
status correction and tracker cleanup. This document owns the longer-lived
destination, coding guardrails, outcome gates, and milestone design. Once its
cleanup has been applied, the reconciliation note can be archived without
losing the programme direction recorded here.

Linear is authoritative for current priority, ownership, dependencies, project
status, and milestone status. GitHub is the public implementation and review
record. Repository plans define accepted engineering policy. None of the three
should silently substitute for another.

## Executive direction

RHIME should be a released collection of readable, citable scientific
inversion recipes, not a universal inversion framework.

A scientist should be able to open one named runner, follow the complete
calculation in execution order, open one nearby model function, and change one
ordinary scientific component. They should not first have to learn a compiler,
dependency-injection system, model intermediate representation, registry,
manifest lifecycle, or generic pipeline framework.

The programme must deliver six outcomes:

1. **Trustworthy results.** Modern products use all posterior chains, carry
   diagnostics, preserve scientific labels and units, and fail safely when a
   result is not supportable.
2. **A scientist-ready core.** Standard and multisector RHIME are readable,
   configurable, documented, executable from clean inputs, and demonstrably
   modifiable by their intended users.
3. **Durable runs.** Prepared scientific data, recoverable traces, final
   products, and model-bound replay artifacts have distinct, explicit
   contracts and provenance.
4. **Production model families and deliberate extensibility.** CO2, linked
   CO2/O2, radiocarbon CO2, and nested-domain RHIME graduate from tested model
   seams or reference branches into complete, citable acquisition-to-output
   products. The current experimental multigas model remains usable, and
   multisector nested-domain composition is kept open without introducing a
   universal model framework.
5. **Reproducible scientific adoption.** Verification Games, WUR25, GEMMA, and
   tutorials exercise supported package paths and retain queryable evidence.
6. **An evidence-based compatibility transition.** Legacy execution is
   retired only after safety, product parity, observed migration, and a release
   window have been demonstrated.

The first outcome is a release gate. The next four can make progress in
parallel, subject to their explicit dependencies. The sixth is an outcome of
the programme, not an early cleanup activity.

## What the product is

### The scientist's supported reading path

The normal code-reading and execution story is:

```text
Python arguments or recipe-specific external configuration
                              |
                              v
named procedural recipe
  standard | multisector | CO2 | CO2/O2 | CO2/14CO2 | nested domain
                              |
                              v
resolve -> acquire/reload -> filter/align
                              |
                              v
construct or lazily reload labelled fp_x_flux preparation cache where required
                              |
                              v
basis projection / coherent reduction -> sensitivities
                              |
                              v
backend-neutral labelled prepared scientific handoff
                              |
                              v
recipe-owned input selection -> named materialisation boundary
                              |
                              v
nearby concrete PyMC graph -> sample -> labelled result
                              |
                              v
diagnostics -> reconstruction -> durable products and provenance
```

The top-level recipe owns the scientific order. Shared code owns ordinary
mechanics whose inputs, equations, side effects, and meanings are genuinely the
same. Output code consumes a small scientific-role contract rather than
inferring meaning from PyMC variable names.

### Dependency and ownership direction

| Layer | Owns | Must not own |
| --- | --- | --- |
| Recipe interface | One normalization of Python/config inputs, compatibility spelling translation, recipe selection, visible execution order | Reflected configuration schemas, ambient contexts, executable callables in config |
| `rhime` recipe modules | Concrete standard, multisector, CO2-family, and nested-domain scientific policy and builders | A generic pipeline, compiler, global recipe registry, or framework lifecycle |
| Shared preparation/numerical modules | Retrieval mechanics, filtering algorithms, basis operations, labelled covariance/reduction, array operations | The order or policy of a concrete recipe |
| `fp_x_flux` preparation cache | Labelled, source-resolved footprint-times-flux before spatial-basis projection; identity, provenance, atomic persistence, lazy reload, and reuse | Model state, basis-bound `H`, a replay bundle, final output, or an opaque unlabelled blob |
| Prepared scientific handoff | Backend-neutral labelled arrays, scientific artifacts, units, provenance, reusable cache boundary | A serialized executable model or a promise of bitwise-identical replay |
| `models` | Reusable PyMC/scientific components which have demonstrated common meaning | Imports back into `rhime`, recipe orchestration, or config policy |
| Materialisation/build boundary | Exact selected inputs, coordinated lazy computation, dense/backend conversion, registered coordinates | Hidden computation earlier in the pipeline |
| Sampling | One named sampler boundary and labelled `InferenceData` | Product policy or implicit scientific validity approval |
| Result/postprocessing | Scientific roles, all-chain semantics, diagnostics, reconstruction, product-specific writers | Guessing roles from names or silently applying legacy chain semantics |
| Campaign repositories | Campaign policy, run matrices, scoring, artifact references, narrative analysis | Forked generic inversion, sampling, or output implementations after package parity exists |

The intended package dependency remains `rhime -> models`, never
`models -> rhime`. Residual ownership work should settle exact modules and
temporary compatibility exceptions; it should not reopen the dependency
direction or rename packages without a demonstrated user benefit.

### Supported variation and extension

Use the lowest-ceremony seam that describes the scientific change:

1. change a typed option when the data topology, state sharing, observation
   axes, and output meaning remain the same;
2. pass an ordinary Python likelihood callable for a project-specific
   likelihood variation;
3. copy the short procedural spine and replace one supported preparation or
   model stage for an advanced project-specific variation;
4. add a named in-tree recipe when the data topology, state sharing,
   observation axes, or outputs differ materially, or when the implementation
   must be stable and citable for repeated scientific use.

The advanced prepared-input/custom-builder route remains useful for replay and
structurally different graphs. It is not the conceptual centre of ordinary
RHIME development.

### Known extension constraints

Deferring a universal multigas schema or generic nested/composite graph does
not defer the requirements already visible in the next model families. New
shared seams must be pressure-tested against all of the following:

- the current experimental Ramsden methane/ethane model, including unequal
  observation axes, shared and channel-only states, fixed and uncertain
  couplings, and channel-specific boundary and error models;
- a named CO2/14CO2 recipe required no later than **28 February 2027**, with
  CO2/14CO2/O2 treated as a concrete likely extension rather than a generic
  N-channel promise;
- a multisector nested-domain case in which sector/source and inner/outer
  domain are orthogonal labelled axes; and
- CO2 nested domains as an explicit data-readiness-gated case until supported
  inner/outer footprint and source-specific flux coverage is inventoried by
  site, time, resolution, and footprint mode, not as a capability inferred
  from array shapes. Some 6 km CO2/inert footprint evidence exists; it does
  not yet establish complete production inputs.

The minimum combination invariants are:

- Each observed channel retains its own labelled observation axis, units,
  baseline or boundary policy, error inputs, and sensitivity until its recipe
  deliberately combines them.
- Shared latent states and channel-only states use exact labels. One state may
  feed several visible channel terms without forcing all channels onto one
  observation axis.
- A fixed coupling may be embedded in a sensitivity only when its direction,
  sign, units, scope, reference value, and provenance are explicit. An
  uncertain coupling is a visible sampled parameter applied to a coupling-free
  operator.
- A recipe explicitly chooses independent channel likelihoods or a labelled
  joint covariance, including mixed-unit and cross-channel-block policy.
- Channel/species, sector/source, state, and domain/basis-group remain
  orthogonal concepts. Support only combinations which scientifically exist;
  do not require a dense Cartesian product.
- Outputs retain these identities and covariance relationships through
  scientific-role mappings rather than backend variable-name parsing.
- Recipe-specific handoffs such as a future `Co2C14PreparedInputs` are
  preferred to `GenericMultigasPreparedInputs` or a generic composite model.

An extension passes this architectural pressure test when it can be added as
a named recipe plus scientifically specific preparation and output code,
without copying private runner internals, changing unrelated recipes, parsing
backend suffixes, or forking generic package mechanics into a campaign
repository.

## Production recipe contract

The word *production* should describe a complete scientific product, not only
a merged graph or a successful sampler call. A production recipe has:

- one obvious named entry point and nearby concrete builder;
- an explicitly approved user-facing operational starting boundary, normally
  acquisition/reload; a prepared-input boundary counts only when the recipe's
  milestone names it as the supported product boundary and provides its
  construction/reload workflow;
- typed Python settings and a runnable external configuration or example;
- visible preparation, materialisation, model, sampling, result, and output
  stages;
- component-owned dimensions, units, defaults, and validation;
- focused equation tests against an independent oracle;
- alignment, unit, borrowed-input, and lazy/eager-boundary tests;
- a controlled end-to-end sampler smoke test;
- supported all-chain products, machine-readable diagnostics, and an approved
  validity policy;
- recipe, code, data, basis, configuration, and scientific provenance;
- documented assumptions, limitations, outputs, and citations; and
- a recorded scientist task test or representative real-run acceptance where
  the recipe makes a production or campaign claim.

A prepared-input builder and sampling seam may be valuable and stable before
all of these exist. It should be described as a **model/replay seam**, not as a
complete production recipe. The standard, multisector, CO2-family, and nested
claims in this roadmap must each meet the full operational boundary stated in
their own milestone and runnable configuration; a successful graph build does
not silently narrow that promise.

### Current maturity

| Recipe | Delivered | Remaining production evidence |
| --- | --- | --- |
| Standard | Complete visible acquisition-to-output recipe; typed likelihood settings; public customization seam | All-chain products, automatic diagnostics and validity policy, executable tutorial completion, representative scientist acceptance |
| Multisector | Separate visible acquisition-to-output recipe and sector-aware model/output path | The same safety and adoption gates, plus resolution of remaining sector/product work on its own merits |
| CO2 only | Concrete coherent-reduction builder and public prepared-input materialise/build/sample seam | Acquisition/configuration/result/output contract, grouped reporting, units, diagnostics, real-run and scientist acceptance |
| CO2/O2 | Validated linked-channel handoff, concrete graph, units/provenance work, advanced prepared-input replay seam | The reserved full runner, recipe config, unit-safe reduction policy, tracer-aware outputs/diagnostics, Verification Games acceptance |
| Nested domain | Divergent scientific/reference implementations and a current delivery chain | Modern preparation and complementary support, readable graph/runner, native-domain outputs/config/docs, controlled and scientist acceptance |
| Experimental methane/ethane | Tested and documented Ramsden prepared-input model using the current sampler and shared components | Keep tests, documentation, graph construction, and sampling current as an experimental requirements fixture; no implied production acquisition/output promise |
| CO2/14CO2 | A dated scientific requirement; no canonical delivery item or concrete recipe yet | Freeze equations, units, state sharing, coupling and covariance policy, input availability, outputs, owner, and acceptance plan; deliver a named recipe by 28 February 2027 |
| CO2/14CO2/O2 | A likely extension of the radiocarbon use case | Preserve the channel/state/coupling seams now; scope and deliver a separate named recipe only when the concrete science and data topology are agreed |
| Multisector nested domain | A discussed composition requirement, not yet owned by the current nested chain | Preserve source labels per domain and prove a two-sector synthetic preparation/model/output fixture; require real-data acceptance only when a consumer owns it |

This matrix should be maintained in the canonical model-family plan and
tracker. It prevents `Done` from meaning one thing for a graph issue and
another for a production recipe.

## Enduring coding guidelines

These are programme constraints, not optional preferences. The linked
development documents contain the detailed review rules.

### Readable recipes and components

- Keep every top-level recipe procedural and readable from top to bottom.
- Keep concrete model construction contiguous and in scientific or
  mathematical order.
- Use normal typed functions with scientifically meaningful names and explicit
  arguments.
- Permit scientifically meaningful composite functions such as a baseline,
  likelihood, or source-resolved pollution term.
- Keep a scientific composite beside its recipe until two production recipes demonstrate
  identical equations and option meanings.
- Accept small runner, forwarding, resolver, and builder duplication when it
  keeps a scientific recipe locally understandable.
- Use dataclasses for recognizable scientific values or durable handoffs, not
  to shorten a signature or disguise a computational object.
- Prefer a moderately long keyword-only signature over an ambient context or
  `**kwargs` forwarding through the scientific pipeline.
- Do not introduce a semantic compiler, generic model IR, dependency-injection
  container, universal component hierarchy, stage registry, or generic
  pipeline.

The extraction rule is deliberately conservative for scientific components:
share one after two production recipes demonstrate the same equation, inputs,
option meanings, and side effects. Ordinary retrieval, serialization, xarray,
and numerical utilities may be shared whenever their small contract is stable.

### Configuration and defaults

- Normalize Python and external configuration once at the recipe boundary.
- Route both forms through the same typed resolver and record the resolved
  value, provenance, and compatibility translation.
- Group options by the scientific component which owns them; do not infer the
  external schema from runtime function signatures.
- Reject unknown, irrelevant, or unused settings before retrieval or sampling.
- Preserve recipe and channel sections where the same spelling has different
  meanings.
- For every scientific or numerical default record its units, scientific
  owner, omission behaviour, whether it is
  retained/explicit-required/legacy-only, and its provenance.
- Keep Python callables out of configuration and serialized specs. Persist
  only safe identity and JSON-compatible scientific metadata.

### Labelled data, units, and execution

- Treat xarray inputs as borrowed and potentially Dask-backed; never mutate
  them in place.
- Never hide copying, computation, persistence, loading, densification,
  serialization, or rechunking behind a property or framework layer.
- Retain labels until they have completed alignment and validation.
- At an independently sourced composition boundary, transpose to semantic
  order and use `xr.align(..., join="exact", copy=False)` unless a different
  join is explicit scientific policy.
- Use Pint once at the unit-owning preparation/composition boundary. Treat
  downstream unit strings as descriptive metadata, not repeated validation.
- Indexed dimension coordinates may normally be inspected eagerly. Do not
  coerce a scientific data payload before its named boundary.
- Select the exact inputs required by the recipe, then materialize related
  arrays and lazy coordinates together.
- Use `openghg_inversions.array_ops.to_dense` for dense chunk payloads while
  preserving outer Dask laziness.
- Cross to NumPy only at a named PyMC, eager-kernel, or serialization boundary.
- Treat labelled `fp_x_flux` and source-resolved `fp_x_flux_sectoral` as a
  reusable pre-basis scientific product. Do not rebuild them for every basis
  or allow an unbounded Dask graph to stand in for the CO2 cache.
- For realistic CO2-family runs, construct or lazily reopen the cache before
  basis projection. Keep species/channel, source, site/time, domain/native
  grid, units, filtering, coupling/reference-ratio status, and algorithm
  identity explicit.
- Record chunking, compression, location, checksums, and completion state as
  physical storage metadata, not as the scientific identity of the cache.
- Reject partial or scientifically incompatible caches with an explained miss;
  loading and structural validation must remain lazy and must not densify the
  payload.

### Validation and PyMC

- Validate at the first boundary which owns an assumption, then trust
  package-created intermediates.
- Prefer static typing, xarray alignment, Pint, `CoordRegistry`, and the
  numerical operation itself over repeated bespoke checks.
- Add a bespoke coordinate check only for an invariant which xarray and
  `CoordRegistry` cannot express, with a documented scientific failure and a
  focused test.
- Validate dynamic extension results when they re-enter package code.
- Build direct custom models with `registered_model()` and register labelled
  model data through ordinary components.
- Keep source, state, tracer, and domain meaning in labels and explicit
  mappings rather than array positions or backend-name suffixes alone.
- Fail before sampling when a requested output is incompatible with the
  selected recipe or custom model contract.

### Results, artifacts, and provenance

- Modern derived products must use all posterior chains. Each reduction must
  state whether it preserves `chain`/`draw`, stacks them into a sample axis, or
  summarizes them.
- Retain chain count, draws, divergences, R-hat, bulk ESS, tail ESS, and the
  approved validity decision in machine-readable form.
- Define a small scientific-role vocabulary beside the recipe's model
  contract. Roles describe quantities, dimensions, and units; they do not
  prescribe execution or storage.
- Keep reusable prepared caches, disposable working caches, model-bound replay
  bundles, traces, and final products distinct.
- Never serialize executable Python code.
- Write the useful recoverable trace before, or atomically with, optional large
  products so product failure does not destroy the run's recoverable result.
- Preserve recipe name, package version, scientific references, resolved
  configuration, data and basis provenance, role mapping, and all relevant
  channel/domain/source/state labels.

### Tests, change shape, and release notes

- Test scientific equations against independent oracles, not only package
  helpers.
- Test model variables, coordinates, priors, optional components, predictive
  inventory, roles, units, and unsupported combinations.
- Test strict and deliberately permissive alignment, input non-mutation, and
  absence of hidden lazy execution.
- Test direct-versus-cached `fp_x_flux`, basis-projected `H`, and downstream
  coherent-product equivalence; exact labels, units, and source order; cache
  invalidation; atomic/partial-write behaviour; and bounded graph, memory, and
  runtime on a representative CO2 workload.
- Run the focused experimental Ramsden tests when shared labelled-state,
  likelihood, sampling, or coordinate components change.
- Keep structural refactoring separate from scientific-default or equation
  changes where practical.
- Keep each pull request runnable and use focused pytest/Ruff checks while
  iterating. Preserve Python 3.10 support and run broader compatibility,
  type-check, and full-suite environments through the repository's reviewed
  Slurm tox path.
- Keep executable examples sourced from tested code.
- Add Towncrier fragments for user-visible changes; do not edit the published
  changelog directly during ordinary delivery.

## Programme workstreams

The work is easier to govern as seven durable streams. Streams can run in
parallel; production claims are controlled by the outcome gates below.

| Stream | Destination | Current principal work |
| --- | --- | --- |
| Scientific trust | All-chain products, diagnostics, approved warn/fail/unsupported policy, safe persistence | GitHub #637/#645/#656/#657; trace-first recovery slice of OPE-125 |
| Core workflow and adoption | Scientist-ready standard/multisector recipes, config/default clarity, tutorials, user acceptance | #587, #661/#663-665, OPE-49, OPE-108, tutorial-data project |
| Durable runs | Explicit model-data ownership, mandatory CO2 `fp_x_flux` preparation cache, compact recoverable output, prepared cache versus replay bundle | OPE-105/106 -> OPE-107 -> OPE-125; OPE-55/#415; OPE-91 and a new canonical cache owner |
| Shared scientific mechanics | Reusable labelled covariance, coherent reduction, mismatch models, roles, state/source selection | Shared-foundations project; only work with a named production consumer should gate a recipe |
| Model families | Complete CO2, CO2/O2, radiocarbon, and nested-domain products while retaining the experimental multigas fixture and a multisector nested seam | CO2 project; OPE-91 cutover; a new dated radiocarbon track; #666 and #407 -> #408 -> #409 plus a nested-multisector fixture |
| Scientific operations | Queryable Verification Games runs and artifacts; controlled WUR25/GEMMA evidence | Verification Games renewal, WUR25, GEMMA |
| Compatibility and release | Migration evidence, release-quality automation, version/DOI agreement, bounded retirement | #587 retirement gate, GitHub #351, later residual-removal issue |

Shared foundations are a capability stream, not a serial programme phase. A
foundation item should block only the recipe or campaign which consumes it.
This allows useful CO2, nested-domain, tutorial, and Verification Games work to
proceed without waiting for an abstract definition of “all foundations.”

## Outcome roadmap

### Gate A — truthful baseline

This is the short administrative gate implemented by the architecture
reconciliation. Its requirements are fully defined here; the reconciliation
is the current execution record, not a second source of architecture. Gate A
does not create or scientifically gate a package release and should not become
permanent programme overhead.

**Exit evidence**

- Canonical plans, GitHub, and Linear distinguish implementation merged from
  scientific/user acceptance complete.
- The P0 W4-W6 status, CO2 maturity, PR #659, merged PR #660, and current open
  work are reflected accurately.
- No live item treats the rejected compiler/ADR as a production dependency.
- Every active urgent/high item has one owner, priority, dependency path,
  bounded output, and acceptance evidence location.
- Missing Linear mirrors and parents for current safety and architecture work
  are created or explicitly waived.
- Every reported milestone contains all active work required for its scoped
  outcome; explicitly unscheduled backlog is excluded from its percentage.

### Gate B — trustworthy modern core

This gate blocks every new production-result claim and the next stabilisation
release.

**Exit evidence**

- Every public result route is classified as modern or compatibility-only,
  including `InferenceOutput`/`InversionOutput`, in-memory and serialized
  round trips, basic/PARIS/country/flux/concentration products,
  `output_format="legacy"`, and `--legacy-fixedbasis`.
- Deliberately discrepant two-chain fixtures affect every supported modern
  concentration, flux, country, basic, and PARIS product.
- Ordinary modern paths contain no implicit first-chain selection.
- Chain/draw preservation, stacking, or summarisation is explicit for every
  product.
- Results and persisted artifacts retain chain/draw counts, divergences,
  R-hat, bulk ESS, tail ESS, and a machine-readable validity decision.
- The diagnostics schema names the scientific latent variables being assessed
  and records one of approved valid, warning, invalid, or not-assessable
  states rather than treating a sampler-wide scalar as sufficient.
- One-chain diagnostics are “not assessable,” never silently passed.
- Converged, non-converged, divergent, and pathological fixtures exercise an
  approved warn/fail/unsupported policy before scientific products are
  written.
- A recoverable trace remains available if an optional large product fails.
- Legacy compatibility semantics are isolated and cannot set modern output
  policy.

### Gate C — scientist-ready standard and multisector core

Gate C may progress in parallel with Gate B, but cannot make a production
claim until Gate B passes.

**Exit evidence**

- Standard and multisector tutorials execute in a clean documented
  environment from acquisition through supported output.
- Direct Python and external configuration resolve equivalently; irrelevant
  settings fail before retrieval, and resolved defaults are preserved.
- The stated fresh external cookiecutter-package acceptance is executed, or
  the canonical W2b claim is narrowed explicitly to the evidence which exists.
- Named representative scientists can locate a prior and likelihood, make one
  change, run the workflow, interpret the output, and complete a recorded
  rubric.
- #587 and the P0 plan show only genuine remaining acceptance or retirement
  work.

### Gate D — durable and recoverable runs

This track can proceed alongside recipe delivery. A recipe does not need every
general replay feature to develop, but a large campaign cannot claim durable
reproducibility until the parts it relies on pass.

**D-core exit evidence — required by realistic CO2-family production and any
other consumer which creates large preparation graphs**

- Ownership is decided for reusable prepared data, model-owned assembly,
  `InferenceData.constant_data`, trace, replay bundle, and final product.
- High-volume inputs are not duplicated accidentally in ordinary results.
- Reference-run size, write time, and recovery targets are approved and met;
  benchmarks record the data, code, storage, and compression context.
- CO2 preparation precomputes, atomically persists, and lazily reloads labelled
  native-grid `fp_x_flux` or source-resolved `fp_x_flux_sectoral` before basis
  projection. A representative persisted-and-reloaded run demonstrates a
  bounded Dask graph, memory use, and runtime.
- Cache scientific identity covers species/channel, native domain and grid,
  site/release/time selection and averaging, footprint and flux identities,
  source order, units/conversions, pre-cache filtering, multiplication
  algorithm, coupling/reference-ratio status, and software/schema revisions.
- Identity also distinguishes footprint species and mode, domain group,
  tracer/channel, exact grid-coordinate and mask identity, and exact returned
  footprint/flux record revisions. It is used in the path or catalog key, not
  stored only as descriptive metadata.
- Cache provenance records the exact OpenGHG queries and returned identities,
  transformations, writer revision, and creating run. Partial, corrupt, or
  incompatible products cannot be cache hits, and cache matching explains the
  hit or miss.
- Data and checksummed manifest are published and read back as one atomic
  completion. Path existence alone is never a cache hit; the loader validates
  schema/version, exact dimensions and order, coordinates, labels, units,
  selector, and manifest-to-store identity before reuse.
- Reload is lazy; dimensions, indexes, units, and source order agree exactly;
  payloads, basis-projected `H`, and coherent downstream products agree with
  direct construction to an approved numerical tolerance.
- A compatible native `fp_x_flux` cache can be projected through more than one
  basis without mutation. A cached `H` is a different, basis-bound artifact.
- The basis-independent raw cache is not serialized into
  `RhimePreparedInputs`; that durable handoff checkpoints projected `H`, basis,
  operator, and reference-flux state. Reprojection through a new basis reopens
  the separately owned raw cache.
- The first supported CO2 path should reuse the public OpenGHG time-resolved
  operator and labelled atomic Zarr persistence delivered by OpenGHG PR #1703.
  Irregular-time cases remain gated on still-open #1708 and its upstream
  dependency rather than being assumed supported or reimplemented here.
- Ownership remains explicit: OpenGHG owns the reusable multiplication and raw
  persistence kernel; this package owns labelled projection through its xarray
  adapter and the projected prepared-input contract; campaign/data-preparation
  code owns discovery, catalog, and temporary orchestration. Do not move raw
  cache discovery or lifecycle into model construction.
- Reuse the sparse, source-native projection work already delivered in this
  package by #651/#654; do not add a cache-specific contraction path.
- Linked and radiocarbon channels retain distinct cache identities even when
  states are shared. Multisector preserves the source axis. Nested outer and
  inner products preserve separate native grids, masks, domain identities, and
  no-double-counting provenance.
- The same cache boundary is available to standard, multisector, nested, and
  multigas recipes when it reduces graph size, without forcing every small run
  to persist a cache or every recipe to depend on CO2 preparation.
- A useful trace is written before, or atomically with, optional product work
  so output failure does not erase the recoverable scientific result.

**D-replay exit evidence — required only for consumers which claim offline
model-bound replay**

- A versioned model-bound bundle can reproduce supported postprocessing
  without raw object-store access and reject an incompatible model spec before
  sampling or output writes.
- Cache reuse is decided by the requested scientific preparation and recorded
  cache identity, not merely by two model specs having similar shapes.
- Custom Python graphs have a truthful supported-replay contract or explicit
  opt-out; executable Python is never serialized.

### Gate E — production model families

Each named family passes an independent subgate. Implementation can proceed in
parallel; a production result ships only after Gate B, D-core where applicable,
and that recipe's own Gate E evidence pass. CO2-only does not wait for linked
O2, radiocarbon does not wait for either linked O2 or nested domain, and nested
domain does not wait for a generic composite representation.

**E1 — CO2-only exit evidence**

- CO2-only satisfies the production recipe contract above from its approved
  operational input boundary through durable output.
- Grouped inner/outer reporting, fixed/active state semantics, units, and
  output provenance are complete.
- Its mandatory `fp_x_flux` path passes D-core, including persisted/reloaded
  representative evidence and cache invalidation.
- Verification Games cuts over through OPE-91 with reproducible real-run
  evidence and no private replacement of generic OGI or OpenGHG mechanics.

**E2 — linked CO2/O2 exit evidence**

- CO2/O2 independently satisfies the production recipe contract.
- Linked-channel alignment and coherent reduction are unit-safe and preserve
  distinct observation axes.
- All-chain products, diagnostics, recipe configuration, and a scientist
  walkthrough are present.
- Reusable components are promoted only where at least two production recipes
  demonstrate the same meaning.

**E3 — CO2/14CO2 exit evidence, due no later than 28 February 2027**

- A canonical project, owner, scientific reviewer, delivery issue, and dated
  milestone are created immediately; GitHub #205 is background scope, not an
  executable six-month schedule.
- The recipe records the radiocarbon observation convention, units and
  conversions, fossil/background state sharing, coupling direction and
  reference values, covariance/error treatment, input availability, products,
  and provenance before its graph contract freezes.
- CO2 and 14CO2 keep their observation axes, cache identities, boundary/error
  policy, and scientific roles until the concrete recipe combines them.
- The named recipe, focused analytic/equation tests, controlled sampler case,
  all-chain products/diagnostics, runnable configuration, cache evidence, and
  representative scientist acceptance pass independently of optional O2.

**E4 — optional CO2/14CO2/O2 exit evidence**

- The concrete three-channel equations, state sharing, data topology, units,
  covariance, and outputs are agreed before implementation.
- It is a separate readable recipe if those meanings differ materially from
  E2 or E3. It is not implemented as a universal N-gas engine.

**E5 — nested-domain exit evidence**

- #407 proves one authoritative complementary mask, no double counting,
  explicit one-to-one time alignment, two labelled native basis artifacts,
  explicit priors, and a validated prepared handoff.
- #408 visibly implements the one-sector equation
  `H_outer @ x_outer + H_inner @ x_inner` before the ordinary baseline and
  likelihood, with equation tests and a controlled sampler smoke test.
- Preparation and output contracts retain source/sector labels separately for
  each domain. A two-sector synthetic fixture proves the composable equation
  `sum_sector(H_outer[sector] @ x_outer[sector] +
  H_inner[sector] @ x_inner[sector])` without requiring a generic graph or a
  real-data multisector release in the first nested milestone.
- #409 provides native-grid domain-aware all-chain outputs, diagnostics,
  config, documentation, provenance, and scientist acceptance.
- Evidence from PRs #359/#600 is reproducibly retained as an oracle or
  explicitly rejected as unsuitable.
- Ownership of any registered real-data regression case is decided with the
  scientists who use the model.
- CO2-nested fails specifically as unsupported while the required inner-domain
  data inventory is incomplete. Its readiness gate names footprint and flux
  products, owners/access, site/time overlap, footprint mode per source,
  grids/calendars/units, masks, provenance/licensing, and a reduced fixture.
  Missing inner data fails explicitly and is never zero-filled; structural
  composability is not presented as scientific or data readiness.

### Gate F — reproducible campaign adoption

Campaign projects retain their own scientific milestones. Gate F defines the
shared operational claim across them.

**Verification Games**

- A versioned run contract and one authoritative result/artifact catalog are
  accepted.
- One recent workflow can be launched, queried, retried, evaluated, and linked
  to a report without notebook state.
- The campaign workspace retains only campaign policy, orchestration, scoring,
  artifact registration, and narrative analysis after package parity.
- Old code, notebooks, tasks, and data are retired only after explicit parity,
  provenance, and retention gates.

**WUR25**

- Controlled full-mole-fraction closure includes the fixed/inferred mismatch
  dependencies which actually block the cutover.
- The no-baseline control, fitted baseline comparison, and covariance audit
  retain separate hypotheses and evidence.
- Seasonal/full matrices begin only after the preceding scientific gates pass;
  protected observations are not used to tune the covariance under test.

**GEMMA**

- Protocol, observation network, comparison outputs, and CO2 workflow
  dependencies are frozen before input production.
- Validated inputs precede the 2013-2025 and 2026 runs.
- Headline and paper-ready analysis is linked to immutable run, diagnostic,
  product, code, and data identities.

### Gate G — compatibility transition and retirement

Legacy removal is not required for the first safe modern release.

**Exit evidence**

- Real INI/SLURM users and required legacy products are named.
- Migration exercises record the revision, configuration, products,
  differences, user feedback, and decision.
- Modern routes never silently select legacy execution or legacy first-chain
  output semantics.
- A communicated deprecation window and release notes exist.
- A new bounded residual-removal issue names exactly which APIs, paths,
  products, tests, and documentation are removed; closed #416 is not revived.
- Removal ships in a later release after at least one safe modern release has
  provided an observed migration window.

## Dependency and release sequence

```text
Gate A governs truthful programme reporting.

Gate B        Gate C        Gate D-core/replay        Gate E1-E5        Gate F
trust         adoption      durable capabilities      named recipes     campaigns
   \             |                |                       /                /
    +------------+----------------+----------------------+----------------+
                         consumer-specific shipping gate

Gate C + required D evidence + observed migration -> Gate G later removal
```

Gates B-F may start and progress in parallel. A modern production-result claim
ships only when Gate B, the recipe's own Gate E subgate, its consumed D-core or
D-replay capabilities, the release-quality envelope, and any claimed campaign
acceptance have passed. Gate A is an administrative reporting prerequisite,
not a substitute for those scientific release gates.

Shared scientific components and the Verification Games run-system work may
advance in parallel. Their dependencies should point to concrete consumer
gates rather than imposing a total order on the programme.

The intended release cuts are readiness-based:

1. **Stabilisation release.** Gates B and C for standard and multisector,
   applicable D-core evidence, executable tutorials, and the release-quality
   envelope. Retain the explicit legacy route. Do not advertise incomplete
   CO2 or nested recipes as production.
2. **Model-family releases.** CO2, CO2/O2, CO2/14CO2, and nested domain may ship
   in separate minor releases as their own Gate E and consumer-specific Gate F
   evidence become ready; none is forced to wait for another family. CO2
   production always includes its D-core `fp_x_flux` evidence.
3. **Radiocarbon deadline.** A supported CO2/14CO2 recipe must ship no later
   than 28 February 2027. It may join the CO2 release or use its own minor cut,
   but cannot be blocked on optional O2 or nested-domain delivery.
4. **Retirement release.** Gate G passes after an observed deprecation window.
   A major-version boundary is a natural option, but numbering is a maintainer
   release decision.

Except for the radiocarbon commitment, these are outcome cuts rather than
calendar estimates. Other target dates should be added when an owner accepts
the scope and capacity.

## Review of the existing milestone system

### Repository roadmap findings

The repository plans contain useful acceptance detail, but their stage labels
no longer describe current implementation status:

| Roadmap surface | Current evidence | Required treatment |
| --- | --- | --- |
| P0 W0-W6 | W0, W1, W2.0, W2a, W2b's in-repository proof, W3a, W4, W4b, W5, and W6 have merged delivery | Mark implementation stages accurately. Keep external-consumer, safety, tutorial, and scientist acceptance as separate evidence gaps rather than reopening merged refactors |
| P0 W2b | The repository contains a package-shaped public-API test, while the written criterion calls for a fresh cookiecutter project, dependency installation, and external test/lint execution | Execute the stated external test or narrow the claim explicitly; do not call the two forms of evidence equivalent |
| P0 dependency diagram | It says W4-W6 follow W2b, but those implementation stages are already merged | Preserve as history if useful and replace it as the live schedule with the outcome dependencies in this document |
| P0 W5b/W7 | Replay, documentation, user review, and the retirement decision remain genuine work | Map replay to Gate D and user/retirement evidence to Gates C/G |
| Model-family expansion | The plan correctly permits families to proceed in parallel, but its tracker list does not distinguish merged CO2 graphs from incomplete products and does not yet cover the radiocarbon deadline, current Ramsden fixture, mandatory CO2 cache, or nested-multisector seam | Maintain the recipe maturity matrix, add these known constraints, and apply Gate E independently to each named family without introducing a generic multigas/composite representation |
| Architecture reconciliation | It is a dated issue-level migration plan | Use it to complete Gate A, then archive it rather than maintaining a third issue roadmap |
| Release engineering | No architecture plan currently owns the complete release-quality outcome | Apply the release-quality envelope below and track its automation under GitHub #351 |

Ownership, configuration, and persistence decisions should be recipe-local
dependencies. They may be developed in parallel, but a recipe cannot pass its
production gate until its own settings, outputs, provenance, and safety
contract are complete.

### Portfolio findings

As of the evidence date, Linear has eight non-cancelled RHIME-related projects
containing 104 issues. Only 23 issues are assigned to milestones; only 19 of 68
active issues are assigned. Seven projects have no recorded lead, and none of
the projects or milestones has a target date.

Consequently, the displayed percentages do not currently provide a reliable
programme view:

- Shared scientific foundations reports 100% because its only milestone member
  is completed OPE-121, while the other active foundation issues are outside
  the milestone.
- The two CO2 milestones report 0% even though OPE-74/75/77/119 and their core
  implementations are Done; completed work is not assigned to the milestone.
- Tutorial data is marked Backlog despite four Done items and two In Review.
- Nested domain is marked In Progress despite every issue being Todo and
  retains three empty “Moved” milestone tombstones.
- WUR25 has a useful scientific sequence, but OPE-114/115/116 block its active
  cutover while contributing to no milestone.
- P0 and GEMMA have no milestones at all.
- Important cross-project dependencies are soft `relatedTo` links or absent.
  In particular, GEMMA is not blocked explicitly on CO2/output readiness, and
  the tutorials do not roll up into the P0 scientist-adoption gate.
- No active Linear deliverable owns radiocarbon, CO2/14CO2/O2, nested
  multisector, or nested CO2. Existing multigas references intentionally reject
  a generic framework but do not provide the dated radiocarbon delivery path.
- `fp_x_flux` ownership is fragmented across the completed prepared-input seam
  (OPE-82), the VG cutover (OPE-91/OPE-99), replay work (OPE-55), and output
  sizing (OPE-125). No active issue owns the mandatory production CO2 cache
  gate or blocks CO2/GEMMA promotion on its evidence.

### Linear project recommendations

Use milestone names as evidence-bearing outcomes, not issue buckets. Each
project should instantiate only the stages it needs from this small vocabulary:

1. **Contract and prerequisites** — scope, scientific protocol, inputs,
   dependencies, acceptance design;
2. **Implementation and integration** — code, configuration, package and data
   integration;
3. **Controlled verification** — equation/regression/safety tests and
   controlled experiments;
4. **Real-data or scientist acceptance** — representative run and review;
5. **Adoption and reproducibility** — supported operation, documentation,
   queryable artifacts, handoff;
6. **Retirement and closure** — deprecation, parity, retention, and final
   closure.

| Linear project | Recommended outcome milestones |
| --- | --- |
| P0 `run_rhime` | Trustworthy modern results; scientist-ready standard/multisector; durable output/replay where owned; compatibility decision |
| Shared scientific foundations | Reusable scientific mechanics verified in named production consumers; persistence/role work only where a current consumer owns acceptance |
| CO2 model family | Core model seam (record delivered); mandatory production `fp_x_flux` cache; CO2-only recipe verification; linked CO2/O2 verification; real VG/scientist acceptance and promotion |
| Radiocarbon model family | Contract and data readiness; CO2/14CO2 implementation and controlled verification; real-data/scientist acceptance by 28 February 2027; optional CO2/14CO2/O2 scope as a separate later outcome |
| Nested-domain family | Data/provenance readiness; contract/preparation; graph/runner; native outputs and scientist acceptance; two-sector structural fixture. Remove the three empty moved milestones after preserving destination links |
| Tutorial data | Versioned portable bundle; tutorial integration and clean-environment acceptance. Set the project to In Progress while review is active |
| Verification Games renewal | Keep the current future-run, thin-workspace, and retirement sequence; add OGI production gates as blockers |
| WUR25 | Keep controlled closure, matched comparison, and covariance milestones; assign OPE-114/115/116 to controlled closure |
| GEMMA | Protocol/network/readiness; validated inputs; historical and drought runs; paper-ready analysis, with CO2 cache/output and any consumed radiocarbon blockers explicit |

The P0 project needs either a broader name which truthfully includes safety and
durability, or only its readability/adoption work should remain there while a
current project owns Gates B and D. Do not leave the work split between
unparented GitHub issues and unmilestoned Linear backlog.

Create three bounded canonical delivery items during Gate A:

1. an urgent CO2-family issue for production `fp_x_flux` precomputation,
   atomic persistence, lazy reload, invalidation, numerical parity, and
   production-shaped performance evidence; it blocks CO2 promotion, OPE-91
   final cutover, and consuming GEMMA runs without waiting for full replay;
2. a dated radiocarbon parent with a scientific owner and separate children
   for contract/data readiness, recipe implementation, controlled
   verification, and real-data/scientist acceptance; and
3. a nested data-readiness child plus a bounded two-sector synthetic pressure
   test. Treat nested CO2 as unsupported/data-blocked until named footprints
   and a consumer exist.

The existing experimental Ramsden model needs a small explicit maintenance
owner or checklist under shared-component changes, not a production project.

### GitHub milestone recommendations

The repository's GitHub milestones should not serve as a second schedule:

| Existing GitHub milestone | Current signal | Recommendation |
| --- | --- | --- |
| #4-#7: data input, docs, data output, errors/testing | Old category buckets with open items and no outcome dates | Triage issues into current Linear projects/backlog, then close the category milestones |
| #8: 0.4 release | Past due, six closed and one open item | Rehome or close the remaining item, then close the historical release milestone |
| #9: Flexible inversions M1 | Ten closed and two open; implementation sequence largely historical | Move remaining documentation/PARIS work to current outcomes, record M1 as historical |
| #10: high-resolution RHIME | Five open items; current nested-domain chain is valid | Retain as the public nested-domain outcome, but do not count parent #666 and its child issues as independent delivery progress |
| #11: linked CO2/O2 | Three open items while a core linked recipe has landed through different work | Re-scope to the genuinely remaining ratio/config/output contract or close in favour of the Linear CO2 production milestones |
| #12: reconstruction/run bundles | Two open items and still current | Map explicitly to Gate D and Linear OPE-55/OPE-21 |
| #13: legacy cleanup | Marked complete because #416 closed, while the live retirement gate is unmet | Keep as historical; track the future bounded removal under Gate G rather than reopening it |

The old GitHub M1-M5 numbering no longer describes delivery order: CO2 work
landed before high-resolution RHIME, and “legacy cleanup complete” does not mean
legacy retirement is accepted. GitHub milestones may remain useful public
implementation groupings, but Linear and this outcome roadmap own schedule and
dependencies.

GitHub #205 records broad tracer/isotope ambition, while #624 and #634 record
concrete coordinate and performance pressure from composable/multigas models.
They are useful engineering evidence but cannot stand in for the Linear
radiocarbon commitment, the CO2 cache gate, or the nested-multisector fixture.

## Milestone entry, exit, and governance

A milestone may start only when:

- the project lead and an assigned gate/exit-review issue identify accountable
  owners and reviewers;
- required issues are attached, bounded, prioritized, and assigned;
- outcome, evidence location, dependencies, risks, scientific/data access, and
  predecessor gates are recorded;
- real blockers use `blockedBy`, not only `relatedTo`; and
- a target date or next formal review date exists.

A milestone may close only when:

- all required issues are Done or explicitly transferred to a named later
  milestone;
- code review and any required scientific/user review are accepted;
- test results, immutable run identities, diagnostics, products, provenance,
  and comparison evidence are linked;
- downstream owners acknowledge cross-project handoffs;
- documentation, release note, tutorial/runbook, and retirement obligations
  for the milestone are complete; and
- the completion note records remaining limitations and follow-ups.

Portfolio rules:

- Assign each active delivery leaf issue to one project milestone. Leave
  explicitly unscheduled backlog outside milestone progress. Track umbrella
  parents for scope, but do not count parent and children as independent
  progress.
- Represent cross-project constraints with blocker edges.
- Do not publish percentages until membership is complete and estimation is
  consistent; publish state counts, critical path, and evidence status in the
  interim.
- Give every project a lead. Because Linear milestones have no owner field,
  use an assigned gate/exit-review issue for milestone accountability.
- Require at least one Started or In Review issue before marking a project In
  Progress.
- Reconcile project status, milestone membership, dependencies, and dated
  evidence weekly while a project is active.
- Keep cancelled compiler projects as decision history, but remove empty
  migration tombstones from active reporting.

## Deliberate deferrals and present obligations

The programme still defers:

- a universal multigas schema, channel registry, or generic N-channel
  likelihood engine;
- a generic nested/composite graph representation;
- automatic recipe routing, a universal runner, a production backend
  protocol, or one configuration engine inferred across all recipes;
- one mandatory cache storage class or format for every workload; and
- a CO2-nested production claim until complete inner/outer footprint and flux
  readiness is evidenced and a scientific consumer exists.

It does **not** defer:

- the labelled channel/state/coupling/covariance invariants needed by current
  CO2/O2 and experimental Ramsden work;
- keeping the Ramsden model's focused tests, documentation, graph, and sampler
  seam current when shared components change;
- the named CO2/14CO2 recipe and its 28 February 2027 deadline;
- mandatory precomputation and persistence of `fp_x_flux` for production CO2
  preparation, with a reusable cache contract and bounded Dask behaviour; or
- source-preserving nested preparation and a two-sector structural fixture.

This boundary deliberately buys concrete extension capacity without making a
generic framework a programme deliverable.

## Release-quality envelope

Architecture completion is not enough to publish a release. GitHub #351
already identifies gaps in the release process; the package remains versioned
`0.6.0` on `devel`, `v0.6.0` is the latest tag, the changelog has a large
Unreleased section, and unassembled Towncrier fragments remain. Release
quality therefore needs explicit programme ownership.

Before every releasable cut:

- advertised recipes have passed the trustworthy-core and applicable
  production-recipe gates;
- focused, full/slow, supported Python (including 3.10), supported OpenGHG
  compatibility, type, and lint checks pass at an immutable revision;
- relevant reviewed/registered real inversion cases pass with documented
  scientific tolerances;
- sdist and wheel build, install cleanly, and pass import and CLI smoke tests;
- documentation and pinned executable tutorials build and run;
- output failure/recovery and rollback procedures are exercised;
- Towncrier fragments are assembled before publishing the GitHub release, and
  no unassembled fragments remain;
- package version, tag, changelog, GitHub release, PyPI artifact, migration
  guidance, and DOI agree; and
- deprecations and removals name their user-visible effects and migration
  window.

## Programme definition of done

The programme is complete when:

1. The default standard and multisector recipes narrate their whole scientific
   workflows and pass the production contract.
2. CO2, linked CO2/O2, CO2/14CO2, and nested-domain recipes each pass their
   independent contract before being advertised as production; radiocarbon is
   delivered by 28 February 2027 without waiting for optional O2.
3. A scientist can locate, change, run, interpret, and cite an ordinary
   component without learning a framework.
4. Configuration, scientific defaults, validation, package ownership, and
   compatibility exceptions have one explicit owner each.
5. Prepared labelled inputs remain backend-neutral until named execution
   boundaries, with no hidden mutation or computation; production CO2 uses a
   persisted, lazily reloaded, identity-checked `fp_x_flux` cache before basis
   projection.
6. All modern products use explicit all-chain semantics and automatically
   retain diagnostics and validity decisions.
7. Unsupported results fail specifically before expensive or irreversible
   product work; useful traces remain recoverable.
8. Native `fp_x_flux` caches, basis-bound `H`, prepared handoffs, versioned
   model-bound replay bundles, traces, and final products are distinct and can
   reproduce their supported claims truthfully.
9. Verification Games and scientific campaigns retain queryable run,
   diagnostic, artifact, and report identities while reusing released package
   mechanics.
10. Every active project and milestone reports truthful membership, ownership,
    dependencies, and acceptance evidence.
11. Every release passes the release-quality envelope and has consistent
    version, migration, package, and DOI records.
12. Legacy removal occurs only after observed user migration and a communicated
    deprecation window.
13. Shared-component changes keep the experimental Ramsden fixture working,
    and a two-sector nested fixture proves that channel, sector, state, and
    domain remain orthogonal without a generic model representation.

No generic compiler, registry, universal component protocol, pipeline
framework, package-wide validation layer, or speculative package reorganization
is needed to complete this programme.
