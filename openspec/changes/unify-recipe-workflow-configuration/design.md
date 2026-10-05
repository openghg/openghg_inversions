# Design

> Draft for review. Names describe proposed boundaries, not implemented APIs.
> See [proposal.md](proposal.md) for scope and
> [the spec](specs/recipe-workflow-contract/spec.md) for behavioral requirements.

## Context

After PRs #772, #784, and #778, standard/multisector full runners, the prepared
runner, and `_standard_stages.py` still independently compose scientific phases.
CO2 staged sampling already reuses ordinary/cached prepared runners. Configuration
is split between `RhimeRunnerSetup`, `Co2RunSetup`, and a further `Co2StageSetup`
wrapper; the facade repeats selection for individual operations. Moving these
owners into `recipes` alone would preserve the duplication.

Follow [RHIME development guidance](../../../docs/development/rhime_model_development.rst):
ordinary callable components, explicit forwarding, readable procedural runners,
and incremental scientific delivery. Nested currently has direct full/prepared
routes without the generic stage suite and need not acquire it in this change.
The [architecture principles proposed in PR #793](https://github.com/openghg/openghg_inversions/blob/92be84ec2e697f2a7c2dbf8a6f0266b9965c0cb7/docs/development/architecture_principles.rst)
inform the roles below: responsibility follows reasons to change, interfaces
serve actual consumers, and handoffs state scientific meaning and ownership.
They do not prescribe a class or module for every responsibility.

## Goals / Non-Goals

**Goals:** Make shared scientific execution the foundation. Keep route wrappers
short, full runners readable, and checkpoint persistence optional. Normalize
configuration and stage calls only where they support existing workflows.

**Non-Goals:** A universal pipeline, builder, prepared-input type, inference mode
union, capability registry, mutable lifecycle, or nullable record for every
possible recipe. No new optimizer, scientific recipe, CLI command, diagnostic
policy, configuration syntax, or numerical prepared-input/posterior codec migration.
Staged metadata compatibility may change under the explicit reset below.
Import isolation and a possible `_model_building.py` split remain separate
investigations. Graph-free replay does not require replay without importing PyMC.

## Decisions

### 1. Recipe operations are canonical; route wrappers sequence them

The scientific owner provides ordinary operations for the phases its recipe
supports. For standard/multisector, one scientific preparation operation visibly
orders filtering, retained-site reconciliation, basis construction, sensitivities,
and labelled assembly. Acquisition remains independently callable for supplied
merged data.
Filtering is part of preparation, rather than a separate high-level workflow
step; existing ordinary helpers can remain useful inside that procedure.
Construction owns input requirements, coordinated materialization, the concrete
model, and its output contract. Sampling retains the recipe's matched policy;
cached CO2 keeps its graph, CompoundStep, conditional predictions, and trace
annotation together. Results/products have one family constructor, accepting
live construction information or the authenticated saved output contract.

Choose operation boundaries around coherent scientific decisions and handoff
contracts. The phase diagram does not prescribe a module or class per phase;
one owner may provide several ordinary operations. Keep mathematically coupled
decisions together and preserve visible scientific composition.

```text
inputs/acquisition --> scientific preparation --> model construction --> sampling
         |                       |                                         |
         v                       v                                         v
   merged checkpoint      prepared checkpoint                       results/products
      (optional)                                                           ^
                                                                           |
                                                              authenticated replay
```

The first three operations correspond to inputs/acquisition, scientific
preparation, and model construction in the six-layer responsibility account.
The checkpoint arrows denote persistence at handoffs; full execution uses the
same values in memory without requiring either checkpoint write.

Full runners visibly call these operations with in-memory handoffs. Prepared
routes join at construction and authenticated saved-output routes at products,
without historical graph-based role recovery. Final product writes and
explicitly enabled cache/basis saves remain supported. Calling the same low-level
functions from separate scientific orchestration procedures is insufficient;
short entry-point sequences forwarding to canonical operations are sufficient.
An opaque executor would obscure scientific order without removing more policy.

Preserve established direct Python signatures and returns, including CO2's trace
return. Likelihood callables, explicit options/conflict rejection, and prepared
complete-model builders retain their supported contracts. Built-in construction
materializes related inputs together; a custom complete-model builder retains
ownership of potentially lazy canonical inputs. No callable serialization or
CLI registration is needed. Numerical components receive resolved values
explicitly rather than an ambient configuration object.

### 2. Stage ownership and configuration are scoped to adopters

Initially standard, multisector, and ordinary/cached CO2 adopt the common staged
calling contract. Select the concrete workflow once at the CLI boundary; its
named `prepare`, `prior_predictive`, `sample`, and `postprocess` operations accept
explicit artifact paths and invocation options, without a repeated model
selector or heterogeneous setup union. A small structural interface describes
these actual callers; no inheritance, registration, generic execution loop, or
stage history is required. Diagnosis remains separately callable from posterior,
optional envelope, and convergence options.

Standard and multisector have separate concrete stage owners associated with
their scientific recipes, replacing `_standard_stages.py` as a combined workflow
owner. Share homogeneous preparation algorithms, inference/output mechanics,
and artifact/authentication/report helpers. Do not share a complete scientific
workflow controlled by a standard/multisector switch. Small wrapper duplication
is acceptable when it keeps each recipe understandable. Implementations import
shared mechanics from their owners, never from the dispatcher.

Adopting workflows use one immutable resolved record containing recipe choice,
one authoritative set of sampler choices, one output policy, and typed recipe
options. One family resolver serves its configuration-resolving routes. Existing
source loading and path/alias/override rules precede resolution.

Use consistent names for consistent roles. The following names are design
choices for review; they do not require a new class for every row.

| Role | Proposed name or existing representation | Responsibility and boundary |
| --- | --- | --- |
| Resolved recipe configuration | `RecipeConfig` | Declarative recipe selection and resolved acquisition/scientific, sampler, and output choices for adopting routes; no live model, sampler, or numerical checkpoint payload |
| Recipe-specific choices | `StandardOptions`, `MultisectorOptions`, `Co2Options` | Options whose meaning belongs to that recipe; forward the values needed by each scientific operation explicitly |
| Sampling choices | `SamplerOptions` | Immutable settings, including nested keyword choices; no live step, mutable cache, or backend execution state |
| Product policy | Existing `RhimeOutputSpec` where suitable | One authoritative choice of products and reporting/writing options; derived result metadata must not duplicate that authority |
| Numerical handoff | Existing concrete merged/prepared input types | Phase-complete scientific values, labels, units, provenance, and explicit borrowing/materialization contracts; not a configuration record |
| Invocation options | Explicit operation arguments | Per-call artifact paths, destination, and overrides; no independent scientific resolution or stage-only setup wrapper |
| Runtime state | Existing build results, model, `RhimeSampler`, matched steps, and caches | Constructed or adapted for the invocation from resolved choices; build results associate the live graph with its scientific output meaning, and coupled cached graph/inference state stays with its recipe |
| Durable output contract | Versioned scientific roles and authenticated identities | Information needed for replay after the live graph is gone; encoded by its artifact owner, not by serializing runtime dataclasses |

`Config` denotes resolved orchestration choices, `Options` a coherent set of
choices, and existing `Inputs` types the numerical handoffs. Keep
family names where needed to disambiguate options; do not rename established
public scientific types merely for cosmetic uniformity. Share a record when
its meanings agree, not because its fields happen to look similar. In particular,
do not create generic input/result containers or one options record per phase.

Remove CO2's stage-only normalization wrapper rather than nesting the old setup
inside the new options. Likewise, standard's derived result run specification
must not retain a second authoritative output policy. Prepared inputs own actual
retained labels; derive aligned invocation values without mutating requested
choices retained for provenance. Merely renaming the old setups does not meet
these ownership requirements.

Runtime sampler, step, and cache objects are constructed or adapted for each
invocation. Overrides must not mutate resolved choices or affect subsequent
invocations, including through nested keyword dictionaries. Existing direct
Python sampler interfaces remain supported outside this resolved configuration
contract. Do not deep-copy borrowed numerical arrays, hide materialization, or
introduce a sampler framework to enforce configuration immutability.

An all-purpose record would force unrelated recipes into fields they do not need. Independent
builders/direct runners continue to accept explicit scientific inputs without
this record; future staged adoption can implement only the supported operations.
No dummy methods or sampler/output settings are required to deliver a model.
The broad rule still holds: equivalent supported routes reuse their recipe's
scientific operations, whether or not they adopt this staged interface.

### 3. Keep merged and prepared checkpoints around scientific preparation

| Handoff | Remaining scientific work |
| --- | --- |
| Acquired, external, or reloaded merged data | The complete scientific preparation operation, then construction/inference/products |
| Fully prepared inputs | Construction, sampling, and products |
| Prepared inputs plus authenticated posterior/output information | Products without acquisition, model-input materialization, or graph construction |

Retain the existing optional `save_merged_data` / `reload_merged_data` mechanism
illustrated by `fixedbasisMCMC`. Its acquisition output precedes the recipe's
configured observation filters, basis work, and sensitivity construction;
acquisition may already have performed averaging and other processing. Full
and staged execution use the same cache operation and existing formats/options,
with explicit stage output containment where applicable. They then call the
same complete preparation operation. Saving a merged cache is opt-in; fully
prepared inputs remain the staged inference handoff.

**Breaking change for the next minor release:** Remove the newer staged filtered
snapshot `merged-data/merged-data.nc` and its `merged_data` path/digest entries
from new preparation manifests, including when optional acquisition caching is
enabled. That cache retains its existing persistence contract instead of
repurposing the retired manifest entries. Preparation no longer unconditionally
writes the filtered artifact. Do not replace it with an unconditional pre-filter
write or silently give old filtered files the pre-preparation cache meaning. Filtered
merged data is an internal handoff; no filtered checkpoint, public filtering
stage, resume phase argument, or phase-detection machinery is introduced.

Replay of supported artifacts authenticates prepared inputs, posterior, output
bindings, and optional affine artifacts without requiring any merged snapshot.
Pre-refactor staged manifests need not remain usable under the compatibility
reset below. Numerical prepared-input/posterior formats and the older optional
acquisition cache retain their separate contracts. CO2 staging already uses
coherent prepared inputs and affine companions without a merged snapshot, so it
acquires no new checkpoint requirement. Validate external handoffs at their owner;
trust locally constructed intermediates.

### 4. Retained-site policy follows the full runners in three phases

Current full preparation permits a valid subset from acquisition, drops sites
absent from a compatible merged cache, and removes sites emptied by filtering.
The staged rejection is broader than a filtering-only difference:
`tests/test_staged_workflow.py::test_prepare_fails_when_a_requested_site_was_dropped`
returns TAC alone from retrieval before filtering. Reconcile all three phases
with full preparation.
For each retained set, select every corresponding per-site option using the same
indices, retaining requested configuration separately. Empty sets fail before
basis/inference. Invalid returned labels/metadata, incompatible caches, and
malformed prepared artifacts retain their existing validation or cache fallback
policy; subset reconciliation does not make invalid data valid.

Replace accidental staged rejection coverage with parity cases for acquisition,
cache reload, and filtering, using unequal per-site options and empty-set cases.
This is an explicit acceptance correction in implementation documentation/release
notes, without a new strictness setting. Inventory other route differences,
including the prepared aggregation-error/output gate; preserve supported boundary
policies with an explicit rationale and regression check rather than broadening
scientific output capability during consolidation.

### 5. Readiness reporting preserves family exception boundaries

| Event | Standard / multisector | Ordinary / cached CO2 |
| --- | --- | --- |
| Returned finite predictive evidence | Existing readiness assessment | Existing readiness assessment |
| Returned empty/non-finite evidence | Existing `fail` check | Existing `fail` check |
| Existing caught build/predictive `KeyError` or `ValueError` | Existing `fail` check, no predictive artifacts | Propagate error; no readiness check or new destination |
| External input loading/authentication outside readiness catch, or artifact serialization error | Propagate error | Propagate error |

Keep the current catch scopes: standard/multisector authenticate outside the
readiness catch and serialize outside it; CO2 builds/predicts before creating its
destination. Returned invalid evidence still saves existing predictive/report
artifacts, including CO2's prior manifest. Shared mechanics must not convert
execution errors into universal readiness failures. Strict-mode exit policy
applies to a returned check; no new common exception type or diagnostic redesign
is needed. Convergence remains
independently owned, with the spec's existing threshold and unknown semantics.

### 6. Keep identity, authentication, and replay ownership explicit

**Breaking boundary for the next minor release:** The staged workflow has no
known consumers, so this refactor need not preserve its existing setup APIs,
manifest/binding contracts, or scientific identity hashes. Retire the historical
standard/multisector version-1 graph-recovery path. No old callable-name aliases,
pre-refactor replay fixtures, or artifact migration implementation are required.
Preserve established scientific Python APIs, numerical artifact formats, and
product contracts; this is not a numerical schema migration.

Identity remains a deliberate family projection of resolved scientific choices,
not `asdict` of incidental runtime records. Keep requested preparation and
retained-run sample/replay projections distinct, retain existing sampler/output/
transport exclusions, and report recorded sampling provenance. Hashes may change
at this boundary; do not build a callable registry to preserve old module names.
Within a supported identity contract, representation changes alone must not
silently redefine scientific identity. Stage destinations do not re-resolve science.

Keep schema versions explicit. Each family declares the schema and identity
contracts it supports; writers identify their contract and readers select that
contract before loading posterior data or creating product destinations.
Incompatible staged metadata changes require an identifiable new contract, not
silent reuse of a schema version with altered meaning. A family may support
selected older versions going forward through ordinary version-specific readers
and identity projections at the artifact/authentication boundary. Those readers
must retain strict content/binding checks and graph-free replay. No generic
migration framework or speculative older-version implementation is required now.

Being accepted by a shared loader does not make a version supported by a family.
Unsupported versions, malformed bindings, and attempts to downgrade binding
authentication fail before posterior loading or product writes, without graph
fallback. Supported standard/multisector contracts require bound saved output
information; ordinary/cached CO2 requires graph-free replay and independently
authenticated affine companions. Existing version numbers need not be preserved
as acceptance requirements across this breaking boundary.

Shared artifact/path/digest and manifest/binding mechanics keep independent
owners. Recipes own scientific compatibility and supported versions; version
readers decode recorded contracts rather than duplicate scientific preparation,
construction, or products. Keep provenance collection independently shared and
the revision requirement local to CO2. Shared metadata codec work in #785 is separate.

### 7. Design checks for independent scientific delivery

These examples test the architecture; they are not OPE-207 features or acceptance
scenarios. No MAP implementation, new recipe, task, command, or output format is
required by this change.

A future MAP experiment can call an existing sampler-free model builder (such as
`build_rhime_co2`) with prepared inputs and explicit scientific choices, run a
local optimizer, and construct its own result. It need not invent sampler options,
`InferenceData`, checkpoint methods, or acquire the common resolved record.
That experiment must decide its scientific target/parameterization, eligible
models, optimizer, result, and any persistence contract locally; this design does
not promise every ordinary/cached graph is a suitable MAP target.

Similarly, a future prepared-only recipe can ship a builder and a direct runner
that visibly calls construction and its supported inference/result operations.
It needs no acquisition path or staged suite. If full or staged routes are later
added, they reuse those scientific operations rather than implement another
workflow. Shared infrastructure follows an actual supported route.

## Risks / Trade-offs

- **Uniform configuration hides duplicate science** -> Trace consumers of each
  operation and require real cross-route parity, not just call-order mocks.
- **Checkpoint consolidation repeats work or accepts invalid inputs** -> Test
  phase-specific resume and retained metadata alongside boundary rejection.
- **Consolidation changes errors, capabilities, or saved identities** -> Preserve
  characterized family exception scopes and output gates; document the staged
  compatibility reset and test supported contracts' strict authentication.
- **Namespace migration recreates old owners** -> Move the canonical operations
  and wrappers together, with established scientific API compatibility imports
  forwarding to them.

## Migration Plan

1. Agree this planning PR; OPE-207 remains open. Create implementation tasks only
   after review, linking acceptance requirements to their validation evidence.
2. Characterize supported routes, options, identities, checkpoint phases, and the
   family readiness boundaries above. Include a short route -> canonical
   operation -> owner table as review evidence, not an executable manifest.
   Review each configuration/handoff role against a concrete reason to change.
   Reconcile actual devel owners with #773–#776: acquisition, inference, model
   components, and recipe composition.
3. Consolidate each existing recipe's scientific operations and migrate its
   consumers together. Establish separate standard/multisector stage owners and
   scoped configuration/calling contracts; remove obsolete parallel owners.
   During implementation review, trace where a retained-site policy change would
   be made and how full and staged routes receive it. Then trace a checkpoint
   encoding change through its persistence/authentication owners. Updating callers,
   documentation, and tests is normal; duplicating policy across callers is not.
4. Validate parity with real controlled preparation, basis/sensitivities,
   source-resolved multisector data, and retained subsets in all three phases.
   Compare selected model inputs, roles, deterministic terms/log probability at
   controlled parameters, and matched sampling policy; add small real-sampling
   checks for existing ordinary/cached variants. Compare products from one
   posterior, including units, dimensions, and conditional reconstruction.
   Supplement parity with existing independent reference calculations or
   pre-refactor expected values for representative deterministic quantities,
   reusing tests and fixtures where possible. Expected values establish regression
   stability; independent calculations provide correctness evidence. Validate the
   retained-site correction against its agreed policy. Check numerical correctness
   and ownership/execution properties separately; route agreement alone can retain
   a shared error.
5. Verify full execution with checkpoint serializers forbidden, prepared
   execution with preparation forbidden, and graph-free replay with construction/
   sampling forbidden. Check invocation overrides leave original and subsequent
   sampler choices unchanged. Validate supported schema dispatch and rejection of
   unsupported or malformed contracts. Run workflow, configuration, identity,
   saved-output/replay tests, changed-path Ruff, and whitespace checks. Review ordinary docstrings
   for extracted public operations: the decision owned, required input state and
   phase meaning, validation ownership, borrowing/materialization effects, returns,
   and failures. Verify optional merged caching precedes preparation, default
   staged preparation omits the retired snapshot/manifest entries, and supported
   replay does not require that snapshot. Update API/user documentation and add
   a Towncrier removal fragment announcing the checkpoint removal and staged
   compatibility reset for the next minor release, alongside the retained-site note.
6. Close OPE-207 after implementation and validation, then complete OPE-165's
   linked integration. Linked remains one joint CO2/O2 recipe, with channel axes,
   covariance, and unit policy local to that scientific feature; #779 has already
   merged its persistence support. The unified `run` CLI remains a separate follow-up.
7. After implementation, sync durable requirements into main specs and archive
   the complete change with its design, tasks, and validation evidence. PR-specific
   migration detail stays in that history; explicit supported-version policy
   remains part of the durable contract. No sync or archive happens in this planning PR.

Land a family's migration only when validated, rather than retain competing
permanent workflows. Established configuration, numerical handoff, and product
contracts remain stable; replay of pre-refactor staged envelopes is outside the
rollback guarantee. Announce the compatibility boundary explicitly.
