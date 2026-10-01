# Design

> Status: Draft for review. Names below describe the proposed contracts, not
> implemented APIs. This document, the proposal, and the scenarios can be
> revised together in the spec PR. No implementation tasks are captured yet.

## Context

See [proposal.md](proposal.md) for motivation and
[the behavioral contract](specs/recipe-workflow-contract/spec.md) for acceptance.
The baseline is devel after PRs #772, #784, and #778. Full standard/multisector
runners, the prepared-input runner, and `_standard_stages.py` independently
compose preparation or the construction/sampling/output tail. They call the
same lower-level scientific functions but own separate forwarding and boundary
decisions. Full runners reconcile their run specification with retained sites;
staged preparation instead rejects a missing requested site after filtering or
assembly. Existing tests encode both behaviors. CO2 staged sampling already
reuses its ordinary/cached prepared runners and should retain that reuse.

Configuration also differs. Standard/multisector resolution creates
`RhimeRunnerSetup(run_spec, sampler, data_args)`, with output
policy nested in `run_spec`. CO2 resolution creates
`Co2RunSetup(preparation_kwargs, runner, runner_kwargs, sampler)`; staging then
separately resolves outputs, reconstruction, and source reporting and wraps it
in `Co2StageSetup`. Eight facade operations repeat family selection and casts.

The current #773–#776 stack relocates the earlier owners, retaining those
configuration shapes. Its recipe namespace is a suitable eventual owner, but
the stack needs reconciliation with the already merged staged extraction and
CO2 workflow. Namespace migration alone does not settle this design.

## Goals / Non-Goals

**Goals:** Give each recipe one execution implementation reused across full,
prepared-input, and staged routes. Keep the full runner visibly procedural,
checkpoint I/O optional, and standard/multisector stage owners separate.
Configuration and the calling interface support this scientific ownership.
Prove parity and give linked integration the same concrete extension boundary.

**Non-Goals:** A universal builder across different scientific recipes, universal
prepared-input type, multigas configuration schema, workflow registry, automatic
discovery, execution graph, dependency injection, or mutable pipeline lifecycle.
Do not change the
scientific runners' established numerical/return contracts or absorb the
separate unified `run` CLI and diagnostic-policy work.

## Decisions

### 1. Canonical in-memory operations belong to each recipe

Each recipe owns ordinary callable operations for the scientific phases it
executes. These are the canonical implementation, not adapters to independent
historical full/prepared/staged pipelines:

- **Preparation:** Own the recipe's applicable phases in one readable procedure.
  For standard/multisector these are filtering, retained-site handling, basis
  construction, sensitivities, and labelled assembly. Acquisition
  remains separately callable, so supplied merged data bypasses acquisition and
  cache I/O. Existing lower-level preparation functions remain useful directly.
- **Construction:** Own the selected input requirements, coordinated
  materialization, concrete graph, and output contract at one boundary. Do not
  repeat their selection inside the prepared runner or a staged build helper.
- **Sampling:** Reuse ordinary sampling and the recipe's matched sampling
  procedure. Cached CO2 retains its graph/step ordering, conditional predictions,
  and trace annotation together in its existing scientific owner.
- **Results/products:** Reuse the same family result and product constructors
  whether the caller has a live build or an authenticated saved output contract.
  Stage-specific filenames, manifest writing, and reporting surround those
  constructors; they do not assemble a second scientific output recipe.

Each full runner explicitly calls its applicable preparation, construction,
sampling, and result/product operations in scientific order using in-memory
handoffs. The preparation procedure remains nearby and visibly orders filtering, basis,
sensitivities, and assembly. Prepared execution joins at construction; saved
posterior execution joins at result/product construction under the replay rules
below. Checkpoint serializers must not be required by ordinary full execution.
Requested final output writes and explicitly enabled cache/basis saves remain
supported.

```text
Acquisition or merged input
             |
             v
    Recipe preparation --> Prepared inputs --> Construction --> Sampling
                                 ^                                |
                                 |                                v
                         Prepared checkpoint              Result/products
                                                                  ^
                                                                  |
                                                       Authenticated replay
```

A short visible sequence of calls may appear in several entry points. It must
forward to the same recipe operations, with no route-local scientific choices,
extra transformations, or independent scientific compatibility policy. Merely
calling the same low-level numerical functions does not establish shared recipe execution.
An opaque shared executor would hide the scientific narrative and is unnecessary.

Direct Python customization remains explicit. Built-in construction materializes
the selected inputs together; complete-model builders on the prepared route
retain their existing ownership of potentially lazy canonical inputs. Preserve
likelihood callables, their options/conflict rules, and standard compatibility
arguments where currently supported. Do not serialize executable callables or
invent CLI support for them. Public runner signatures and return types remain:
in particular, existing direct CO2 runners can return a trace without being
forced to produce a `RhimeResult` or write staged products.

### 2. Checkpoints enter at named scientific boundaries

Checkpoint wrappers load/authenticate inputs, invoke the canonical operations,
and persist/report outputs. They own filesystem effects and artifact envelopes;
they do not own filtering, model construction, sampler selection, or product
equations. Keep checkpoint meanings distinct:

| Handoff | Remaining scientific work |
| --- | --- |
| Acquired, external, or reloaded pre-filter merged data | Filtering and remaining preparation, then inference/products |
| Existing staged filtered merged data | Basis/sensitivities/assembly and remaining execution; no repeated filtering |
| Fully prepared inputs | Construction, sampling, and products; no acquisition or preparation |
| Prepared inputs plus authenticated posterior/output information | Results/products, with historical v1 role recovery only where required |

The old `fixedbasisMCMC` save-merged-data option illustrates optional persistence
of a handoff, not another executor to retain. The present ordinary merged cache
is saved earlier than the staged `merged-data/merged-data.nc`, which contains
filtered data. Preserve their formats and phase meanings. Use the checkpoint's
known producer/provenance or an explicit input boundary to choose the remaining
work; do not guess that an arbitrary merged file has already been filtered.
Filtered-stage resume needs an explicit Python phase boundary; it is not an
existing automatic interpretation of `reload_merged_data`. Do not repurpose
ordinary merged reloads, add schema fields, or promise a new installed command.
Supported resume routes must use the same remaining recipe operations.

The preparation operation may return the concrete filtered merged handoff needed
for existing checkpoint persistence alongside prepared inputs. This does not
require callbacks, checkpoint hooks, an execution-state container, a universal
checkpoint schema, or new installed commands. Retain existing filenames,
manifest fields, and authentication. Validate independently supplied handoffs at
their owning boundary; trust locally constructed intermediates.

### 3. Consistent scientific policy, with explicit supported differences

For standard and multisector, use the full runners' retained-site policy for all
routes: filtering may remove empty sites, subsequent operations use the retained
observations, and effective sites/averaging periods stay aligned. An empty
retained set fails before basis construction or inference. Requested choices
remain available for provenance rather than being mutated into the retained set.

Removing the staged missing-requested-site rejection is an intentional behavior
correction. Replace its current rejection test with cross-route retained-site
regressions and document the correction in implementation docs/release notes.
Do not add a new strictness setting merely to preserve this accidental divergence.
Canonical retained-site handling must not turn arbitrary missing or malformed
prepared data into an accepted run.

Inventory other route policies before consolidating. External prepared-artifact
validation, Python-only customization, sampler defaults, and output destinations
can differ for concrete boundary reasons; each retained difference needs an
explicit rationale and regression check. Scientific compatibility and output
capability decisions belong to the consuming recipe. In particular, characterize
the prepared runner's aggregation-error/output gate against the actual model
output contract before moving it; do not silently broaden output support or
introduce a new route-specific scientific restriction. A blanket exception for
all historical route differences is insufficient.

### 4. One resolved configuration record, with typed family options

Use a small immutable `ResolvedRecipe[Options]` record with recipe identity,
`sampler`, common `output` policy, and typed family-owned `options`. This is the
normalized configuration boundary, not a container for live models, posterior
arrays, caches, or stage execution state.

Immutability means the resolved choices remain authoritative and execution does
not mutate them. The existing sampler and keyword mappings do not require a new
immutable implementation. Invocation-specific sampler/output changes use derived
values; obtaining those values must not deep-copy or materialize borrowed
scientific arrays. A frozen outer record alone is not permission to mutate
nested sampling mappings.

Standard/multisector options own acquisition/filter/basis choices, period/site
choices, and concrete scientific model settings. CO2 options own the prepared
artifact request, ordinary/cached scientific choices, optional affine artifact,
and source-to-sector reporting. Linked options later own their joint-channel
configuration. Do not accumulate nullable fields for every family in the common
record or represent all scientific options as an untyped catch-all mapping.

Remove `Co2StageSetup`'s extra normalization/wrapping layer. Do not put a whole
old `Co2RunSetup` into `options`, since that would retain a second sampler owner.
Do not keep a second authoritative output field inside standard's nested run
specification. Existing result/output adapters can construct `RhimeRunSpec`
from resolved values when they need that output contract; it is a derived
product, not another configuration source.

Alternatives considered: renaming the existing setup classes leaves split
resolution untouched; wrapping them behind compatibility adapters hides it;
forcing every family into standard's run specification incorrectly treats
artifact-derived CO2 observation metadata as acquisition configuration. A small
common record shares only concepts already common to the concrete workflows.

### 5. A family resolves complete configuration once

Each family owns one normalization/resolution implementation used by its
configuration-resolving full, prepared-input, and staged routes. It resolves
common sampling/output policy together with the family's preparation and
scientific choices. Source loading remains explicit:
the existing loaders retain source-relative path rules, aliases, and override
precedence, then supply the loaded mapping to that resolver. No stage function
parses configuration again or infers accepted options from runner signatures.

The staged factory accepts a selected recipe and the loaded parameter mapping;
the CLI continues to choose its explicit config/parameter source before calling
it. The existing boundary determines file-relative paths once. Configuration-resolving
Python entry points use the same family resolver for their existing parameter
inputs. Ordinary scientific runners/builders that accept explicit prepared inputs
and scientific arguments retain their direct callable contracts; this design
does not route those calls through configuration parsing.

Retain existing defaults and accepted option sets where the difference is an
explicit supported boundary policy as described above. Apply it once at
resolution; do not silently accept an output option on a route that previously
rejected it. Equivalent resolved scientific choices must produce the same
scientific behavior. An explicit stage destination creates an effective output
policy for that invocation without re-normalizing science or mutating the
record. Saved sampler settings remain authoritative for replay provenance.

Concrete full runners continue unpacking and forwarding scientifically named
values to ordinary functions. They must not thread the entire common record
through numerical/model components as an ambient context. This preserves the
existing readable procedural shells while eliminating separate stage-only
configuration ownership.

### 6. Separate concrete standard and multisector stage owners

Standard and multisector each have their own concrete staged implementation,
beside or clearly associated with their recipe. Remove the combined
`_standard_stages.py` owner of complete workflows. Neither stage implementation
accepts a separate standard/multisector selector or calls a dispatcher to obtain
scientific helpers. Each invokes its own canonical recipe operations directly.

Share the existing preparation algorithms, sampling mechanics, result/product
mechanics, and artifact/authentication/report helpers wherever their contracts
and policies are identical. Small parameterized helpers for homogeneous
mechanics are appropriate; a shared prepare-to-output procedure that switches
recipes is not. Similar wrappers can remain short and explicit instead of using
inheritance or a flag-driven scientific executor. This preserves locality while
avoiding duplicate scientific policy within any recipe.

### 7. Resolve a narrow staged wrapper at the CLI boundary

One explicit selector chooses a built-in family. Its staged factory resolves
configuration and returns a concrete implementation of a small structural
`StagedWorkflow` interface. The implementation holds the same resolved record;
it does not copy it into another setup/context object.

The common file-backed operations are `prepare`, `prior_predictive`, `sample`, and
`postprocess`. Their signatures name actual stage inputs: artifact paths,
manifests, output destination, draw count, and check options where applicable.
Recipe identity is already resolved, so operations receive neither a separate
`model` selector nor a heterogeneous setup union. Scientific inputs inside the
implementation retain their concrete family types. This external checkpoint
interface is distinct from the in-memory scientific operations: its methods
authenticate/load, invoke those operations, and save/report.

```text
CLI arguments + explicit configuration source
                    |
                    v
       select family + resolve configuration
                    |
                    v
       concrete workflow + resolved recipe
                    |
                    v
        named operation with explicit paths
                    |
                    v
       canonical recipe execution operations
```

A structural protocol documents/types this real calling boundary; it does not
require inheritance or registration. Implementations wrap the same canonical
recipe operations used by full/prepared execution. Interface objects have no
stage history, automatic next-step execution, materializing properties, or
generic `execute(stage, **kwargs)` loop. Shared diagnosis stays separately
callable because its inputs are the posterior, optional authentication envelope,
and convergence policy, not a family scientific setup.

A bare module convention would be smaller if signatures already matched; the
current separate standard/multisector `model` argument and CO2 signatures do
not. Since these staged internals can change, normalize the actual calling
contract rather than add eight adapters preserving the old mismatch.

### 8. Keep identity serialization and replay policy explicit

Family identity functions continue projecting the same scientifically relevant
values into their existing JSON encoding. Never hash `asdict` of the new common
record: its shape and common sampler/output fields would invalidate historical
identities. Characterize existing hash values before changing representation.
Use explicit compatibility projections for effective configuration and saved
manifest fields, preserving their names, content meanings, and schemas.
Preserve the requested preparation identity and the existing sample/replay
projection of the retained run specification. Site-dropping cases derive
effective metadata from prepared observations, without mutating requested
configuration or creating a second resolver. Existing successful configurations
must retain their historical hashes; do not redefine identity for all routes
to make the new configuration record convenient.

Shared artifact/path/digest and manifest/binding authentication remain independent
owners. A family invokes those mechanics and owns its supported versions and
scientific compatibility decisions. Standard/multisector version 2 authenticates
saved output bindings without a graph and passes the saved contract to the same
result/product operations as full execution. Genuine historical version 1 uses
the canonical recipe construction operation to recover missing roles, then
those same result/product operations, without resampling. CO2 accepts sample
version 1, remains graph-free, and separately authenticates affine artifacts.
The common envelope loader's acceptance of a
version does not grant that version to every family.

Preserve the common provenance collector and CO2-local revision requirement.
Do not expand these changes into a diagnostic or serialization redesign; #785's
shared metadata codec investigation is separate.

### 9. Preserve compatibility where callers actually depend on it

The compatibility boundary includes installed command arguments/stdout/exit
policy, config formats and meanings, durable artifacts, saved-output contracts,
and established scientific Python runner signatures/results. The retained-site
correction above is the explicit staged acceptance change; preserved command
syntax and artifact contracts do not require keeping that divergence. New staged
helper signatures and setup classes can be replaced and documented without permanent
aliases solely for hypothetical use.

Inventory exported configuration helpers and known downstream uses before
removing a symbol outside that newly introduced staged surface. Any required
temporary shim converts at the API edge to the canonical resolver; it must not
retain a second internal configuration implementation. This does not authorize
a broad public scientific API rename.

### 10. Carry these owners into recipes; keep linked work bounded

Implement against the actual devel owners and reconcile them with #773–#776.
When `recipes` lands, move the canonical scientific operations, configuration,
stage interface, separate concrete stage implementations, and shared mechanics
to their appropriate owners. Existing required `rhime` compatibility imports
follow that owner; do not restore the old monolith
or keep competing scientific execution/configuration/stage implementations in
both namespaces. Acquisition follows #773, reusable inference follows #774,
and model components follow #775; recipe composition stays with #776's concrete
scientific families. Namespace migration alone does not satisfy shared execution.

OPE-207 establishes the contract using existing standard, multisector, ordinary
CO2, and cached CO2 behavior. OPE-165 supplies the actual linked stage and output
implementation using #779's distinct durable prepared type. It remains one
joint CO2/O2 recipe; unequal observation axes, cross-channel covariance, units,
and linked roles stay with its scientific owner. A linked selector entry is
added when that supported implementation lands, not as a speculative placeholder.
The first linked route retains OPE-165's same-unit/common-scale limit; OPE-86
owns heterogeneous-unit scientific transformation.

### 11. Establish parity at every reused scientific boundary

Use controlled inputs and real scientific operations to compare preparation,
construction, and products across full, merged/prepared-input, and staged routes.
Call-order mocks are useful routing checks but do not prove numerical or policy
parity. Include retained-site changes with unequal per-site metadata, actual
basis/sensitivity calculations, and source-resolved multisector layouts.

For the same prepared inputs and resolved choices, compare model input selection,
roles, deterministic forward terms and log probability at controlled parameter
values, and matched sampling policy. Run small real-sampler smoke checks for
existing supported routes/variants. Compare deterministic products using one
posterior across ordinary construction and authenticated saved replay, allowing
only named transport/report differences. Do not demand identical trajectories
from independently sampled runs or identical graphs across distinct recipes.

Use targeted forbidden-operation checks alongside parity: full execution works
with checkpoint serializers forbidden; prepared execution forbids acquisition,
filtering, basis, and sensitivity reconstruction; graph-free replay forbids
model materialization/construction and sampling. Preserve borrowed inputs and
existing Dask execution boundaries. Repeat replay/authentication coverage for
both standard and multisector, plus ordinary/cached CO2 and affine bindings.

## Risks / Trade-offs

- **Uniform configuration hides duplicate scientific workflows** -> Review
  full/prepared/staged callers of each canonical recipe operation and require
  real cross-route parity, not merely a matching record or protocol.
- **Retained-site correction broadens malformed-input acceptance** -> Separate
  valid filtering drops from missing/corrupted artifact data, retain boundary
  validation, and check aligned metadata and empty-set failure explicitly.
- **Merged checkpoints repeat filtering** -> Preserve the pre-filter versus
  filtered handoff boundary and test resume from each supported producer.
- **Shared execution hides the scientific narrative** -> Keep concrete callable
  operations near procedural runners, with no lifecycle or generic executor.
- **Uniform shape disguises unresolved historical wrappers** -> Review field
  ownership and resolver callers directly; every resolved record has one sampler
  and output owner, with no nested old stage setup.
- **Configuration cleanup changes accepted options or defaults** -> Compare
  full/prepared/staged resolved choices and explicit rejection rules, including
  ordinary/cached variants, path handling, and explicit overrides.
- **New representation changes historical hashes** -> Compare existing family
  identity fixtures and replay saved manifests through explicit projections.
- **Interface becomes a scientific framework** -> Keep only the named external
  operations; retain explicit scientific calls and immutable configuration.
- **Namespace and interface work conflict** -> Integrate the actual merged
  owners in the stack rather than applying the pre-#778 staged tree unchanged.
- **Direct customization loses its numerical boundary** -> Preserve explicit
  likelihood/complete-model arguments and lazy-input ownership in shared operations.

## Migration Plan

1. Agree this draft in its PR. OPE-207 remains open: a merged specification is
   not proof that the refactor is implemented. Only after agreement, capture
   implementation tasks with `$openspec-propose`; this PR supplies no tasks.
2. Characterize full/prepared/staged scientific choices, checkpoint phases,
   configuration, identity hashes, CLI, and saved replay. Record supported
   boundary differences and the explicit retained-site correction.
3. Consolidate each existing recipe's preparation/construction/sampling/products
   and migrate all its consumers together. Establish real parity as each family
   is migrated; retain the readable full runner and reuse existing CO2 operations.
4. Give standard/multisector separate concrete stage owners; establish the common
   resolved record/resolver and stage interface around canonical execution.
   Remove stage-only setup wrappers and obsolete parallel workflows, preserving
   independent diagnostics and authentication.
5. Run full/prepared/staged parity, configuration, ordinary/cached installed
   workflow, output/replay, and serialization checks with explicit float64.
   Run changed-path Ruff and whitespace checks, review affected docstrings,
   and update user/API docs and a unique Towncrier fragment for the implementation.
6. Integrate the canonical owners with the recipes migration. Close OPE-207 only
   after its implementation and acceptance checks pass; then complete OPE-165's
   linked integration. PR #779's persistence work can progress independently.

Until a family migration is validated, retain its existing implementation on
the implementation branch rather than land two permanent paths. Since CLI,
configuration identities, and artifact schemas stay stable, a failed structural
migration can be rolled back without changing saved files or user configurations.
