# Design

> Status: Draft for review. Names below describe the proposed contracts, not
> implemented APIs. This document, the proposal, and the scenarios can be
> revised together in the spec PR. No implementation tasks are captured yet.

## Context

See [proposal.md](proposal.md) for motivation and
[the behavioral contract](specs/recipe-workflow-contract/spec.md) for acceptance.
The baseline is devel after PRs #772, #784, and #778. Standard/multisector
resolution creates `RhimeRunnerSetup(run_spec, sampler, data_args)`, with output
policy nested in `run_spec`. CO2 resolution creates
`Co2RunSetup(preparation_kwargs, runner, runner_kwargs, sampler)`; staging then
separately resolves outputs, reconstruction, and source reporting and wraps it
in `Co2StageSetup`. Eight facade operations repeat family selection and casts.

The current #773–#776 stack relocates the earlier owners, retaining those
configuration shapes. Its recipe namespace is a suitable eventual owner, but
the stack needs reconciliation with the already merged staged extraction and
CO2 workflow. Namespace migration alone does not settle this design.

## Goals / Non-Goals

**Goals:** Give common configuration concepts one owner, give file-backed
execution one small calling contract, and keep each family's scientific recipe
directly readable. Make linked integration a concrete additional implementation,
not another layer of conditionals in every public operation.

**Non-Goals:** A shared scientific model builder, universal prepared-input type,
multigas configuration schema, workflow registry, automatic discovery, execution
graph, dependency injection, or mutable pipeline lifecycle. Do not change the
scientific runners' established numerical/return contracts or absorb the
separate unified `run` CLI and diagnostic-policy work.

## Decisions

### 1. One resolved configuration record, with typed family options

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

### 2. A family resolves complete configuration once

Each family owns one normalization/resolution implementation used by its full
and staged routes. It resolves common sampling/output policy together with the
family's preparation and scientific choices. Source loading remains explicit:
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

Full and staged routes can retain existing distinct defaults and accepted
option sets. Apply those documented route policies once at resolution; do not
silently accept an output option on a route that previously rejected it.
The shared record guarantees a single resolved policy, not globally identical
defaults. An explicit stage destination creates an effective output policy
for that invocation without re-normalizing science or mutating the record.
Saved sampler settings remain authoritative for postprocessing provenance.

Concrete full runners continue unpacking and forwarding scientifically named
values to ordinary functions. They must not thread the entire common record
through numerical/model components as an ambient context. This preserves the
existing readable procedural shells while eliminating separate stage-only
configuration ownership.

### 3. Resolve a narrow family workflow at the CLI boundary

One explicit selector chooses a built-in family. Its staged factory resolves
configuration and returns a concrete implementation of a small structural
`StagedWorkflow` interface. The implementation holds the same resolved record;
it does not copy it into another setup/context object.

The common operations are `prepare`, `prior_predictive`, `sample`, and
`postprocess`. Their signatures name actual stage inputs: artifact paths,
manifests, output destination, draw count, and check options where applicable.
Recipe identity is already resolved, so operations receive neither a separate
`model` selector nor a heterogeneous setup union. Scientific inputs inside the
implementation retain their concrete family types.

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
       procedural family scientific functions
```

A structural protocol documents/types this real calling boundary; it does not
require inheritance or registration. Family implementations contain readable
procedures and reuse existing scientific functions. Interface objects have no
stage history, automatic next-step execution, materializing properties, or
generic `execute(stage, **kwargs)` loop. Shared diagnosis stays separately
callable because its inputs are the posterior, optional authentication envelope,
and convergence policy, not a family scientific setup.

A bare module convention would be smaller if signatures already matched; the
current separate standard/multisector `model` argument and CO2 signatures do
not. Since these staged internals can change, normalize the actual calling
contract rather than add eight adapters preserving the old mismatch.

### 4. Keep identity serialization and replay policy explicit

Family identity functions continue projecting the same scientifically relevant
values into their existing JSON encoding. Never hash `asdict` of the new common
record: its shape and common sampler/output fields would invalidate historical
identities. Characterize existing hash values before changing representation.
Use explicit compatibility projections for effective configuration and saved
manifest fields, preserving their names, content meanings, and schemas.

Shared artifact/path/digest and manifest/binding authentication remain independent
owners. A family invokes those mechanics and owns its supported versions and
scientific compatibility decisions. Standard/multisector version 2 authenticates
saved output bindings without a graph; genuine historical version 1 reconstructs
missing roles. CO2 accepts sample version 1, remains graph-free, and separately
authenticates affine artifacts. The common envelope loader's acceptance of a
version does not grant that version to every family.

Preserve the common provenance collector and CO2-local revision requirement.
Do not expand these changes into a diagnostic or serialization redesign; #785's
shared metadata codec investigation is separate.

### 5. Preserve compatibility where callers actually depend on it

The compatibility boundary includes installed command arguments/stdout/exit
policy, config formats and meanings, durable artifacts, saved-output contracts,
and established scientific Python runner signatures/results. New staged helper
signatures and setup classes can be replaced and documented without permanent
aliases solely for hypothetical use.

Inventory exported configuration helpers and known downstream uses before
removing a symbol outside that newly introduced staged surface. Any required
temporary shim converts at the API edge to the canonical resolver; it must not
retain a second internal configuration implementation. This does not authorize
a broad public scientific API rename.

### 6. Carry these owners into recipes; keep linked work bounded

Implement against the actual devel owners and reconcile them with #773–#776.
When `recipes` lands, move the canonical configuration, stage interface, family
implementations, and private shared mechanics together. Existing required
`rhime` compatibility imports follow that owner; do not restore the old monolith
or keep competing configuration/stage implementations in both namespaces.

OPE-207 establishes the contract using existing standard, multisector, ordinary
CO2, and cached CO2 behavior. OPE-165 supplies the actual linked stage and output
implementation using #779's distinct durable prepared type. It remains one
joint CO2/O2 recipe; unequal observation axes, cross-channel covariance, units,
and linked roles stay with its scientific owner. A linked selector entry is
added when that supported implementation lands, not as a speculative placeholder.
The first linked route retains OPE-165's same-unit/common-scale limit; OPE-86
owns heterogeneous-unit scientific transformation.

## Risks / Trade-offs

- **Uniform shape disguises unresolved historical wrappers** -> Review field
  ownership and resolver callers directly; every resolved record has one sampler
  and output owner, with no nested old stage setup.
- **Configuration cleanup changes accepted options or defaults** -> Compare
  existing full/staged resolved choices and rejection rules, including ordinary
  and cached variants, path handling, and explicit overrides.
- **New representation changes historical hashes** -> Compare existing family
  identity fixtures and replay saved manifests through explicit projections.
- **Interface becomes a scientific framework** -> Keep only the named external
  operations; retain explicit scientific calls and immutable configuration.
- **Namespace and interface work conflict** -> Integrate the actual merged
  owners in the stack rather than applying the pre-#778 staged tree unchanged.
- **Stochastic comparisons hide or invent regressions** -> Compare resolved
  scientific values and deterministic products from the same saved posterior;
  use installed sampling smoke checks without demanding identical trajectories.

## Migration Plan

1. Agree this draft in its PR. OPE-207 remains open: a merged specification is
   not proof that the refactor is implemented. Only after agreement, capture
   implementation tasks with `$openspec-propose`; this PR supplies no tasks.
2. Characterize current resolution, CLI, identity hashes, and saved replay
   contracts. Establish the single common record and family resolution owners,
   migrating full and staged consumers without changing scientific semantics.
3. Replace repeated staged dispatch with the resolved family interface. Remove
   the stage-only wrapper and obsolete internal paths; preserve independent
   diagnostics and authentication ownership.
4. Run relevant full/staged configuration, ordinary/cached installed workflow,
   output/replay, and serialization checks with explicit float64. Run changed-path
   Ruff and whitespace checks, review affected docstrings, and update user/API
   docs and a unique Towncrier fragment for the implementation.
5. Integrate the canonical owners with the recipes migration. Close OPE-207 only
   after its implementation and acceptance checks pass; then complete OPE-165's
   linked integration. PR #779's persistence work can progress independently.

Until a family migration is validated, retain its existing implementation on
the implementation branch rather than land two permanent paths. Since CLI,
configuration identities, and artifact schemas stay stable, a failed structural
migration can be rolled back without changing saved files or user configurations.
