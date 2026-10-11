# Proposal

## Why

Full and saved recipe routes still couple scientific preparation to later model
choices and require callers to reconcile configuration, numerical paths and
manifests. Replace the OPE-207 proposal accepted in [#787](https://github.com/openghg/openghg_inversions/pull/787)
with reusable preparation and one handoff per saved operation. The unmerged
[#798](https://github.com/openghg/openghg_inversions/pull/798) is prototype evidence,
not a completed implementation or required artifact contract.

## What Changes

- Keep one readable scientific sequence per family. Full execution calls science
  in memory; optional `save_preparation`/`save_samples` write numerical artifacts
  and the complete manifest last. Saving introduces no publisher or workflow engine.
- Give complete `RhimeConfig` requests a preparation record alongside the existing
  model, sampler and output owners. Move existing preparation fields/defaults
  rather than duplicating them. Group legacy INI options into ordinary mappings;
  resolve each saved phase independently and bind model choices to prepared facts.
- Retain `RhimeSampler`. New live inference accepts an instance or `"default"`,
  with omission selecting a fresh recipe-appropriate default per invocation.
  Isolate mutable settings, preserve supported container values and reject coordinate
  overrides. Replay results use `sampler=None` and recorded metadata, with no implicit
  sampler construction or additional sampler-settings class.
- Return one absolute manifest path from preparation and sampling. Reuse compatible
  graph-time priors/likelihoods through scientific checks rather than full-config
  equality. Preserve coherent CO2 reduction and affine association constraints.
- Save bounded output meaning, actual inference provenance and the existing output
  contract. Resolve relative dependencies and check digests; replay needs no original
  configuration, model construction, runtime sampler or separate output-binding sidecar.
- Deliver a complete standard proof before multisector, ordinary CO2 and cached CO2
  adoption. Each phase owns callers, examples, documentation and meaningful validation.
- **BREAKING:** Reset staged APIs/envelopes in 0.8 without historical hash emulation.
  Preserve numerical codecs, products and direct scientific contracts. Consume the
  earlier filtered-checkpoint/cache removal; acquisition replay remains deferred to
  #829. Do not restore removed layers to satisfy the old workflow plan.

## Capabilities

### New Capabilities

- `recipe-execution`: Concrete scientific execution, grouped phase choices,
  explicit sampler defaults/array ownership, checks and route parity.
- `prepared-recipe-handoff`: Saved preparation, production facts, compatible reuse
  and common file-reference/save-completion rules.
- `sampled-recipe-handoff`: Saved inference meaning and products from one handoff
  plus current output policy, without model or sampler reconstruction.

### Modified Capabilities

There are no durable workflow capability specs on inspected `devel` `edeffaff`.
These capabilities replace [the archived superseded proposal](../archive/2026-10-11-unify-recipe-workflow-configuration/proposal.md).
That change was archived without syncing its requirements; its historical
contract is not a current capability or an implementation prerequisite.
They also explicitly amend the landed [configuration change](../resolve-rhime-configuration/specs/rhime-configuration/spec.md):
flat preparation fields become one owned group, and phase-only consumers no longer
require complete configuration. Historical completed tasks remain baseline evidence,
not evidence that this amendment is implemented.

The three new capability documents live under this active change as proposed
spec deltas. `openspec/specs/` is the home for current capability specs after
implementation and reconciliation; `openspec/changes/archive/` retains completed
or superseded change history. Moving a change to the archive and syncing its
requirements are separate operations. This replacement remains active with its
implementation tasks unchecked.

## Impact

Planning only: [design.md](design.md) gives the concrete API sketches, metadata
inventory, scientific order and saving semantics; [tasks.md](tasks.md) defines four
implementation delivery phases and their acceptance gates. All implementation
checkboxes remain unchecked. No runtime code, new file format or dependency ships.

Reuse merged #773/#774 and #822–#825. Reconcile the pending #826/#830/#832 and
#834–#837 work before implementing over it. #840 implements #838's explicit basis
paths; #839's native flux/operator-loading work remains separate. #808 retains its
bounded integrity/compatibility contract. Neither a new acquisition codec nor all
CO2 design decisions block the first standard route.

OPE-207 remains coordination until its deliveries are validated and merged.
OPE-164 provides initial CO2 staging evidence; OPE-165 retains linked CO2/O2 work.
Nested stages and MAP remain outside implementation scope. Sync/archive only after
actual scenario coverage; planning approval alone is not completion.
