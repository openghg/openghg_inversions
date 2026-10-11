# Proposal

## Approval and delivery status

Approved for implementation by the maintainer on 11 October 2026. Begin with a
bounded implementation PR stacked on #841; subsequent deliveries follow the
acceptance gates in [tasks.md](tasks.md). This records approval of the plan, not
completion of its implementation or an additional OpenSpec lifecycle state.
The change remains active, implementation tasks stay unchecked until their
acceptance evidence exists, and its deltas are not yet current main specs.

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
  Standard/multisector omitted model BC use inherits the preparation choice;
  BC frequency remains a baked preparation fact.
- Retain `RhimeSampler`. New live inference accepts an instance or `"default"`,
  with omission selecting a fresh recipe-appropriate default per invocation.
  Isolate mutable settings, preserve supported container values and reject coordinate
  overrides. Replay results use `sampler=None` and recorded metadata, with no implicit
  sampler construction or additional sampler-settings class.
- Return one absolute manifest path from preparation and sampling. Reuse compatible
  graph-time priors/likelihoods through scientific checks rather than full-config
  equality. Preserve coherent CO2 reduction and affine association constraints.
- Save bounded output meaning, actual inference provenance and the existing output
  contract. Initially use NetCDF numerical files and JSON manifests, preserving
  direct numerical Zarr codecs outside these handoffs. Resolve relative dependencies
  and check digests; replay needs no original
  configuration, model construction, runtime sampler or separate output-binding sidecar.
- Deliver a complete standard proof before multisector, ordinary CO2 and cached CO2
  adoption. Each phase owns callers, examples, documentation and meaningful validation.
- **BREAKING:** Reset staged APIs/envelopes in 0.8 without historical hash emulation.
  Preserve numerical codecs, products and direct scientific contracts. Consume the
  earlier filtered-checkpoint/cache removal; acquisition replay remains deferred to
  #829. Do not restore removed layers to satisfy the old workflow plan.

## Capabilities

### New Capabilities

- `recipe-execution`: Concrete scientific execution,
  explicit sampler defaults/array ownership, checks and route parity.
- `prepared-recipe-handoff`: Saved preparation, production facts, compatible reuse
  and common file-reference/save-completion rules.
- `sampled-recipe-handoff`: Saved inference meaning and products from one handoff
  plus current output policy, without model or sampler reconstruction.

### Modified Capabilities

- `rhime-configuration`: [Explicit deltas](specs/rhime-configuration/spec.md)
  amend the [current configuration capability](../../specs/rhime-configuration/spec.md):
  preparation fields move into one `PreparationConfig` group; complete and
  phase-only resolution share owners without requiring unrelated sections.

These capabilities replace [the archived superseded proposal](../archive/2026-10-11-unify-recipe-workflow-configuration/proposal.md).
That change was archived without syncing its requirements; its historical
contract is not a current capability or an implementation prerequisite.

The three new capabilities and configuration amendment live under this active
change as proposed deltas. Main specs in `openspec/specs/` describe delivered
behavior; `openspec/changes/archive/` retains completed or superseded history.
This replacement remains active with all implementation tasks unchecked. Its
deltas SHALL NOT be synced merely because planning is approved.

## Reconciled lifecycle baseline

The 11 October 2026 closeout uses merged `devel` `80b0ebf0`:

| Current capability | Merged implementation | Reconciliation evidence |
| --- | --- | --- |
| `co2-output-reconstruction` | #715 and #734 | Affine value, codec and prepared binding; `test_affine_flux_map.py`, `test_affine_flux_map_io.py`, `test_co2_affine_output.py` |
| `retained-state-reconstruction` | #732 | Directional basis operations and output consumers; `test_retained_reconstruction.py` and recorded profile decision |
| `rhime-configuration` | #824/#825, reconciled through #826/#830 | Configuration and explicit preparation; `test_rhime_configuration.py`, `test_rhime_resolved_run.py`, `test_merged_data.py`, `test_basis_dataset_inputs.py` |

Their completed changes are archived after syncing. Configuration's historical
wrapper/cache requirements were reconciled with #826's landed removals before
promotion. This is an inspection of implementation, existing tests and recorded
merge evidence, not a new runtime test run. The original scientific reconstruction
requirements are preserved. Pending #832–#841 changes are not promoted into the
main specs. When rebasing that stack, carry its spec edits as active deltas or
update the current capability with the corresponding landed implementation;
do not recreate or append new requirements to an archived change.

## Impact

Sampler selection has one deliberate public migration: omitted optional arguments
still request defaults, but explicit `None` must become `"default"` or be omitted
at live entry points. `None` consistently denotes absence on results; replay uses
saved provenance. No `None`-means-default compatibility adapter is retained.

Planning only: [design.md](design.md) gives the concrete API sketches, metadata
inventory, scientific order and saving semantics; [tasks.md](tasks.md) defines four
implementation delivery phases and their acceptance gates. All implementation
checkboxes remain unchecked. No runtime code, new file format or dependency ships.

Reuse merged #773/#774, #822–#826 and #830. Reconcile pending #832 and
#834–#837 before implementing over them. #840 implements #838's explicit basis
paths; #841 implements #839's native flux/operator-loading work and remains under
review. #808 retains its
bounded integrity/compatibility contract. Neither a new acquisition codec nor all
CO2 design decisions block the first standard route.

OPE-207 remains coordination until its deliveries are validated and merged.
OPE-164 provides initial CO2 staging evidence; OPE-165 retains linked CO2/O2 work.
Nested stages and MAP remain outside implementation scope. Sync/archive only after
actual scenario coverage; planning approval alone is not completion.
