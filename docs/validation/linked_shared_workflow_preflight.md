# Linked shared-workflow prerequisites: validation

This follow-up preserves #1005 at `61356750` in its original worktree and starts
a fresh `codex/linked-shared-workflow` worktree from that exact commit. It adds
independent acquisition fixtures and stronger existing linked checkpoint
bindings while the shared-model lead defines and reviews the linked API.
It does not implement or claim a linked shared-baseline model yet. The dated
[workflow plan](../plans/linked_shared_workflow_20261102.md) records remaining
integration and the distinct corrected-delivery/fresh-native routes.

## Acquisition fixture

`tests/test_linked_surface_acquisition.py` exercises the maintained
`MergedData.from_options` / `acquire_surface_data` / `prepare_surface_site`
path with only external getter I/O patched. It checks both tracers, explicit
site/inlet/window selectors, selected input versions, separate signed source
responses, a retained leading observation gap, and borrowed lazy transport.
Absolute O2 values near 209300 ppm retain 0.0001 ppm perturbations through native
mol/mol BC conversion. A separate case converts O2 observations exactly once
and changes only the explicit scalar reference. Missing directional endpoints
and incompatible flux units fail. Equivalent Pint unit spellings are accepted.

Five new acquisition cases passed. The relevant broader acquisition run passed
159 cases across this module, `test_surface_acquisition.py`,
`test_surface_centered_preparation.py` and `test_surface_acquisition_support.py`.
No corrected production payload was computed or changed by these tests.

## Checkpoint identity and graph-free replay

New linked manifests retain the existing format version and add exact prepared
and posterior content IDs using the existing file/Zarr hashing convention.
Readers retain compatibility with earlier records lacking those fields.
Loaded preparations carry their actual input identity; normal sampling forwards
it to the writer, including when the source manifest was written by an older
version. Source changes during inference, rewritten preparation records with
the same generation, and swapped posterior bytes fail before publishing output.

The existing shared target check also rejects ancestor/descendant overlap with
an input directory. This prevents writing a checkpoint inside its referenced
Zarr store and invalidating that store's bytes. Exact paths, hard links and
resolved symlink targets retain their previous protection.

The first bounded follow-up run passed 32 linked persistence, configuration-owner
and acquisition cases. The replacement cases use the existing no-inference
fixture to forbid model/sampler construction. Eight focused replacement and
directory-containment cases then passed. Python 3.12/3.13 tox validation is
pending the registered Slurm run.

An independent gpt-6.1-sol high-effort review identified the during-inference
identity and nested-Zarr-output cases; both were addressed with regressions.
The final review reported no remaining actionable findings. An existing Zarr
round-trip in that reviewer's partial test run was interrupted; the parent
bounded run completed. Ruff and whitespace checks pass.

## Remaining integration

The model lead's reviewed exact dependency is required before integrating linked
shared recipe selection, its complete resolved numerical codec, ordinary/cached
builders and graph-free concentration/physical-flux output roles. No builder
signature, additive O2 effective ratio or linked model mathematics was guessed.
The planned real-input synthetic-target replay is not launched. Native timing
choices will consume reviewed #1002; archived delivery timing is not silently
reinterpreted. No shared production data, campaign output or issue state changed.

## Reviewed model integration follow-up

The prerequisite-only description above records the initial checkpoint at
`2b6be2be`. The integration now consumes reviewed model API #1007 at
`fe068ad135876db993bc86fdc6322cac2621cfe4` and maintained compact recording #992
at `44c1ff7601a20050d469cca643d60cc07e4effba` as local branch dependencies.
It adds explicit `co2_o2_shared_baseline` recipe selection, frozen v4 numerical
preparation, ordinary/matched cached inference, graph-free native basic products,
and the direct-template surface acquisition adapter. Existing recipe defaults
remain unchanged. [Usage](../usage/linked_shared_workflow.rst) describes the
explicit direct-template assumption and the different physical/atmospheric
accounting. Shared PARIS products remain explicitly unsupported.

Final bounded validation passed 94 preparation/codec tests and 85 shared
workflow, shared products, surface templates, legacy linked handoff and
configuration-owner tests. Earlier acquisition/preparation coverage passed
64 tests. A synthetic lazy surface → frozen preparation → NetCDF checkpoint →
load → configured prior-predictive run produced two finite joint draws.

Independent review found and prompted fixes for codec-envelope identity,
matched-sampler default tuning, repeated lazy transport materialization,
configuration/preparation recipe mismatch, and frozen/native operator identity.
The final regression suite and zero/injected sampling controls are submitted to
Slurm from an immutable source snapshot. Their terminal evidence will be added
when available. No real-data campaign acceptance is claimed by this record.

The earlier configuration validation job 19885280 failed five tests due to
obsolete test doubles/imports. Its corrective job 19885522 passed all 45 affected
tests on Python 3.12 and 3.13. The prerequisite job 19885936 is tracked separately;
it precedes the complete integration and is not a substitute for final checks.
