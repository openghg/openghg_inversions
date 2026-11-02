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
directory-containment cases then passed. The registered prerequisite Slurm run 19885936 completed with exit code 0:
390 tests passed on Python 3.12 and 390 on Python 3.13. Its batch MaxRSS was
4,414,248 KiB (about 4.2 GiB). This earlier worktree run precedes the final
immutable integration snapshot.

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

Final bounded validation passed 95 preparation/codec tests and 85 shared
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

Final integration validation remains tracked by immutable-source job 19886130
and synthetic control array 19886136, both submitted from commit
`4df5eede2b82e760aaf6223272c387891fc7aa4e`. At the prerequisite wakeup, the
former was running and the latter was pending with ordinary Priority scheduling.
Their durable terminal callbacks remain armed; no final results are inferred
from the successful prerequisite run.


## Initial synthetic array and target correction

Array 19886136 completed all five tasks, each executing zero and injected
controls through preparation, configured sampling, saved trace reopening and
product export. Peak batch RSS ranged from 621,700 to 1,122,020 KiB. All ten
fits reported zero divergences, but several 250-draw chains reported elevated
R-hat or low effective sample size. These are execution checks, not convergence
acceptance.

Review of recovery summaries exposed an example target error. Its initial
observations were raw physical source totals, while the shared recipe subtracts
the frozen outer departure means. The multiplicative zero case consequently
estimated atmospheric deltas of about -0.167 ppm CO2 and +0.190 ppm O2 instead
of the stated zero. The injected-minus-zero estimates were about +0.351 ppm and
-0.447 ppm, consistent with the intended +0.35/-0.45 ppm injections; the offset
came from the target definition.

The example now uses the reviewed graph-free shared reconstructor to generate
synthetic concentrations at the known selected positive flux truth and the
stated atmospheric deltas, after freezing native support and transport. Three
regressions check the zero forward target and exact injected shifts for all
outer modes. The updated preparation/template suite passed 54 tests. Independent
read-only review cleared this example correction; production model mathematics
and calibration are unchanged.

A replacement immutable-snapshot control array uses 1000 tuning and 1000 retained
draws per chain and records baseline intervals, divergences, R-hat and effective
sample sizes through the maintained saved-trace diagnostics operation. Its 4 GiB
request is based on the first array's measured peak below 1.1 GiB. This record
will distinguish the corrected controls from the initial target-definition run.


## Final terminal evidence

Job 19886130 completed successfully: 430 tests passed on each of Python 3.12
and 3.13. Corrected array 19886194 completed all five tasks, including ten saved
sampling/product runs. The [completed-control report](linked_shared_workflow_results.md)
records baseline and state truth intervals, positivity, atmospheric accounting,
resource evidence and the two remaining R-hat flags. Its machine-readable
companion preserves the full numerical summaries. No scientific convergence
or real-data acceptance is inferred solely from scheduler completion.

## Zero-outer repeated serialization fix

The VG acceptance input exposed a production CF-codec defect after this initial
integration: a legitimate empty `physical_outer_state` MultiIndex lost unused
level domains on decoding; the next load rejected its empty level coordinates.
The shared serialization owner now accepts empty domains only for zero-row
gathers and restores their declared level domains and coordinate metadata.
Nonempty gathers with empty level domains still fail validation.

Four shared-preparation regressions perform three complete NetCDF/Zarr save/load
cycles for zero and nonempty outer selections with MultiIndex retained states.
They check exact scientific arrays, state/prior identity and empty-index level
metadata. Shared-codec tests also cover unused and fully empty level domains,
coordinate metadata and malformed nonempty gathers. All 231 broader shared-codec,
linked-preparation and general-preparation tests passed; Ruff and whitespace
checks passed. Independent read-only review found no actionable issues and
checked simultaneous empty/nonempty indexes, mixed labels, encoding, borrowed
input ownership and lazy payload identity.

The exact 1302-row VG artifact was loaded and saved/reopened three more times
using production preparation manifests, with `xr.testing.assert_identical` on
the complete scientific trees. No rows, targets or source artifacts changed.
Retest evidence and new manifests are in the worktree's
`inversion_outputs/linked-codec-fix/retest.json` and `cycle-{0,1,2}/`.
This is a codec check. The selected artifact's zeroed outer sensitivities remain
a separate scientific-input concern owned by the VG worker and model lead.

The immutable codec-fix commit `4fb0ac9a6045e7870f3adbf09ac3cef18faa2911`
completed CI-equivalent validation in Slurm job 19886351: 231 tests passed on
Python 3.12 and 231 on Python 3.13, exit code 0, elapsed 3 minutes 37 seconds,
batch MaxRSS 3808672 KiB (about 3.6 GiB). Logs are retained at
`inversion_outputs/linked-codec-fix/slurm-19886351.out`. Together with the exact
1302-row repeated-roundtrip retest above, this closes the production codec
validation. It does not resolve the separately owned scientific-input issue.
