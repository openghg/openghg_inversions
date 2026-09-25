# Issue 443 PyMC 6 and ArviZ 1 DataTree Migration

Status: first implementation pass complete; compatibility validation remains

Evidence date: 22 September 2026

Reviewed base: `origin/devel` at `804c61c0`, after PR #670 completed
all-chain derived outputs and PR #714 removed the direct HBMCMC implementation

The first pass implements the atomic dependency/runtime migration, shared trace
persistence, bounded postprocessing tidyings, ArviZ 1 statistics compatibility,
and user-facing documentation updates. Remaining acceptance work is to add
genuine pre-migration artifact fixtures and run the reviewed Python 3.12/3.13
Slurm test matrix.

## Decision

Adopt `xarray.DataTree` directly as the runtime and public trace container when
moving to PyMC 6 and ArviZ 1.

Do not introduce a trace wrapper, vendor the old ArviZ loader, or support a
permanent `InferenceData | DataTree` protocol. `InversionOutput` is already the
domain wrapper around a trace, canonical inputs, basis functions, and run
metadata. Wrapping its trace again would reproduce DataTree indexing,
selection, mutation, attributes, and persistence without creating a useful
scientific boundary.

Keep the existing `trace` and `idata` field names during the migration, while
changing their types to `xr.DataTree`. A later naming cleanup is not required
for compatibility with the new stack.

## Updated scope after HBMCMC removal

PR #714 deleted `fixedbasisMCMC`, `inferpymc`, their trace adapters, their
standalone trace writer, and their legacy result dictionaries. Issue #443
therefore has no fixed-basis sampler or postprocessing branch to migrate.

The live runtime surface is:

- standard and multisector RHIME sampling;
- CO2-family sampling, custom group insertion, and scientific annotations;
- the experimental Ramsden result;
- staged prior-predictive, sampling, diagnostics, and postprocessing commands;
- `RhimeResult`, `InversionOutput`, model-coordinate restoration, and shared
  serialization; and
- generic, PARIS, country, diagnostics, sigma, and modern legacy-format
  postprocessing.

`run_hbmcmc.py` is only an inbound parser and parameter translator. Its
DataTree acceptance criterion is one wrapper-to-`run_rhime` regression; it
must not gain container-specific compatibility code.

## Existing design to preserve

The durable `InversionOutput` schema is already DataTree-native:

```text
/trace/<group>
/inv_inputs
/basis_functions
```

Standalone trace artifacts place trace groups directly below the root. Keep
both layouts unchanged. A standalone trace and a complete `InversionOutput`
are distinct artifact types and should have distinct loaders and clear
wrong-artifact errors rather than format auto-detection.

Most postprocessing already works with ordinary `xr.Dataset` views exposed by
`InversionOutput.trace_dataset()` and `model_data()`. DataTree leaf handling
should remain concentrated at that boundary. ArviZ should remain at the
statistics and diagnostics boundary, not own project persistence.

Issue #657 has already removed implicit first-chain selection from modern
derived outputs. The migration must preserve native `chain` and `draw`
dimensions and the completed all-chain behaviour; it does not reopen that
scientific policy.

## Small preparatory tidyings

These changes reduce the migration surface without adding a new abstraction:

1. Add one exact `InversionOutput.trace_group(name) -> xr.Dataset` accessor.
   Use it for the few product paths that still access `.posterior` or
   `.constant_data` directly.
2. Replace substring-based group discovery in `convert_idata_to_dataset()`
   with an explicit group list and a trace-neutral helper such as
   `merge_trace_groups()`. Retain a forwarding alias only if public-use
   evidence justifies it.
3. Route RHIME standalone trace writes through shared `save_trace()` and
   `load_trace()` helpers backed by `save_datatree()` and
   `open_datatree_loaded()`. Remove the separate backend loop in
   `rhime.outputs`.
4. Fold the container-specific retained-draw reset from top-level
   `_sampling.py` into `rhime.sampling`. RHIME is now its only caller, so the
   shared-looking module no longer reflects real ownership.
5. Keep coordinate and MultiIndex restoration as one named operation over
   direct DataTree children. Reassign transformed child datasets explicitly so
   group and variable attributes survive.

Do not add a generic tree visitor, trace protocol, postprocessing layer, or
adapter class.

## Delivery sequence

### 1. Current-stack characterization

Land a small PR against the guarded PyMC 5 / ArviZ 0.x stack before changing
dependencies.

- Commit a real ArviZ 0.x standalone NetCDF trace fixture.
- Commit a real schema-v1 `InversionOutput` fixture with root/group attrs and
  a MultiIndex; add a small Zarr fixture only if it exercises behaviour not
  covered by NetCDF.
- Freeze group paths, scientific-role and units metadata, coordinate
  restoration, all-chain derived products, modern legacy-format output, and
  representative statistics.
- Add clear failure tests for loading a standalone trace as a complete
  inversion artifact and vice versa.

### 2. Atomic dependency and runtime migration

Rebuild the implementation from current `devel`; do not continue PR #559's
stale branch. Remove the temporary `pymc<6` and `arviz<1` guards only in the PR
that converts every live trace path.

- Resolve a bounded PyMC 6 / ArviZ 1 range and refresh the lock file for Python
  3.12 and 3.13.
- Convert sampling return values, predictive updates, burn slicing, retained
  draw coordinates, coordinate restoration, CO2 group insertion, result
  types, and staged workflows to native DataTree operations.
- Change `InversionOutput.trace`, `RhimeResult.idata`, CO2-family results, and
  `RamsdenResult.idata` to `xr.DataTree` without renaming the fields.
- Replace InferenceData construction and persistence with project-owned
  DataTree helpers while preserving existing on-disk group paths.
- Apply the bounded postprocessing tidyings above.
- Migrate ArviZ calls in the same dependency PR, but keep their compatibility
  logic in an identifiable statistics-focused commit or workstream.

Half-migrated dependency states are not supported. A separate cleanup PR is
appropriate only if the atomic migration reveals a concrete remaining need.

## Statistics compatibility

The container change is mechanical; ArviZ 1 statistics are a separate source
of scientific risk within the same atomic dependency migration.

- Verify `az.summary` dimensions, coordinates, variable selection, and names.
- Verify `az.rhat` Dataset output and the legacy formatter's scalar extraction.
- Update `az.hdi` arguments and assert the new interval dimension and bound
  coordinates explicitly.
- Replace or locally implement the removed `az.r2_score` behaviour, with
  frozen empty-input and numerical-result tests.
- Keep `chain` and `draw` until each named statistic boundary; do not flatten
  them in the generic group-access helper.

## Acceptance criteria

- PyMC sampling and prior/posterior predictive updates return and retain an
  `xr.DataTree` through standard, multisector, CO2-family, staged, and Ramsden
  paths.
- Burn removal resets retained draw coordinates consistently across applicable
  groups.
- Scientific coordinates, MultiIndexes, root/group/variable attrs, recipe
  identity, roles, units, and covariance annotations survive prediction and
  round-trip persistence.
- Standalone NetCDF/Zarr traces and complete `InversionOutput` artifacts retain
  their existing layouts and load through explicit project-owned APIs.
- Genuine pre-migration fixtures load under the new stack without importing or
  vendoring `InferenceData`.
- Generic, PARIS, country, flux, concentration, diagnostics, sigma, and modern
  legacy-format output regressions pass with deliberately discrepant chains.
- ArviZ summary, R-hat, HDI, and R-squared results match the frozen contract or
  have an explicitly reviewed scientific change.
- `run_hbmcmc.py` still translates a supported old INI file and reaches the
  normal RHIME DataTree path; no HBMCMC-specific trace code exists.
- Changed paths pass focused pytest and Ruff checks, `git diff --check`, the
  supported Python 3.12/3.13 matrix, and the repository's reviewed Slurm tox
  environments.

## Issue and PR disposition

Issue #443 can close when this decision is recorded and the characterization
and atomic-migration follow-ups are linked. PR #559 should close as
superseded; its useful tests and compatibility findings should be reapplied to
current `devel`, not recovered through a high-risk rebase.

Issue #657 is complete and remains the decision record for all-chain derived
outputs. DataTree work must preserve that contract rather than reopening it.

## Out of scope

- Reintroducing or emulating the removed HBMCMC execution path.
- Removing `run_hbmcmc.py` or the modern legacy NetCDF formatter.
- Renaming every public `idata` field in the same migration.
- Changing likelihood equations, priors, sampler defaults, output schemas, or
  diagnostics policy beyond required ArviZ 1 compatibility.
- Designing a backend-independent trace abstraction before a second supported
  runtime creates evidence for one.
