# Issue 405 PARIS Sector Outputs

## Scope

Track implementation progress for GitHub issue #405: sector-aware RHIME outputs and PARIS-compatible total outputs.

## Decisions

- The next release should keep the current PARIS schema as the default. Users must pass an explicit latest-template option to produce the new PARIS concentration v04 and flux v03 outputs.
- Site/platform metadata should be resolved from observation metadata first. Likely sources are `ObsData.metadata`, `ObsData.data.attrs`, and, as fallback, `openghg_defs` site definitions through the `openghg` dependency.
- The preferred multisector output should include per-sector variables where the PARIS template supports them. Total PARIS outputs should still be generated from reconstructed sector flux fields, not from summed scale factors.
- Latest-template country outputs use the canonical 22-country EUROPE v03 list by default so the ordinary output matches the supplied CDL. Pass `country_selections=None` to use every country from another domain file, or pass an explicit list of names or codes for a selected domain.
- Multisector country traces should project each retained basis and prior flux to countries before applying posterior scaling traces. This keeps country-only statistics lazy without constructing a draw-wise latitude/longitude flux field.
- Total country posterior covariance applies to both single-sector and multisector output. Multisector output additionally includes per-sector and cross-sector covariance variables.
- NetCDF writing must preserve the explicit `units` and `calendar` attributes on time-bounds variables required by the supplied CDL templates, even though xarray's CF encoder normally removes duplicate bounds metadata.

## Branch Plan

1. `codex/405-paris-template-version`: add this planning note, import the uploaded CDL templates, and add template-version plumbing while preserving the current default.
2. Single-sector latest PARIS branch: implement explicit latest-template concentration and flux outputs, including `platform`, observation `index`, `time_bnds`, new `mf_*` names, flux `time_bnds`, `cell_area`, and latest country variable names.
3. Sector reconstruction branch: reconstruct sector prior/posterior flux traces with each sector's own prior flux, sum reconstructed flux traces for total outputs, and move RHIME multisector diagnostics onto the postprocessing helper.
4. Multisector total PARIS branch: route multisector `output_format="paris"` through total PARIS output creation.
5. Per-sector PARIS branch: add sector-specific PARIS flux variables and document any sector diagnostics that do not fit cleanly in the product files. Per-sector concentration remains a separate design item because multisector PARIS concentration output is not currently produced.

## Progress

- [x] Recorded decisions and branch plan.
- [x] Added latest CDL templates in-tree.
- [x] Added explicit latest single-sector PARIS products.
- [x] Added sector-aware flux reconstruction.
- [x] Routed multisector PARIS outputs through total product creation.
- [x] Added per-sector PARIS flux variables, sector coordinates, country totals, and covariance diagnostics.
- [x] Projected multisector country totals directly from basis regions and retained lazy Dask arrays until statistics require dense chunks.
- [x] Kept the canonical EUROPE country list as the latest-template default while allowing explicit all-domain or selected-country output.
- [x] Added a marked end-to-end multisector test covering local data preparation, sampling, postprocessing, NetCDF writing, and CDL-derived schema validation.
- [ ] Designed per-sector PARIS concentration variables for multisector concentration output.

## Test Plan

- Focused unit tests for template-version selection and legacy default behavior.
- Single-sector latest-output tests for required v04/v03 variables and dimensions.
- Synthetic multisector tests where total prior/posterior flux and country traces equal the sum of reconstructed sector values.
- NetCDF write/read smoke tests for latest concentration and flux products, including explicit time-bounds metadata.
- A `slow` full-pipeline test using two distinct source labels for numerically identical fluxes, with output variables, dimensions, dtypes, and attributes checked against the flux v03 CDL.
- Focused pytest, Ruff, and Pyright checks on changed files, then full `tox -p` before review.
