# Issue 405 PARIS Sector Outputs

## Scope

Track implementation progress for GitHub issue #405: sector-aware RHIME outputs and PARIS-compatible total outputs.

## Decisions

- The next release should keep the current PARIS schema as the default. Users must pass an explicit latest-template option to produce the new PARIS concentration v04 and flux v03 outputs.
- Site/platform metadata should be resolved from observation metadata first. Likely sources are `ObsData.metadata`, `ObsData.data.attrs`, and, as fallback, `openghg_defs` site definitions through the `openghg` dependency.
- The preferred multisector output should include per-sector variables where the PARIS template supports them. Total PARIS outputs should still be generated from reconstructed sector flux fields, not from summed scale factors.

## Branch Plan

1. `codex/405-paris-template-version`: add this planning note, import the uploaded CDL templates, and add template-version plumbing while preserving the current default.
2. Single-sector latest PARIS branch: implement explicit latest-template concentration and flux outputs, including `platform`, observation `index`, `time_bnds`, new `mf_*` names, flux `time_bnds`, `cell_area`, and latest country variable names.
3. Sector reconstruction branch: reconstruct sector prior/posterior flux traces with each sector's own prior flux, sum reconstructed flux traces for total outputs, and move RHIME multisector diagnostics onto the postprocessing helper.
4. Multisector total PARIS branch: route multisector `output_format="paris"` through total PARIS output creation.
5. Per-sector PARIS branch: add sector-specific PARIS variables and a documented sector diagnostics dataset for variables that do not fit cleanly in the product files.

## Progress

- [x] Recorded decisions and branch plan.
- [x] Added latest CDL templates in-tree.
- [x] Added explicit latest single-sector PARIS products.
- [x] Added sector-aware flux reconstruction.
- [x] Routed multisector PARIS outputs through total product creation.
- [ ] Added per-sector PARIS variables and diagnostics.

## Test Plan

- Focused unit tests for template-version selection and legacy default behavior.
- Single-sector latest-output tests for required v04/v03 variables and dimensions.
- Synthetic multisector test where total posterior flux equals the sum of reconstructed sector posterior flux fields.
- NetCDF write/read smoke tests for latest concentration and flux products.
- Focused `uv run pytest` and `uv run ruff` checks on changed files, then full `tox -p` before review if time and resources allow.
