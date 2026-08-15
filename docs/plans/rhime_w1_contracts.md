# RHIME W1 behavioural and usability contracts

- **Status:** Executable characterization baseline for OPE-43
- **Scope:** Current public RHIME behaviour only. Production code and future
  customization choices are outside this baseline.

## Authoritative command

Every contract below is selected by one command:

```bash
tox -e py310-openghgCur -- -m rhime_contract
```

There is no second, prose-only contract suite. Parameterized node IDs below
name the base node; every parameter case is selected by the marker.

## Executable contract map

| Contract | Executable pytest node ID |
| --- | --- |
| Successful single-sector acquisition, exact observation `long_name` attributes, scientific stage/timing order, distinctive Normal prior parameters and state dimension, modern in-memory output, filename, and serialized round-trip | `tests/test_rhime.py::test_run_rhime_api_smoke` |
| Successful source-resolved multi-sector acquisition, sector priors, model output, and in-memory diagnostics | `tests/test_rhime.py::test_run_rhime_multisector_api_smoke` |
| Standard CLI parses a real config, applies keyword-over-config precedence, and routes winning scientific, model, sampler, and output values through the runner | `tests/test_rhime.py::test_cli_run_rhime_passes_config_and_overrides` |
| Multi-sector CLI forwarding of its configuration unchanged | `tests/test_rhime.py::test_cli_run_rhime_multisector_passes_config` |
| Config normalization and early unknown-parameter failure | `tests/test_rhime.py::test_rhime_normalises_legacy_output_format_aliases`; `tests/test_rhime.py::test_run_rhime_rejects_unknown_parameter_before_data_preparation` |
| Standard and source-resolved prepared layouts | `tests/test_rhime.py::test_prepare_rhime_inputs_single_sector_reloads_merged_data`; `tests/test_rhime.py::test_prepare_rhime_inputs_multisector_keeps_source_dimension` |
| Scalar averaging-period normalization reaches shared preparation unchanged | `tests/test_rhime.py::test_run_rhime_leaves_scalar_averaging_period_for_shared_preparation` |
| Indexed site/time identity and site-indicator alignment | `tests/test_prepared_inputs_serialisation.py::test_site_indicator_is_derived_from_measurement_sites` |
| NetCDF and Zarr prepared-input dimensions, indexed and auxiliary coordinates, values, basis metadata, and replay without preparation | `tests/test_prepared_inputs_serialisation.py::test_real_prepared_inputs_save_load_and_run_without_repreparation` |
| Current Dask eager boundaries and preservation of the caller's lazy arrays/chunks | `tests/test_rhime.py::test_rhime_dask_materialization_boundaries` |
| Exact standard PyMC variable and dimension inventory | `tests/test_rhime.py::test_build_rhime_model_contains_expected_variables` |
| Exact multi-sector PyMC variable and dimension inventory | `tests/test_rhime.py::test_build_rhime_multisector_model_contains_expected_variables` |
| Optional global offset variables and dimensions | `tests/test_rhime.py::test_build_rhime_model_accepts_global_scalar_offset` |
| Predictive variable selection and prior/posterior-predictive calls | `tests/test_rhime.py::test_rhime_sampler_runs_pymc_sampling_and_predictive_steps` |
| Accepted standard output modes, rejection of unknown modes, and rejection of the multi-sector legacy combination | `tests/test_rhime.py::test_make_output_spec_accepts_supported_output_modes`; `tests/test_rhime.py::test_run_rhime_rejects_unsupported_output_format`; `tests/test_rhime.py::test_output_path_validation_rejects_multisector_legacy_output` |
| Case-normalized output naming and the historical derived filename | `tests/test_rhime.py::test_make_output_spec_normalizes_filename_convention_case`; `tests/test_rhime.py::test_derived_output_filename_can_use_legacy_convention` |
| `none` mode has no output filesystem side effects | `tests/test_rhime.py::test_run_rhime_from_prepared_inputs_defaults_sampler_and_skips_none_output_writes` |
| Standard and multi-sector modern output-bundle contents | `tests/test_rhime.py::test_make_standard_output_bundle_returns_outputs_without_mutating_result`; `tests/test_rhime.py::test_make_multisector_output_bundle_returns_modern_inv_out` |
| Selected real basic product variables; selected latest-PARIS schema fields, dtypes, covariance shape, and NetCDF reloads; and selected real legacy product variables and NetCDF reload | `tests/test_rhime.py::test_basic_output_processes_modern_output`; `tests/test_rhime.py::test_latest_paris_output_processes_modern_output`; `tests/test_rhime.py::test_standard_legacy_output_uses_modern_inversion_output` |
| Real multi-sector latest-PARIS total/sector flux variables and sector-diagnostic mean variables, with NetCDF reloads | `tests/test_rhime.py::test_make_multisector_output_bundle_builds_latest_paris_flux` |
| Default concrete-model inversion-output identity, selected provenance, burn metadata, inputs, flux, and basis-matrix round-trip | `tests/test_rhime.py::test_default_model_inversion_output_save_load_roundtrip` |
| Standalone trace write, metadata, and serialized measurement coordinates | `tests/test_rhime.py::test_save_inferencedata_preserves_burn_attrs_and_resets_multiindex_coords` |
| Selected `run_hbmcmc.py` translation fields, exact config-copy contents, legacy mode, and historical filename convention | `tests/test_run_hbmcmc_shim.py::test_run_hbmcmc_main_routes_to_run_rhime` |
| Direct current `fixedbasisMCMC(...)` legacy schema, dimensions, values, historical filename, and NetCDF write | `tests/test_postprocessing.py::test_hbmcmc_postprocessing_preserves_expected_vars_attrs_and_coords` |
| Direct current `fixedbasisMCMC(...)` trace and modern inversion-output paths, dimensions, values, and serialization | `tests/test_postprocessing.py::test_inv_out_and_trace_outputs_preserve_downstream_dims_and_custom_paths` |

## Current Dask boundary

The current implementation is not lazy through sampling. Preparation's NaN
check computes Dask-backed `H` and `H_bc`. Model construction then computes
the Dask-backed `H`, `H_bc`, `mf`, `mf_error`, and `min_error` arrays while
registering PyMC data, before sampling begins. Those operations leave the
caller's Dask arrays and chunk layout in place. The Dask callback and named
materialization counters in
`tests/test_rhime.py::test_rhime_dask_materialization_boundaries` assert each
part of this statement.

## Built-model inventory

The standard built model currently contains `Y`, `error`, `min_error`, `hx`,
`x`, `mu`, `hbc`, `bc`, `mu_bc`, `sigma_site_index`,
`sigma_period_index`, `sigma`, `epsilon`, and `y`. The multi-sector model
replaces standard `hx`/`x`/`mu` with source-suffixed design, state, and
contribution variables and retains total `mu`. A global offset adds
`site_indicator`, scalar `offset_latent`, and observation-aligned `offset`.
Prior and posterior predictive generation selects built-model variable `y`.

These inventories are copied from the exact `model.named_vars` and
`model.named_vars_to_dims` equality assertions in the mapped model tests; they
are not inferred from helper names or documentation.

## Ordinary scientist variation

`tests/test_rhime.py::test_run_rhime_api_smoke` is the executable example. It
starts from ordinary acquisition arguments, supplies
`x_prior={"pdf": "normal", "mu": 1.25, "sigma": 0.125}`, builds the model,
checks the actual PyMC `NormalRV` parameters and `region` dimension, samples
deterministically, writes the modern output, reloads it, and compares inputs
and trace values. Future builder behavior is intentionally not characterized
by W1.

Scientific-user review:

1. Find the standard or multi-sector public entry point in the first two rows.
2. Follow the asserted timing labels through preparation, model, sampling, and
   output.
3. Change an ordinary prior as in the single-sector executable example.
4. Run the authoritative command.
5. Inspect the returned `RhimeResult` and the asserted serialized artifact.
