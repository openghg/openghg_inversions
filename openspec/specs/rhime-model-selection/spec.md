# RHIME model selection

> Status: Current supported behavior at `42a32f62`; documentation baseline,
> no runtime change. Future recipe renaming and workflow proposals are outside
> this specification.

## Purpose

Define how the current Python selector, installed run commands, and established
aliases reach concrete recipes without changing their input contracts.

## Requirements

### Requirement: Select a concrete Python recipe

`openghg_inversions.rhime.run` SHALL default to `model="standard"` and accept
`standard`, `multisector`, `nested`, `co2`, `co2_cached_sigma`, `co2_o2`, and
`co2_o2_cached_sigma`. It SHALL forward the supplied recipe arguments unchanged
and return that recipe's result. An unsupported model name SHALL raise
`ValueError`.

#### Scenario: Preserve recipe validation

- **WHEN** the selector receives a standard or multisector request combining
  resolved `config` with `config_file`
- **THEN** the selected recipe rejects the ambiguous request with `ValueError`

#### Scenario: Reject an unknown model

- **WHEN** the caller supplies an unsupported model name
- **THEN** selection raises `ValueError` identifying that name

### Requirement: Preserve the CO2 prepared-input boundary

Selection of any of the four CO2-family models SHALL use that recipe's existing
prepared-input interface. Selection SHALL NOT supply an acquisition route or
accept `config_file` on behalf of those prepared-input recipes.

#### Scenario: Select CO2 without compatible arguments

- **WHEN** a CO2-family model is selected without its required `prepared_inputs`
- **THEN** the selected recipe raises `TypeError`
- **AND WHEN** `config_file` is supplied to that selected recipe
- **THEN** it raises `TypeError` rather than interpreting an INI file

### Requirement: Preserve installed run-command selection

The `rhime` and `openghg-inversions` executables SHALL expose `run --model` for
`standard`, `multisector`, and `nested`, defaulting to `standard`. Their run
commands SHALL forward the selected model, configuration path, effective date
bounds, and noncolliding decoded keyword overrides to the Python selector.
JSON `start_date` and `end_date` overrides SHALL take precedence over positional
dates. CO2-family models SHALL remain outside this configuration-driven CLI
selector.

Current reserved-key limitation: JSON `model` or `config_file` entries collide
with separately supplied keywords and raise `TypeError` before selection runs.
They are not supported override keys.

#### Scenario: Select a recipe through either executable

- **WHEN** either executable receives an allowed model, explicit dates, an INI
  path, and noncolliding keyword overrides such as `draws`
- **THEN** it passes those values and the selected model to `rhime.run`
- **AND** its parser rejects a CO2-family `--model` choice

#### Scenario: Override positional date bounds

- **WHEN** a run command supplies positional dates and JSON `start_date` or
  `end_date` overrides
- **THEN** it forwards the JSON values for the overridden dates
- **AND** positional values remain effective for dates without a JSON override

### Requirement: Retain established public aliases

During the documented compatibility period through the 0.9 cycle, the established
runner aliases SHALL refer to their current implementations, except for the
explicit 0.8 removal of Python `run_rhime` with import guidance to `run_standard`.
This supersedes the earlier alias-preservation promise for that name.
`run_rhime_multisector` SHALL refer to `run_multisector`, and
`run_rhime_nested` to `run_nested`. The corresponding shared aliases SHALL remain
`RhimeMergedData` to `MergedData`, `RhimePreparedInputs` to `PreparedInputs`,
`assemble_rhime_inputs` to `assemble_inputs`. The 0.8 retirement of `RhimeRunSpec`
also removes `with_prepared_sites` and `with_prepared_rhime_sites`; this supersedes
their alias-preservation promise. Retained sites belong to the prepared handoff.

#### Scenario: Removed standard Python alias

- **WHEN** a caller imports `run_rhime` in 0.8
- **THEN** it receives a helpful import error directing it to `run_standard`
- **AND** the `run-rhime` CLI command preserves its written files and logging

#### Scenario: Import a retained established public name

- **WHEN** a caller imports one of these established names and its current name
- **THEN** both names refer to the same implementation or handoff type

## Source and test evidence

- `openghg_inversions/rhime/api.py`: `run` and its compatibility-period docstring;
  `openghg_inversions/rhime/__init__.py` and
  `openghg_inversions/inversion_data/__init__.py`: public exports.
- `openghg_inversions/cli.py`: `build_parser`, `_command_kwargs`,
  `_run_command`, `main`, and `rhime_main`. The CLI test below covers noncolliding
  overrides; date precedence and reserved-key collisions follow directly from
  `_command_kwargs` and `_run_command`.
- `tests/test_public_run_api.py`:
  `test_established_imports_are_same_implementations`,
  `test_unknown_model_rejected`,
  `test_carbon_recipes_require_their_actual_prepared_inputs`,
  `test_selector_preserves_real_recipe_configuration_validation`,
  `test_both_executables_select_models_and_preserve_overrides`, and
  `test_cli_default_and_prepared_only_model_boundary`.
