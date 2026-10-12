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
commands SHALL forward explicit date bounds, configuration path, and decoded
keyword overrides to the Python selector. CO2-family models SHALL remain outside
this configuration-driven CLI selector.

#### Scenario: Select a recipe through either executable

- **WHEN** either executable receives an allowed model, explicit dates, an INI
  path, and keyword overrides
- **THEN** it passes those values and the selected model to `rhime.run`
- **AND** its parser rejects a CO2-family `--model` choice

### Requirement: Retain established public aliases

During the documented compatibility period through the 0.9 cycle, the established
runner aliases SHALL refer to their current implementations:
`run_rhime` to `run_standard`, `run_rhime_multisector` to `run_multisector`, and
`run_rhime_nested` to `run_nested`. The corresponding shared aliases SHALL remain
`RhimeMergedData` to `MergedData`, `RhimePreparedInputs` to `PreparedInputs`,
`assemble_rhime_inputs` to `assemble_inputs`, and `with_prepared_rhime_sites` to
`with_prepared_sites`.

#### Scenario: Import an established public name

- **WHEN** a caller imports one of these established names and its current name
- **THEN** both names refer to the same implementation or handoff type

## Source and test evidence

- `openghg_inversions/rhime/api.py`: `run` and its compatibility-period docstring;
  `openghg_inversions/rhime/__init__.py` and
  `openghg_inversions/inversion_data/__init__.py`: public exports.
- `openghg_inversions/cli.py`: `build_parser`, `main`, and `rhime_main`.
- `tests/test_public_run_api.py`:
  `test_established_imports_are_same_implementations`,
  `test_unknown_model_rejected`,
  `test_carbon_recipes_require_their_actual_prepared_inputs`,
  `test_selector_preserves_real_recipe_configuration_validation`,
  `test_both_executables_select_models_and_preserve_overrides`, and
  `test_cli_default_and_prepared_only_model_boundary`.
