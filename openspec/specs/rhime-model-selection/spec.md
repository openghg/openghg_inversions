# RHIME model selection

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

The canonical `rhime` executable SHALL expose `run --model` for
`standard`, `multisector`, `nested` and configured `co2`, defaulting to `standard`.
For standard, multisector and nested, its run command SHALL forward the selected
model, configuration path, effective date
bounds, and noncolliding decoded keyword overrides to the Python selector.
For those three recipes, JSON `start_date` and `end_date` overrides SHALL take precedence over positional
dates. Configured CO2 execution SHALL preserve its own configuration,
date/output override and component-section override behavior through
`rhime run --model co2`; command-name retirement SHALL NOT change that recipe's
prepared-input or product contract. Cached and linked CO2-family choices SHALL
remain outside this configuration-driven CLI selector.

Current reserved-key limitation for standard, multisector and nested: JSON
`model` or `config_file` entries collide
with separately supplied keywords and raise `TypeError` before selection runs.
They are not supported override keys.

#### Scenario: Select a recipe through the canonical executable

- **WHEN** `rhime run` receives standard, multisector or nested, explicit dates, an INI
  path, and noncolliding keyword overrides such as `draws`
- **THEN** it passes those values and the selected model to `rhime.run`
- **AND** its parser rejects cached or linked CO2-family `--model` choices

#### Scenario: Override positional date bounds

- **WHEN** a standard, multisector or nested run command supplies positional dates and JSON `start_date` or
  `end_date` overrides
- **THEN** it forwards the JSON values for the overridden dates
- **AND** positional values remain effective for dates without a JSON override

### Requirement: Bounded installed-command retirement

`rhime` SHALL be the canonical installed executable. The following temporary
executable and subcommand names MAY be retired, with these supported replacements
for their corresponding scientific routes:

| Retire | Canonical replacement |
| --- | --- |
| `openghg-inversions` executable | `rhime` with the same supported subcommands, including staged and product operations |
| `run-rhime` subcommand under either executable | `rhime run --model standard`, or `rhime run` |
| `run-rhime-multisector` subcommand under either executable | `rhime run --model multisector` |
| `run-rhime-nested` subcommand under either executable | `rhime run --model nested` |
| `run-co2` subcommand under either executable | `rhime run --model co2` for configured CO2 execution |

After this retirement, `rhime` SHALL be the sole installed executable and the
retired command spellings SHALL NOT require forwarding aliases. While retained,
the temporary spellings SHALL continue their established behavior. This
permission SHALL NOT imply that removal is already implemented.

For this command-name migration, canonical routes SHALL preserve effective
configuration and the meanings of current positional dates, `--kwargs` and
date/output overrides, including their recipe-specific precedence. Positional
dates and `--kwargs` remain open to revision through a separately specified
argument-syntax amendment; this command map SHALL NOT fix the final argument
design. Command renaming alone SHALL NOT introduce replacement syntax or change scientific
execution, result contents, written outputs, export defaults, logging or artifact
contracts. Retained staged/product operations SHALL remain available under
`rhime`. Retiring the executable SHALL NOT rename the Python package or
distribution, or change package installation commands.

This exception SHALL NOT authorize Python import/type/constructor renaming,
signature or result-wrapper changes, or removal of other command operations.
Retirement SHALL preserve the
[migration evidence guarantees](../hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence).

#### Scenario: Migrate a full-run command name

- **WHEN** a caller replaces one of the four listed full-run subcommands with its canonical `rhime run --model` route
- **THEN** the same effective configuration, date/output overrides and scientific recipe are used
- **AND** the resulting outputs and artifact meanings are unchanged

#### Scenario: Retire the second executable without losing staged operations

- **WHEN** `openghg-inversions` is no longer installed
- **THEN** its supported staged and product operations remain available under `rhime`
- **AND** the Python package, distribution and installation commands keep their names

### Requirement: Retain established public aliases

The established runner aliases SHALL refer to their current implementations,
except for the
explicit 0.8 removal of Python `run_rhime` with import guidance to `run_standard`.
This supersedes the earlier alias-preservation promise for that name.
`run_rhime_multisector` SHALL refer to `run_multisector`, and
`run_rhime_nested` to `run_nested`. The corresponding shared aliases SHALL remain
`RhimeMergedData` to `MergedData`, `RhimePreparedInputs` to `PreparedInputs`,
`assemble_rhime_inputs` to `assemble_inputs`. The 0.8 retirement of `RhimeRunSpec`
also removes `with_prepared_sites` and `with_prepared_rhime_sites`; this supersedes
their alias-preservation promise. Retained sites belong to the prepared handoff.

These named Python aliases SHALL remain supported. Installed command aliases
SHALL remain supported except for the exact executable/subcommand exceptions in
`Bounded installed-command retirement` above;
retirement of the HBMCMC namespace or temporary parameter/output-format
translation SHALL NOT remove them. Their support SHALL NOT be inferred to end
at a version boundary. Scientific recipe selection and independent
output/artifact capabilities SHALL retain their separate contracts.

#### Scenario: Removed standard Python alias

- **WHEN** a caller imports `run_rhime` in 0.8
- **THEN** it receives a helpful import error directing it to `run_standard`
- **AND** canonical `rhime run --model standard` preserves the standard recipe's written files and logging

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
