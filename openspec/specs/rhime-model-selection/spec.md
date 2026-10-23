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

### Requirement: Retire named Python aliases

The following five aliases SHALL be retired in favor of their canonical public
names. This supersedes their earlier identity-alias preservation promises:

| Retire | Canonical replacement | Defining module within `openghg_inversions` |
| --- | --- | --- |
| `run_rhime_multisector` | `run_multisector` | `rhime.multisector` |
| `run_rhime_nested` | `run_nested` | `rhime.nested` |
| `RhimeMergedData` | `MergedData` | `inversion_data.acquisition` |
| `RhimePreparedInputs` | `PreparedInputs` | `inversion_data.prepared_inputs` |
| `assemble_rhime_inputs` | `assemble_inputs` | `rhime.preparation` |

Retirement SHALL remove those alias names from their defining modules and
public re-exports without retaining forwarding aliases solely for naming
compatibility. While still present before retirement, they SHALL refer to their
canonical implementations. The retirement requirement SHALL NOT imply that
deletion is already implemented.

The canonical callables and types SHALL remain supported with their scientific
behavior, arguments, constructor behavior and result contents unchanged. Alias
retirement SHALL NOT rename or relocate their canonical definitions, change
persisted schema identifiers/versions, serialized fields, coordinates or
scientific variables, or alter artifact codecs and borrowed-array/execution
guarantees. In particular, the prepared-input schema identifier
`openghg_inversions.rhime_prepared_inputs` SHALL NOT be renamed because its
Python alias retires.

This finite exception SHALL NOT authorize removal or renaming of other Python
imports, types or constructors. Retirement SHALL preserve the
[migration evidence guarantees](../hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence).
Installed command names follow only the separate exact exceptions in
`Bounded installed-command retirement` above.

The previously specified removal of Python `run_rhime` with guidance to
`run_standard` remains unchanged. The retirement of `RhimeRunSpec`,
`with_prepared_sites` and `with_prepared_rhime_sites` also retains its separate
contract. Retained sites belong to the prepared handoff.

#### Scenario: Removed standard Python alias

- **WHEN** a caller imports `run_rhime` in 0.8
- **THEN** it receives a helpful import error directing it to `run_standard`
- **AND** canonical `rhime run --model standard` preserves the standard recipe's written files and logging

#### Scenario: Use a canonical replacement after alias retirement

- **WHEN** a caller replaces one of the five listed aliases with its canonical name
- **THEN** it uses the same scientific callable or data type with unchanged arguments and result contents
- **AND** existing saved artifacts retain their schema identity and scientific meaning

#### Scenario: A retired alias is no longer a public access path

- **WHEN** the five-alias retirement is implemented
- **THEN** their old names are unavailable from the defining modules and public re-exports
- **AND** no forwarding alias restores those old names solely for naming compatibility

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
