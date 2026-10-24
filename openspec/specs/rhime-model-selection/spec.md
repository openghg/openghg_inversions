# RHIME model selection

## Purpose

Define canonical Python recipe selection and installed commands, their scientific
input/result boundaries, and the exact retired access paths.

## Requirements

### Requirement: Select a concrete Python recipe

`openghg_inversions.rhime.run` SHALL default to `model="standard"` and accept
`standard`, `multisector`, `nested`, `co2`, `co2_o2`, and
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

Ordinary `co2` selection SHALL accept the configured `run_co2` product request
with supplied `prepared` inputs, a configured prepared-input path, or explicit
canonical inputs and a coherent reduction. It SHALL consume the configured
model, sampling and output choices and return `Co2Output` with its complete
annotated trace. It SHALL NOT acquire data or infer native covariance or
projection choices.

Linked `co2_o2` and `co2_o2_cached_sigma` selection SHALL retain their
`prepared_inputs` scientific interfaces and inference DataTree returns.
Selection SHALL NOT accept `config_file` or add acquisition on their behalf.
The ordinary `co2_cached_sigma` selector SHALL be unsupported; matched inference
is selected through ordinary CO2's scientific model choices.

#### Scenario: Select configured ordinary CO2

- **WHEN** `model="co2"` receives a supported configured request and one of its supported coherent input routes
- **THEN** execution returns the configured scientific result and requested products without acquisition
- **AND** the complete trace and reconstruction preserve that route's scientific meaning

#### Scenario: Select linked CO2 without compatible arguments

- **WHEN** a linked CO2/O2 model is selected without its required `prepared_inputs`, or with `config_file`
- **THEN** selection rejects the incompatible arguments rather than adding configuration-driven acquisition

### Requirement: Preserve installed run-command selection

The canonical `rhime` executable SHALL expose `run --model` for
`standard`, `multisector`, `nested` and configured `co2`, defaulting to `standard`.
For standard, multisector and nested, its run command SHALL forward the selected
model, configuration path, effective date
bounds, and noncolliding decoded keyword overrides to the selected scientific recipe.
For those three recipes, JSON `start_date` and `end_date` overrides SHALL take precedence over positional
dates. Configured CO2 execution SHALL preserve its own configuration,
date/output override and component-section override behavior through
`rhime run --model co2`; command-name retirement SHALL NOT change that recipe's
prepared-input or product contract. Cached and linked CO2-family choices SHALL
remain outside this configuration-driven CLI selector.

For standard, multisector and nested, `model` and `config_file` SHALL NOT be
supported JSON override keys. A request containing either SHALL fail with
`TypeError` before scientific execution rather than override recipe selection
or the separately supplied configuration path.

#### Scenario: Select a recipe through the canonical executable

- **WHEN** `rhime run` receives standard, multisector or nested, explicit dates, an INI
  path, and noncolliding keyword overrides such as `draws`
- **THEN** it executes the selected scientific recipe with those effective inputs
- **AND** its parser rejects cached or linked CO2-family `--model` choices

#### Scenario: Override positional date bounds

- **WHEN** a standard, multisector or nested run command supplies positional dates and JSON `start_date` or
  `end_date` overrides
- **THEN** it forwards the JSON values for the overridden dates
- **AND** positional values remain effective for dates without a JSON override

### Requirement: Bounded installed-command retirement

`rhime` SHALL be the sole installed executable. The following executable and
subcommand names SHALL be unavailable, with these canonical replacements for
their corresponding scientific routes:

| Retire | Canonical replacement |
| --- | --- |
| `openghg-inversions` executable | `rhime` with the same supported subcommands, including staged and product operations |
| `run-rhime` subcommand under either executable | `rhime run --model standard`, or `rhime run` |
| `run-rhime-multisector` subcommand under either executable | `rhime run --model multisector` |
| `run-rhime-nested` subcommand under either executable | `rhime run --model nested` |
| `run-co2` subcommand under either executable | `rhime run --model co2` for configured CO2 execution |

The retired executable SHALL NOT be installed and the retired subcommand names
SHALL NOT be accepted under `rhime`. No forwarding commands SHALL restore those
names solely for naming compatibility.

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

- **WHEN** a modern distribution is installed
- **THEN** only `rhime` is installed and supported staged/product operations remain available there
- **AND** the four retired full-run subcommands are unsupported
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

The five old names SHALL be unavailable from their defining modules and public
re-exports. No forwarding alias SHALL restore them solely for naming compatibility.
The canonical callables and types SHALL remain supported with their scientific
behavior, arguments, constructor behavior and result contents unchanged.

Python naming SHALL NOT itself require changes to serialized fields, coordinates,
scientific variables or artifact meaning. Supported artifact formats and versions,
and conversion or rejection of unsupported historical data, SHALL follow their
own persistence contracts; this naming requirement SHALL NOT promise historical
file compatibility or freeze a schema identifier or version. Borrowed-array and
explicit execution guarantees SHALL remain in force.

This finite alias exception SHALL NOT authorize removal or renaming of other
Python imports, types or constructors. The separately approved
[prepared-input factory retirement](../rhime-configuration/spec.md#requirement-retire-the-prepared-input-compatibility-factory)
is a bounded constructor removal, not a sixth alias. Retirement SHALL preserve the
[migration evidence guarantees](../hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence).
Installed command names follow only the separate exact exceptions in
`Bounded installed-command retirement` above.

Python `run_rhime`, `RhimeRunSpec`, `with_prepared_sites` and
`with_prepared_rhime_sites` SHALL be unavailable. Standard Python execution uses
`run_standard`; retained sites belong to the prepared handoff.

#### Scenario: Removed standard Python alias

- **WHEN** a caller requests standard execution through `rhime run --model standard` or `rhime run`
- **THEN** it executes the standard recipe, whose Python counterpart is `run_standard`
- **AND** effective inputs, written files, export defaults and logging retain their meanings
- **AND** the obsolete Python `run_rhime` name is unavailable

#### Scenario: Use a canonical replacement after alias retirement

- **WHEN** a caller replaces one of the five listed aliases with its canonical name
- **THEN** it uses the same scientific callable or data type with unchanged arguments and result contents
- **AND** naming alone does not change scientific artifact contents or their independently specified support

#### Scenario: A retired alias is no longer a public access path

- **WHEN** a caller requests one of the five old names
- **THEN** their old names are unavailable from the defining modules and public re-exports
- **AND** no forwarding alias restores those old names solely for naming compatibility
