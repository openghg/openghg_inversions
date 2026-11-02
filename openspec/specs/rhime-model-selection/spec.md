# RHIME model selection

## Purpose

Define canonical Python recipe selection and installed commands, their scientific
input/result boundaries and supported operations.

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

### Requirement: Canonical installed commands

`rhime` SHALL be the sole installed executable. Its canonical full-run routes
SHALL be `run --model standard` (also `run`), `run --model multisector`,
`run --model nested` and configured `run --model co2`. Supported staged and
product operations SHALL remain available under `rhime`.

These routes SHALL preserve effective configuration, current positional dates,
`--kwargs`, recipe-specific date/output override precedence, scientific execution,
result contents, written outputs, export defaults, logging and artifact contracts.
Positional dates and `--kwargs` remain open to separately agreed argument-syntax
revision; command names SHALL NOT prescribe that future design. The Python
package, distribution and installation commands SHALL retain their names.

#### Scenario: Execute through a canonical command

- **WHEN** a caller selects a supported full, staged or product route under `rhime`
- **THEN** it executes the selected scientific operation with its effective configuration and overrides
- **AND** scientific results, written outputs and artifact meanings retain their declared contracts

### Requirement: Canonical public Python interfaces

Standard, multisector and nested execution SHALL use `run_standard`,
`run_multisector` and `run_nested` respectively. Canonical acquired and prepared
scientific values SHALL use `MergedData` and `PreparedInputs`; input assembly
SHALL use `assemble_inputs`. Their arguments, constructor behavior, scientific
results and borrowed-array/execution guarantees SHALL follow their owning
configuration, scientific-input and model contracts.

Python names SHALL NOT by themselves require changes to serialized fields,
coordinates, scientific variables or artifact meaning. Supported artifact formats
and versions, including explicit conversion or rejection of unsupported data,
SHALL follow their persistence contracts rather than public Python vocabulary.

#### Scenario: Use the canonical scientific interface

- **WHEN** a caller constructs canonical acquired/prepared inputs and selects a supported Python recipe
- **THEN** it uses the declared scientific arguments and result contents
- **AND** Python naming alone does not establish stored-format compatibility or freeze a schema identifier
