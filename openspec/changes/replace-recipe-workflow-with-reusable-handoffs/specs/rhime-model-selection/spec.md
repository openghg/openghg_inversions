## MODIFIED Requirements

### Requirement: Preserve installed run-command selection

The `rhime` and `openghg-inversions` executables SHALL expose `run --model` for
`standard`, `multisector`, `nested`, and ordinary `co2`, defaulting to `standard`.
Standard, multisector and nested commands SHALL forward the selected model,
configuration path, effective date bounds, and noncolliding decoded keyword
overrides to the Python selector. For these recipes, JSON `start_date` and
`end_date` overrides SHALL take precedence over positional dates.

Ordinary CO2 CLI execution SHALL call `run_co2` with its TOML
configuration, positional date overrides, output-path override and decoded
component-section overrides. It SHALL consume configured model, sampler and
output choices and write requested products without requiring manifests or
acquiring data. `run-co2` SHALL be an alias for this configured CLI route.
The Python `rhime.run(model="co2", ...)` selector SHALL call the same configured
product runner and return `Co2Output` with its complete annotated trace DataTree. Cached and linked
CO2-family models SHALL remain outside the configuration-driven CLI selector.

Current reserved-key limitation for standard, multisector and nested: JSON
`model` or `config_file` entries collide with separately supplied keywords and
raise `TypeError` before selection runs. They are not supported override keys.

#### Scenario: Select a recipe through either executable

- **WHEN** either executable receives standard, multisector or nested, explicit
  dates, an INI path, and noncolliding keyword overrides such as `draws`
- **THEN** it passes those values and the selected model to `rhime.run`
- **AND** its parser rejects cached or linked CO2-family `--model` choices

#### Scenario: Override positional date bounds

- **WHEN** a standard, multisector or nested run command supplies positional
  dates and JSON `start_date` or `end_date` overrides
- **THEN** it forwards the JSON values for the overridden dates
- **AND** positional values remain effective for dates without a JSON override

#### Scenario: Run configured ordinary CO2 through either spelling

- **WHEN** either executable receives `run --model co2` or `run-co2`, a CO2 TOML
  configuration, date/output overrides and current component-section choices
- **THEN** both spellings call the same configured product runner
- **AND** requested products and the complete annotated trace use the selected
  scientific preparation and effective model, sampler and output choices
- **AND** trace-only callers compose the existing materialize, eager-build and
  sample operations rather than a second ordinary runner


### Requirement: Preserve the CO2 prepared-input boundary

Ordinary CO2 selection SHALL call `run_co2` with its configured product request
and a saved coherent preparation or explicitly supplied canonical inputs and
coherent reduction. It SHALL NOT acquire data or infer native covariance and
projection choices. Both Python and CLI selection SHALL consume the same model,
sampler and output choices and return/write the corresponding products.
Trace-only scientific callers SHALL compose the existing materialization,
eager construction, sampling and annotation operations. The redundant ordinary
`run_rhime_co2`, bundled `build_rhime_co2` and unshipped `run_co2_from_config`
entry points SHALL NOT retain aliases.

Cached and linked CO2-family selection SHALL retain the existing `prepared_inputs`
interfaces and trace returns. Those recipes SHALL NOT accept `config_file` on
behalf of their prepared-input scientific runners.

#### Scenario: Select CO2 without compatible arguments

- **WHEN** a cached or linked CO2-family model is selected without its required `prepared_inputs`
- **THEN** the selected recipe raises `TypeError`
- **AND WHEN** `config_file` is supplied to that selected recipe
- **THEN** it raises `TypeError` rather than interpreting an INI file

#### Scenario: Select configured ordinary CO2 in Python

- **WHEN** Python selection receives `model="co2"`, an ordinary CO2 request and
  its coherent prepared inputs
- **THEN** it calls `run_co2`, consumes the model/sampler/output choices and
  returns the scientific result with the full trace DataTree and requested products
- **AND** it uses the same recipe as `rhime run --model co2`
