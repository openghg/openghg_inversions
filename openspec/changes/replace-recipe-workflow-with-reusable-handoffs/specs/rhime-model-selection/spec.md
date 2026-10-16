## MODIFIED Requirements

### Requirement: Preserve installed run-command selection

The `rhime` and `openghg-inversions` executables SHALL expose `run --model` for
`standard`, `multisector`, `nested`, and ordinary `co2`, defaulting to `standard`.
Standard, multisector and nested commands SHALL forward the selected model,
configuration path, effective date bounds, and noncolliding decoded keyword
overrides to the Python selector. For these recipes, JSON `start_date` and
`end_date` overrides SHALL take precedence over positional dates.

Ordinary CO2 CLI execution SHALL call `run_co2_from_config` with its TOML
configuration, positional date overrides, output-path override and decoded
component-section overrides. It SHALL consume configured model, sampler and
output choices and write requested products without requiring manifests or
acquiring data. `run-co2` SHALL be an alias for this configured CLI route.
The Python `rhime.run(model="co2", prepared_inputs=...)` selector SHALL retain
its direct scientific runner and annotated trace return. Cached and linked
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
- **AND** direct Python CO2 runner returns remain unchanged
