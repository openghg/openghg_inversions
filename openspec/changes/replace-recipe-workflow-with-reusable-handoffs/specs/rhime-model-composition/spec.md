## MODIFIED Requirements

### Requirement: One owner of model and likelihood construction

Standard and multisector requests SHALL reject a custom `likelihood_builder`
combined with configured built-in mismatch/likelihood settings before acquisition
in ordinary runners or materialization in prepared-input execution. Their eager
builders `build_standard_model` and `build_multisector_model` SHALL take numerical
inputs, `RhimeModelSpec` and materialized prior parameters without output
configuration and SHALL NOT accept `model_builder`. Materialization SHALL be an
explicit runner/caller operation; selected related inputs and enabled priors MAY
be computed jointly to preserve shared Dask execution. Construction SHALL NOT
compute lazy payloads or introduce lazy/eager builder variants or another saved
workflow phase.

`run_standard(prepared=...)` and `run_multisector(prepared=..., config=...)` SHALL
use their respective acquired-data and saved-execution scientific operations
without a complete-model-builder hook. The superseded
`run_rhime_from_prepared_inputs` entry point and multisector
`build_multisector_rhime_model_result` adapter SHALL be removed with their
consumers. Whole-model customization SHALL directly compose scientific functions
with explicit inputs rather than use a run-specification context. The lower-level
`build_multisector_rhime_model` SHALL likewise not accept `model_builder`.

#### Scenario: Reject competing likelihood selections

- **WHEN** ordinary or prepared-input standard/multisector execution supplies a
  custom likelihood alongside a configured built-in selection
- **THEN** it raises `ValueError` before acquisition or model-input materialization

#### Scenario: Direct complete-model composition

- **WHEN** a caller needs a complete graph beyond the built-in recipe
- **THEN** the caller directly composes its scientific construction function,
  materialization, sampling and compatible output operations without a runner
  builder hook or output-bearing context

#### Scenario: Reject competing builder ownership

- **WHEN** a standard or multisector runner receives `model_builder`, with or
  without `likelihood_builder`
- **THEN** it rejects the unsupported runner option before constructing a model
  or invoking a likelihood; complete-model functions are composed directly
