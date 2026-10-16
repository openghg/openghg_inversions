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
with explicit inputs rather than use a run-specification context. The superseded
raw `build_multisector_rhime_model` function SHALL be removed; the single eager
`build_multisector_model` SHALL construct its scientific graph directly rather
than forward into a second public builder. The optional `sigma_alignment`
argument SHALL remain available, as in the standard eager builder.

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

## ADDED Requirements

### Requirement: One full labelled flux state for standard and multisector

Standard and multisector construction SHALL expose one complete physical scaling
state `x`, one two-dimensional sensitivity `hx` and one total flux concentration
contribution `mu`. Their forward calculation SHALL apply that sensitivity to the
same full state used by scientific output consumers. Multisector states SHALL
retain source-local labels on a gathered state coordinate without rectangular
padding. Packing rectangular sensitivities for construction SHALL NOT mutate
prepared inputs or retained basis artifacts; already gathered state coordinates
SHALL retain their scientific meaning and ordering.

Independent source blocks SHALL retain their selected prior families, parameters,
reparameterization and active/fixed policies. Private prior variables and recorded
sector-prior identities MAY differ by block. Fixed and zero-effect entries SHALL
remain in the full physical state. New graphs SHALL NOT create per-sector
full-state, sensitivity or contribution aliases solely to preserve old names.
Standard and multisector SHALL share the concrete flux calculation without
introducing a generic execution framework or an optional representation switch.

#### Scenario: Different priors on ragged source blocks

- **WHEN** multisector sources have unequal state counts and different supported
  priors or active/fixed policies
- **THEN** one full source-labelled `x` contains every physical state and one
  labelled linear calculation gives the complete flux concentration contribution
- **AND** structurally pruned backend sensitivity columns select their matching
  full-state entries without changing the scientific `H @ x` calculation
- **AND** its joint density and forward calculation preserve the declared block
  priors, fixed values and labelled source routing

#### Scenario: Shared construction preserves preparation ownership

- **WHEN** standard or multisector construction consumes eager scientific inputs
- **THEN** the same concrete full-state flux operation is used
- **AND** model-only packing leaves the caller's preparation and retained basis
  values, source order and lazy unselected arrays unchanged

#### Scenario: CO2 retains its coherent shared-state interpretation

- **WHEN** ordinary CO2 applies its full correlated retained state and reconstructs
  native or country flux
- **THEN** the coherent prior, effective sensitivity, affine intercept and
  retained-state-conditional reconstruction remain unchanged
- **AND** common labelled state representation does not substitute independent
  source scaling or standard multiplicative reconstruction for the affine map
