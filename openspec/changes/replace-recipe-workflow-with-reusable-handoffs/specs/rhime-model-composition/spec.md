## MODIFIED Requirements

### Requirement: One owner of model and likelihood construction

Standard and multisector requests SHALL reject a custom `likelihood_builder`
combined with configured built-in mismatch/likelihood settings before acquisition
in ordinary runners or materialization in prepared-input execution. Their eager
builders `build_standard_model` and `build_multisector_model` SHALL take numerical
inputs, `RhimeModelSpec` and materialized prior parameters without output
configuration and SHALL NOT accept `model_builder`.

Materialization SHALL be an explicit runner/caller operation. Selected related
inputs and enabled priors SHALL preserve shared Dask execution at that named
boundary. Construction SHALL NOT compute lazy payloads or require a separate
lazy/eager builder variant or saved-workflow phase. The optional `sigma_alignment`
argument SHALL remain available on the canonical eager builders.

`run_standard(prepared=..., config=...)` and
`run_multisector(prepared=..., config=...)` SHALL use the same scientific
operations as acquired-data and saved execution without a complete-model-builder
hook. Whole-model customization SHALL directly compose scientific functions with
explicit inputs, materialization, sampling and compatible output operations.
`run_rhime_from_prepared_inputs`, `build_multisector_rhime_model_result` and
`build_multisector_rhime_model` SHALL be unavailable; no run-specification context
or forwarding adapter SHALL preserve those interfaces.

#### Scenario: Reject competing likelihood selections

- **WHEN** an ordinary or prepared-input standard/multisector run supplies a custom likelihood alongside a configured built-in selection
- **THEN** it raises `ValueError` before acquisition or model-input materialization

#### Scenario: Reject a complete-model runner hook

- **WHEN** a standard or multisector runner receives `model_builder`, with or without `likelihood_builder`
- **THEN** it rejects that unsupported option before constructing a model or invoking a likelihood
- **AND** whole-model callers can directly compose the supported scientific operations

#### Scenario: Construct an eager multisector model

- **WHEN** a caller supplies eager numerical inputs, resolved `RhimeModelSpec` and materialized priors to `build_multisector_model`
- **THEN** it constructs the scientific model without output configuration or an intermediate compatibility builder
- **AND** supported explicit `sigma_alignment` is preserved

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


### Requirement: Boundary state coordinates do not collide with flux metadata

Built-in standard, multisector, nested, ordinary and cached CO2 construction
SHALL qualify boundary state dimensions, MultiIndex levels and state-dependent
auxiliaries with `bc_`, preserving names that already have that prefix. Linked
CO2/O2 SHALL retain its existing `co2_` and `o2_` boundary prefixes. Flux state
names, source labels, supplied `sector` auxiliaries and gathered-state packing
SHALL retain their existing behavior. This regression fix SHALL NOT introduce
flux qualification, a source/sector identity redesign or inverse output renaming.

Borrowed input arrays SHALL retain their local names and numerical payloads.
Saved trace groups SHALL retain qualified boundary labels and coordinate
ownership metadata. Within-component label disagreements and conflicting
ownership metadata on a registered name SHALL be rejected. Observation and
scalar coordinates SHALL remain shared; conflicting scalar values SHALL raise
an explicit shared-coordinate error.

#### Scenario: Flux and boundary priors carry different source metadata

- **WHEN** a flux state and a boundary prior use the same local auxiliary name
  with different dimensions or labels
- **THEN** both scientific coordinate sets survive model construction and saved
  trace restoration under distinct boundary-qualified names
- **AND** existing `region`, `source`, `bc_region` and linked-channel trace names
  remain unchanged where they already identify the corresponding state

#### Scenario: Correlated activity validation consumes lazy auxiliaries

- **WHEN** correlated-prior and activity coordinates are compared before selecting
  active, fixed or partially active states
- **THEN** the eager correlated-state preparation boundary SHALL explicitly
  materialize both coordinate datasets together before label comparison
- **AND** shared Dask tasks SHALL run once without mutating borrowed inputs
