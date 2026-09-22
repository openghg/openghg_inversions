# Spec Delta

## Purpose

Defines the labelled affine quantity operation and durable coherent-CO2 payload required to replay retained-state output reconstruction after staged inference.

## ADDED Requirements

### Requirement: Labelled centred affine quantity operation
The system SHALL expose a public mathematical value for a reconstructed quantity of interest \(q\),

\[
q(\alpha)=q_{ref}+S_q(\alpha-\alpha_{ref}),
\]

with public terms `reference_output` for \(q_{ref}\), `sensitivity` for \(S_q\), and `reference_state` for \(\alpha_{ref}\). Public APIs and documentation SHALL use **sensitivity** or **sensitivity matrix**, not **design** or **design matrix**. The operation SHALL preserve labelled state and output coordinates, compatible units, arbitrary reference-state values, and all non-state sample dimensions.

#### Scenario: Apply the operation to posterior samples
- **WHEN** a labelled state array has chain, draw, or other non-state dimensions
- **THEN** applying the operation preserves those dimensions and returns values on the declared output coordinates equal to the centred affine equation

#### Scenario: Use a non-unit reference state
- **WHEN** the reference state contains values other than one
- **THEN** the operation uses those values exactly and does not substitute a mean-one scaling convention

#### Scenario: Reject incompatible labelled data
- **WHEN** state labels are missing, duplicated, reordered incompatibly, or have incompatible units
- **THEN** validation fails before positional multiplication

### Requirement: Separate reconstruction and model-input boundaries
The system SHALL keep output-only reconstruction arrays outside `Co2PreparedInputs` and its `inv_inputs` dataset. A reconstruction companion SHALL bind to one exact prepared-input artifact and SHALL obtain the authoritative coherent-CO2 reference state from that artifact. The companion SHALL persist the exact `reference_output` and `sensitivity` required for replay.

#### Scenario: Bind a matching pair
- **WHEN** the prepared inputs and reconstruction companion have matching content identity, retained-state coordinates, units, affine convention, and projection provenance
- **THEN** the system constructs the declared affine operations using the prepared `alpha_prior_mean` as `reference_state`

#### Scenario: Reject a mismatched pair
- **WHEN** any required identity, state coordinate, unit, or affine convention differs
- **THEN** loading or binding fails before reconstruction

#### Scenario: Accept an externally produced reference state
- **WHEN** an external producer includes a reference state in its incoming bundle
- **THEN** import requires exact labelled agreement with the authoritative prepared value and does not retain a second source of truth

### Requirement: Versioned named reconstruction payload
The companion SHALL use a versioned schema containing named affine quantity maps, output labels, units, uncertainty scope, projection/reconstruction provenance, and source and sector metadata needed to interpret the maps. It SHALL round-trip through the supported staged-artifact formats without reopening native preparation inputs or private sidecars. It SHALL NOT contain raw footprint-times-flux arrays, native covariance, or a dense native-by-native covariance matrix.

#### Scenario: Round-trip the companion
- **WHEN** a companion is saved and reloaded through NetCDF or Zarr
- **THEN** names, arrays, dimensions, MultiIndexes, units, scopes, identities, and JSON-safe provenance are preserved

#### Scenario: Reject an incomplete or prohibited payload
- **WHEN** a required map element is absent or a prohibited native array is present
- **THEN** schema validation identifies the offending element and rejects the payload

### Requirement: Projection-strategy-neutral retained grid map
The companion SHALL store the actual affine retained-state-to-grid map rather than relying on a projection-strategy name. Bucket-preserving coherent preparation and exact supplied-restriction producers SHALL enter through the same payload contract. Construction and validation of a supplied restriction remain outside this capability.

#### Scenario: Produce a bucket-preserving map
- **WHEN** coherent preparation uses bucket-preserving prolongation
- **THEN** the persisted affine map equals the direct map derived from the retained basis and reference flux

#### Scenario: Import a supplied-restriction map
- **WHEN** a producer provides an exact map with compatible identity metadata
- **THEN** the system imports, round-trips, and applies it without assuming that its covariance-natural prolongation equals the bucket basis

### Requirement: Explicit reconstruction scope
A grid reconstructed from the affine map SHALL carry machine-readable `retained_state_conditional` scope. It represents the affine lift of retained-state values and SHALL NOT be described as complete observation-conditioned native-grid inference.

#### Scenario: Apply the retained grid map
- **WHEN** retained-state samples are passed to the grid affine operation
- **THEN** every reconstructed value retains `retained_state_conditional` scope

#### Scenario: Avoid manufacturing native uncertainty
- **WHEN** the payload has no complete native residual action or samples
- **THEN** the system does not add native residual variance, generate native residual draws, or claim complete native-grid uncertainty

### Requirement: Functional residual extension data
For a named quantity of interest, the companion MAY carry labelled unresolved quantity covariance \(C_{qq}\), unresolved quantity-observation cross-covariance \(C_{qy}\), or an explicit `retained_exact` declaration. These data SHALL have distinct typed quantity and observation axes, units, identities, and likelihood-representation provenance. This capability SHALL only validate and preserve these data; it SHALL NOT condition on observations, calculate complete posterior moments, or format scientific outputs from them.

#### Scenario: Preserve residual blocks
- **WHEN** a compatible producer supplies \(C_{qq}\) and \(C_{qy}\) for a declared quantity
- **THEN** save, load, and binding preserve their values, axes, units, and observation/likelihood identities exactly enough for a later consumer

#### Scenario: Reject ambiguous residual data
- **WHEN** residual blocks lack required axes, units, observation identity, or likelihood-representation provenance
- **THEN** validation rejects the declared quantity without affecting affine-only quantities

### Requirement: Preserve source and sector meaning
The companion SHALL preserve OpenGHG source provenance separately from scientific sector labels or mappings. This capability SHALL NOT treat source as an independent public state-vector dimension or perform source-to-sector aggregation.

#### Scenario: Round-trip source and sector metadata
- **WHEN** several inferred sources contribute to one reported sector
- **THEN** their separate provenance and the labelled reporting relationship survive serialization without being applied

### Requirement: Independent affine evaluation
The implementation SHALL be evaluated with direct dense affine calculations and a public Verification Games-compatible fixture. Acceptance SHALL test the retained-state affine lift, not complete functional conditioning or output-format parity.

#### Scenario: Compare with a dense oracle
- **WHEN** a small labelled case uses an arbitrary non-unit reference state
- **THEN** the public operation and persisted replay match an independently calculated centred affine result

#### Scenario: Compare producer strategies
- **WHEN** bucket-preserving and supplied-restriction fixtures are loaded through the public schema
- **THEN** each reproduces its independently supplied retained-state conditional grid map without strategy-specific consumer logic

### Requirement: Owner-controlled specification
All artifacts for this change SHALL remain `Draft` until the specification owner explicitly approves finalisation. Finalisation SHALL record the approval date and change all planning-artifact statuses together without substantive edits. After finalisation, these artifacts SHALL NOT be changed without the owner's explicit consent.

#### Scenario: Discover a conflict after finalisation
- **WHEN** implementation reveals a conflict with a finalised requirement or task
- **THEN** work records the conflict and obtains owner consent before changing the planning artifacts
