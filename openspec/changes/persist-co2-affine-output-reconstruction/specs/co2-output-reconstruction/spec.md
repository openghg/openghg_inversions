# Spec Delta

## Purpose

Defines the identity-bound affine native-scaling and flux value required to replay coherent-CO2 grid reconstruction after staged inference without reopening native preparation inputs.

## ADDED Requirements

### Requirement: Affine native-scaling and flux map
The system SHALL expose one public affine native-flux value, provisionally `AffineFluxMap`, containing labelled native mean \(m\), signed reference flux \(F\), and covariance-natural prolongation \(U^*\). Given retained state \(\alpha\) and authoritative reference state \(\alpha_{ref}\), its public `state_to_native` and `state_to_flux` operations SHALL evaluate

\[
\bar x(\alpha)=m+U^*(\alpha-\alpha_{ref}),\qquad
\bar f(\alpha)=F\bar x(\alpha).
\]

The operations SHALL preserve exact labelled state/native alignment, compatible units, arbitrary reference-state values, and all non-state sample dimensions. Public APIs and documentation SHALL use **prolongation**, `state_to_native`, and `state_to_flux`; they SHALL NOT call \(U^*\), \(FU^*\), or a derived output action a **sensitivity** or **design matrix**.

#### Scenario: Reconstruct posterior grid values
- **WHEN** a labelled retained state has chain, draw, or other non-state dimensions
- **THEN** `state_to_native` and `state_to_flux` preserve those dimensions and equal the centred affine equations on the declared native coordinates

#### Scenario: Use a non-unit reference state
- **WHEN** the authoritative reference state contains values other than one
- **THEN** both operations use those values exactly and do not substitute a mean-one scaling convention

#### Scenario: Reject incompatible labelled data
- **WHEN** state or native labels are missing, duplicated, reordered incompatibly, or have incompatible units
- **THEN** validation fails before positional multiplication or broadcasting

### Requirement: Separate strategy-neutral ingredients
The durable affine flux value SHALL preserve \(m\), \(F\), and \(U^*\) as separate ingredients. It SHALL NOT require or persist a precomposed \(FU^*\) array. Bucket-preserving preparation SHALL represent exact \(U^*=U_{bucket}\) through the retained bucket operator. A supplied-restriction producer SHALL provide exact labelled \(U^*\) explicitly and independently of \(F\). Both representations SHALL satisfy the same `state_to_native` and `state_to_flux` behavior.

#### Scenario: Produce a bucket-preserving value
- **WHEN** coherent preparation uses bucket-preserving prolongation
- **THEN** reconstruction reuses the retained bucket operator and flux without flattening their product

#### Scenario: Import a supplied-restriction value
- **WHEN** a producer provides exact labelled \(m\), \(F\), and explicit \(U^*\) with compatible identity metadata
- **THEN** the system imports, round-trips, and applies them without assuming \(U^*=U_{bucket}\) or requiring precomputed \(FU^*\)

#### Scenario: Compare equivalent strategies
- **WHEN** bucket and explicit prolongation representations contain the same mathematical operation
- **THEN** their native-scaling and flux reconstructions are label-wise equal

### Requirement: Separate reconstruction and model-input boundaries
The system SHALL keep affine flux reconstruction data outside `Co2PreparedInputs.inv_inputs` and `CoherentGaussianReduction`. The affine flux value SHALL bind to one exact prepared-input artifact and SHALL obtain the authoritative coherent-CO2 reference state from that artifact.

#### Scenario: Bind a matching pair
- **WHEN** prepared inputs and affine flux data have matching content identity, retained/native coordinates, units, affine convention, and projection provenance
- **THEN** the system constructs the bound value using prepared `alpha_prior_mean` as `reference_state`

#### Scenario: Reject a mismatched pair
- **WHEN** any required identity, coordinate, unit, affine convention, or projection provenance differs
- **THEN** loading or binding fails before reconstruction

#### Scenario: Accept an externally supplied reference state
- **WHEN** an external producer includes a reference state in its incoming bundle
- **THEN** import requires exact labelled agreement with the authoritative prepared value and does not retain a second source of truth

### Requirement: Versioned reconstruction payload
The affine flux value SHALL round-trip through a versioned staged-artifact schema preserving native mean, signed reference flux, tagged prolongation representation, dimensions, MultiIndexes, units, retained-state-conditional scope, intrinsic source labels, identities, and JSON-safe projection/reconstruction provenance. The payload SHALL NOT contain raw footprint-times-flux arrays, \(\Pi\), native covariance, dense native-by-native covariance, precomposed \(FU^*\), named derived-quantity maps, quantity-specific residual covariance blocks, or reporting-sector mappings.

#### Scenario: Round-trip the payload
- **WHEN** an affine flux value is saved and reloaded through NetCDF or Zarr
- **THEN** its ingredient values, representation kind, labels, units, scope, identities, and provenance are preserved exactly enough to reproduce both public operations

#### Scenario: Reject an incomplete or prohibited payload
- **WHEN** a required ingredient is absent or a prohibited native, precomposed, derived-quantity, residual, or reporting-policy element is present
- **THEN** schema validation identifies the offending element and rejects the payload

### Requirement: Aggregate before posterior samples
The affine flux value SHALL expose its labelled native ingredients through a bounded contraction boundary so downstream consumers can combine an aggregate functional with \(F\) and \(U^*\) before applying retained-state samples. It SHALL NOT require consumers to construct native-grid-by-sample flux solely to calculate country or other aggregate outputs. OPE-169 SHALL NOT persist the resulting aggregate-specific affine map.

#### Scenario: Derive a compact country operation
- **WHEN** a country consumer supplies labelled membership, area, physical conversion, and selection data
- **THEN** it can form country reference values and a country-by-state action before chain or draw dimensions are introduced

#### Scenario: Produce a requested grid output
- **WHEN** a consumer explicitly requests native-grid scaling or flux samples
- **THEN** the corresponding public operation introduces native-grid and sample dimensions at that named product boundary

### Requirement: Explicit reconstruction scope
Native grids produced by the affine flux value SHALL carry machine-readable `retained_state_conditional` scope. They represent conditional means given retained state and SHALL NOT be described as complete observation-conditioned native-grid inference.

#### Scenario: Apply retained-state reconstruction
- **WHEN** retained-state samples are passed to `state_to_native` or `state_to_flux`
- **THEN** every reconstructed value retains `retained_state_conditional` scope

#### Scenario: Avoid manufacturing unresolved uncertainty
- **WHEN** only the affine flux value is available
- **THEN** the system does not add unresolved variance, generate unresolved residual draws, or claim complete native-grid or aggregate uncertainty

### Requirement: Preserve intrinsic source meaning
The affine flux value SHALL preserve native source labels, order, and provenance carried intrinsically by \(m\), \(F\), and \(U^*\). It SHALL NOT treat source as an independent public retained-state axis when the retained state is gathered, and SHALL NOT own or apply reporting-sector mappings.

#### Scenario: Round-trip source-aware ingredients
- **WHEN** reconstruction spans several native sources and a gathered retained state
- **THEN** exact source/native alignment survives serialization and reconstruction without introducing a padded public source-state dimension

### Requirement: Independent affine evaluation
The implementation SHALL be evaluated with direct dense calculations on small cases, bucket-versus-explicit parity, aggregate-before-samples checks, and a public Verification Games-compatible supplied-restriction fixture. Acceptance SHALL test retained-state conditional native scaling and flux, not complete functional conditioning or output-format parity.

#### Scenario: Compare with a dense oracle
- **WHEN** a small labelled case uses arbitrary \(m\), \(F\), \(U^*\), and non-unit reference state
- **THEN** both public operations and persisted replay match independently calculated centred results

#### Scenario: Compare producer strategies
- **WHEN** bucket-preserving and supplied-restriction fixtures are loaded through the public schema
- **THEN** each reproduces its independently supplied retained-state conditional native scaling and flux without strategy-specific consumer logic

### Requirement: Owner-controlled specification
All artifacts for this change SHALL remain `Draft` until the specification owner explicitly approves finalisation. Finalisation SHALL record the approval date and change all planning-artifact statuses together without substantive edits. After finalisation, these artifacts SHALL NOT be changed without the owner's explicit consent.

#### Scenario: Discover a conflict after finalisation
- **WHEN** implementation reveals a conflict with a finalised requirement or task
- **THEN** work records the conflict and obtains owner consent before changing the planning artifacts
