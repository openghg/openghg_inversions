# Prepared-input compatibility factory retirement

## ADDED Requirements

### Requirement: Retire the prepared-input compatibility factory

`PreparedInputs.from_legacy_inputs` SHALL be unavailable, without a forwarding
factory or the addition of its legacy arguments to the canonical constructor.
The supported replacement SHALL remain
`PreparedInputs(inv_inputs=..., basis_functions=..., site_metadata=...)`.

Canonical construction and xarray preparation SHALL preserve scientific input
values, site/averaging-period ordering, observation-aligned release coordinates,
borrowed/lazy numerical arrays and existing basis source/path provenance,
including the xarray interface's `"supplied"` fallback for an unannotated basis.
This factory retirement SHALL NOT change existing persisted schema or codec
behavior, scientific variables, units or supported artifacts.

This separately approved constructor removal SHALL NOT extend the
[five-alias retirement](../../../../specs/rhime-model-selection/spec.md#requirement-retire-named-python-aliases)
or authorize removal of other constructors or basis capabilities. It SHALL
preserve the [migration evidence guarantees](../../../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence).

#### Scenario: Construct canonical prepared inputs

- **WHEN** a caller constructs `PreparedInputs` from canonical inversion inputs, a basis and labeled site metadata
- **THEN** the scientific inputs, metadata ordering and borrowed-array guarantees retain their meanings
- **AND** supported serialization retains its existing schema and codec behavior

#### Scenario: Prepare supplied xarray inputs after factory retirement

- **WHEN** xarray preparation receives supported stationary or moving release coordinates and a supplied basis
- **THEN** observation-aligned coordinates and existing basis source/path provenance are retained
- **AND** an unannotated supplied basis retains the `"supplied"` provenance fallback without mutation of the caller's basis
- **AND** the removed factory is not a public construction path

