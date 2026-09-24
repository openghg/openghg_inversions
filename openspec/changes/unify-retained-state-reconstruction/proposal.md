# Proposal

## Why

Retained-state reconstruction has two overlapping public vocabularies: basis `interpolate` hides source summation and eager conversion, while `AffineFluxMap` uses directional `state_to_native` and `state_to_flux` operations. OPE-184 should make the output meaning and execution boundary explicit without changing the approved affine equations or making ragged multisource states a special case.

## What Changes

- Expose directional native-scaling and signed-flux reconstruction on the retained basis API, using `state_to_native` and `state_to_flux` alongside the existing affine names. Keep `prolongation` for the underlying retained-to-native linear map and use *reconstruction* for the user-facing operation.
- Preserve native sources when the basis or retained flux is source-resolved. Require an explicit source sum for total-grid products; retain support for different region counts per source and shared states with source-resolved flux.
- Use the flux retained with `BasisFunctions` as the authoritative normal postprocessing flux. Keep time-axis adaptation and any dense conversion needed for serialization at a named output boundary.
- Migrate in-tree `interpolate` callers and deprecate the overlapping methods with a documented replacement. Check direct external use before choosing removal timing; do not keep a second permanent application API solely for speculative compatibility.
- Compare gathered, sourcewise, and expanded-prolongation application before choosing the multisource reconstruction kernel. Keep observation sensitivity separate.

## Capabilities

### New Capabilities

- `retained-state-reconstruction`: Directional basis reconstruction, source and flux semantics, compatibility, and execution behavior shared conceptually with affine reconstruction.

### Modified Capabilities

None. The approved OPE-169 affine equations and planning artifacts remain unchanged.

## Impact

Public `BasisOperator` and `BasisFunctions` methods, their postprocessing callers, focused tests, and usage/API documentation. No new model framework, output router, country adapter, affine persistence, or native posterior uncertainty is proposed.
