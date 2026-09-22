# Proposal

> Status: Final. Approved by the specification owner on 2026-09-22. This document must not be modified without that owner's explicit consent.

## Why

Staged coherent-CO2 inversions need the same durable native-grid reconstruction capability that `FluxWeightedBasis` supplies to standard and multisector inversions. Coherent preparation computes the native mean and covariance-natural prolongation needed for that reconstruction, but the observation-side reduction does not retain them and cannot recover them after preparation data have been released.

## What Changes

- Add one public affine native-flux value, `AffineFluxMap`, representing the strategy-neutral ingredients

  \[
  \bar x(\alpha)=m+U^*(\alpha-\alpha_{ref}),\qquad
  \bar f(\alpha)=F\bar x(\alpha),
  \]

  with public operations `state_to_native` and `state_to_flux`.
- Persist native mean \(m\), signed reference flux \(F\), and covariance-natural prolongation \(U^*\) separately rather than persisting precomposed output-specific maps such as \(FU^*\) or country-by-state products.
- Represent bucket-preserving \(U^*\) with the existing bucket `BasisOperator` and supplied-restriction \(U^*\) with an exact labelled, chunked array through one application contract.
- Keep the reconstruction value outside `Co2PreparedInputs.inv_inputs` and `CoherentGaussianReduction`, bind it to one exact prepared-input artifact, and obtain the authoritative retained reference state from that artifact.
- Mark reconstructed grids as `retained_state_conditional`; this change does not claim complete observation-conditioned native-grid inference.
- Require aggregate consumers to contract country or other output functionals with \(F\) and \(U^*\) before applying posterior samples. OPE-169 does not persist those derived maps.

Generic affine quantity-of-interest maps and complete unresolved functional moments remain useful follow-up abstractions, but they are not implemented by this change. OPE-68 owns complete country moments, OPE-164 owns staged output routing, OPE-24 owns backend-neutral quantity roles, OPE-31 owns supplied-\(\Pi\) construction and \(U^*\) derivation, and OPE-40 owns generic coherent-product persistence.

## Capabilities

### New Capabilities

- `co2-output-reconstruction`: An identity-bound, durable affine native-scaling and flux reconstruction value for coherent CO2.

### Modified Capabilities

None.

## Impact

- Adds a public value analogous to `FluxWeightedBasis`, extended with native centring and a covariance-natural prolongation.
- Adds versioned NetCDF/Zarr round trips, prepared-input binding, dense-oracle tests, bucket/supplied parity fixtures, and explicit memory evaluation.
- Leaves `Co2PreparedInputs.inv_inputs` as the model-input boundary and `CoherentGaussianReduction` as the likelihood/reduction boundary.
- Leaves country definitions, source-to-sector reporting transforms, generic quantity maps, residual covariance blocks, and scientific output formatting to their existing follow-up owners.
