# Proposal

## Why

Scientific datasets cross component boundaries with structure that their
`xr.Dataset` annotations do not express. Named contracts make those assumptions
discoverable and distinguish independently supplied data from trusted products,
so consumers need not repeat checks of established guarantees.

## What Changes

- Specify named static contracts with colocated, identity-preserving validation.
- Document structural and numerical guarantees together. Validators check
  structure only and never compute; producers establish numerical guarantees
  by construction, required algorithmic work, or an explicit eager check where
  supplied data enters.
- Keep ordinary xarray containers and the existing borrowed-data trust model.
- Make type docstrings the canonical guarantee description and document how
  producers establish those guarantees without duplicate numerical computation.
- Make parameter-dependent roles recoverable by consumers, and specify
  mechanical contract discovery, shared conformance coverage and extension of
  shared contracts.
- Describe the static-marker pattern in developer guidance and implement the
  centered-boundary pilot with class-owned metadata schemas and operations.
  Lazy-stage contracts establish structure; explicit numerical operations own
  their additional finite-value, mass and weight requirements.

## Capabilities

### New Capabilities

- `explicit-data-contracts`: Guarantees exposed by named labelled-data contracts,
  their validation, and their producer/consumer handoffs.

### Modified Capabilities

None. Existing scientific-input and model-composition specifications retain
ownership of scientific values, alignment and execution behavior. This change
describes how those guarantees become explicit in selected static interfaces.

## Impact

The declaration mechanism and the first centered-boundary application are
implemented: `RawCenteredSensitivity`, `BoundCenteredSensitivity` and
`BoundAdditiveSensitivity` retain ordinary Dataset runtime values and own
class-level validation and scientific operations. Their inspectable metadata
schemas use `xarray_validate` 0.0.5; library choice remains an application
decision rather than a general contract requirement. Shared conformance tests
and executable checker probes cover this pilot, tracked by
[#1033](https://github.com/openghg/openghg_inversions/issues/1033) and
[#1034](https://github.com/openghg/openghg_inversions/issues/1034).
The proposal remains active pending promotion. Concrete covariance and
numerical-contract cases remain follow-up work in
[#1036](https://github.com/openghg/openghg_inversions/issues/1036).

## Non-goals

No repository-wide dataset migration, runtime wrappers, mutation interception,
warning subsystem, generic component framework, or mandatory validation library.
