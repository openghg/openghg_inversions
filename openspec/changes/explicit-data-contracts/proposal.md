# Proposal

## Why

Scientific datasets cross component boundaries with structure that their
`xr.Dataset` annotations do not express. Named contracts make those assumptions
discoverable and distinguish independently supplied data from trusted products,
so consumers need not repeat checks of established guarantees.

## What Changes

- Specify named static contracts with colocated, identity-preserving validation.
- Describe structural and numerical guarantees together, established by
  validation or trusted construction, with concrete-value checks at an explicit
  execution boundary and no full claim from partial validation.
- Keep ordinary xarray containers and the existing borrowed-data trust model.
- Specify explicit contract declaration, shared conformance coverage and
  extension of shared contracts.
- Describe the recommended static-marker pattern in developer guidance; leave
  validator-library selection and concrete scientific schemas to application
  work.

## Capabilities

### New Capabilities

- `explicit-data-contracts`: Guarantees exposed by named labelled-data contracts,
  their validation, and their producer/consumer handoffs.

### Modified Capabilities

None. Existing scientific-input and model-composition specifications retain
ownership of scientific values, alignment and execution behavior. This change
describes how those guarantees become explicit in selected static interfaces.

## Impact

This PR proposes the contract and adds developer guidance; it does not implement
contract classes, change runtime behavior or add dependencies. The centered
boundary handoffs are the first intended application, tracked by
[#1031](https://github.com/openghg/openghg_inversions/issues/1031) and its children.
The proposal contributes the general principles to
[#1032](https://github.com/openghg/openghg_inversions/issues/1032); concrete
boundary schemas and executable checker probes remain application work.
Concrete covariance and numerical-contract cases are recorded in
[#1036](https://github.com/openghg/openghg_inversions/issues/1036).

## Non-goals

No repository-wide dataset migration, runtime wrappers, mutation interception,
warning subsystem, generic component framework, or mandatory validation library.
