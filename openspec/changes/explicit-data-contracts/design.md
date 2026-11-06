# Design

## Context

See [the proposal](proposal.md) for motivation. Existing development guidance
already assigns validation to input owners and treats numerical arrays as
borrowed. The missing interface is a static name for an established dataset
contract. This change captures the agreed pattern; it does not implement it.

## Goals / Non-Goals

Keep scientific contracts discoverable without changing xarray's runtime
representation or recreating its alignment and selection API. The specification
covers observable guarantees; developer guidance explains the typing technique.
Concrete boundary schemas and executable type-checker probes remain application
work.

## Decisions

### Keep ordinary xarray objects

Recommend a phantom subtype whose class-level validator returns the input
through a static cast after checking its structure. Trusted producers return
the same annotation for layouts they construct. The pseudocode belongs in
[the validation guide](../../../docs/development/validation_and_xarray.rst).

A wrapper record would make fields explicit, but would also separate related
arrays or require forwarding dataset operations. A plain alias would retain
xarray convenience without distinguishing a checked contract in static calls.
The phantom subtype supplies that distinction without runtime construction.
Do not add instance methods or rely on runtime subtype checks.

### Separate validation from preparation

Normalization, exact alignment and unit conversion remain explicit scientific
operations. A structural validator checks the resulting contract without
repairing inputs or executing payloads. Use existing xarray and Pint operations
where they already own the requirement; avoid duplicate schema checks at each
layer. Numerical checks stay at their named eager boundary.

The promising [schema pilot](https://github.com/openghg/openghg_inversions/pull/905)
is a candidate implementation source. Its `xarray_validate` dependency is not
required by the contract. Choose its bounded adoption when implementing a
concrete schema; error-collection mode alone does not establish Dask laziness.

### Trust package construction

Binding and supported row transformations establish the contracts they return.
Tests should exercise those guarantees rather than adding complete validation
to every dataclass reconstruction. Independently supplied data is checked at
the existing boundary that accepts it. This pattern does not create a new
custom-data API or require every intermediate to acquire a special type.

## Risks / Trade-offs

- Mutation or a schema-changing xarray operation can leave a stale marker:
  callers own validity, and package transformations test the contracts they
  return. Mutation policing is outside this design.
- Inherited `Self` annotations, casts and `Any` can overstate guarantees:
  document the limitation and use focused checker probes for each adopted
  handoff. The marker is a useful interface claim, not a proof.
- Lazy payloads can hide numerical failures: describe structural guarantees
  separately from the value checks performed after explicit materialization.

## Delivery

Keep this change active as a specification proposal. No runtime or serialized
format changes occur in this PR. Apply the pattern first to centered boundary
preparation/binding and then its model consumers, tracked by
[#1033](https://github.com/openghg/openghg_inversions/issues/1033) and
[#1034](https://github.com/openghg/openghg_inversions/issues/1034).
Promote the specification after application evidence is available.
