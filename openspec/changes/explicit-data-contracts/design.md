# Design

## Context

See [the proposal](proposal.md) for motivation. Existing development guidance
assigns validation to input owners and treats numerical arrays as borrowed.
The missing interface is a static name for established scientific guarantees.
This change captures the agreed pattern; it does not implement it.

## Goals / Non-Goals

Keep scientific contracts discoverable without changing xarray's runtime
representation or recreating its alignment and selection API. The specification
covers observable guarantees; developer guidance explains the typing technique.
Concrete scientific schemas and executable type-checker probes remain
application work. Do not introduce a type for every intermediate or validation
phase, or invent new public ingress paths.

## Decisions

### Keep ordinary xarray objects

Use a phantom subtype of `xr.DataArray`, `xr.Dataset` or `xr.DataTree` whose
class-level `validate` returns the identical input through a static cast after
checking the promised guarantees. Trusted producers return that annotation for
outputs they construct. The pseudocode belongs in
[the validation guide](../../../docs/development/validation_and_xarray.rst).

The runtime value remains an ordinary xarray object. Changing `__class__` would
mutate borrowed data; constructing a real subtype would retain that class
through some schema-breaking operations and lose it through others. Neither
provides reliable evidence of validity. A plain alias, conversely, would not
distinguish contracts in static calls.

Require a staticmethod or classmethod validator; other class-level helpers are
allowed. New instance methods and properties would type-check but be absent
from the actual object. Do not use `isinstance` to determine contract validity.
The restriction on runtime class changes applies to establishing the contract;
it does not prohibit explicit conversions between xarray container kinds.

### Declare contracts in one small mechanism

Use one package-defined `@xr_contract` decorator and registry in a small
top-level module containing only declaration machinery. The decorator checks
at declaration time that the class subclasses a supported xarray container and
has its own staticmethod or classmethod `validate`. Concrete schemas stay beside
their producers. A structural Protocol matching a method named `validate` would
also match unrelated classes and does not provide explicit discovery.

Preserve the exact decorated subtype with a union-bound type parameter,
rather than constraints that can promote it to a base container. Proposed
Python 3.12+ signature:

```python
def xr_contract[T: xr.DataArray | xr.Dataset | xr.DataTree](cls: type[T]) -> type[T]: ...
```

The registry is the sole declaration inventory. Shared tests explicitly import
the intended contract modules before comparing that inventory with examples
supplied by each contract's tests; a registry alone cannot discover modules
that were never imported. Fail on missing examples. Check identity for full
validation and preservation of laziness for structural checks. A numerical
contract's valid eager example must not be confused with a structure-only lazy
example. Concrete checker probes verify base-value rejection, validator and
producer return acceptance, and compatibility with the base container type.

### Keep related arrays together and extend shared contracts

Replace the motivating `CenteredOperands` record with a raw centered Dataset
contract in the boundary application. Its binding already packs anomaly and
mass into a Dataset for joint selection and expansion. Binding can be an
ordinary function or class-level operation. An ordinary component record such
as `AdditiveBoundary(choice, data: BoundAdditive)` holds the contracted value;
it does not repeat its validation or join the xarray contract registry.
A record with independent cross-field invariants can still own its composition
checks; this mechanism does not define every possible Python data contract.

An additive bound contract extends the shared centered contract by subtyping.
Its validator establishes the shared guarantees and additional centering-weight
requirements, reusing the shared validation logic. The shared schema must allow
these extra variables. A trusted producer that adds them with `assign` casts
the constructed result to the extended type, since the inherited return type
does not establish the stronger claim.

### Contracts own numerical properties; boundaries own execution

Describe both structural and numerical guarantees on the contract. Validation
of arbitrary supplied numerical values happens after the owner explicitly
materializes related arrays together. Validation returns that same supplied
object; it does not hide compute or replacement. Trusted construction can
establish numerical guarantees even for lazy results. Laziness alone says
nothing about whether a guarantee has been established.

A structure-only check cannot return a type promising unestablished numerical
properties. It can return no narrowed type, or a weaker type if a real handoff
needs one. Avoid a flag such as `check_values=False` that retains a full-contract
return annotation. Reuse required numerical work to enforce the scientific
policy rather than adding a redundant preliminary decomposition.

Concrete covariance cases illustrate the distinction:

| Guarantee | Owner and establishment |
| --- | --- |
| Square layout with matching row/companion-coordinate labels in order | Covariance contract; structural checks of eager indexes |
| Symmetry and positive semidefiniteness within a declared tolerance | Covariance contract; numerical validation or trusted construction |
| Compatibility with this model's observation labels | Composition boundary; exact alignment |
| Positive definiteness required for a solve | Stronger contract or numerical operation; required factorization |

A companion dimension might use `_2`; the existing covariance-coordinate helper
uses a configurable name with `_cov` as its default. The concrete contract must
settle that convention. PSD permits singular matrices; it must not silently
become a positive-definite requirement. Finite-value policy likewise belongs to
the specific contract. Existing `NativeCovarianceAction` semantics illustrate
trusting PSD construction without globally certifying a matrix-free operator;
they do not require registering that Protocol as an xarray contract. Existing
spectral preparation in `observation_error.py` and `models/scalar_sigma.py`
provides opportunities to reuse eigenvalues already needed by the algorithm.
These cases are tracked in [#1036](https://github.com/openghg/openghg_inversions/issues/1036).

### Separate validation from preparation and trust construction

Normalization, exact alignment and unit conversion remain explicit scientific
operations. Reuse xarray and Pint where they already express the requirement.
The promising [schema pilot](https://github.com/openghg/openghg_inversions/pull/905)
is a candidate implementation source; `xarray_validate` is not required by the
contract. Error-collection mode alone does not establish Dask laziness.

Binding and supported transformations establish their returned guarantees.
Tests exercise their outputs against the validator. Consumers trust them;
complete validation does not belong in every dataclass reconstruction.
Independently supplied data is checked at the existing boundary accepting it.

## Risks / Trade-offs

- Mutation can leave a stale marker: callers own validity; mutation policing
  is outside this design.
- Inherited xarray `Self` can retain the contract annotation through operations
  such as dropping a required variable. This is normal typing behavior, not
  proof of preservation. Package transformations establish their guarantees
  through validation or tested construction; casts and `Any` offer no proof.
- DataTree child indexing returns a general `DataTree | DataArray`. A parent
  contract can describe child relationships but does not automatically narrow
  the static types of indexed children.
- Partial validation can overstate numerical guarantees: keep weaker checks
  explicit and reserve the full type for validation or trusted construction
  that establishes every promise.

## Delivery

Keep this change active as a specification proposal. No runtime or serialized
format changes occur in this PR. Apply the declaration mechanism and checker
probes with centered boundary preparation/binding and its consumers, tracked by
[#1033](https://github.com/openghg/openghg_inversions/issues/1033) and
[#1034](https://github.com/openghg/openghg_inversions/issues/1034). Covariance is
follow-up application work under the same umbrella. Promote the specification
after application evidence is available.
