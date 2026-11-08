# Design

## Context

See [the proposal](proposal.md) for motivation. Existing development guidance
assigns validation to input owners and treats numerical arrays as borrowed.
The missing interface is a static name for established scientific guarantees.
This change captures the agreed pattern and implements its first
centered-boundary pilot.

## Goals / Non-Goals

Keep scientific contracts discoverable without changing xarray's runtime
representation or recreating its alignment and selection API. The specification
covers observable guarantees; developer guidance explains the typing technique.
Centered-boundary schemas and executable type-checker probes implement the
first application. Other scientific handoffs remain application work. Do not
introduce a type for every intermediate or validation phase, or invent new
public ingress paths.

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
has its own staticmethod or classmethod `validate`. Concrete schemas live in
`contracts.py` modules within their owning packages, beside their producers.
The decorator rejects declarations whose defining module is not a `contracts`
module. A structural Protocol matching a method named `validate` would
also match unrelated classes and does not provide explicit discovery.

Preserve the exact decorated subtype with a union-bound type parameter,
rather than constraints that can promote it to a base container. Proposed
Python 3.12+ signature:

```python
def xr_contract[T: xr.DataArray | xr.Dataset | xr.DataTree](cls: type[T]) -> type[T]: ...
```

The registry is the sole declaration inventory. Shared tests discover
`contracts.py` files under the package and import those modules before comparing
the registry with examples from each contract's tests. Discovery walks filenames,
not imports of every package module. Importing a discovered module still runs
parent-package initialization, so contract modules and their parents must be
considered when assessing import cost and optional dependencies.

Fail on missing examples. Each example supplies its validator arguments; the
shared tests check identity on valid inputs and that validation leaves lazy
examples unexecuted. Numerical guarantees are tested with the producers and
eager checks that establish them, not by the shared conformance tests. Concrete
checker probes verify base-value
rejection, validator and producer return acceptance, and compatibility with the
base container type.

A misplaced declaration is rejected when its module is imported. This cannot
detect a misplaced declaration in a module that is never imported; universal
placement enforcement would require an additional source-level check. The
current design makes discovery automatic for the designated module convention,
without claiming that stronger enforcement.

### Document guarantees and recover parameter-dependent roles

The type's docstring is the canonical statement of what consumers may assume,
including numerical tolerance policies. Producers and accepting boundaries
document how they establish those guarantees from their input preconditions.
Tests provide examples and review checks the reasoning; merely loading data or
passing example tests does not establish a numerical property for all valid
inputs. A standalone validator is a structural check, not a compulsory step in
production construction.

Validation arguments are not retained by a cast. A configurable observation
dimension must therefore be identifiable from the value or a fixed convention,
such as a role recorded during preparation or the first dimension of a named
variable. The validator checks that convention without adding metadata. The
guide's pseudocode uses the first dimension of the anomaly variable. Passing
additional context onward is allowed, but the static type does not encode an
arbitrary earlier argument's value or its relationship to that context.

### Keep related arrays together and extend shared contracts

Replace the motivating `CenteredOperands` record with a raw centered Dataset
contract in the boundary application. Its binding already packs anomaly and
mass into a Dataset for joint selection and expansion. The pilot owns binding
on `BoundCenteredTransport.bind(raw, observations, ...)`, and weight alignment
on `BoundAdditiveTransport.bind_weights(bound, weights)`. An ordinary component
record such as `AdditiveBoundary(choice, data: BoundAdditiveTransport)` holds
the contracted value;
it does not repeat its validation or join the xarray contract registry.
A record with independent cross-field invariants can still own its composition
checks; this mechanism does not define every possible Python data contract.

An additive bound contract extends the shared centered contract by subtyping.
Its validator establishes the shared guarantees and additional centering-weight
requirements, reusing the shared validation logic. The shared schema must allow
these extra variables. A trusted producer that adds them with `assign` casts
the constructed result to the extended type, since the inherited return type
does not establish the stronger claim.

### Validators check structure; producers establish numerical properties

The contract docstring describes numerical properties alongside structure, but
`validate` checks only structure. It never computes, so it behaves the same on
eager and lazy data and the shared conformance tests need no execution modes.
Numerical properties are established by whatever produces the value, in one of
three ways:

- By construction, where the property follows from the producer's input
  preconditions, such as PSD from `A Aᵀ`. Tests check outputs numerically and
  review checks the reasoning: example outputs cannot show that the property
  holds for every valid input, and loading data does not establish finiteness.
- By numerical work the algorithm already requires. Preparation needing an
  eigenbasis checks PSD with its required eigenvalues and returns a trusted
  result; it does not first call a separate PSD check repeating the
  decomposition.
- By an explicit eager check at the boundary accepting independently supplied
  data. That boundary materializes related arrays together, runs its check
  alongside `validate`, and only then returns the contract type.

A consumer of a value established in any of these ways does not check it again.
Do not force earlier materialization or add an intermediate just to obtain an
annotation. For concrete applications,
review which scans and decompositions run before and after adoption, and use
focused execution tests to catch duplicated work. Required new numerical work
on changed values is distinct from rechecking an established input.

Concrete covariance cases illustrate the distinction:

| Guarantee | Owner and establishment |
| --- | --- |
| Square layout with matching row/companion-coordinate labels in order | Covariance contract `validate`; structural checks of eager indexes |
| Symmetry and positive semidefiniteness within a declared tolerance | Documented on the covariance contract; established by the producer's construction, its required decomposition, or the eager check at the supplied-data boundary |
| Compatibility with this model's observation labels | Composition boundary; exact alignment |
| Positive definiteness required for a solve | Stronger contract or numerical operation; required factorization |

PSD permits singular matrices; it must not silently become a positive-definite
requirement. A factorization is usable as a check only when its failure behavior
establishes the actual policy: Cholesky success alone is not a rank-tolerance or
symmetry check. Finite-value policy likewise belongs to the specific contract
and is established in the same ways.
Concrete coordinate conventions and covariance application examples are tracked
in [#1036](https://github.com/openghg/openghg_inversions/issues/1036).

### Separate validation from preparation and trust construction

Normalization, exact alignment and unit conversion remain explicit scientific
operations. Reuse xarray and Pint where they already express the requirement.
The centered-boundary pilot uses `xarray_validate` 0.0.5, following the
[schema pilot](https://github.com/openghg/openghg_inversions/pull/905).
Each contract exposes an inspectable `schema(observation_dim=...)` class-level
operation. Declarative schemas own variables, real dtypes, dimensions and
attributes; callbacks own eager index and scientific metadata relationships.
The general contract does not mandate a validation library. Error-collection
mode alone does not establish Dask laziness; execution tests cover lazy payloads
and auxiliary coordinates.

The raw and bound boundary contracts promise structural guarantees only. Each
centered family's `add` checks the jointly materialized anomaly, mass and
reference once while calculating the fixed reference shift, and weighted
additive `add` checks its weights once while normalizing them. Row selection
can turn nonzero raw mass into zero bound mass, so that check belongs to the
fitted rows. These operations reject lazy inputs rather than computing them;
no extra eager stage or contract type is introduced. Replacing raw weights
removes only axes the old weights owned, so Dataset alignment cannot silently
truncate supplied weights.

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
- `validate` returns the full contract type after checking only structure. An
  input boundary that calls it but omits its eager numerical check overstates
  the documented guarantees. The producer-establishment requirement makes that
  check the boundary's responsibility, and review of input boundaries enforces
  it.

## Delivery

Keep this change active pending specification promotion. The declaration
mechanism, shared conformance tests, centered-boundary preparation/binding and
consumers, and checker probes implement the first pilot, tracked by
[#1033](https://github.com/openghg/openghg_inversions/issues/1033) and
[#1034](https://github.com/openghg/openghg_inversions/issues/1034).
`CenteredOperands` is replaced by ordinary Datasets with the raw/bound contract
annotations and class-level operations; this changes internal Python APIs.
Covariance remains follow-up application work under the same umbrella.
