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

Fail on missing examples. Each example supplies the validation callable, its
arguments and execution requirements: full validation checks identity on valid
supported inputs; structural validators accept lazy examples without compute;
validators requiring concrete values reject unsupported lazy inputs without
compute. There is no requirement for every numerical contract to expose a
separate structural validator. Concrete checker probes verify base-value
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
inputs. A standalone validator is an executable check, not a compulsory step in
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
nothing about whether a guarantee has been established. A full validator that
requires concrete payloads documents that input precondition and rejects lazy
payloads it needs to inspect without computing them.

A structure-only check cannot return a type promising unestablished numerical
properties. It can return no narrowed type, or a weaker type if a real handoff
needs one. Avoid a flag such as `check_values=False` that retains a full-contract
return annotation. Reuse required numerical work to enforce the scientific
policy rather than adding a redundant preliminary decomposition. For example,
preparation needing an eigenbasis accepts structurally prepared input, checks
PSD using its required eigenvalues and returns a trusted result. It must not
first call a standalone PSD validator that repeats the decomposition.

Introduce a numerical contract only at a useful handoff after its guarantees
are naturally established. Do not force earlier materialization or add a fully
validated intermediate just to obtain an annotation. For concrete applications,
review which scans and decompositions run before and after adoption, and use
focused execution tests to catch duplicated work. Required new numerical work
on changed values is distinct from rechecking an established input.

Concrete covariance cases illustrate the distinction:

| Guarantee | Owner and establishment |
| --- | --- |
| Square layout with matching row/companion-coordinate labels in order | Covariance contract; structural checks of eager indexes |
| Symmetry and positive semidefiniteness within a declared tolerance | Covariance contract; numerical validation or trusted construction |
| Compatibility with this model's observation labels | Composition boundary; exact alignment |
| Positive definiteness required for a solve | Stronger contract or numerical operation; required factorization |

PSD permits singular matrices; it must not silently become a positive-definite
requirement. A factorization is usable as a check only when its failure behavior
establishes the actual policy: Cholesky success alone is not a rank-tolerance or
symmetry check. Finite-value policy likewise belongs to the specific contract.
Concrete coordinate conventions and covariance application examples are tracked
in [#1036](https://github.com/openghg/openghg_inversions/issues/1036).

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
