Validation and labelled-array patterns
======================================

Scientific code should make its equations and data flow easy to inspect.
Validate in proportion to the failure prevented, and establish compatibility
once at the boundary that first combines independently prepared data. Do not
make every numerical function distrust canonical objects produced locally by
this package.

Choose the validation boundary
------------------------------

Public APIs should document their required dimensions, coordinates, units,
and numerical assumptions. Put checks at the boundary which owns those
assumptions:

.. important::

   Coordinate validation has a fixed order: use labelled xarray operations
   such as ``xr.align(..., join="exact")`` while preparing or composing data,
   then rely on ``CoordRegistry`` when labelled arrays enter a registered PyMC
   model. Do not add a custom coordinate check which duplicates either
   mechanism. A bespoke check requires a documented invariant that neither
   xarray nor the registry can express, the scientific failure it prevents,
   and a focused test demonstrating that gap.

* use static types for ordinary Python type contracts; do not add routine
  ``isinstance`` checks only to improve errors for callers which ignore type
  checking;
* normalize configuration and user choices once at the runner boundary;
* align and normalize independently sourced xarray inputs at ingestion,
  preparation, or a named scientific composition boundary;
* check results from dynamic extensions, such as custom strategies, where
  they re-enter package code;
* let the operation that needs an invariant enforce it: check a positive
  batch size before batching, and let a required Cholesky factorization,
  together with a condition estimate from its factor, report a matrix which
  cannot be used as positive definite (see `Singular and nearly singular
  matrices`_); and
* validate serialized or cached artifacts in their loader against a published
  schema, rather than repeating those checks in every downstream kernel.

After such a boundary, trust locally constructed intermediates. In particular,
avoid repeatedly checking types, dimensions, finite values, derived unit
attributes, construction identities, or solve residuals at each layer. Avoid
full eigendecompositions merely to pre-validate a factorization which the
algorithm performs anyway. A one-use ``_require_*`` helper is usually a sign
that validation is obscuring the equations.

Keep a production check when it defines scientific policy, protects an I/O or
extension boundary, prevents a library's permissive behaviour from changing
the calculation's meaning, or gives a materially clearer failure before
expensive or irreversible work. Otherwise, an equation-level test is often
the clearer home for the invariant.

Singular and nearly singular matrices
-------------------------------------

Do not infer rank from whether a Cholesky factorization fails. Whether an
exactly singular matrix passes depends on the LAPACK build and on the values:
Apple Accelerate, which macOS NumPy wheels use, often factors one with a tiny
positive last pivot, where OpenBLAS usually fails. A wrong answer computed
from that factor need not look wrong later. For a redundant bucket basis, the
noisy inverse of the Gram matrix gave a restriction whose retained covariance
was well conditioned, so downstream checks accepted an invalid result in about
a third of trials.

When rank matters, factorize once and check the reciprocal condition estimate
that LAPACK ``dpocon`` computes from the factor, as
``openghg_inversions.basis.covariance_products`` does for both the bucket
Gram matrix and :math:`C_\alpha`. This needs no eigendecomposition. Choose
the threshold for the calculation that consumes the matrix, and test the
singular case with a factorization that accepts it, so the test does not pass
only because the local LAPACK happens to reject it. The investigation, with
the history and trial counts, is recorded on `pull request 1055
<https://github.com/openghg/openghg_inversions/pull/1055#issuecomment-6817880924>`_.
Use an eigendecomposition when the eigenvalues are themselves needed, for
example to check positive semidefiniteness of a matrix that may legitimately
be singular.

Make established contracts visible in types
-------------------------------------------

Use a named contract at a handoff where an ordinary ``xr.DataArray``,
``xr.Dataset`` or ``xr.DataTree`` annotation leaves important guarantees
implicit. The type's docstring is the canonical description of the variables,
dimensions, indexed coordinates, units, relationships and numerical properties
that consumers can assume, including precise tolerance policies. Keep its
validation colocated. Producer and input-boundary docstrings explain how their
operations establish these guarantees from their own preconditions.
The validator checks structure only; numerical properties are established by
the producer, as described below.

The :download:`explicit data contracts proposal
<../../openspec/changes/explicit-data-contracts/specs/explicit-data-contracts/spec.md>`
captures this pattern. :func:`openghg_inversions.xr_contracts.xr_contract` is
the declaration decorator, and
``openghg_inversions.models.boundary.contracts`` holds the centered boundary
transport contracts, the first application. The following pseudocode
illustrates the recommended static marker for a structural contract;
``check_bound_layout`` and ``select_and_expand_transport`` stand for
domain-specific validation and preparation.

.. code-block:: python

   from typing import cast

   import xarray as xr

   @xr_contract
   class BoundTransport(xr.Dataset):
       """Observation-aligned transport with ordinary Dataset runtime values.

       The first dimension of H_bc_anomaly is the observation dimension.
       H_bc_mass shares its ordered dimensions and indexed coordinates.
       """

       __slots__ = ()

       @staticmethod
       def validate(
           data: xr.Dataset, *, observation_dim: str
       ) -> "BoundTransport":
           """Check the layout and return the same borrowed dataset."""
           check_bound_layout(data, observation_dim=observation_dim)
           return cast("BoundTransport", data)

   def bind_transport(raw, observations) -> BoundTransport:
       """Construct observation-aligned transport in canonical order."""
       bound = select_and_expand_transport(raw, observations)
       return cast(BoundTransport, bound)

Where an existing input boundary needs a standalone check, call the validator
and use its typed return value. A trusted producer instead establishes
the contract through its preparation operations. ``cast`` only records that
claim for the type checker; it performs no validation. In this example, a
consumer recovers the observation dimension from ``data.H_bc_anomaly.dims[0]``.
``check_bound_layout`` checks that convention against ``observation_dim``;
the cast does not retain validator arguments. Other contracts may record roles
in metadata during preparation. Validation checks existing information rather
than adding it. A validator returning ``None`` does not by itself narrow the
caller's variable.

Keep ordinary xarray representations when their shared coordinates,
alignment and joint selection are useful. A static marker requires no runtime
wrapper or special constructor. Its runtime object has no new instance methods
or properties: adding these to a phantom subtype would allow calls that
type-check but fail at runtime. Use class-level helpers for additional behavior,
and do not use ``isinstance`` to decide whether the contract holds. An ordinary
record may hold contracted fields without repeating their validation.

Declare contracts through the package decorator in ``contracts.py``
modules within their owning packages. Shared tests discover those files and
import their modules without a hand-maintained import list. Each contract's
tests supply valid examples, including lazy ones, and required validator
arguments. Validation leaves lazy examples uncomputed.

A family-specific contract can extend a shared contract by subtyping. Its
validator establishes both shared and additional guarantees; the shared schema
must permit the extension's extra data. A trusted producer adding that data
casts its constructed result to the stronger type. DataTree parent contracts
can describe child relationships, but indexing a child does not automatically
give it a narrower static contract type.

Package-owned transformations must preserve or establish the contract they
return. Inherited xarray ``Self`` annotations do not prove that a selection,
rename or reduction preserves the schema; these operations may retain a stale
contract annotation. Validate independently supplied results or use known
construction covered by tests that pass its outputs to the validator. Do not
add complete validation in every ``__post_init__`` or dataclass reconstruction.
Mutating an object so that its declared contract no longer holds is caller
error; interception or mutation warnings are not required.

Use this pattern where it clarifies a real handoff, rather than introducing a
special type for every intermediate. Keep normalization, alignment and unit
conversion explicit, following the guidance below.

A contract's docstring can document numerical properties, but its validator
never checks them. For example, a covariance contract can document matching row
and column labels, symmetry and positive semidefiniteness within a declared
tolerance. Its validator checks the labels, using eager indexed coordinates, and
leaves payloads lazy. The producer of each value establishes the numerical
properties in one of three ways:

* By construction from the producer's own input preconditions, such as PSD from
  ``A @ A.T``. Tests check outputs numerically, but example outputs alone do not
  prove the construction argument, so review checks the reasoning. Loading data
  does not establish finiteness.
* By numerical work the algorithm already requires. A preparation function
  that needs eigenvectors checks PSD using the eigenvalues from that same
  decomposition, rather than first calling a separate check that repeats it.
* By an explicit eager check at the boundary accepting independently supplied
  data. That boundary materializes related arrays together, runs the check
  alongside the validator, and only then returns the contract type.

Consumers do not check these properties again. Do not introduce an intermediate
or force earlier materialization merely to obtain a contract annotation. At
each adopted handoff, review which scans and decompositions run and which
repeated checks disappear.

Positive semidefiniteness permits singular covariance matrices: a solve requiring
positive definiteness owns that stronger requirement. Reusing numerical work as
a check requires reliable failure behavior for the stated policy; Cholesky
success alone does not establish symmetry or a specified rank tolerance.
Compatibility with a particular model's observation labels belongs to the
composition boundary combining them.

Centered-boundary application
-----------------------------

The first application is
:class:`~openghg_inversions.models.boundary.RawCenteredSensitivity`,
:class:`~openghg_inversions.models.boundary.BoundCenteredSensitivity` and its
:class:`~openghg_inversions.models.boundary.BoundAdditiveSensitivity` extension.
Their schemas use ``xarray_validate`` for declarative variables, dtypes,
dimensions and attributes, with callbacks for unit and index relationships;
the library choice belongs to this application, not to every contract. The
contracts are structural: each centered family's ``add`` performs the one
numerical check of the jointly materialized values. The class docstrings list
the guarantees and operations; see :doc:`/usage/centered_boundary_conditions`
for the scientific model.

Transpose, align, compute
-------------------------

For independently prepared labelled arrays, the usual strict pattern is:

.. code-block:: python

   mean = mean.transpose(*native_dims)
   sensitivity = sensitivity.transpose(observation_dim, *native_dims)
   basis = basis.transpose(*native_dims, state_dim)

   # xr.dot otherwise uses an inner join and could silently discard cells.
   mean, sensitivity, basis = xr.align(
       mean,
       sensitivity,
       basis,
       join="exact",
       copy=False,
   )

   mean, sensitivity, basis = dask.compute(
       to_dense(mean),
       to_dense(sensitivity),
       to_dense(basis),
   )

``transpose`` states the semantic dimension order and naturally rejects
missing or extra dimensions. Use ``...`` only when extra dimensions are part
of the contract. ``xr.align(..., join="exact", copy=False)`` is the ordinary
strict compatibility check; always use the arrays it returns. Select
``inner``, ``outer``, ``left``, ``right``, or ``override`` only when that join
is an explicit scientific policy, and comment non-obvious choices. Use
``xr.broadcast`` when broadcasting is intentional.

Exact alignment compares indexes when they exist, but does not require every
dimension to have an index. When later code selects or dispatches by label,
require an indexed dimension coordinate once at that public boundary.

Keep labelled xarray operations until labels have performed their alignment
role. Convert to NumPy at a named eager, SciPy, PyMC, or serialization
boundary. Materialize related Dask-backed arrays together so shared graphs are
not executed repeatedly. Indexed dimension coordinates are normally eager;
avoid inspecting lazy auxiliary-coordinate payloads merely for validation.

Units
-----

OpenGHG normally supplies unit metadata, and importing OpenGHG enables
pint-xarray. At the preparation or composition boundary for independently
sourced quantities, quantify them, convert compatible quantities to the
calculation's canonical units, and dequantify if the numerical kernel requires
ordinary arrays. Pint owns dimensional compatibility and conversion; xarray
independently owns coordinate alignment.

After that conversion, a kernel may propagate or derive ``units`` attributes
to document known output algebra (for example, covariance has squared
sensitivity units). Treat those strings as descriptive metadata, not as
dimensional validation. Do not construct unit strings and then revalidate them
throughout a kernel as a substitute for Pint. If a kernel accepts dequantified
values, document their canonical-unit contract and trust the preparation
boundary. Handle missing or incompatible units once where the independent data
is normalized.

Tests rather than production assertions
---------------------------------------

Use tests to establish scientific identities against an independent oracle,
strict and intentionally permissive alignment behaviour, unit conversion and
incompatibility at the unit boundary, lazy execution boundaries, and genuine
algorithm failures such as rank deficiency. Avoid tests whose only purpose is
corrupting a package-created intermediate dataclass unless that object is a
supported external or deserialization boundary.

During review, ask:

* Where was this input first normalized?
* Is this check defending a real boundary or distrusting local construction?
* Can xarray alignment, Pint, static typing, or the numerical operation express
  the requirement directly?
* Does the check unexpectedly compute lazy data?
* Does it obscure the scientific equation?
* Would a test document the invariant more clearly?
