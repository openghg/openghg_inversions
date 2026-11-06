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
  batch size before batching, and let a required Cholesky factorization report
  a matrix which cannot be used as positive definite; and
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

Make established contracts visible in types
------------------------------------------

Use a named contract at a handoff where an ordinary ``xr.Dataset`` annotation
leaves important structure implicit. Keep the contract description and its
validation together. Describe the variables, dimensions, indexed coordinates,
units and relationships that consumers can assume, distinguishing structural
guarantees from numerical checks that await materialization.

The :download:`explicit data contracts proposal
<../../openspec/changes/explicit-data-contracts/specs/explicit-data-contracts/spec.md>`
captures this pattern; it does not indicate that contract types have already
been implemented. The following pseudocode illustrates the recommended static
marker. ``check_bound_layout`` and ``select_and_expand_transport`` stand for
domain-specific validation and preparation, not existing package functions.

.. code-block:: python

   from typing import cast

   import xarray as xr

   class BoundTransport(xr.Dataset):
       """Static contract; runtime values remain ordinary Datasets."""

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

At an existing boundary accepting independently supplied data, call the
validator and use its typed return value. A trusted producer instead establishes
the contract through its preparation operations. ``cast`` only records that
claim for the type checker; it performs no validation. A validator returning
``None`` does not by itself narrow the caller's variable.

Keep ordinary Dataset or DataTree representations when their shared coordinates,
alignment and joint selection are useful. A static marker requires no runtime
wrapper or special constructor. Do not attach new instance behavior to a phantom
subtype or use ``isinstance`` to decide whether its contract holds.

Package-owned transformations must preserve or establish the contract they
return. Inherited xarray ``Self`` annotations do not prove that a selection,
rename or reduction preserves the schema. Use tests of supported transformations
to establish this, rather than complete validation in every ``__post_init__``
or dataclass reconstruction. Mutating an object so that its declared contract no
longer holds is caller error; interception or mutation warnings are not required.

Use this pattern where it clarifies a real handoff, rather than introducing a
special type for every intermediate. Keep normalization, alignment and unit
conversion explicit, following the guidance below. Structural validation
preserves lazy payloads; finite-value and other numerical checks belong to their
named execution boundary.

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
