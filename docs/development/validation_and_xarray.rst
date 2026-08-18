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

Do not reproduce unit algebra with attribute-string construction and then
revalidate those derived strings throughout a kernel. If a kernel accepts
dequantified values, document their canonical-unit contract and trust the
preparation boundary. Handle missing or incompatible units once where the
independent data is normalized.

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
