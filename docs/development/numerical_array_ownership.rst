Numerical array ownership and execution
=======================================

Xarray objects carry labels, metadata, and a numerical backend. That backend
may be an eager NumPy or sparse array, or a Dask computation that has not run
yet. Understanding ownership and execution separately makes scientific code
easier to read and avoids surprising memory use.

This page explains those distinctions. The :doc:`rhime_model_development`
guide covers scientific composition, and :doc:`validation_and_xarray` explains
where alignment, units, and input checks belong. Reviewed capability
specifications define the behavior promised by a particular API or artifact.

Borrowing is different from a snapshot
--------------------------------------

A borrowed array is an existing object used without taking ownership of its
storage. Callers can inspect it and construct derived arrays; they must respect
the owner's mutation contract. Ordinary access to a borrowed array does not
promise an independent snapshot.

A shallow xarray copy can separate attributes or container structure while
sharing numerical storage. A NumPy copy normally allocates new numeric
storage, whereas a Dask copy can still share graph keys and upstream work.
``copy(deep=True)`` therefore does not have one universal meaning across
backends, and a frozen dataclass does not freeze the arrays it contains.

An independent snapshot is a stronger promise: the API has deliberately
materialized or copied the required data and specifies what is independent.
Copying small coordinate vectors can protect a cached numerical factor from
later aliasing. Repeatedly copying a large scientific field on property access
adds cost without necessarily creating independent lazy storage.

Labels can be checked before values are computed
------------------------------------------------

Indexed dimension coordinates normally have eager labels because xarray uses
them for selection and alignment. Checking an ordinary indexed ``time``,
``site``, ``lat``, or ``lon`` coordinate is different from computing a data
variable. Auxiliary coordinates can remain lazy; inspecting their full values
can execute a graph.

Xarray transpose and alignment preserve labelled scientific meaning before an
eager numerical boundary. For example, independently prepared quantities can
be aligned with ``xr.align(..., join="exact", copy=False)`` before contraction.
The arrays returned by alignment are the ones used in the calculation. Details
and examples are in :doc:`validation_and_xarray`.

``.data`` exposes the underlying array backend. ``.to_numpy()`` and
``.as_numpy()`` make a NumPy conversion explicit; ``.values`` can obscure that
intent. Whether a conversion executes work depends on the backend and on which
array or coordinate is being inspected.

Sparse storage and lazy execution are separate
----------------------------------------------

A sparse array can already be computed, and a dense array can still be lazy.
:func:`~openghg_inversions.array_ops.to_dense` handles these cases explicitly:

* An eager sparse payload becomes a dense NumPy-backed DataArray.
* Sparse Dask chunks are converted by a lazy block operation. The outer Dask
  collection and chunk layout are retained until execution.
* A Dask array with dense chunks is returned unchanged.

Consequently, ``compute()`` alone need not produce NumPy data: it can return
an eager sparse payload. A scalar reduction can be computed and then densified
before extracting a Python scalar, without densifying the original field.
Likewise, a writer that needs dense chunk payloads can receive lazily densified
chunks without forcing the whole field into memory before writing.

Related computations can share work
-----------------------------------

Dask collections often share retrieval, filtering, alignment, and contraction
tasks. Computing each result separately can execute those shared tasks several
times. A workflow can submit the related arrays together at the boundary that
needs concrete values:

.. code-block:: python

   import xarray as xr
   from dask import compute
   from openghg_inversions.array_ops import to_dense

   mean, sensitivity = xr.align(mean, sensitivity, join="exact", copy=False)
   mean, sensitivity = compute(to_dense(mean), to_dense(sensitivity))

The compute call is an explicit execution boundary. It does not replace the
caller's borrowed arrays with the returned values. Selecting the required
arrays first also avoids computing unrelated extension variables merely
because they share a Dataset.

Joint computation, persistence, and a working cache are different choices.
Persistence retains computed intermediates for reuse; an on-disk working cache
adds storage, invalidation, and failure-handling decisions. Neither follows
automatically from using Dask. The original Zarr-cache proposal remains an
unadopted experiment in the :download:`dated numerical ownership investigation
<../plans/numerical_data_ownership_and_execution_boundaries.md>`.

An eager kernel has an explicit memory boundary
------------------------------------------------

PyMC data registration, a SciPy factorization, and a serialization writer are
examples of boundaries that may require concrete or backend-compatible data.
The owning operation determines which quantities are needed and when they
become resident. Labels remain useful until alignment and scientific selection
are complete.

Dask chunks participate in a scheduler graph. A manual loop over an already
materialized NumPy array only limits additional workspace; it does not restore
out-of-core execution. Describing an algorithm as eager and blocked is clearer
than implying that its loop is still a Dask computation.

Artifact ownership is also explicit. A lazy writer can execute the graph while
saving. An eager loader can load inside the file context and return arrays that
do not depend on a closed file. The artifact owner still defines its schema,
index restoration, metadata meaning, and compatibility policy; generic storage
mechanics do not supply those scientific decisions.

Evidence for execution behavior
--------------------------------

Numerical equality alone cannot show whether an accessor unexpectedly computed
an array, changed chunks, or repeated shared work. Where execution is part of
a supported contract, focused tests can observe Dask tasks, retained input
backends and chunks, explicit materialization, and on-disk round trips.
Measurements of peak memory and task counts complement value checks for large
workflows. Their fixture, revision, and execution boundary matter; a small
benchmark does not establish a production memory guarantee.
