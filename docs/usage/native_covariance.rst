Native Covariance Projection
============================

The native covariance projection API prepares the covariance and observation
products needed to reduce a gridded scaling state. It keeps three concerns
separate: basis geometry, native covariance, and the scientific definition of
the retained state. It is currently a low-level preparation API; integration
with the RHIME likelihood is follow-up work.

Component overview
------------------

The classes have the following distinct roles.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Component
     - Responsibility
   * - ``BasisOperator``
     - Owns basis geometry and retained-state labels. For a single source,
       ``basis_matrix`` is the bucket prolongation :math:`U_{\mathrm{bucket}}`.
       A gathered multisource matrix is a spatial template that the projection
       code expands onto an explicit native source dimension.
   * - ``InvertibleNativeCovarianceAction``
     - Structural interface for applying :math:`B` and solving systems in
       :math:`B` without constructing a dense native covariance matrix.
   * - ``SeparableExponentialCovariance``
     - Concrete latitude/longitude covariance action. Optional class labels
       make cells in different classes uncorrelated.
   * - ``IndependentSourceCovariance``
     - Composes ordered, source-specific spatial covariances into a
       block-diagonal native covariance. Its blocks currently must be
       ``SeparableExponentialCovariance`` instances. It satisfies the same
       covariance action interface, so product calculations use the common
       :math:`B` apply/solve path; multisource basis geometry is still expanded
       separately.
   * - ``RetainedProjectionStrategy``
     - Policy interface that chooses a coherent restriction :math:`\Pi` and
       covariance-natural prolongation :math:`U_*`.
   * - ``PreserveBucketProlongation``
     - Current structural implementation of the strategy interface. It fixes
       :math:`U_* = U_{\mathrm{bucket}}` and derives the compatible
       :math:`\Pi_U`.
   * - ``RetainedProjection``
     - Frozen value dataclass returned by a strategy. It carries
       :math:`(\Pi, U_*)` and the strategy identifier; it is not a protocol.
       Its contained xarray objects remain mutable.
   * - ``project_native_covariance``
     - Validates and combines :math:`H`, :math:`B`, the basis geometry, and a
       projection strategy.
   * - ``NativeCovarianceProducts``
     - Frozen result and serialization dataclass containing the labelled
       reduced covariance and observation products. Its contained xarray
       objects and provenance mapping remain mutable.

The data flow is:

.. code-block:: text

   BasisOperator ──> U_bucket ─┐
                              ├─> RetainedProjectionStrategy
   covariance action ──> B ───┘              │
                                             v
                              RetainedProjection(Pi, U_*)
                                             │
   native sensitivity H ─────────────────────┤
   covariance action B ──────────────────────┤
                                             v
                              project_native_covariance
                                             │
                                             v
                              NativeCovarianceProducts

``FluxWeightedBasis`` remains a data-preparation wrapper that pairs a basis
operator with a flux field. It does not own :math:`\Pi`. By the time this API
is called, :math:`H` already contains footprint times prior flux, so both the
native state and retained state are multiplicative scalings.

Projection notation
-------------------

For a centred native perturbation :math:`x`, retained coefficients are

.. math::

   \alpha = \Pi x,

with retained covariance

.. math::

   C_\alpha = \Pi B \Pi^\mathsf{T}.

The covariance-natural prolongation associated with a chosen restriction is

.. math::

   U_* = B \Pi^\mathsf{T} C_\alpha^{-1}.

In general, :math:`U_*`, :math:`U_{\mathrm{bucket}}`, and
:math:`\Pi^\mathsf{T}` are different operators. The initial strategy preserves
the established bucket-scaling interpretation by choosing

.. math::

   \Pi_U =
   \left(U_{\mathrm{bucket}}^\mathsf{T} B^{-1}
   U_{\mathrm{bucket}}\right)^{-1}
   U_{\mathrm{bucket}}^\mathsf{T} B^{-1}.

This gives :math:`\Pi_U U_{\mathrm{bucket}} = I` and
:math:`U_* = U_{\mathrm{bucket}}`. The resulting products are
:math:`C_\alpha`, :math:`H U_*`, :math:`H B \Pi^\mathsf{T}`, and either dense
:math:`H B H^\mathsf{T}` or its diagonal.

Observation batching
--------------------

``observation_batch_size`` is an eager execution setting, not an xarray or
Dask chunk size. The native sensitivity has already been materialized before
batching begins. Each batch applies :math:`B` to a group of columns from
:math:`H^\mathsf{T}` and then forms the corresponding output block. This limits
the temporary native-grid-by-observation working set.

The dense result is still concatenated into a complete
:math:`H B H^\mathsf{T}` matrix and therefore requires quadratic
observation-space storage. Requesting the diagonal avoids that dense output.
The batch size is an execution choice and does not participate in the
scientific content identity.

API reference
-------------

See :mod:`openghg_inversions.native_covariance`,
:mod:`openghg_inversions.source_covariance`, and
:mod:`openghg_inversions.basis.covariance_products` for the public interfaces.
