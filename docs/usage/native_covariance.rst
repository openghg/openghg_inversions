Native Covariance Projection
============================

The native covariance projection API prepares labelled covariance and
observation product blocks for reducing a gridded scaling state. It keeps basis
geometry, native covariance, and retained-state semantics separate, and never
constructs a dense native-grid covariance matrix. This low-level operation is
not connected to RHIME input preparation or likelihoods, and
its product blocks are inputs to the separate :doc:`coherent_reduction`
operation rather than a complete coherent-reduction result by themselves.

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
       ``basis_matrix`` is the bucket coarse-to-fine map (prolongation)
       :math:`U_{\mathrm{bucket}}`.
       A gathered multisource matrix is a spatial template; covariance
       projection expands it onto a canonical explicit native source
       dimension through an internal adapter. In both cases, the transpose
       does not define the retained restriction :math:`\Pi`.
   * - ``InvertibleNativeCovarianceAction``
     - Structural interface for a labelled, self-adjoint positive-definite
       :math:`B`, including its compatible inverse solve, without constructing
       a dense native covariance matrix. The calculation trusts this semantic
       contract rather than globally certifying a matrix-free action.
   * - ``SeparableExponentialCovariance``
     - Concrete latitude/longitude covariance action. Optional class labels
       set cross-class covariance to zero; this alone does not assert
       probabilistic independence outside a joint Gaussian model.
   * - ``ClassMixedExponentialCovariance``
     - Class-blocked covariance with a kernel chosen per class: classes named
       in ``class_kernels`` use a dense ``GeodesicExponentialKernel``, the
       others the separable kernel. See :ref:`geodesic-class-blocks`.
   * - ``IndependentSourceCovariance``
     - Composes ordered, source-specific spatial covariances into a
       block-diagonal native covariance. Its blocks must be
       ``SeparableExponentialCovariance`` or
       ``ClassMixedExponentialCovariance`` instances. Covariance application
       and solves use the common action interface; only multisource basis
       expansion needs a separate source-aware path.
   * - ``RetainedProjectionStrategy``
     - Pi-first policy interface that chooses the authoritative labelled
       restriction :math:`\Pi`. Each strategy owns the inputs that define
       :math:`\Pi`; the calculation derives its covariance-natural
       prolongation :math:`U_*`.
   * - ``PreserveBucketProlongation``
     - Compatibility strategy holding a basis operator's
       :math:`U_{\mathrm{bucket}}`. It derives the compatible authoritative
       restriction :math:`\Pi_U`; the calculation still derives :math:`U_*` from
       that returned restriction.
   * - ``SuppliedRestriction``
     - Supplied-:math:`\Pi`-first strategy holding a physical restriction
       chosen independently of :math:`B`. It checks the restriction's labels
       and values, optionally requires it to preserve positivity, and records
       that provenance on the returned restriction.
   * - ``absolute_weighted_regional_mean_restriction``
     - Builds the sparse source-blocked regional mean weighted by
       :math:`|q|`, the production policy described below, for use with
       ``SuppliedRestriction``.
   * - ``RetainedProjection``
     - Frozen value dataclass returned by a strategy. It carries
       :math:`\Pi` and the strategy identifier; it is not a protocol, and its
       contained xarray object remains mutable.
   * - ``project_native_covariance``
     - Validates and combines :math:`H`, :math:`B`, and a projection
       strategy.
   * - ``NativeCovarianceProducts``
     - Frozen in-memory result dataclass containing the labelled product
       blocks. Its contained xarray objects remain mutable. Durable identity,
       schema design, and persistence are not part of this API.

The data flow is:

.. code-block:: text

   BasisOperator ──> U_bucket ──> PreserveBucketProlongation ─┐
   physical Pi ─────────────────> SuppliedRestriction ────────┤
                                                              v
   covariance action ──> B ───────────> RetainedProjectionStrategy
                                             │
                                             v
                              RetainedProjection(Pi)
                                             │
   native sensitivity H ─────────────────────┤
   covariance action B ──────────────────────┤
                                             v
                              project_native_covariance
                                             │
                                             v
                              NativeCovarianceProducts

``FluxWeightedBasis`` is a data-preparation wrapper that pairs an operator
with a flux field for sensitivity projection and flux reconstruction. It does
not define :math:`\Pi`, apply native covariance, or own covariance transforms.
By the time this API is called, :math:`H` already contains footprint times
prior flux, so both the native and retained states are multiplicative
scalings.

Projection and centring
-----------------------

Let the native scaling state have mean :math:`m` and covariance :math:`B`:

.. math::

   \operatorname{E}[x] = m,
   \qquad
   \operatorname{Cov}(x) = B,
   \qquad
   \delta x = x - m.

For a retained restriction :math:`\Pi`, define

.. math::

   \alpha = \Pi x,
   \qquad
   \delta\alpha = \alpha - \Pi m = \Pi\,\delta x,
   \qquad
   C_\alpha = \Pi B \Pi^\mathsf{T}.

The covariance-natural prolongation associated with that restriction is

.. math::

   U_* = B \Pi^\mathsf{T} C_\alpha^{-1}.

The full affine lift is therefore

.. math::

   \widehat{x}(\alpha)
   = m + U_*\delta\alpha
   = m + U_*(\alpha - \Pi m),

not the uncentred expression :math:`U_*\alpha`. For a joint Gaussian model
this lift is :math:`\operatorname{E}[x\mid\alpha]`. Without Gaussianity it is
the linear-Bayes lift determined by the first two moments.

In general, :math:`U_*`, :math:`U_{\mathrm{bucket}}`, and
:math:`\Pi^\mathsf{T}` are different operators. ``PreserveBucketProlongation``
preserves the established bucket-scaling interpretation by choosing

.. math::

   \Pi_U =
   \left(U_{\mathrm{bucket}}^\mathsf{T} B^{-1}
   U_{\mathrm{bucket}}\right)^{-1}
   U_{\mathrm{bucket}}^\mathsf{T} B^{-1}.

This gives :math:`\Pi_U U_{\mathrm{bucket}} = I` and
:math:`U_* = U_{\mathrm{bucket}}`; the latter identity follows from the
calculation's generic derivation :math:`U_* = B\Pi^\mathsf{T}C_\alpha^{-1}`, not
from a strategy-supplied lift. The projection returns :math:`C_\alpha`,
:math:`H U_*`, :math:`H B \Pi^\mathsf{T}`, and either dense
:math:`H B H^\mathsf{T}` or its diagonal.

Supplied physical restriction
-----------------------------

``SuppliedRestriction`` accepts a labelled :math:`\Pi` with dimensions
``(state_dim, *native_dims)``, including an explicit native source dimension
for source-blocked covariance. Its unique state coordinate labels every
retained product, its native coordinates must align exactly with :math:`H`,
and its values must be finite and real. No basis prolongation is needed. Full row rank is checked by
the positive-definite factorization of :math:`C_\alpha`. The calculation then
derives

.. math::

   C_\alpha = \Pi B \Pi^\mathsf{T},
   \qquad
   U_* = B \Pi^\mathsf{T} C_\alpha^{-1},
   \qquad
   H_\alpha = H U_*,

so :math:`\Pi U_* = I` and :math:`B\Pi^\mathsf{T} = U_* C_\alpha`. The
strategy is policy-neutral: it neither builds nor prefers a particular
weighting. Absolute-flux-weighted regional means and signed flux-total
restrictions are both supplied by the caller.

``absolute_weighted_regional_mean_restriction`` builds the current production
policy from per-source integer region maps and signed per-cell weights
:math:`q` (for example prior flux times cell area). For source :math:`s`,
region :math:`r`, and native cell :math:`i`,

.. math::

   \Pi_{(s,r),(s',i)} =
   \mathbf 1[s=s']\,\mathbf 1[i\in r]\,
   \frac{|q_{si}|}{\sum_{j\in r}|q_{sj}|}.

Each row is exactly nonnegative and sums to one by construction, so the result
preserves positivity, as a LogNormal retained state needs, without any
tolerance. Using
absolute weights is not the same as scaling the net signed regional flux. For a
region of two cells with :math:`q = (-2, +1)`, the retained scaling is the
positive weighted mean :math:`\alpha = (2x_1 + x_2)/3`. Scaling the net flux
:math:`-2x_1 + x_2` against its prior value :math:`-1` would instead give
:math:`2x_1 - x_2`, which can be negative even when both native scalings are
positive.

Zero and positive region labels are active, so zero-based labels such as those
from ``range`` work directly; negative labels mark inactive cells, which receive
no weight. Relabel a map that marks inactive cells with zero, for example with
``labels.where(labels > 0, -1)``. The retained state is a
``(source, region_in_source)`` MultiIndex that numbers each source's active
regions ``0, 1, ...`` in ascending label order. For maps without inactive
cells, this numbering matches ``MultiSourceBucketBasisOperator``. The payload is
sparse, with at most one weight per active cell.

For positive native scaling :math:`x`, the retained scaling
:math:`\alpha = \Pi x` is guaranteed positive exactly when :math:`\Pi` is
nonnegative with positive weight in every row; such a :math:`\Pi` preserves
positivity. A LogNormal retained state needs this, so pass
``require_positivity_preserving=True`` for one, and the strategy then refuses a
signed restriction or an all-zero row. By default, signed weights are accepted,
but every restriction still needs full row rank: an all-zero row makes
:math:`C_\alpha` singular, and the retained Cholesky factorization rejects it.
The derived lift :math:`U_*` may be signed in either case, because it is a
linear conditional mean rather than a positive scaling map. The returned
restriction records ``restriction_source="supplied"``, whether positivity
preservation was required, and the ``nonnegative``, ``positive_row_sums``,
``row_normalized``, and ``positivity_preserving`` diagnostics.
Durable persistence of this record belongs to the artifact-I/O layer.

The retained solve also records a cheap LAPACK reciprocal 1-norm condition
estimate while its Cholesky factor is already available. A restriction whose
retained covariance is numerically ill-conditioned is rejected with a
basis-design diagnostic. This is not an eigendecomposition of the
observation-sized covariance, nor a requirement that the unresolved
aggregation covariance be positive definite or Cholesky-factorable before
observation and model-error covariance are added. The unresolved covariance
may be singular, but LRPD preparation and dense aggregation-error resolution
still require it to be positive semidefinite within numerical tolerance.

What projection does not construct
----------------------------------

The exact Gaussian reduction also needs

.. math::

   B_\perp
   = B - U_* C_\alpha U_*^\mathsf{T},

and the centred conditional observation model

.. math::

   y \mid \alpha \sim \mathcal N\!\left(
   Hm + H U_*(\alpha - \Pi m),
   R + H B_\perp H^\mathsf{T}
   \right).

To see where :math:`B_\perp` comes from, write the unresolved native part as

.. math::

   e = x - m - U_*(\alpha - \Pi m).

Then :math:`\operatorname{Cov}(e, \alpha) = B\Pi^\mathsf{T} - U_* C_\alpha = 0`,
so :math:`e` is independent of :math:`\alpha` when :math:`x` is Gaussian, and
:math:`\operatorname{Cov}(e) = B - U_* C_\alpha U_*^\mathsf{T} = B_\perp`.
Observations see :math:`e` through :math:`H`, which gives the unresolved
observation covariance :math:`A = H B_\perp H^\mathsf{T}`.

The :doc:`coherent_reduction` operation owns the first exact solve-based
transformation, centred forward model, and unresolved observation covariance.
These product blocks are inputs to that work; they must not be described or
persisted as though they were already the complete result. Arbitrary
reporting-function products :math:`Q`, low-rank-plus-diagonal numerical views,
and likelihood integration are also outside this API.

.. _geodesic-class-blocks:

Dense geodesic class blocks
---------------------------

``SeparableExponentialCovariance`` uses
:math:`\exp(-|\Delta\phi|/\ell_\phi - |\Delta\lambda|/\ell_\lambda)` in
degrees. That distance is an L1 distance, so diagonal correlations decay
faster than north-south or east-west ones, and a longitude length fixed in
degrees means a shorter east-west length in km at high latitude.
``ClassMixedExponentialCovariance`` replaces the kernel of chosen classes by

.. math::

   K_c(i, j) = \exp\!\left(-\frac{d_{\mathrm{gc}}(i, j)}{\ell_c}\right),

with :math:`d_{\mathrm{gc}}` the haversine great-circle distance between cell
centres on a sphere of radius ``earth_radius_km`` (by default
:math:`111 \cdot 180/\pi` km, so one degree of arc is 111 km, matching the
km-to-degree conversion of separable lengths). This kernel is positive definite
on the sphere for every :math:`\ell_c > 0`, isotropic in local km, and
non-separable. Other classes keep the separable kernel, and cells in different
classes stay uncorrelated:

.. math::

   B = \sigma^2 \sum_c M_c K_c M_c .

.. code-block:: python

   from openghg_inversions.native_covariance import (
       ClassMixedExponentialCovariance,
       GeodesicExponentialKernel,
   )

   gpp = ClassMixedExponentialCovariance(
       lat, lon,
       class_labels=basis_class,            # e.g. 1 inner land, 2 sea, 4xx outer land
       class_kernels={1: GeodesicExponentialKernel(500.0)},
       sigma=0.5,
       latitude_correlation_length=500 / 111,
       longitude_correlation_length=500 / (111 * np.cos(np.deg2rad(50))),
   )
   ff = gpp.with_sigma(0.6)  # same kernel, shared dense factor

Per-source configuration goes through ``IndependentSourceCovariance``, whose
blocks may mix separable and class-mixed sources with their own lengths.

Each dense class of :math:`n` cells is built once and stored as one
Fortran-ordered :math:`n \times n` float64 array, :math:`8n^2` bytes (1.1 GB
for 11,832 cells), that holds :math:`K_c` in its upper triangle and the unit
lower-triangular Cholesky factor in its lower triangle, so no second dense copy
is kept. Construction runs the Cholesky factorization and a LAPACK reciprocal
condition estimate, and raises ``numpy.linalg.LinAlgError`` if the block
is not numerically positive definite; ``geodesic_diagnostics`` reports the
estimate and ``minimum_eigenvalues()`` the smallest eigenvalue by Lanczos
iterations on the cached factor. Apply costs :math:`2n^2` flops per right-hand
side (``dsymm``); solve uses two triangular solves for dense classes and the
matrix-free conjugate gradients of the class-blocked separable action,
restricted to the separable classes. ``with_sigma`` changes the amplitude
without rebuilding the block. ``to_dataset`` records the separable
configuration, class labels and class kernels, keyed by the class labels found
in the grid and ordered by class. ``same_unit_kernel`` (also a method) tells
whether two actions have the same unit-variance kernel, ignoring ``sigma`` and
class-label metadata. ``IndependentSourceCovariance`` writes schema version 2
when any source is class-mixed (version 1, unchanged, otherwise), and on
restore sources with the same unit kernel share one dense factor.

Units and persistence
---------------------

Native and retained scaling perturbations are dimensionless. Accordingly,
:math:`\Pi`, :math:`U_*`, and :math:`C_\alpha` carry units ``1``;
:math:`H U_*` and :math:`H B\Pi^\mathsf{T}` inherit the sensitivity units;
and :math:`H B H^\mathsf{T}` (including its diagonal view) carries their
square as descriptive output metadata. Dimensional compatibility and
conversion of independently sourced quantities belong to the preparation
step and use OpenGHG's Pint registry; these derived attributes are not a
substitute for quantification. This low-level result is in-memory only;
durable identities, schema compatibility, and DataTree/NetCDF I/O are separate
serialization concerns.

Memory use and lazy inputs
--------------------------

The structured covariance action stores axis factors rather than dense native
:math:`B`, but the product calculation is otherwise eager. Callers may pass a
sensitivity with sparse or Dask-backed data; ``project_native_covariance``
densifies and computes it. The covariance action owns the native grid: each
application of :math:`B` aligns its right-hand side exactly, so the
restriction and sensitivity grids are checked where they meet :math:`B`.
A strategy materializes its own inputs:
``PreserveBucketProlongation`` densifies its basis prolongation, whose values
it needs. A restriction may remain sparse or Dask-backed until it is selected,
when it is computed once. A sparse
restriction stays sparse; only one retained-state right-hand-side block at a
time is densified before applying :math:`B`. The computed restriction is reused
across those blocks, avoiding repeated execution of its lazy graph.

For the January production shape, :math:`d = 361` retained rows over
:math:`N = 96\,768` native source cells, a dense float64 restriction would need
about 279 MB. The source-blocked sparse restriction stores at most one weight
per native cell (the January production restriction has 51,806), and each
densified block of 64 rows needs about 50 MB. ``observation_batch_size`` sets
the size of these restriction blocks as well as the observation batches. The dense outputs
:math:`B\Pi^\mathsf{T}` and :math:`U_*` remain about 279 MB each, because a
non-local covariance makes them dense. For
native size :math:`N`, retained size
:math:`d`, and observation count :math:`M`, important dense storage includes
:math:`M N` for :math:`H`, :math:`N d` for each native-by-retained array,
:math:`d^2` for retained products, and either :math:`M^2` for dense
:math:`H B H^\mathsf{T}` or :math:`M` for its diagonal.

``observation_batch_size`` is an eager execution setting, not an xarray or
Dask chunk size. Each batch applies :math:`B` to a group of columns from
:math:`H^\mathsf{T}`, limiting the temporary :math:`N`-by-batch working set.
The same setting bounds explicit custom-restriction right-hand-side blocks.
Dense batches still fill a preallocated complete quadratic observation
covariance. Changing the batch size changes execution only, not the scientific
inputs or requested numerical form.

Correspondence with ``verification-games``
-------------------------------------------

This API ports only the reusable lower-level prototype calculation.
The recent 14-site production runs define retained coefficients with a
source-blocked, absolute-prior-flux-weighted mean: the regional mean above with

.. math::

   q_{si} = \bar F^{\mathrm{UOB\,BASE}}_{si} A_i.

Each row averages native scaling factors within exactly one source and one
active region. If a production map marks inactive cells with zero, relabel
them to a negative value before calling the builder. This physical restriction is
chosen independently of :math:`B`; the covariance then determines
:math:`C_\alpha` and :math:`U_*`. The prototype also supports a distinct
signed-flux-total policy, but that is not the current 14-site production
setting; it remains an alternative policy that a caller can supply to
``SuppliedRestriction``. Both policies use the supplied-:math:`\Pi`-first
strategy.

.. list-table::
   :header-rows: 1
   :widths: 31 21 48

   * - Prototype capability
     - Current status
     - Correspondence
   * - Separable exponential covariance action, optional class blocking, and
       batched :math:`\Pi B\Pi^\mathsf{T}`,
       :math:`H B\Pi^\mathsf{T}`, and :math:`H B H^\mathsf{T}` products
     - Faithful port
     - Re-expressed as labelled xarray actions and product objects; the API also
       supports ordered independent-source blocks.
   * - Choose a physical restriction :math:`\Pi` first, then derive its
       covariance-natural lift. Current 14-site production uses the
       absolute-prior-flux-weighted regional mean above.
     - Faithful port
     - ``SuppliedRestriction`` accepts the same physical :math:`\Pi` and
       reproduces the prototype's :math:`C_\alpha`, :math:`H_\alpha`, and
       unresolved observation covariance for separable and class-blocked
       covariance. The compatibility-oriented ``PreserveBucketProlongation``
       remains available for bucket bases.
   * - Cross-source covariance, arbitrary reporting functionals :math:`Q`,
       :math:`B_\perp` products, coherent likelihood reduction, reconstruction,
       and covariance approximation
     - Not ported
     - These remain prototype or downstream work and must not be inferred from
       the presence of covariance product blocks.

API reference
-------------

See :mod:`openghg_inversions.native_covariance`,
:mod:`openghg_inversions.source_covariance`, and
:mod:`openghg_inversions.basis.covariance_products` for the public interfaces.
