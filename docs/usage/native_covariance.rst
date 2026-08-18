Native Covariance Projection
============================

The native covariance projection API prepares labelled covariance and
observation product blocks for reducing a gridded scaling state. It keeps basis
geometry, native covariance, and retained-state semantics separate, and never
constructs a dense native-grid covariance matrix. This is the low-level OPE-17
boundary: it is not connected to RHIME input preparation or likelihoods, and
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
       ``basis_matrix`` is the bucket prolongation :math:`U_{\mathrm{bucket}}`.
       A gathered multisource matrix is a spatial template; the basis-side
       ``native_prolongation`` adapter expands it onto a canonical explicit
       native source dimension. In both cases, the transpose does not define
       the retained restriction :math:`\Pi`.
   * - ``InvertibleNativeCovarianceAction``
     - Structural interface for a labelled, self-adjoint positive-definite
       :math:`B`, including its compatible inverse solve, without constructing
       a dense native covariance matrix. The kernel trusts this semantic
       contract rather than globally certifying a matrix-free action.
   * - ``SeparableExponentialCovariance``
     - Concrete latitude/longitude covariance action. Optional class labels
       set cross-class covariance to zero; this alone does not assert
       probabilistic independence outside a joint Gaussian model.
   * - ``IndependentSourceCovariance``
     - Composes ordered, source-specific spatial covariances into a
       block-diagonal native covariance. Its blocks currently must be
       ``SeparableExponentialCovariance`` instances. Covariance application
       and solves use the common action interface; only multisource basis
       expansion needs a separate source-aware path.
   * - ``RetainedProjectionStrategy``
     - Pi-first policy interface that chooses the authoritative labelled
       restriction :math:`\Pi`. The kernel derives its covariance-natural
       prolongation :math:`U_*`.
   * - ``PreserveBucketProlongation``
     - Current strategy implementation. It uses
       :math:`U_{\mathrm{bucket}}` to derive the compatible authoritative
       restriction :math:`\Pi_U`; the kernel still derives :math:`U_*` from
       that returned restriction.
   * - ``RetainedProjection``
     - Frozen value dataclass returned by a strategy. It carries
       :math:`\Pi` and the strategy identifier; it is not a protocol, and its
       contained xarray object remains mutable.
   * - ``project_native_covariance``
     - Validates and combines :math:`H`, :math:`B`, basis geometry, and a
       projection strategy.
   * - ``NativeCovarianceProducts``
     - Frozen in-memory result dataclass containing the labelled product
       blocks. Its contained xarray objects remain mutable. Durable identity,
       schema design, and persistence are deferred to OPE-40.

The data flow is:

.. code-block:: text

   BasisOperator ──> U_bucket ─┐
                              ├─> RetainedProjectionStrategy
   covariance action ──> B ───┘              │
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
:math:`\Pi^\mathsf{T}` are different operators. The initial strategy preserves
the established bucket-scaling interpretation by choosing

.. math::

   \Pi_U =
   \left(U_{\mathrm{bucket}}^\mathsf{T} B^{-1}
   U_{\mathrm{bucket}}\right)^{-1}
   U_{\mathrm{bucket}}^\mathsf{T} B^{-1}.

This gives :math:`\Pi_U U_{\mathrm{bucket}} = I` and
:math:`U_* = U_{\mathrm{bucket}}`; the latter identity follows from the
kernel's generic derivation :math:`U_* = B\Pi^\mathsf{T}C_\alpha^{-1}`, not
from a strategy-supplied lift. OPE-17 returns :math:`C_\alpha`,
:math:`H U_*`, :math:`H B \Pi^\mathsf{T}`, and either dense
:math:`H B H^\mathsf{T}` or its diagonal.

The retained solve also records a cheap LAPACK reciprocal 1-norm condition
estimate while its Cholesky factor is already available. A restriction whose
retained covariance is effectively rank deficient at float64 precision is
rejected with a basis-design diagnostic. This is not an eigendecomposition of
the observation-sized covariance, nor a requirement that the unresolved
aggregation covariance be independently factorable before observation and
model-error covariance are added.

What OPE-17 does not construct
------------------------------

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

The :doc:`coherent_reduction` operation owns the first exact solve-based
transformation, centred forward model, and unresolved observation covariance.
The OPE-17 product blocks are inputs to that work; they must not be described or
persisted as though they were already the complete result. Arbitrary
reporting-function products :math:`Q`, low-rank-plus-diagonal numerical views,
and likelihood integration are also outside this API.

Units and persistence boundary
------------------------------

Native and retained scaling perturbations are dimensionless. Accordingly,
:math:`\Pi`, :math:`U_*`, and :math:`C_\alpha` carry units ``1``;
:math:`H U_*` and :math:`H B\Pi^\mathsf{T}` inherit the sensitivity units;
and :math:`H B H^\mathsf{T}` (including its diagonal view) carries their
square as descriptive output metadata. Dimensional compatibility and
conversion of independently sourced quantities belong to the preparation
boundary and use OpenGHG's Pint registry; these derived attributes are not a
substitute for quantification. This low-level result is in-memory only. OPE-40 owns durable identities,
typed coordinate persistence, schema compatibility, and DataTree/NetCDF I/O.

Memory and execution boundary
-----------------------------

The structured covariance action stores axis factors rather than dense native
:math:`B`, but the numerical product kernel is otherwise eager. The upstream
pipeline may pass a related sensitivity and canonical basis prolongation with
sparse or Dask-backed payloads; the explicit projection boundary densifies
them where needed and materializes them together. A custom restriction may
also remain sparse or Dask-backed until it is selected, when it is materialized
and densified once. The eager restriction is then reused across retained-state
right-hand-side blocks, avoiding repeated execution of its lazy graph. For
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

OPE-17 deliberately ports only the reusable lower-level prototype boundary.
The recent 14-site production runs define retained coefficients with a
source-blocked, absolute-prior-flux-weighted mean. For source :math:`s`, region
:math:`r`, and native cell :math:`i`, let

.. math::

   q_{si} = \bar F^{\mathrm{UOB\,BASE}}_{si} A_i,
   \qquad
   \Pi_{(s,r),(s',i)} =
   \mathbf 1[s=s']\,\mathbf 1[i\in r]\,
   \frac{|q_{si}|}{\sum_{j\in r}|q_{sj}|}.

Thus each row is nonnegative, sums to one, and averages native scaling factors
within exactly one source and one active region. This physical restriction is
chosen independently of :math:`B`; the covariance then determines
:math:`C_\alpha` and :math:`U_*`. The prototype also supports a distinct
signed-flux-total policy, but that is not the current 14-site production
setting. Both policies require a supplied-:math:`\Pi`-first OGI strategy.

.. list-table::
   :header-rows: 1
   :widths: 31 21 48

   * - Prototype capability
     - OPE-17 status
     - Correspondence
   * - Separable exponential covariance action, optional class blocking, and
       batched :math:`\Pi B\Pi^\mathsf{T}`,
       :math:`H B\Pi^\mathsf{T}`, and :math:`H B H^\mathsf{T}` products
     - Faithful port
     - Re-expressed as labelled xarray actions and product objects; OPE-17 also
       supports ordered independent-source blocks.
   * - Choose a physical restriction :math:`\Pi` first, then derive its
       covariance-natural lift. Current 14-site production uses the
       absolute-prior-flux-weighted regional mean above.
     - Deliberate difference
     - The first OPE-17 strategy is compatibility-oriented: it uses
       :math:`U_{\mathrm{bucket}}` internally to derive :math:`\Pi_U`. The
       strategy interface itself is Pi-first, so later policies may supply a
       physical restriction directly.
   * - Cross-source covariance, arbitrary reporting functionals :math:`Q`,
       :math:`B_\perp` products, coherent likelihood reduction, reconstruction,
       and covariance approximation
     - Not ported
     - These remain prototype or downstream OPE-18 work and must not be inferred
       from the presence of OPE-17 product blocks.

API reference
-------------

See :mod:`openghg_inversions.native_covariance`,
:mod:`openghg_inversions.source_covariance`, and
:mod:`openghg_inversions.basis.covariance_products` for the public interfaces.
