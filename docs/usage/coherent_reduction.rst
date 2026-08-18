Coherent Gaussian Reduction
===========================

The coherent-reduction API turns one labelled native covariance projection
into an exact retained Gaussian model. It is a backend-neutral preparation
operation: PyMC and analytic model builders may consume its result, but the
operation does not construct either backend.

Let the native scaling state and observation model be

.. math::

   x \sim \mathcal N(m, B),
   \qquad
   y = Hx + \epsilon,
   \qquad
   \alpha = \Pi x.

Here :math:`\epsilon` is independent of :math:`x`, and :math:`R` denotes its
valid observation-plus-model-error covariance in the prepared observation
representation.

The low-level :doc:`native_covariance` operation constructs
:math:`C_\alpha = \Pi B\Pi^\mathsf{T}`,
:math:`H_\alpha`, and :math:`HBH^\mathsf{T}`. Coherent reduction constructs
those linked products locally, then returns

.. math::

   H_\alpha = HB\Pi^\mathsf{T}C_\alpha^{-1}

without constructing an inverse, then returns the centred conditional model

.. math::

   y \mid \alpha \sim \mathcal N\!\left(
      Hm + H_\alpha(\alpha - \Pi m),
      R + A
   \right),

.. math::

   A = HBH^\mathsf{T} - H_\alpha C_\alpha H_\alpha^\mathsf{T}.

All three retained quantities---the prior, effective forward operator, and
unresolved covariance---come from the same product set.

Preparing the exact result
--------------------------

Pass the native covariance, canonical basis prolongation, labelled native mean,
and sensitivity through one operation:

.. code-block:: python

   from openghg_inversions.coherent_reduction import reduce_native_gaussian

   reduction = reduce_native_gaussian(
       covariance=covariance,
       basis_prolongation=basis_prolongation,
       state_dim="state",
       native_mean=native_mean,
       native_sensitivity=native_sensitivity,
       observation_dim="observation",
   )

   retained_mean = reduction.retained_mean
   retained_covariance = reduction.retained_covariance
   effective_operator = reduction.effective_observation_operator
   unresolved_covariance = reduction.unresolved_observation_covariance

The operation is a named eager boundary. Related Dask-backed mean, sensitivity,
and basis-prolongation payloads are densified lazily where necessary and
materialized together. The covariance product blocks are built and reduced
locally, so a sensitivity cannot be substituted after projection. The result
fields are eager xarray arrays with explicit retained and observation labels.

Boundary contract and limits
----------------------------

The public operation transposes inputs into their declared scientific roles
and performs one exact xarray alignment before computing. This is deliberate:
xarray dot products otherwise use an inner join and could silently omit native
cells with unmatched labels. Products created by that operation are trusted by
the nearby equation kernel rather than validated a second time.

Unit conversion belongs to the upstream OpenGHG/pint-xarray preparation
boundary. The native mean and basis describe dimensionless scaling, and the
sensitivity must already be expressed in the desired observation units. The
reducer propagates those prepared unit attributes; it does not compare
manufactured unit strings as a substitute for Pint dimensional analysis.
The current single-DataArray interface requires all stacked observations to
have one common prepared unit. Thus the present linked CO2/O2 case may use
jointly converted ppm channels, but ppm CO2 must not yet be stacked directly
with delta(O2/N2) observations expressed in per meg.

The retained restriction must be full rank, so :math:`C_\alpha` is positive
definite. Covariance-product construction uses the existing retained-space
Cholesky factor to estimate its reciprocal condition cheaply and rejects an
effectively rank-deficient retained basis; it does not choose a pseudoinverse.
The unresolved covariance :math:`A` is the exact aggregation-error covariance
for the Gaussian model above. It is symmetrized to remove floating-point
asymmetry, but is not separately eigendecomposed or clipped. Likelihood
construction owns checking/factorizing the scientifically relevant total
:math:`R + A`; a reasonable :math:`R` commonly absorbs harmless roundoff in
:math:`A`, but cannot repair a genuinely unstable retained solve.
Any later low-rank-plus-diagonal approximation should therefore be assessed
through the total likelihood covariance and its log density, especially when
:math:`R` contains little or no model-mismatch error.

This exact contract is Gaussian. Reusing its moments for a LogNormal retained
state requires an explicit moment/linear-Bayes closure, and a positive sampled
state requires a separately established nonnegative physical restriction.
The default covariance-natural restriction is not guaranteed to preserve
positivity.

This operation does not add observation error :math:`R`, construct a
likelihood, approximate :math:`A` as low-rank plus diagonal, serialize the
result, or reconstruct arbitrary native functionals. Those are separate
scientific or persistence boundaries. In particular, durable coherent-result
identity and storage belong to OPE-40. Until then, the in-memory result must be
kept as one handoff: do not substitute :math:`A` independently or add a second
error term representing the same unresolved flux. Adding another covariance
component assumes its error is independent (zero cross-covariance) unless the
cross terms are represented explicitly.

See :mod:`openghg_inversions.coherent_reduction` for the public interface.
