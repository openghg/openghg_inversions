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

The :doc:`native_covariance` projection constructs
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

The current function accepts a basis prolongation :math:`U`. With the default
strategy, it constructs the compatible restriction :math:`\Pi_U` using
:math:`B`, so that the covariance-derived prolongation :math:`U_*` equals the
supplied basis prolongation. A custom projection strategy can already choose
:math:`\Pi`, although callers must still supply the basis argument. A future
interface may accept either operator directly and construct its compatible
counterpart. The restriction and prolongation are related through :math:`B`;
they are not generally transposes of one another.

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

Calling ``reduce_native_gaussian`` computes any Dask-backed mean, sensitivity,
and basis-prolongation data. Related arrays are computed together, and the
returned xarray arrays are eager with explicit retained-state and observation
labels.

Preparing CO2 model inputs
--------------------------

The CO2-only model has a dedicated public handoff that keeps the linked
reduction products together with canonical RHIME observations, basis data,
and site metadata. Load the canonical inputs from a checkpoint produced by
the :doc:`rhime` workflow::

   from openghg_inversions.inversion_data import RhimePreparedInputs
   from openghg_inversions.rhime.co2 import prepare_co2_inputs

   canonical_inputs = RhimePreparedInputs.load("base-prepared-inputs.zarr")
   co2_inputs = prepare_co2_inputs(canonical_inputs, reduction)

The returned
:class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact uses at most
512 low-rank modes by default, capped at the observation count. Pass
``aggregation_error_rank=None`` to keep
``reduction.unresolved_observation_covariance`` as an exact dense aggregation
covariance. The artifact records one representation, and the CO2 runners use
it without a separate mode argument.
This handoff does not alter the prepared-input contracts of the standard,
multisector, or ``run_hbmcmc.py`` paths.

For larger cases, request an explicit low-rank-plus-diagonal (LRPD) rank at the
handoff::

   co2_inputs = prepare_co2_inputs(
       canonical_inputs,
       reduction,
       aggregation_error_rank=40,
   )

This factorization is downstream of coherent reduction: it does not make the
reduction approximate or change the retained prior and effective operator.
Let :math:`n` be the observation count and :math:`r` the retained rank. The
current preparation path first materializes the complete dense
:math:`n \times n` covariance and computes a full dense eigendecomposition,
requiring :math:`O(n^2)` peak storage and :math:`O(n^3)` decomposition work.
The saved factor and diagonal use :math:`O(nr+n)` storage. For an ordinary
diagonal-plus-factor likelihood, Woodbury evaluation uses
:math:`O(nr^2+r^3)` work. Reducing :math:`r` therefore shrinks the artifact and
downstream structured-likelihood work; it does not make preparation
matrix-free or avoid the dense eigendecomposition.

The default 512-mode cap is not an adequacy claim; selecting or overriding the
LRPD rank remains a caller-owned numerical and scientific decision. LRPD
preparation preserves the dense covariance diagonal up to accepted roundoff
and diagonal-tail clipping, reported by ``diagonal_preservation_error``. It
does not automatically certify that a rank is adequate for a particular
likelihood. Compare the approximate and dense total likelihood covariance or
log density over representative error profiles.
Inspect the recorded retained-spectrum and reconstruction metrics directly::

   diagnostics = co2_inputs.provenance["aggregation_error"]
   retained_fraction = diagnostics["retained_positive_spectral_fraction"]
   relative_error = diagnostics["relative_frobenius_reconstruction_error"]
   diagonal_error = diagnostics["diagonal_preservation_error"]

These diagnostics describe the covariance approximation; they do not certify
that the chosen rank is adequate for a particular likelihood.

Assumptions and limitations
---------------------------

The native coordinates of the mean, sensitivity, and basis prolongation must
match exactly. The function checks this before computing because xarray dot
products would otherwise use an inner join and could silently omit unmatched
native cells.

The native mean and basis describe dimensionless scaling factors. Convert the
sensitivity to the required observation units with OpenGHG's Pint registry
before calling the function. All rows of the current observation DataArray
share one ``units`` attribute, so differently represented observations must
not be stacked directly. For example, ppm rows cannot currently be combined
with delta(O2/N2) rows expressed in per meg.

The retained states must not be redundant. The projection obtains a cheap
condition estimate while factorizing :math:`C_\alpha` and rejects a numerically
ill-conditioned retained covariance. This usually means that two or more
retained basis states describe nearly the same native variation.

For the Gaussian model above, :math:`A` is the aggregation-error covariance.
The function does not add observation or model-error covariance :math:`R`;
likelihood construction must use the total :math:`R + A`. The reduction does
not validate :math:`A` independently. When :math:`A` is converted to LRPD
during CO2 preparation, or resolved as dense input at runner and serialization
boundaries, it must be symmetric and positive semidefinite within its
scale-based numerical tolerance; it may be singular. :math:`R` is not used to
rescue a materially indefinite :math:`A`. LRPD preparation retains only modes
above that tolerance and carries the discarded marginal variance in the
diagonal tail. Small negative tail entries caused by accepted roundoff are
clipped to zero. Assess any low-rank-plus-diagonal approximation using the
total likelihood covariance and its log density, particularly when
model-mismatch error is small.

The conditional model is exact for a Gaussian native state and error
independent of that state. Using the resulting moments with a LogNormal
retained state is a moment-matched approximation. A positive sampled state
requires a separately established nonnegative physical restriction; the
default restriction is not guaranteed to preserve positivity.

The function does not construct a likelihood, approximate :math:`A`, serialize
the result, or reconstruct arbitrary native quantities. Do not add another
error term representing the same unresolved flux. Adding covariance components
also assumes that their errors are independent unless their cross-covariances
are included explicitly.

See :mod:`openghg_inversions.coherent_reduction` for the public interface.
