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

The retained projection strategy supplies :math:`\Pi`, by one of two routes:

* ``PreserveBucketProlongation(U)`` holds a basis prolongation :math:`U` and
  constructs the compatible restriction :math:`\Pi_U` using :math:`B`, so that
  the covariance-derived prolongation :math:`U_*` equals :math:`U`.
* ``SuppliedRestriction(Pi)`` holds a physical restriction chosen
  independently of :math:`B`; no basis prolongation is needed.

Either way, the restriction and prolongation are related through :math:`B`;
they are not generally transposes of one another.

Preparing the exact result
--------------------------

Pass the native covariance, a retained projection strategy, labelled native
mean, and sensitivity through one operation. The bucket strategy holds a basis
operator's canonical prolongation; ``SuppliedRestriction`` instead holds a
physical restriction (see :doc:`native_covariance`):

.. code-block:: python

   from openghg_inversions.basis import PreserveBucketProlongation
   from openghg_inversions.coherent_reduction import reduce_native_gaussian

   reduction = reduce_native_gaussian(
       covariance=covariance,
       strategy=PreserveBucketProlongation(basis_prolongation),
       state_dim="state",
       native_mean=native_mean,
       native_sensitivity=native_sensitivity,
       observation_dim="observation",
   )

   retained_mean = reduction.retained_mean
   retained_covariance = reduction.retained_covariance
   effective_operator = reduction.effective_observation_operator
   unresolved_covariance = reduction.unresolved_observation_covariance

Calling ``reduce_native_gaussian`` computes any Dask-backed mean and
sensitivity together. The strategy computes its own inputs, and the
returned xarray arrays are eager with explicit retained-state and observation
labels.

Preparing CO2 model inputs
--------------------------

The CO2-only model has a dedicated public handoff that keeps the linked
reduction products together with canonical RHIME observations, basis data,
and site metadata. Load the canonical inputs from a checkpoint produced by
the :doc:`rhime` workflow::

   from openghg_inversions.inversion_data import PreparedInputs
   from openghg_inversions.rhime.co2 import prepare_co2_inputs

   canonical_inputs = PreparedInputs.load("base-prepared-inputs.zarr")
   co2_inputs = prepare_co2_inputs(canonical_inputs, reduction)

The returned
:class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact uses at most
512 low-rank modes by default, capped at the observation count. Pass
``aggregation_error_rank=None`` to keep
``reduction.unresolved_observation_covariance`` as an exact dense aggregation
covariance. The artifact records one representation, and the CO2 runners use
it without a separate mode argument.
This handoff does not alter the prepared-input contracts of the standard,
or multisector paths.

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

Marginalising low-information states
------------------------------------

Some retained states barely affect the observations. A state with a tiny
signal and a wide LogNormal prior can still sit far in the prior tail during
sampling, where the signal becomes large and the sampler diverges. Such states
can be integrated out before sampling with an information threshold on
:class:`~openghg_inversions.models.StateActivity`.

For retained prior standard deviation :math:`s_j = \sqrt{(C_\alpha)_{jj}}`
and prior observation-error or model-mismatch variance :math:`\sigma_o^2`,
the prior-predicted information of state :math:`j` is

.. math::

   I_j = \sum_o \frac{\left((H_\alpha)_{oj}\, s_j\right)^2}{\sigma_o^2}.

It uses only the prior and the operator, never observed values. States with
:math:`I_j < I_{\min}` are marginalised. Split the retained states into kept
:math:`K` and marginalised :math:`D`, with
:math:`G = C_{DK} C_{KK}^{-1}` and Schur complement
:math:`S = C_{DD} - G C_{KD}`. The exact model given :math:`\alpha_K` is

.. math::

   y \mid \alpha_K \sim \mathcal N\!\left(
      Hm + H'_K(\alpha_K - \Pi_K m),\; R + A + H_D S H_D^\mathsf{T}
   \right),
   \qquad
   H'_K = H_K + H_D G.

This is the coherent reduction with restriction :math:`\Pi_K`; when the
marginalised states are prior-uncorrelated with the kept ones it equals
reducing again with the smaller bucket basis. The prior predictive covariance
of :math:`y` is unchanged. The returned reduction keeps the full state axis
and sets the marginalised effective-operator columns to zero, so model
construction removes them in the same way as zero-sensitivity states::

   from openghg_inversions.coherent_reduction import marginalise_low_information_states
   from openghg_inversions.models import StateActivity

   selection = marginalise_low_information_states(
       reduction,
       observation_variance=mismatch_variance,  # sigma_o**2 on the observation axis
       policy=StateActivity(group_coord="source", min_information={"o2_ocean": 0.1}),
   )
   co2_inputs = prepare_co2_inputs(
       canonical_inputs,
       selection.reduction,
       information_selection=selection,
   )

``min_information`` is either one number, applied to every state, or a
mapping from labels of ``group_coord`` to thresholds. A mapping applies the
threshold only to those groups. For example, it can screen only ocean states,
with separate CO2-ocean and O2-ocean groups in linked CO2/O2 models. The
default ``None`` keeps every state, and the input reduction is returned
unchanged. :func:`~openghg_inversions.models.resolve_state_activity` rejects
a policy that still carries a threshold, because fixing a state at model time
would discard its variance instead of moving it into :math:`A`.

``selection.information``, ``selection.threshold``,
``selection.marginalised`` and ``selection.unresolved_variance_increase``
(the change in :math:`\operatorname{diag} A`) are returned with the
reduction. ``selection.to_dataset()`` collects them for saving, and
``selection.provenance`` summarizes them as JSON. With
``information_selection``, the CO2 handoff saves them as
``state_information``, ``state_information_threshold``,
``state_information_marginalised`` and
``aggregation_error_variance_increase``. It also records the summary under
``provenance["information_threshold"]`` and marks marginalised states
inactive, with fixed value equal to their retained prior mean. Linked CO2/O2
preparation receives covariance blocks rather than a reduction, so pass
``selection.provenance`` in its ``provenance`` argument.

Marginalised regions receive no posterior update. Retained-state flux and
country reconstructions hold them at the prior mean. When they are
prior-correlated with kept states, the exact conditional mean would also
regress them on the kept states, and the reconstruction leaves that term out.
Country uncertainty computed only from retained states omits the
marginalised states' variance, because that variance is now in :math:`A`.
Use :math:`A`-based functionals when the uncertainty of those regions
matters. Calibrate :math:`I_{\min}` on synthetic observations: an information of
``0.1`` means one prior standard deviation of the state changes the
observations by about a third of one noise standard deviation in total.

Assumptions and limitations
---------------------------

The native coordinates of the mean and sensitivity must match exactly. The
function checks this before computing them, because xarray dot products would
otherwise use an inner join and could silently omit unmatched native cells.
The strategy's input, a basis prolongation for ``PreserveBucketProlongation``
or a restriction for ``SuppliedRestriction``, must use the same native grid.
The covariance action checks that grid when it applies or solves with that
input, which may happen after the mean and sensitivity have been computed.

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
requires a separately established positivity-preserving physical restriction,
such as ``SuppliedRestriction`` with ``require_positivity_preserving=True``. The
bucket-compatible restriction derived by ``PreserveBucketProlongation`` is not
guaranteed to preserve positivity.

The function does not construct a likelihood, approximate :math:`A`, serialize
the result, or reconstruct arbitrary native quantities. Do not add another
error term representing the same unresolved flux. Adding covariance components
also assumes that their errors are independent unless their cross-covariances
are included explicitly.

See :mod:`openghg_inversions.coherent_reduction` for the public interface.
