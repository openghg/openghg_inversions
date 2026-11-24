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

Complete native functionals
---------------------------

The same reduction splits the native state as
:math:`x = m + U_*(\alpha - \Pi m) + x_\perp`, where
:math:`U_* = B \Pi^T C_\alpha^{-1}` and :math:`x_\perp` is independent of
:math:`\alpha` with covariance :math:`B_\perp = B - U_* C_\alpha U_*^T`. A
native linear functional :math:`Q`, such as a country total, therefore splits
as :math:`Q x = Q m + D(\alpha - \Pi m) + Q x_\perp` with :math:`D = Q U_*`.
Retained-state draws give only the first two terms. :math:`Q x_\perp` carries
both variance and a data-driven mean update, because it is correlated with the
unresolved observation error :math:`H x_\perp`; both are largest when retained
regions cross the functional's borders.

Pass the functionals to the reduction to prepare the products that condition
:math:`Q x_\perp`::

   reduction = reduce_native_gaussian(
       ...,
       functionals=country_functionals,  # (functional_dim, *native_dims)
       functional_dim="country_functional",
   )
   products = reduction.functional_products

``products`` holds :math:`Q m`, :math:`D`, :math:`Q B_\perp H^T` and
:math:`Q B_\perp Q^T`. They are computed from bounded blocks of
:math:`B Q^T`; :math:`B` is never densified. Use one functional row per
reported term, for example one country total per source, so that the joint
covariance of terms is available and sums are formed per draw.

For a posterior draw of :math:`\alpha`, any other parameters :math:`\beta`
and the noise parameters, with residual
:math:`r = y - y_0 - H_\alpha(\alpha - \Pi m) - G\beta`,

.. math::

   Q x_\perp \mid \alpha, \beta, y \sim
   N\left(Q B_\perp H^T (A + R)^{-1} r,\;
   Q B_\perp Q^T - Q B_\perp H^T (A + R)^{-1} H B_\perp Q^T\right).

Under the prior, :math:`Q x_\perp \sim N(0, Q B_\perp Q^T)` independently of
:math:`\alpha`. For a Gaussian retained state and the exact :math:`A`, the
complete functional mean and covariance equal the native-grid Gaussian
posterior for any retained partition. With a LogNormal retained state the
conditioning on each draw remains exact under the reduced model, but the
reduced model is the moment-matched approximation described in
`Assumptions and limitations`_, so the result is not the native-grid
posterior of a LogNormal native field.
:func:`openghg_inversions.postprocessing.unresolved_functionals.complete_functional_draws`
adds the conditional mean and one conditional draw to each retained draw. It
takes :math:`(A + R)^{-1}` as a labelled solve whose observation labels
must equal the residuals':
:func:`~openghg_inversions.postprocessing.unresolved_functionals.dense_solve`
for one fixed dense covariance, whose labelled columns are aligned to its rows
and checked for symmetry, or
:func:`~openghg_inversions.postprocessing.unresolved_functionals.fixed_ou_solve`
for the fit's low-rank-plus-diagonal :math:`A` with each draw's fixed-OU
amplitudes, selected by chain and draw label. ``FixedOuLowRank`` keeps no
observation labels, so ``fixed_ou_solve`` trusts the supplied
``observation_index`` for row order: it checks the row count and each row's
site (from ``site_index``, such as the fit's ``ou_site_index``, or the index's
``site`` level) against the covariance, but not the time order within a site. Residuals
must carry the units of :math:`H`.

For CO2 country totals, build :math:`Q` with
:func:`~openghg_inversions.postprocessing.co2_flux_outputs.co2_country_functionals`
and pass an
:class:`~openghg_inversions.postprocessing.unresolved_functionals.UnresolvedFunctionalConditioning`
to :func:`~openghg_inversions.postprocessing.co2_flux_outputs.co2_country_flux_outputs`.
Its outputs then carry ``uncertainty_scope="complete_native"`` and the
conditioning description; without it they remain
``"retained_state_conditional"`` and omit the unresolved native state.

The products record the
:func:`~openghg_inversions.coherent_reduction.reduction_identity` of their
reduction. The reduced arrays alone cannot identify the native problem: if
one observation sees either of two cells that :math:`B` and :math:`\Pi`
treat alike, swapping the cells leaves :math:`C_\alpha`, :math:`H_\alpha`
and :math:`A` unchanged but reverses the sign of :math:`Q B_\perp H^T`. The
reduction therefore records a
:func:`~openghg_inversions.coherent_reduction.native_reduction_identity`,
a hash of the native :math:`H`, :math:`\Pi`, :math:`m`, the prolongation
:math:`U_*` and, when the covariance action provides ``to_dataset()``, its
description of :math:`B`. The reduction identity combines it with a hash of
:math:`C_\alpha`, :math:`H_\alpha` and the exact :math:`A`, so anyone who
stores the native identity and the reduced arrays, such as a prepared
artifact, can recompute it and detect a changed :math:`A`. The conditioning
must carry the identity recorded when the fit was prepared, and the two must
be equal. :math:`Q m` and :math:`D` do not depend on :math:`H`, so products
built with another observation operator are caught only by the identity.
The identity is an exact hash, not a tolerance: take it from the same
``reduce_native_gaussian(functionals=...)`` call that produced the products,
rather than recomputing it from another reduction. In addition, the
products' :math:`Q m` and :math:`D` must equal the map's own :math:`A F m` and
:math:`A F U_*`, which rejects products prepared for another functional,
partition or covariance.

Use the same :math:`A` in the solve as in the products. The products use the
exact :math:`B_\perp`, so pairing them with a truncated low-rank :math:`A`
mixes two covariance models: the conditional mean update can be inflated and
the conditional covariance can lose positive semidefiniteness. Negative
eigenvalues up to ``clipping_tolerance`` (default ``1e-6``) relative to the
largest eigenvalue of the prior functional covariance
:math:`Q B Q^T = Q B_\perp Q^T + D C_\alpha D^T` are treated as roundoff and
clipped before drawing. That scale is used because the drawn covariances are
formed by subtracting terms of that size and can themselves be zero, as when
the retained states resolve every functional. The most negative relative
eigenvalue is recorded as ``conditional_covariance_min_eigenvalue``. A larger
negative eigenvalue raises an error instead of producing ``complete_native``
output. A
:class:`~openghg_inversions.models.fixed_ou.FixedOuLowRank` prepared with a
full-width factor of the exact :math:`A` gives the exact update with sampled
OU amplitudes.

When retained states are marginalised into :math:`A`,
:func:`openghg_inversions.coherent_reduction.marginalise_functional_states`
moves their functional contribution into :math:`Q x_\perp`, matching a
reduction with the kept rows of :math:`\Pi`. For CO2 country outputs, pass
full-restriction products and declare the marginalised states on the
conditioning; they must be exactly the fit's inactive states, which are
otherwise held fixed without their variance and make complete uncertainty
undefined. Activity is read from the ``<flux_scale>_is_active`` mask that the
fitted model records for every activity policy, including all-active; a trace
without it is rejected.

Staged CO2 outputs
~~~~~~~~~~~~~~~~~~

The staged CO2 pipeline reports complete country totals when the prepared
artifact carries the country products:

1. Build :math:`Q` from the signed reference flux the affine reconstruction
   will carry with
   :func:`~openghg_inversions.postprocessing.co2_flux_outputs.co2_flux_country_functionals`
   and the reporting country file, and pass it to
   :func:`~openghg_inversions.coherent_reduction.reduce_native_gaussian` with
   ``functional_dim="country_functional"``.
2. :func:`~openghg_inversions.rhime.co2.co2_preparation.prepare_co2_inputs`
   with ``aggregation_error_rank=None`` saves ``reduction.functional_products``
   in the prepared artifact's ``country_functionals`` node (schema version 4)
   and summarizes them under ``provenance["country_functionals"]``. Only
   full-restriction products on the ``country_functional`` axis are accepted,
   and their ``reduction_identity`` must equal the artifact's
   :attr:`~openghg_inversions.rhime.co2.co2_preparation.Co2PreparedInputs.reduction_identity`.
   The artifact records the reduction's ``native_reduction_identity`` and
   recomputes the reduction identity from it and its stored
   ``alpha_prior_covariance``, ``H`` and dense :math:`A`. This ties the
   products to the native observation operator, which the :math:`Q m` and
   :math:`D` checks cannot, and to the stored :math:`A`. The node stores the
   products' identity, and load requires the restored artifact to reproduce
   it, so an operator or :math:`A` changed after preparation is rejected. The unresolved
   covariance must be symmetric positive semidefinite.
3. Replay (:func:`openghg_inversions.rhime.co2.handoffs.postprocess`, or
   :func:`~openghg_inversions.rhime.co2.execution.run_co2`) rebuilds each
   posterior draw's residual ``mf - model_mean`` and the fit's own
   :math:`(A + R)^{-1}` with
   :func:`~openghg_inversions.rhime.co2.co2_unresolved.co2_unresolved_conditioning`,
   restricts the products to the requested countries, passes the artifact's
   reduction identity, and labels country outputs ``"complete_native"``. The
   trace must carry the fitted model's ``<flux_scale>_is_active`` mask. Without products they stay
   ``"retained_state_conditional"``. Native-grid flux is unchanged.

The products use the exact :math:`B_\perp`, so the joint covariance of the
products and the observations is consistent only with the exact :math:`A`.
An artifact with products must therefore store dense :math:`A`, and the fit
samples the same :math:`A` it is conditioned on. The solve follows the
recorded likelihood. Additive-sigma and site-sigma fits use the stored
:math:`A` plus ``epsilon**2 - diag(A)`` for each draw. Fixed-OU fits, with
fixed or inferred amplitudes, rebuild the fit's
:class:`~openghg_inversions.models.fixed_ou.FixedOuLowRank` from the recorded
observation error, OU site index and time scales (numeric hours, or a
timedelta), with each draw's amplitudes. Other likelihoods raise an error
rather than claiming complete outputs. ``unresolved_conditioning`` records
the :math:`A` and noise model as JSON. A conditional-covariance eigenvalue
below the clipping tolerance raises (see above), so no complete label is
reported.
Country sector totals are summed from the same complete (country, source)
draws. States marginalised at preparation are declared from
``state_information_marginalised`` when the artifact records it.

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
of :math:`y` is unchanged.

The selection keeps the full-restriction reduction, and the prepared artifact
stores that reduction's :math:`C_\alpha`, :math:`H_\alpha` and :math:`A`
together with the marginalised mask. Only :math:`C_\alpha`,
:math:`H_\alpha` and the mask are needed to integrate the states out, so
:func:`~openghg_inversions.coherent_reduction.marginalisation_terms` derives
the fitted :math:`H'` (with zero marginalised columns, which model
construction removes like zero-sensitivity states) and the factor
:math:`F = H_D S^{1/2}` of the increase :math:`A' - A = F F^\mathsf{T}`
after loading.
:attr:`~openghg_inversions.rhime.co2.co2_preparation.Co2PreparedInputs.fitted_inputs`
applies them: it replaces ``H``, ``fixed_prior_contribution`` and the
aggregation error, adding :math:`F F^\mathsf{T}` to a dense :math:`A` or
:math:`F` as extra columns of a low-rank factor. The CO2 runners, the
concentration reconstruction and the complete-country observation solve read
that view. Keeping the full arrays under their usual names preserves the
invariant that ``alpha_prior_covariance`` and ``H`` are the full-restriction
:math:`C_\alpha` and :math:`H_\alpha`, and keeps the artifact's
``reduction_identity`` equal to the country products'.
:attr:`~openghg_inversions.coherent_reduction.InformationSelection.marginalised_reduction`
gives the same fitted view of a reduction in memory::

   from openghg_inversions.coherent_reduction import marginalise_low_information_states
   from openghg_inversions.models import StateActivity

   selection = marginalise_low_information_states(
       reduction,
       observation_variance=mismatch_variance,  # sigma_o**2 on the observation axis
       policy=StateActivity(group_coord="source", min_information={"o2_ocean": 0.1}),
   )
   co2_inputs = prepare_co2_inputs(
       canonical_inputs,
       reduction,  # the full-restriction reduction, which is selection.reduction
       information_selection=selection,
   )
   fit_view = co2_inputs.fitted_inputs  # H', H m - H' Pi m and A + F F.T

``min_information`` is either one number, applied to every state, or a
mapping from labels of ``group_coord`` to thresholds. A mapping applies the
threshold only to those groups. For example, it can screen only ocean states,
with separate CO2-ocean and O2-ocean groups in linked CO2/O2 models. The
default ``None`` keeps every state. :func:`~openghg_inversions.models.resolve_state_activity` rejects
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
Complete country outputs (see `Staged CO2 outputs`_) include it exactly: the
full-restriction country products are marginalised with the same mask, and
the observation solve uses the fitted :math:`A'`. Calibrate :math:`I_{\min}` on synthetic observations: an information of
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

The function does not construct a likelihood, approximate :math:`A`, or
serialize the result. It reconstructs native quantities only as the
linear functionals requested through ``functionals``. Do not add another
error term representing the same unresolved flux. Adding covariance components
also assumes that their errors are independent unless their cross-covariances
are included explicitly.

See :mod:`openghg_inversions.coherent_reduction` for the public interface.
