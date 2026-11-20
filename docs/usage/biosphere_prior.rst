Calibrating retained biosphere uncertainty
==========================================

The Python components in :mod:`openghg_inversions.biosphere_prior` change one
aggregate uncertainty direction in an existing retained flux-state prior.
Calibration changes only the arithmetic covariance; the model keeps the
retained prior's marginals. Omitting ``biosphere_prior`` from the CO₂ or linked
CO₂/O₂ runners preserves their existing prior. A calibration is a sensitivity
intervention, not a configured production shrinkage factor.

This mode keeps the reduced transport, affine flux reconstruction, reporting
functionals and aggregation-error covariance fixed. It does not regenerate a
coherent reduction from a changed native-grid prior. Such a native-prior change
would require rebuilding the reduction and unresolved covariance together.

One aggregate residual direction
--------------------------------

Let :math:`x` be the retained coefficients with existing arithmetic mean
:math:`m` and covariance :math:`B`. Supply labelled affine flux functionals

.. math::

   GPP(x) &= GPP_{ref} + w_{GPP}^{T}(x-m),\\
   TER(x) &= TER_{ref} + w_{TER}^{T}(x-m).

GPP is signed negative uptake; TER is positive respiration. Define
:math:`n=w_{GPP}+w_{TER}` and :math:`g=-w_{GPP}+w_{TER}`. The net and gross
corrections are :math:`n^T(x-m)` and :math:`g^T(x-m)`. No logarithm of signed
GPP or potentially negative net flux is taken. Both original forward columns
and their output reconstruction remain separate.

The supplied boolean ``biosphere`` selector identifies coefficients allowed to
change. Form :math:`C` from :math:`n^T` and a selector for every other state.
After dividing each nonzero control row by its prior SD, pivoted QR identifies
independent controls. Solves
on the independent, standardized controls give

.. math::

   r &= g-C^T\operatorname{solve}(CBC^T,CBg),\\
   v &= Br,\qquad s^2=r^TBr,\\
   B_{new} &= B+(\lambda^2-1)vv^T/s^2.

Here :math:`r^T(x-m)` is the residual of the best *linear* prior prediction of
gross exchange from the controls. For Gaussian states this is a conditional
variance construction. For lognormal states it does not fix the true nonlinear
conditional distribution.

The arithmetic identities are

.. math::

   E[x]=m,\qquad CB_{new}=CB,\qquad r^TB_{new}r=\lambda^2s^2.

Thus net variance and its covariance with every coefficient are preserved,
as are the protected-state block and its cross-covariances. Just one covariance
direction changes. Separate GPP and TER marginal variances generally change;
requiring both to stay fixed would generally conflict with this intervention.
Expressing the baseline in net/gross coordinates retains its original
net/gross covariance; setting those coordinates independent is another prior.

Explicit calibration
---------------------

:func:`~openghg_inversions.biosphere_prior.calibrate_biosphere_prior` requires
exactly one of:

* ``residual_sd_multiplier``: dimensionless :math:`\lambda`;
* ``target_residual_sd``: scalar with declared flux units, giving
  :math:`\lambda=SD_{target}/s`;
* ``target_gross_sd``: scalar *marginal* gross SD with declared flux units.

For the last choice, the regression stays fixed and

.. math::

   V_{explained} &= g^TBg-s^2,\\
   Var_{new}(gross) &= V_{explained}+\lambda^2s^2.

A marginal variance below :math:`V_{explained}` is infeasible. Zero residual
variance has no direction to calibrate; only the explicit identity multiplier
is accepted in that case. This dense
implementation requires positive-definite arithmetic and sampling covariances;
a zero residual-SD target is singular and rejected. No jitter, projection or
correlation clipping is applied.

With ``residual_sd_multiplier=1`` the existing covariance is returned exactly,
and with the same marginals this recovers the existing *joint distribution*.
A factor such as ``0.5`` is a sensitivity setting requiring scientific
justification, not a recommended production default. Calibration uses prior
moments, never posterior fit or a validation truth field.

.. _sampled-marginals:

Sampled marginals
-----------------

The model samples a
:class:`~openghg_inversions.correlated_state.CorrelatedPrior`: arithmetic
moments joined by a Gaussian copula, with a marginal family per state. A
calibration keeps the retained prior's marginals. A whitened
:math:`z\sim N(0,I)` is correlated as :math:`u=Lz` and each state is
:math:`x_i=F_i^{-1}(\Phi(u_i))`, where :math:`F_i` has the state's arithmetic
mean and SD. The latent correlations :math:`LL^T` are chosen pair by pair so
that the arithmetic covariance is reproduced exactly.

``"lognormal"`` is the default. With it on every state the prior is
:class:`~openghg_inversions.correlated_state.CorrelatedLognormalPrior` and the
model is built exactly as before:

.. math::

   S_{ij}&=\log(1+B_{new,ij}/(m_i m_j)),\\
   a_i&=\log(m_i)-S_{ii}/2,\\
   x&=\exp(a+Lz),\qquad LL^T=S,\quad z\sim N(0,I).

The logarithm arguments and positive definiteness of :math:`S` must be valid.
Positive-definite arithmetic covariance alone does not guarantee valid
multivariate lognormal moments. Incompatible requests fail clearly. Arithmetic
mean/covariance are preserved exactly, but preserving net mean and variance
does not preserve all its higher moments.

The other marginals have the same mean and SD with less right skew:

- ``"normal"``: :math:`x_i=m_i+s_i u_i`. Symmetric, so a posterior constrained
  only in a sum of states (net flux) is not pulled towards small gross terms,
  but scalings can be negative (about 6% probability at CV 0.63). Mainly a
  test and comparison family.
- ``"weibull"`` and ``"gamma"``: positive and defined for any CV. A mean-one
  lognormal at CV 0.63 has median 0.85; the Weibull 0.89 and gamma 0.87. In
  OPE-188 July 2021 fits, Weibull GPP/TER marginals removed about two thirds
  of the lognormal's gross shrinkage relative to a Gaussian in CO2-only fits
  and about half in linked fits; gamma removed a fifth to a quarter. Gamma
  states are held at their quantile for latent :math:`u=8` above it
  (probability 6e-16), because the JAX inverse incomplete gamma overflows
  near :math:`u=10`; both samplers use the same map.

A pair with a normal state, or two lognormal states, has a closed-form latent
correlation. Weibull and gamma pairs solve a 60-term Hermite series for it.
A pair covariance no latent correlation can reach (for example strong negative
correlation between very skewed states), or a latent correlation that is not
positive definite, fails clearly without repair.

Labelled inputs and model use
------------------------------

``baseline`` supplies existing labelled means and covariance. ``weights`` has
exactly ``GPP`` and ``TER`` on ``process`` and the complete prior state axis;
its ``units`` describe flux totals per dimensionless coefficient. ``reference``
has the same process labels and declares compatible flux units. Pint converts
reference values and absolute targets into the weights' units. The boolean
``biosphere`` selector uses the same labelled state axis. Source names, region
counts and positional GPP/TER pairing have no special meaning.

For example, given independently prepared labelled arrays:

.. code-block:: python

   from openghg_inversions.biosphere_prior import calibrate_biosphere_prior
   from openghg_inversions.correlated_state import CorrelatedPrior
   from openghg_inversions.models.biosphere import resolve_retained_flux_prior

   baseline = CorrelatedPrior(retained_mean, retained_covariance)
   from openghg_inversions.rhime.co2 import build_co2_model

   calibrated = calibrate_biosphere_prior(
       baseline,
       biosphere=biosphere_selector,
       weights=reporting_weights,
       reference=reference_totals,
       target_residual_sd=requested_sd,  # scalar DataArray with units
   )
   model = build_co2_model(
       H,
       retained_prior=resolve_retained_flux_prior(baseline, calibrated),
       fixed_prior_contribution=affine_intercept,
       observations=observations,
       observation_error=observation_error,
       aggregation_error=aggregation_error,
   )

The ordinary/cached CO₂ and CO₂/O₂ graph builders take one ``retained_prior``,
a ``CorrelatedPrior``. ``resolve_retained_flux_prior`` checks once that the
calibration was made from that baseline, rejects a different one, and returns
the baseline's mean and marginals with the calibrated covariance.
``build_co2_from_materialized``, ``build_rhime_co2_cached_sigma``,
``build_rhime_co2_o2`` and the linked prepared sampling runners instead take a
separate ``biosphere_prior`` argument and make that check against their
prepared baseline. The :doc:`configured saved-calibration workflow
<configured_co2_priors>` also loads explicitly selected artifacts through the
CO2 and linked TOML/JSON model owners.
The linked recipes apply it once to shared coefficients, independently of how
many observation channels use them.

Activity policies fix explicit inactive output coefficients. The model samples
the joint marginal of the active states; all fixed states require no latent
draw. The functional moments describe the unfixed prior; fixing coefficients
subsequently removes their output variance and should be recorded separately.

Audit, serialization and validation
-----------------------------------

The returned ``BiospherePrior.data`` holds the inputs (baseline mean and
covariance, selector, functional weights and reference) and the results
(calibrated covariance and residual gross weights). Its attributes record the
resolved policy, requested value, reporting units and multiplier. The
provisional ``openghg_inversions.biosphere_prior/v1`` schema may change.
``BiospherePrior.functionals()`` derives GPP, TER, net, gross and residual gross
means, SDs and correlations before and after calibration. For any other linear
functional :math:`l`, compute :math:`Var(l^Tx)=l^TB l` from the saved covariance.
These analytic moments are the primary calibration evidence; prior draws are
secondary checks.

Calibration is an explicit dense eager boundary over already-retained states.
Related lazy functional inputs are computed together; supplied arrays remain
borrowed and unchanged. There is no native-grid scalability claim.

Recipe trace annotation automatically adds a ``biosphere_prior`` group. After
direct PyMC sampling, call
``annotate_biosphere_prior(trace, calibrated)`` from
``openghg_inversions.models.biosphere`` explicitly. It records
``biosphere_prior_scope`` in ``rhime_model_metadata``; the ``prior`` entry is
unchanged. Save the
group with the ordinary trace codec to preserve gathered MultiIndexes:

.. code-block:: python

   import xarray as xr
   from openghg_inversions.biosphere_prior import BiospherePrior
   from openghg_inversions.serialization import save_trace, load_trace

   save_trace(
       xr.DataTree.from_dict({"biosphere_prior": calibrated.to_dataset()}),
       "calibration.nc",
   )
   restored = BiospherePrior.from_dataset(
       load_trace("calibration.nc")["biosphere_prior"].to_dataset()
   )

The loader checks the schema, labels, units and policy. It also checks that
the saved covariance is the residual-direction change of the baseline for the
saved multiplier, and that it preserves the protected and net covariances. It
does not reselect controls or resolve calibration targets. Existing
``flux_scaling`` names, coefficient units, affine output reconstruction and
sampling diagnostics remain available.

The runnable :download:`calibration CLI <../../examples/biosphere_prior/calibrate.py>`
reads a ``biosphere_prior`` group containing the five uncalibrated input
arrays ``baseline_mean``, ``baseline_covariance``, ``biosphere``, ``weights``
and ``reference`` (it need not contain an existing calibration) and requires either ``--multiplier``
or ``--residual-sd`` with ``--sd-units``. For example:

.. code-block:: console

   python -m examples.biosphere_prior.calibrate input.nc calibrated.nc --multiplier 1

The repository tests cover analytic identities, label/unit changes, positive
support, exact active marginals, serialization, and full/enhancement/linked
ordinary-versus-cached densities.
