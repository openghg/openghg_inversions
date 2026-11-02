Calibrating retained biosphere uncertainty
==========================================

The Python components in :mod:`openghg_inversions.biosphere_prior` change one
aggregate uncertainty direction in an existing retained flux-state prior.
The positive correlated lognormal family is the primary option. A conditional
Gaussian biosphere construction is available as a signed-template diagnostic.
Omitting ``biosphere_prior`` from the CO₂ or linked CO₂/O₂ builders preserves
their existing prior. There is no configured production shrinkage factor.

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
independent controls. The report records the number of redundant rows. Solves
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

With ``residual_sd_multiplier=1`` the existing covariance is returned exactly.
With the same lognormal family this recovers the existing *joint distribution*.
Switching to the Gaussian diagnostic with the same moments is not distributional
equivalence. A factor such as ``0.5`` is a sensitivity setting requiring
scientific justification, not a recommended production default. Calibration
uses prior moments, never posterior fit or a validation truth field.

Positive and diagnostic families
--------------------------------

For positive arithmetic means, the lognormal option reuses
:class:`~openghg_inversions.correlated_state.CorrelatedLognormalPrior`:

.. math::

   S_{ij}&=\log(1+B_{new,ij}/(m_i m_j)),\\
   a_i&=\log(m_i)-S_{ii}/2,\\
   x&=\exp(a+Lz),\qquad LL^T=S,\quad z\sim N(0,I).

The logarithm arguments and positive definiteness of :math:`S` must be valid.
Positive-definite arithmetic covariance alone does not guarantee valid
multivariate lognormal moments. Incompatible requests fail clearly. Arithmetic
mean/covariance are preserved exactly, but preserving net mean and variance
does not preserve all its higher moments.

For ``family="gaussian"``, partition the coefficients into selected biosphere
:math:`b` and protected others :math:`o`. With the requested covariance blocks,

.. math::

   x_o &\sim \text{original protected lognormal marginal},\\
   R &= \operatorname{solve}(B_{oo},B_{ob})^T,\\
   V &= B_{bb}-R B_{ob},\\
   x_b &= m_b+R(x_o-m_o)+\operatorname{chol}(V)z_b.

Independent standard normals :math:`z_b` supply the biosphere residuals. This
matches all requested arithmetic moments, including nonzero cross-covariance.
When :math:`B_{bo}\ne0`, biosphere marginals are mixtures, not multivariate
Gaussian distributions. Negative coefficients are allowed and should be
reported; these signed template weights are not positive physical processes.
Truncating them would change the calibrated prior.

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
       retained_prior=baseline,
       biosphere_prior=calibrated,
       fixed_prior_contribution=affine_intercept,
       observations=observations,
       observation_error=observation_error,
       aggregation_error=aggregation_error,
   )

The same ``biosphere_prior`` argument is accepted by ordinary/cached CO₂ and
CO₂/O₂ graph builders, ``build_co2_from_materialized``,
``build_rhime_co2_cached_sigma``, ``build_rhime_co2_o2`` and the linked prepared
sampling runners. These are explicit Python inputs; TOML/INI configuration and
saved model-choice schemas do not load a calibration automatically. Pass the
original baseline alongside the calibration; a different baseline is rejected.
The linked recipes apply it once to shared coefficients, independently of how
many observation channels use them.

For enhancement-only inputs, calibrate the full prior first, then call
``calibrated.lognormal_marginal(enhancement_state_labels)``. This selects exact
arithmetic and latent marginal moments without a Schur complement. Use the
enhancement product's own sensitivity, intercept and aggregation covariance.
Do not copy the full-model observation covariance into that product.

Activity policies still fix explicit inactive output coefficients. The Gaussian
diagnostic retains its full latent draw before replacing fixed outputs, thereby
preserving the active marginal mixture; it may sample unused latent directions.
A report describes the unfixed prior; fixing coefficients subsequently removes
their output variance and should be recorded separately.

Audit, serialization and validation
-----------------------------------

The returned ``BiospherePrior.data`` contains baseline mean/covariance,
resulting covariance, selector, functional weights/reference, residual weights,
unchanged functional means, analytic SDs, and the full functional correlation
matrix. Its JSON ``calibration_report`` records requested policy/units, resolved
factor, explained gross variance, control rank, maximum control/protected
covariance changes, and arithmetic/lognormal validity. For any other linear
functional :math:`l`, compute :math:`Var(l^Tx)=l^TB l` from the saved covariance.
These analytic moments are the primary calibration evidence; prior draws are
secondary checks.

Calibration is an explicit dense eager boundary over already-retained states.
Related lazy functional inputs are computed together; supplied arrays remain
borrowed and unchanged. There is no native-grid scalability claim.

Recipe trace annotation automatically adds a ``biosphere_prior`` group. After
direct PyMC sampling, call
``annotate_biosphere_prior(trace, calibrated)`` from
``openghg_inversions.models.biosphere`` explicitly. A marginal-only workflow
should record the parent calibration and its selected state labels. Save the
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

The loader recomputes the policy and checks cached numerical results. Existing
``flux_scaling`` names, coefficient units, affine output reconstruction and
sampling diagnostics remain available.

The runnable :download:`calibration CLI <../../examples/biosphere_prior/calibrate.py>`
reads the same group from an input artifact and requires either ``--multiplier``
or ``--residual-sd`` with ``--sd-units``. For example:

.. code-block:: console

   python -m examples.biosphere_prior.calibrate input.nc calibrated.nc --multiplier 1

The repository tests cover analytic identities, label/unit changes, positive
support, cross-covariance in the Gaussian diagnostic, exact marginals,
serialization, and full/enhancement/linked ordinary-versus-cached densities.
