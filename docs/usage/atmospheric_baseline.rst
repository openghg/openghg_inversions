Outer-region flux states held or marginalised
=============================================

By default the CO2 recipes sample every retained flux state, including the
outer-region states, uncentred and beside any boundary correction. The
outer-flux option selects outer-region states on the flux prior. The
``"fixed"`` mode holds them at their prior mean with zero uncertainty.
The ``"marginalise"`` mode integrates them out of the likelihood and carries
their uncertainty as an observation-error term.

Outer states are the flux states whose ``basis_group`` coordinate is in
``outer_groups``. Their entries in ``flux_scaling`` retain the prior mean in
both modes. In ``"marginalise"`` mode those entries are placeholders in
the raw trace.

``"fixed"`` mode
----------------

``"fixed"`` holds the selected outer states at their prior mean, with zero
uncertainty. With ``fixed_sources`` (for example ``("ocean",)``) only outer
states whose ``source`` coordinate matches are held, and the other outer
states stay sampled as usual; without it every outer state is held.

This is a what-if for a known outer prior, not a stand-in for leaving the
outer flux unsampled. Nothing is added to the observation error for the held
states, so the variance their prior would contribute,
:math:`\operatorname{diag}(H_O\Sigma_{OO}H_O^T)`, is left out of the model.
When the outer prior mean is close to the truth, holding it there can look
better than sampling it; that is the benefit of the known prior, not evidence
that the outer flux need not be sampled. Use ``"marginalise"`` to leave the
outer flux unsampled fairly.

``"marginalise"`` mode
----------------------

For selected outer states :math:`O` uncorrelated with the other states,
``"marginalise"`` includes :math:`H_O m_O` in the mean and adds their
prior covariance as seen by the observations,

.. math::

   W = R + A + H_O \Sigma_{OO} H_O^T,

to the likelihood. The new term has rank at most the number of selected
states. It is stored as a factor :math:`H_O L_O` with
:math:`L_O L_O^T = \Sigma_{OO}`, appended to the low-rank aggregation-error
factor of :math:`A`. For dense :math:`A`, the full term is added. Without
aggregation error, the new factor supplies the entire term. All CO2 recipes
that take ``outer_flux``
(ordinary, cached fixed-OU and linked CO2/O2) use it. In linked recipes one
term spans both channels, so it also carries the CO2/O2 covariance the shared
outer states induce.

For a linear Gaussian model with uncorrelated outer states this is exact: the
posterior of the other states equals their marginal posterior when the outer
states are sampled uncentred (the default). Correlated outer states are
conditioned on the others; see below.

Under the default LogNormal-copula flux prior it is an approximation, not a
pure reparameterisation of sampling the outer states:
:math:`\Sigma_{OO}` is the prior's arithmetic covariance, so the outer
contribution is replaced by the Gaussian with the same mean and covariance,
and the posterior can differ. In one WUR BASE July 2021 test (5314
observations, 18 marginalised outer states), NUTS with ``"marginalise"``
moved the inner NET_BIOSPHERE and TOTAL posterior means by +90 Tg CO2/yr
(0.05 posterior SD, about two Monte Carlo standard errors) relative to
sampling the outer states, and left FF unchanged. Exact solves of the same
problem with a Gaussian prior agreed to 2e-9 Tg CO2/yr. This comparison
quantifies the approximation for that case and prior.

Taking the outer states out of the sampler makes the state smaller, but it
need not make NUTS faster: the likelihood gains a low-rank term, and in the
test above sampling took slightly longer than sampling the outer states.

The outer ocean can be marginalised the same way instead of being held with
zero uncertainty, with ``fixed_sources=("ocean",)``.

Correlated outer states
-----------------------

``"fixed"`` requires zero prior covariance between the selected outer states
and the states outside the selection.

``"marginalise"`` accepts them. With cross-covariance :math:`\Sigma_{OI}`
between the selected states and all other states :math:`I`, the outer states
are replaced by their best linear predictor from :math:`I` under the prior's
arithmetic moments,

.. math::

   x_O = m_O + B(x_I - m_I) + r, \qquad B = \Sigma_{OI}\Sigma_{II}^{+},

and :math:`r` is integrated out. So

- the design of the other states becomes :math:`H_I + H_O B`;
- the fixed term gains :math:`-H_O B m_I`;
- the error term is :math:`H_O\Sigma_{O|I}H_O^T` with
  :math:`\Sigma_{O|I} = \Sigma_{OO} - B\Sigma_{IO}`.

The prior of the other states is unchanged, so a LogNormal-copula inner prior
stays positive. For a Gaussian prior this is exact. Under the LogNormal copula
:math:`r` is uncorrelated with :math:`x_I` but not independent of it, a
moment-level approximation of the same kind as the coherent reduction. States
that state activity or exact-zero pruning hold fixed enter at their fixed
values. A state with no direct response but correlated with the outer states
gains a column through :math:`H_O B`, so it is sampled rather than pruned.
The selected outer entries in ``flux_scaling`` are prior-mean placeholders.
Given the other states, :math:`m_O + B(x_I - m_I)` is the conditional prior
mean in the Gaussian model and the linear predictor in the LogNormal
approximation. Conditioning on the observations also updates the outer
residual. Posterior reconstruction is tracked in
`#1129 <https://github.com/openghg/openghg_inversions/issues/1129>`_.

Building a model
----------------

The builder resolves the choice on the ``retained_prior`` it is given, so the
held states and the covariance check always refer to the prior the model
samples:

.. code-block:: python

   from openghg_inversions.models.outer_flux import OuterFluxChoice
   from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model

   outer_flux = OuterFluxChoice("marginalise", ("outer",))

   cached = build_co2_cached_sigma_model(
       flux_sensitivity, retained_prior=flux_prior,
       fixed_prior_contribution=fixed_prior_contribution,
       observations=observations, observation_error=observation_error,
       aggregation_error=aggregation_error, tau_hours=tau_hours,
       site_amplitude_prior=site_amplitude_prior,
       boundary=boundary, outer_flux=outer_flux,
   )

:func:`~openghg_inversions.rhime.co2.co2_model.build_co2_model` takes the same
``outer_flux`` argument. Any boundary, or none, may be used with it. For
correlated marginalisation the builder adjusts the design before pruning
exact-zero columns. It then sets the selected states inactive at their prior
mean and adds any marginalised covariance to ``aggregation_error``. Every label in
``outer_groups`` and ``fixed_sources`` must match a state, and a
``state_activity`` policy that explicitly fixes a held state at a value other
than its prior mean is rejected. Run metadata records the
mode and selectors.

Reporting with marginalised outer flux
--------------------------------------

CO2 ``basic`` reporting provides fitted concentrations and saved predictive
draws where available. It uses the fitted affine intercept, including any
adjustment for correlated outer states. The full native flux map and the
combined PARIS concentration-and-flux export require posterior outer flux
draws.

To report an inner-domain total, supply an exact-grid country or reporting
region mask in ``outputs.country_file``. Use
``outputs.paris_postprocessing_kwargs.country_selections`` to request specific
names. Every requested quantity must have zero loading on each marginalised
state through the bound affine map, whose prolongation can couple
geographically separate regions. The same check runs on saved samples using
the fitted state labels. These totals retain the affine map's
retained-state-conditional uncertainty scope.

Linked CO2/O2
-------------

The linked recipes share one flux state and prior between the CO2 and O2
channels, so each builder resolves the choice once on that shared prior.
``"fixed"`` sets the selected states to their prior means. ``"marginalise"``
adds a joint covariance term spanning both channels, including the covariance
between CO2 and O2 induced by the selected states. Both modes are available
in the ordinary and cached linked recipes. For marginalised outer flux,
use the saved joint concentration predictions. Linked basic and PARIS
products require the fitted design and intercept for their component
reconstruction.

.. code-block:: python

   from openghg_inversions.models.outer_flux import OuterFluxChoice
   from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model

   model = build_co2_o2_model(..., outer_flux=OuterFluxChoice("marginalise", ("outer",)))
