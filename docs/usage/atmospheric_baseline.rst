Outer-region flux states held or marginalised
=============================================

By default the CO2 recipes sample every retained flux state, including the
outer-region states, uncentred and beside any boundary correction. The
outer-flux option selects the outer-region states once on the flux prior and
holds chosen ones at their prior mean, either dropping their prior
uncertainty (``"fixed"``) or carrying it as an observation-error term
(``"marginalise"``). Existing model defaults are unchanged.

Outer states are the flux states whose ``basis_group`` coordinate is in
``outer_groups``. They stay in the ordinary ``flux_scaling`` state, so CO2
outputs, state activity and the :doc:`retained flux prior <flux_prior>` work
as they do without the option.

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

``"marginalise"`` holds the selected outer states :math:`O` at their prior
mean :math:`m_O`, so :math:`H_O m_O` joins the fixed part of the forward model,
and adds their prior covariance as seen by the observations,

.. math::

   W = R + A + H_O \Sigma_{OO} H_O^T,

to the likelihood. The new term has rank at most the number of selected
states. It is stored as a factor :math:`H_O L_O` with
:math:`L_O L_O^T = \Sigma_{OO}`, appended to the low-rank aggregation-error
factor of :math:`A`; a dense :math:`A` gets the full term added, and a diagonal
or absent :math:`A` becomes low-rank. All CO2 recipes that take ``outer_flux``
(ordinary, cached fixed-OU and linked CO2/O2) use it. In linked recipes one
term spans both channels, so it also carries the CO2/O2 covariance the shared
outer states induce. Run metadata records ``outer_marginal_rank``.

For a linear Gaussian model with uncorrelated outer states this is exact: the
posterior of the other states equals their marginal posterior when the outer
states are sampled uncentred (the default), and it is cheaper because the
outer states leave the sampler. Correlated outer states are conditioned on the
others; see below.
Under the default LogNormal-copula flux prior it is an approximation:
:math:`\Sigma_{OO}` is the prior's arithmetic covariance, so the outer
contribution is replaced by the Gaussian with the same mean and covariance.

The outer ocean can be marginalised the same way instead of being held with
zero uncertainty, with ``fixed_sources=("ocean",)``.

Correlated outer states
-----------------------

``"fixed"`` rejects selected outer states with nonzero prior covariance with
any other state: holding a state at its prior mean while sampling the marginal
prior of the others is exact only when they are uncorrelated.

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
The held outer states still report their prior mean in ``flux_scaling``; their
conditional mean is :math:`m_O + B(x_I - m_I)`.

Building a model
----------------

The state selection is done once per flux prior:

.. code-block:: python

   from openghg_inversions.models.outer_flux import OuterFluxChoice
   from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model

   outer_flux = OuterFluxChoice("marginalise", ("outer",)).resolve_states(flux_prior)

   cached = build_co2_cached_sigma_model(
       flux_sensitivity, retained_prior=flux_prior,
       fixed_prior_contribution=fixed_prior_contribution,
       observations=observations, observation_error=observation_error,
       aggregation_error=aggregation_error, tau_hours=tau_hours,
       site_amplitude_prior=site_amplitude_prior,
       boundary=boundary, outer_flux=outer_flux,
   )

:func:`~openghg_inversions.rhime.co2.co2_model.build_co2_model` takes the same
``outer_flux`` argument. Any boundary, or none, may be used with it. Exact-zero
columns of :math:`H` are pruned first, as in the ordinary recipe, and the held
states are then made inactive at their prior mean; in ``"marginalise"`` mode
their error term is then added to ``aggregation_error``. Run metadata records
the mode and selectors.

Linked CO2/O2
-------------

The linked recipes share one flux state and prior between the CO2 and O2
channels, so they take the same resolved outer states, and the held states
are fixed in the joint state after joint exact-zero pruning:

.. code-block:: python

   from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model

   outer_flux = OuterFluxChoice("fixed", ("outer",)).resolve_states(flux_prior)
   model = build_co2_o2_model(..., outer_flux=outer_flux)
