Outer-region flux states held at their prior mean
=================================================

By default the CO2 recipes sample every retained flux state, including the
outer-region states, uncentred and beside any boundary correction. The
outer-flux option selects the outer-region states once on the flux prior and
holds chosen ones at their prior mean. Existing model defaults are unchanged.

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
that the outer flux need not be sampled. Leaving the outer flux unsampled
fairly would marginalise it, adding :math:`H_O\Sigma_{OO}H_O^T` to the error
covariance; this option does not do that.

Holding a state at its prior mean while sampling the marginal prior of the
others is exact only when the held state is uncorrelated with them, so the
option rejects held outer states with nonzero prior covariance.

Building a model
----------------

The builder resolves the choice on the ``retained_prior`` it is given, so the
held states and the covariance check always refer to the prior the model
samples:

.. code-block:: python

   from openghg_inversions.models.outer_flux import OuterFluxChoice
   from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model

   outer_flux = OuterFluxChoice("fixed", ("outer",))

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
states are then made inactive at their prior mean. Every label in
``outer_groups`` and ``fixed_sources`` must match a state, and a
``state_activity`` policy that explicitly fixes a held state at a value other
than its prior mean is rejected. Run metadata records the
mode and selectors.

Linked CO2/O2
-------------

The linked recipes share one flux state and prior between the CO2 and O2
channels, so each builder resolves the choice once on that shared prior. The
held states are fixed in the joint state after joint exact-zero pruning, with
the same meaning as above: zero uncertainty and nothing added to the error
covariance, in the ordinary and cached linked recipes alike:

.. code-block:: python

   from openghg_inversions.rhime.co2.co2_o2_model import build_co2_o2_model

   model = build_co2_o2_model(..., outer_flux=OuterFluxChoice("fixed", ("outer",)))
