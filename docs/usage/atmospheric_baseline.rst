Atmospheric baseline: boundary level with outer-region flux corrections
=======================================================================

Outer-region fluxes and the boundary condition both raise every observation by
a similar amount, so a CO2 model that samples both cannot tell their common
level apart. The outer-flux option of the ordinary CO2 recipes removes that
level from the outer fluxes and leaves it to the boundary's mean shift, which
becomes the model's one atmospheric level. Existing model defaults are
unchanged.

The option is two choices used together:

* a centred boundary from :doc:`boundary_correction_structures`, additive or
  anomaly scaling, **with** a mean shift prior. Any of its anomaly structures
  may be used; ``anomaly_scale=0`` keeps only the level;
* an :class:`~openghg_inversions.models.outer_flux.OuterFluxChoice`, which
  centres the sampled outer flux columns and holds the other outer states at
  their prior mean.

The recipes reject the outer-flux option without a boundary mean shift. The
level remains an observation-space discrepancy: the data cannot say whether it
comes from the boundary or from outer fluxes, and the option does not make the
remaining boundary and outer patterns independent.

Outer-flux modes
----------------

Outer states are the flux states whose ``basis_group`` coordinate is in
``outer_groups``. They stay in the ordinary ``flux_scaling`` state, so CO2
outputs, state activity and :doc:`biosphere calibration <biosphere_prior>`
work as they do without the option.

``"multiplicative"``
   Samples the outer states, except those whose ``source`` is in
   ``fixed_sources`` (for example ``("ocean",)``), which are held at their
   prior mean. A calibrated biosphere prior is passed as the recipe's
   ``retained_prior`` as usual.
``"fixed"``
   Holds every outer state at its prior mean.
``"additive"``
   Holds every outer state at its prior mean and adds centred concentration
   patterns for net-biosphere (``net_bio``) and fossil-fuel (``FF``) density
   corrections, each with a standard normal coefficient. Use one net
   biosphere correction rather than separate GPP and TER corrections on the
   same support: sampling both would add a direction the data cannot
   allocate. This mode is available from Python only.

Holding a state at its prior mean while sampling the marginal prior of the
others is exact only when the held state is uncorrelated with them, so the
option rejects held outer states with nonzero prior covariance.

Observation-space centring
--------------------------

Give every site equal total weight, spread evenly over that site's rows. Call
the resulting row weights :math:`w`, with :math:`w^T\mathbf 1=1`. For the
sampled outer columns :math:`H_S` with prior mean :math:`m_S`, let
:math:`\mu_S=w^TH_S`. The recipe uses

.. math::

   H'_S=H_S-\mathbf 1\mu_S,\qquad f'=f+(\mu_S m_S)\mathbf 1,

so that

.. math::

   Hx+f=H_Ix_I+H_Fm_F+H'_Sx_S+f'+\left[\mu_S(x_S-m_S)\right]\mathbf 1

for inner states :math:`I` and held outer states :math:`F`. The bracketed
level is what the boundary mean shift absorbs. At the prior mean it is zero,
so the prior mean of the prediction is unchanged.

The mean shift reaches each row through the row's boundary mass fractions,
:math:`M\mathbf 1`, not through exactly 1. The absorption is therefore exact
only when every row's boundary mass sums to one (and, for a by-period mean
shift, within each period), as supported surface preparation guarantees.
Otherwise it is approximate.

The weights and column means come from the rows the option is bound to, so
binding a subset of rows defines a different prior. Only the column means are
computed; no observation-by-observation projection is formed.

Exact-zero columns of the raw :math:`H` are pruned first, as in the ordinary
recipe, and centring is applied to the retained columns. A sampled outer
column that becomes zero only after centring stays sampled: it has no centred
likelihood response, but its prior uncertainty and correlations still enter
the outer flux diagnostics.

Building a model
----------------

The state selection is done once per flux prior, and the row work once per
fit:

.. code-block:: python

   from openghg_inversions.models.boundary import AnomalyScalingChoice
   from openghg_inversions.models.outer_flux import OuterFluxChoice
   from openghg_inversions.rhime.co2.co2_cached_sigma_model import build_co2_cached_sigma_model

   boundary = AnomalyScalingChoice(
       {"pdf": "normal", "mu": 0.0, "sigma": level_sd}, pattern_sd,
   ).bind(observations, boundary_sensitivity)
   outer_states = OuterFluxChoice("multiplicative", ("outer",), ("ocean",)).resolve_states(flux_prior)
   outer_flux = outer_states.bind_rows(observations, flux_sensitivity)

   cached = build_co2_cached_sigma_model(
       flux_sensitivity, retained_prior=flux_prior,
       fixed_prior_contribution=fixed_prior_contribution,
       observations=observations, observation_error=observation_error,
       aggregation_error=aggregation_error, tau_hours=tau_hours,
       site_amplitude_prior=site_amplitude_prior,
       boundary=boundary, outer_flux=outer_flux,
   )

:func:`~openghg_inversions.rhime.co2.co2_model.build_co2_model` takes the same
``boundary`` and ``outer_flux`` arguments and the same likelihood options as
any centred CO2 boundary. Centred graphs need PyTensor ``floatX=float64``.

For additive mode, pass ``additive_design`` on ``(nmeasure, outer_pattern)``
and ``additive_density_sd`` on ``outer_pattern`` to ``bind_rows``. The design
columns are concentration responses to one density standard deviation, in
the observations' units, with an ``outer_source`` label of ``net_bio`` or
``FF`` per column; the widths are converted to ``mol m-2 s-1``.

Atmospheric baseline outputs
----------------------------

With this option ``mu_bc`` is no longer the whole atmospheric baseline, because
it also carries the outer level. Use
:func:`~openghg_inversions.rhime.co2.atmospheric_baseline.reconstruct_atmospheric_baseline`
with the bound ``outer_flux.data``, the raw ``H``, fixed term and prior mean,
and posterior draws of ``flux_scaling`` and ``mu_bc`` to report:

``atmospheric_baseline``
   :math:`f+\mu_{bc}+H_Ox_O-\mu_S(x_S-m_S)`, which equals
   ``modelled_concentration`` less the inner flux contribution and any offset.
``outer_flux_contribution``
   The physical outer flux contribution :math:`H_Ox_O`.
``outer_removed_level``
   :math:`\mu_S(x_S-m_S)`, the part of the outer flux change carried by the
   mean shift.
``outer_centred_correction``
   :math:`H'_S(x_S-m_S)`, the part the likelihood sees directly.

In additive mode it also reports ``outer_pattern_contribution`` and
``outer_density_correction`` in ``mol m-2 s-1``.

Widths and validation
---------------------

The mean shift, anomaly and outer widths are explicit scientific choices;
examples do not define recommended production uncertainties. Matching a
median pointwise prior standard deviation matches that statistic, not the full
concentration covariance. Improving the concentration fit alone does not show
better recovery of inner fluxes or identify the source of a baseline
discrepancy.
