Shared atmospheric level with boundary and outer corrections
============================================================

This opt-in Python recipe gives boundary-condition (BC) and exterior-flux
corrections one shared atmospheric level. It starts from an existing coherent
reduction and an already prepared :doc:`boundary correction
<boundary_pattern_priors>`. Existing model defaults are unchanged.

The shared level is an observation-space discrepancy. Its physical source
cannot be identified uniquely as boundary conditions or exterior flux. Removing
the common mean from correction patterns also does not make their remaining
patterns independent or remove their overlap.

Frozen observation-space centring
---------------------------------

Assign every retained site equal total weight, then distribute its weight
uniformly among that site's retained observations. Let the resulting column
vector be :math:`a`, with :math:`a^T\mathbf 1=1`. For each transported
correction operator :math:`D`, retain its column means and use

.. math::

   m_D=a^TD,\qquad D_c=D-\mathbf1m_D.

Only the column means are computed; no observation-by-observation projection
matrix is formed. BC period gating precedes this operation. In particular,
centring a gated column may give nonzero values outside its original period.
This is the intended removal of its whole-window constant direction.

Prepare and save the shared baseline on the complete comparison support.
Subset the prepared rows afterwards. Predictions on a subset reuse the frozen
means; recomputing weights or means would define a different prior. The
concentration reference :math:`c` stays scalar throughout the inversion.

Baseline accounting
-------------------

Let :math:`H_i,H_o` be the retained inner and outer operators, :math:`m_o` the
outer arithmetic prior mean, :math:`f` the complete fixed affine contribution,
and :math:`b_{BC}` the prescribed physical BC. The prescribed atmospheric
baseline is

.. math::

   b_0=f+b_{BC}+H_om_o.

For sampled multiplicative outer states, the absolute model mean is

.. math::

   \mu=H_ix_i+b_0+\mathbf1\delta+D_{BC,c}v
       +H_{o,c}(x_o-m_o).

The outer sum includes prescribed states that are subsequently fixed, including
ocean in the matched replay. The fixed affine contribution is included once.
This operation retains the prepared coherent operator and aggregation-error
covariance; it does not regenerate a native prior or reduction.

The physical outer deviation :math:`H_o(x_o-m_o)`, its removed mean
:math:`(a^TH_o)(x_o-m_o)`, and the centred atmospheric correction
:math:`H_{o,c}(x_o-m_o)` have different meanings. Retain all three when
reporting flux changes alongside concentration predictions. The level
:math:`\delta` is reported separately.

Outer-prior choices
-------------------

The four matched choices are:

* Fixed outer flux: retain its prior contribution and remove its sampled
  correction states.
* Additive net biosphere and fossil fuel: replace the sampled outer states
  with signed Gaussian density corrections transported by unit-density
  operators. The replay uses six net-biosphere and six fossil-fuel columns.
* Multiplicative GPP, TER and fossil fuel: keep their retained positive
  coefficients and centre their transported deviations from the prior mean.
* The same multiplicative model with explicit
  :doc:`gross-biosphere calibration <biosphere_prior>` applied once to the
  retained prior.

Ocean remains fixed in the replay. Additive GPP and TER corrections on identical
support can be replaced by one net correction with variance equal to the sum
of their independent variances. Sampling both gross corrections would add an
unidentified allocation direction. The additive recipe therefore replaces
the original sampled outer states instead of adding duplicate corrections.

Calibration and widths are caller choices. Matching a median pointwise prior
standard deviation matches that statistic, not the full concentration
covariance. The experimental 5 ppm level width, inherited BC pattern widths,
and matched outer widths are replay specifications, not recommended production
uncertainties. The four-arm experiment does not select scientific defaults.

Preparing and building a model
------------------------------

``prepare_shared_baseline`` in
``openghg_inversions.models.shared_baseline`` accepts the existing labelled
flux sensitivity, retained prior, fixed affine contribution and prepared BC
correction. ``outer`` and ``sampled_outer`` are boolean DataArrays on the
retained-state axis; the latter selects adjustable outer states and excludes
fixed ocean. The accepted ``mode`` values are ``fixed``, ``additive``,
``multiplicative`` and ``multiplicative_shrunk``. ``delta_sd`` is required in
the observation concentration units.

Supply numerical flux, affine and additive operators in the concentration
units declared by the prepared BC reference. The model requires the
observations to declare the same unit label and rejects a mismatch; this
recipe does not silently convert or relabel supplied numerical operators.

For example, with already prepared scientific inputs:

.. code-block:: python

   from openghg_inversions.models.shared_baseline import prepare_shared_baseline
   from openghg_inversions.rhime.co2.shared_baseline_model import (
       build_shared_co2_cached_sigma_model,
   )

   baseline = prepare_shared_baseline(
       flux_sensitivity, retained_prior, fixed_prior_contribution,
       boundary_correction, outer=outer, sampled_outer=sampled_outer,
       mode="multiplicative", delta_sd=level_sd,
   )
   cached = build_shared_co2_cached_sigma_model(
       baseline, observations=observations, observation_error=observation_error,
       aggregation_error=aggregation_error, tau_hours=tau_hours,
       site_amplitude_prior=site_amplitude_prior,
   )

The ordinary ``build_shared_co2_model`` accepts the same baseline and existing
CO2 likelihood options, including a custom likelihood builder. Both routes
consume the same frozen mean. They retain existing observation-error and
aggregation-error policies. This separate recipe currently supports CO2;
linked CO2/O2 construction is outside its supported interface.

For ``multiplicative_shrunk``, additionally pass ``calibrated_prior``: a
resolved positive ``BiospherePrior`` whose baseline mean and covariance match
the supplied retained prior. Its biosphere selector must be contained in
``sampled_outer``. The resolved calibration is stored alongside the transformed
prior; building or reopening the baseline never recalibrates it.

For ``additive``, additionally supply ``additive_design`` on
``(nmeasure, outer_pattern)`` and ``additive_density_sd`` on ``outer_pattern``.
The design columns are unit-density responses multiplied by their explicit
physical density standard deviations. Both arrays have the same unique
pattern labels. An ``outer_source`` coordinate identifies six ``net_bio`` and
six ``FF`` columns; density widths declare their physical units. The frozen
artifact retains these widths and the unit-density operator. Backend Gaussian
coefficients have unit standard deviation; ``outer_density_correction``
restores physical density corrections.

Fixing or removing states requires zero covariance between those states and
the sampled retained states. Unsupported nonzero covariance fails explicitly.
Multiplicative modes preserve nonzero inner/adjustable-outer covariance in
their retained joint marginal. The caller continues to pass the original
prepared aggregation covariance to model construction.

Saving, subsets and contribution outputs
----------------------------------------

Use the existing :mod:`openghg_inversions.serialization` MultiIndex codec to
save the returned Dataset, then reopen and restore its indexes before building
a model. A subset is simply ``baseline.isel(nmeasure=rows)`` with matching
observation and likelihood arrays. Stored column means, scalar reference,
density widths and calibration remain frozen.

After normal CO2 trace-coordinate and absolute-predictive restoration, call
``annotate_shared_baseline_trace(trace, model)`` from
``openghg_inversions.rhime.co2.shared_baseline_model`` to attach the complete
prepared Dataset as the ``shared_baseline`` group before saving the trace.
The replay example demonstrates this saved-model workflow.

Use ``shared_baseline_contribution`` for the complete atmospheric baseline
and ``modelled_concentration`` for absolute total concentration. Inspect
``baseline_delta``, ``bc_raw_correction``, ``bc_removed_mean``,
``bc_centered_correction``, ``outer_raw_correction``, ``outer_removed_mean``
and ``outer_centered_correction`` for their separate terms.
``outer_physical_contribution`` includes the outer prior contribution plus
its raw physical adjustment. These distinctions are needed because centring
changes atmospheric accounting without changing what a physical flux
adjustment transports.

Validation and scientific limits
--------------------------------

The :download:`matched replay report <../validation/shared_baseline_replay.md>` records
input identities, checks, sampling diagnostics and Monte Carlo uncertainty.
Improving concentration fit alone does not establish better recovery of inner
fluxes or identify the physical source of a baseline discrepancy. All replay
arms must use the same repaired endpoints, coherent inputs, likelihood and
prior widths.
