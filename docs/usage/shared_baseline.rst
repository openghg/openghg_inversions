Shared atmospheric level with boundary and outer corrections
============================================================

This opt-in Python recipe gives boundary-condition (BC) and exterior-flux
corrections one shared atmospheric level. It starts from an existing coherent
reduction, the prepared centered boundary transport and a :doc:`boundary
correction structure <boundary_correction_structures>` that supplies the BC
anomaly patterns. Existing model defaults are unchanged.

The atmospheric level replaces any boundary mean shift, so the boundary choice
must omit its mean prior: period contrasts or independent anomalies with
``mean_prior=None``, or anomaly scaling with ``mean_prior=None``. Weighted
additive anomalies are not supported here. ``boundary=None`` keeps only the
prescribed boundary.

The shared level is an observation-space discrepancy. Its physical source
cannot be identified uniquely as boundary conditions or exterior flux. Removing
the common mean from correction patterns also does not make their remaining
patterns independent or remove their overlap.

Observation-space centring
--------------------------

Assign every retained site equal total weight, then distribute its weight
uniformly among that site's retained observations. Let the resulting column
vector be :math:`a`, with :math:`a^T\mathbf 1=1`. For each transported
correction operator :math:`D`, retain its column means and use

.. math::

   m_D=a^TD,\qquad D_c=D-\mathbf1m_D.

For BC patterns, :math:`D_{BC}` is the boundary family's ``anomaly_factor()``,
its anomaly design already scaled by the pattern width. The sampled ``bc_pattern_standardized`` vector
therefore contains independent standard-normal coefficients, and
:math:`D_{BC,c}v` has concentration units. Only nonempty active pattern axes
create sampled states; reconstruction supplies absent corrections as zeros.

Only the column means are computed; no observation-by-observation projection
matrix is formed. BC period gating precedes this operation. In particular,
centring a gated column may give nonzero values outside its original period.
This is the intended removal of its whole-window constant direction.

The weights and means are computed from the rows the choice is bound to.
To fit a subset with the centring of a larger comparison support, bind on the
larger support and subset the bound record with
``dataclasses.replace(baseline, data=baseline.data.isel(nmeasure=rows))``;
binding on the subset itself defines a different prior. The concentration
reference :math:`c` stays scalar throughout the inversion; the prescribed
boundary :math:`b_{BC}=A\mathbf1+cM\mathbf1` keeps its transported shift.

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
ocean. The fixed affine contribution is included once.
This operation retains the prepared coherent operator and aggregation-error
covariance; it does not regenerate a native prior or reduction.

The physical outer deviation :math:`H_o(x_o-m_o)`, its removed mean
:math:`(a^TH_o)(x_o-m_o)`, and the centred atmospheric correction
:math:`H_{o,c}(x_o-m_o)` have different meanings. Retain all three when
reporting flux changes alongside concentration predictions. The level
:math:`\delta` is reported separately.

Outer-prior choices
-------------------

The supported outer choices are:

* Fixed outer flux: retain its prior contribution and remove its sampled
  correction states.
* Additive net biosphere and fossil fuel: replace the sampled outer states
  with signed Gaussian density corrections transported by unit-density
  operators. An explicit design may use any nonempty set of labelled
  net-biosphere and fossil-fuel columns.
* Multiplicative GPP, TER and fossil fuel: keep their retained positive
  coefficients and centre their transported deviations from the prior mean.
* The same multiplicative model with explicit
  :doc:`gross-biosphere calibration <biosphere_prior>` applied once to the
  retained prior.

The shared builders retain the joint physical prior even when centring makes
an outer sensitivity column exactly zero. Such a coefficient has no direct
centred likelihood response, but its uncertainty and correlations still enter
physical outer-flux diagnostics.

Ocean remains fixed. Additive GPP and TER corrections on identical
support can be replaced by one net correction with variance equal to the sum
of their independent variances. Sampling both gross corrections would add an
unidentified allocation direction. The additive recipe therefore replaces
the original sampled outer states instead of adding duplicate corrections.

Calibration and widths are caller choices. Matching a median pointwise prior
standard deviation matches that statistic, not the full concentration
covariance. Atmospheric, BC-pattern and outer widths are explicit scientific
choices; examples do not define recommended production uncertainties.

Binding and building a model
----------------------------

``SharedBaselineChoice(boundary, delta_sd, mode)`` in
``openghg_inversions.models.shared_baseline`` holds the scientific options:
the boundary anomaly choice, the atmospheric width ``delta_sd`` in observation
concentration units, and one of the ``fixed``, ``additive``,
``multiplicative`` and ``multiplicative_shrunk`` outer modes. Its ``bind``
method takes the native observation rows, the labelled flux sensitivity,
retained prior, fixed affine contribution and prepared centered boundary
transport. ``outer`` and ``sampled_outer`` are boolean DataArrays on the
retained-state axis; the latter selects adjustable outer states and excludes
fixed ocean.

Binding uses Pint to convert compatible flux, affine and additive operators to
the concentration units of the boundary reference. At model construction,
observations must declare the same unit label; the builder rejects a mismatch.

For example, with already prepared scientific inputs:

.. code-block:: python

   from openghg_inversions.models.boundary import AnomalyScalingChoice, RawCenteredTransport
   from openghg_inversions.models.shared_baseline import SharedBaselineChoice
   from openghg_inversions.rhime.co2.shared_baseline_model import (
       build_shared_co2_cached_sigma_model,
   )

   choice = SharedBaselineChoice(
       AnomalyScalingChoice(None, pattern_sd), atmospheric_delta_sd, "multiplicative",
   )
   baseline = choice.bind(
       observations, flux_sensitivity=flux_sensitivity, retained_prior=retained_prior,
       fixed_prior_contribution=fixed_prior_contribution,
       boundary_transport=RawCenteredTransport.validate(prepared_inputs),
       outer=outer, sampled_outer=sampled_outer,
   )
   cached = build_shared_co2_cached_sigma_model(
       baseline, observations=observations, observation_error=observation_error,
       aggregation_error=aggregation_error, tau_hours=tau_hours,
       site_amplitude_prior=site_amplitude_prior,
   )

The ordinary ``build_shared_co2_model`` accepts the same bound baseline and
existing CO2 likelihood options, including a custom likelihood builder. Both
routes consume the same mean. They retain existing observation-error and
aggregation-error policies. This recipe supports one CO2 channel; linked
CO2/O2 construction is outside its interface.

For ``multiplicative_shrunk``, additionally pass ``calibrated_prior``: a
resolved positive ``BiospherePrior`` whose baseline mean and covariance match
the supplied retained prior. Its biosphere selector must be contained in
``sampled_outer``. The resolved calibration is recorded alongside the
transformed prior; binding never recalibrates it.

For ``additive``, additionally supply ``additive_design`` on
``(nmeasure, outer_pattern)`` and ``additive_density_sd`` on ``outer_pattern``.
The design columns are unit-density responses multiplied by their explicit
physical density standard deviations. Both arrays have the same unique
pattern labels. A nonempty ``outer_source`` coordinate labels ``net_bio`` or ``FF`` columns;
density widths are converted to ``mol m-2 s-1``. The bound
baseline records these widths and the unit-density operator. Backend Gaussian
coefficients have unit standard deviation; ``outer_density_correction``
restores physical density corrections.

Fixing or removing states requires zero covariance between those states and
the sampled retained states. Unsupported nonzero covariance fails explicitly.
Multiplicative modes preserve nonzero inner/adjustable-outer covariance in
their retained joint marginal. The caller continues to pass the original
prepared aggregation covariance to model construction.

Trace records and contribution outputs
--------------------------------------

There is no separate shared-baseline artifact to save. After normal CO2
trace-coordinate and absolute-predictive restoration, call
``annotate_shared_baseline_trace(trace, baseline)`` from
``openghg_inversions.rhime.co2.shared_baseline_model``. It records the bound
baseline's operators, centring, widths and any calibration as the
``shared_baseline`` group, and its options in the trace metadata, so replay
needs no preparation. The normal ``save_trace``/``load_trace`` codec retains
that group.

Call ``reconstruct_shared_baseline(trace.shared_baseline.to_dataset(),
trace.posterior.to_dataset())`` (or pass ``baseline.data``) to reconstruct
these terms from the sampled coefficients and recorded operators. The graph
retains the canonical scalar ``bc_mean_shift``, ``bc_pattern_standardized`` and (for additive outer
mode) ``outer_pattern_coefficient`` states; it does not save repeated
observation-sized physical diagnostics for every draw.

Use ``shared_baseline_contribution`` for the complete atmospheric baseline
and ``modelled_concentration`` for absolute total concentration. Inspect
``baseline_delta``, ``bc_raw_correction``, ``bc_removed_mean``,
``bc_centered_correction``, ``outer_raw_correction``, ``outer_removed_mean``
and ``outer_centered_correction`` for their separate terms.
The reconstructed ``baseline_delta`` field reports the canonical
``bc_mean_shift`` draw; it is not a second sampled variable.
``bc_raw_correction`` is the transported physical BC correction;
``bc_centered_correction`` enters the atmospheric likelihood, and
``bc_removed_mean`` is their difference.
``outer_physical_contribution`` includes the outer prior contribution plus
its raw physical adjustment. These distinctions are needed because centring
changes atmospheric accounting without changing what a physical flux
adjustment transports.

Validation and scientific limits
--------------------------------

Improving concentration fit alone does not establish better recovery of inner
fluxes or identify the physical source of a baseline discrepancy.

The atmospheric level is the only level: its width is the explicitly
supplied ``delta_sd``, and boundary choices with their own mean prior are
rejected rather than silently replaced.
