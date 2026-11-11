Linked CO2/O2 shared atmospheric baselines
==========================================

The linked Python recipe uses one retained flux draw for both tracers and
separate atmospheric level corrections for CO2 and O2. Within each tracer,
that level is shared between boundary-condition (BC) and outer-flux
corrections. Its physical source is unidentified. This opt-in recipe extends
the :doc:`shared_baseline` equations; existing defaults are unchanged.

The initial supported outer modes are ``fixed``, ``multiplicative`` and
``multiplicative_shrunk``. The latter accepts an already resolved positive
:doc:`biosphere_prior` calibration against the same original joint prior.
Outer ocean stays fixed; inner ocean states retain their existing treatment.
The prior, joint error covariance, fixed coherent affine contribution and
paired tracer responses retain their scientific labels.

.. _linked-shared-baseline-precision:

Inputs and binding
------------------

Start Python with ``PYTENSOR_FLAGS=floatX=float64`` before importing OpenGHG
Inversions, PyMC or PyTensor. If other PyTensor flags are configured, preserve
them and add or update only the ``floatX`` entry. Float64 input arrays alone do
not select the runtime precision: the default float32 runtime is rejected by
the native centred-boundary precision guard.

Start with maintained ``Co2O2PreparedInputs`` scientific arrays. Acquisition
and preparation own source selection, units, filtering and tracer alignment.
O2 sensitivities already contain the source-specific signed oxidative responses;
the shared builder does not apply an additional ratio. Different channel
site/time supports are supported.

The existing linked likelihood requires common concentration units, usually
ppm. Convert values and covariance axes consistently before binding the
shared baseline. Keep one scalar reference and explicit physical level and
pattern widths per tracer. O2 mole fraction in ppm is different from per-meg
O2/N2 or APO; a unit label must not substitute for a physical conversion.

Give each tracer its own :class:`~openghg_inversions.models.shared_baseline.SharedBaselineChoice`
with the same outer mode. Its boundary choice (see
:doc:`boundary_correction_structures`) omits a mean prior: the recipe's single
atmospheric delta per tracer is the only level. ``bind_linked_shared_baseline``
binds both against the joint state:

.. code-block:: python

   from openghg_inversions.models.boundary import AnomalyScalingChoice, RawCenteredSensitivity
   from openghg_inversions.rhime.co2 import bind_linked_shared_baseline
   from openghg_inversions.models.shared_baseline import SharedBaselineChoice

   choices = {
       channel: SharedBaselineChoice(AnomalyScalingChoice(None, pattern_sd[channel]),
                                     delta_sd[channel], "multiplicative")
       for channel in ("co2", "o2")
   }
   baseline = bind_linked_shared_baseline(
       choices, native_observations,
       co2_sensitivity=linked.co2_sensitivity,
       o2_sensitivity=linked.o2_sensitivity,
       retained_prior=linked.retained_prior,
       fixed_prior_contribution=fixed_by_channel,
       boundary_sensitivity={channel: RawCenteredSensitivity.validate(operands[channel])
                           for channel in ("co2", "o2")},
       outer=outer,
       sampled_outer=sampled_outer,
   )

``native_observations``, ``fixed_by_channel`` and the transport are mappings
keyed exactly by ``co2`` and ``o2``. Fixed contributions use the same native
row axis as their tracer sensitivity. Each ``delta_sd`` is a positive number
in the declared channel concentration units; there are no production widths.
``outer`` and ``sampled_outer`` are labelled boolean arrays on the original
joint retained-state axis. The latter selects adjustable outer coefficients
and excludes outer ocean. Removing a state with covariance against sampled
states is explicitly unsupported. Nonzero covariance among retained inner
and sampled outer coefficients is preserved.

The returned mapping contains one bound baseline per channel, preserving its
native row dimension. Binding gives each retained site equal total weight,
uniform within that site's rows, and centres each correction on the bound
rows. BC period gating precedes centring. To fit a subset with the centring
of the full support, bind on the full support and subset each bound
baseline's ``data``.

For ``multiplicative_shrunk``, also pass ``calibrated_prior``. Its original
prior must match exactly, its positive family must be available, and its
biosphere selector must lie inside ``sampled_outer``. Its resolved moments
are used as supplied rather than calibrated against the current rows. This is a
retained-state intervention with the coherent transport, affine term and
unresolved aggregation covariance kept as prepared.

Native O2 precision
-------------------

For normalized surface transport on a complete NESW or twelve-region partition,
:math:`M\mathbf1=1`. With native anomaly :math:`A` and scalar reference
:math:`c`, the prescribed contribution to the centred likelihood is

.. math::

   A\mathbf1,

and the observations are shifted by :math:`cM\mathbf1`.

The recipe computes that expression directly in float64. It does not recover
an anomaly by subtracting a roughly 200,000 ppm O2 background. The reference
never multiplies correction widths. Absolute predictions restore the scalar
once. Both single-channel and linked recipes retain the native centred anomalies.

Build, record and reconstruct
-----------------------------

Both model builders are exported from ``openghg_inversions.rhime.co2``:

.. code-block:: python

   from openghg_inversions.rhime.co2 import (
       build_shared_co2_o2_cached_sigma_model,
   )

   cached = build_shared_co2_o2_cached_sigma_model(
       baseline,
       observations=linked.observations,
       independent_error_sd=linked.independent_error_sd,
       aggregation_error=linked.aggregation_error,
       tau_hours=tau_hours,
       site_amplitude_prior=site_amplitude_prior,
   )

``build_shared_co2_o2_model`` accepts the same scientific arrays and the
existing ordinary linked noise choices. Both builders construct one
``flux_scaling`` state. The two ``co2_bc_mean_shift``/``o2_bc_mean_shift`` variables have
independent physical level priors. Joint aggregation-error cross-channel
entries are preserved. Cached inference requires the existing matched
sigma-then-state sampler; an arbitrary sampler must not consume its quadratic
cache without that update protocol.

The direct recipe binds its choices at construction and records the bound
operators in the trace. The optional :doc:`linked_shared_workflow` snapshots
that same bound record in the linked prepared-input artifact for saved phases
and frozen full-support centring. After normal linked sample-coordinate
and absolute-predictive restoration, call
``annotate_linked_shared_baseline_trace(trace, baseline)``. It records each
bound baseline in the ``shared_baseline_co2`` and ``shared_baseline_o2``
groups. Use ``save_trace``/``load_trace`` for the resulting trace.

``reconstruct_linked_shared_baseline(groups, posterior)``, with the channel-keyed
recorded groups (or ``{channel: baseline[channel].data}``), consumes restored
labelled saved coefficients and returns native-channel Datasets. It builds no
model, retrieves no data and recalibrates nothing. Outputs include absolute
``modelled_concentration``, ``shared_baseline_contribution``, physical outer
contribution, raw and centred BC/outer corrections, their removed means and
``baseline_delta``. Physical flux products lazily expand sampled states with
fixed outer prior means from the recorded original prior; saved traces retain
only the selected ``flux_scaling`` vector. Mismatched pattern labels fail rather
than being silently relabelled. Concentration diagnostics are reconstructed
from saved coefficients and recorded operands; they are not duplicated for every
sampled draw. Model components are padded onto the joint observation axis;
pure reconstruction returns native channel rows.

Why linked additive net biosphere is rejected
---------------------------------------------

The previous CO2-only additive construction collapses GPP and TER on common
unit-density support to one net correction. Different GPP/TER O2 responses
can retain information about their gross allocation even when their CO2 net
is unchanged. A universal net-biosphere oxidative ratio would lose that
direction. Linked ``additive`` therefore raises an explicit error.

A future additive recipe needs either the necessary gross degree of freedom
and a declared joint prior, or an explicit fixed gross-allocation convention
with its transported responses. The model tests include an independent linear
oracle for the condition under which gross allocation remains observable.

Physical covariance and centring support
----------------------------------------

For the selected physical outer covariance ``Sigma``, the two raw removed
means have covariance ``m_co2.T @ Sigma @ m_o2``. Different source responses
need not make them perfectly correlated. Projection removes these constant
physical contributions from the atmospheric likelihood; independent
unattributed delta priors replace that atmospheric uncertainty. Physical flux
draws and raw contributions retain their chosen joint covariance.

The descriptive ``baseline_delta`` product is reconstructed from the
canonical scalar level and is not a separate graph variable. Boundary pattern
designs have concentration units and their standardized coefficients are
dimensionless. The shared prior covers the joint physical state; each channel
retains its own centring means and reference.
