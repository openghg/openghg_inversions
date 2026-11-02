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

Inputs and explicit preparation
-------------------------------

Start with maintained ``Co2O2PreparedInputs`` scientific arrays. Acquisition
and preparation own source selection, units, filtering and tracer alignment.
O2 sensitivities already contain the campaign's signed oxidative responses;
the shared builder does not apply an additional ratio. Different channel
site/time supports are supported.

The existing linked likelihood requires common concentration units, usually
ppm. Convert values and covariance axes consistently before preparing the
shared baseline. Keep one scalar reference and explicit physical level and
pattern widths per tracer. O2 mole fraction in ppm is different from per-meg
O2/N2 or APO; a unit label must not substitute for a physical conversion.

Use ``prepare_boundary_correction`` separately for the two native observation
axes, supplying each channel's float64 ``H_bc_anomaly``, dimensionless
``H_bc_mass`` and scalar ``bc_reference``. Resolve any coefficient frequency
and inversion-start anchor there. The shared recipe replaces that correction's
level with its own single atmospheric delta; it does not sample both levels.

``prepare_linked_shared_baseline`` is exported from
``openghg_inversions.rhime.co2``:

.. code-block:: python

   from openghg_inversions.rhime.co2 import prepare_linked_shared_baseline

   frozen = prepare_linked_shared_baseline(
       co2_sensitivity=linked.co2_sensitivity,
       o2_sensitivity=linked.o2_sensitivity,
       retained_prior=linked.retained_prior,
       fixed_prior_contribution=fixed_by_channel,
       boundary_correction=boundary_by_channel,
       outer=outer,
       sampled_outer=sampled_outer,
       mode="multiplicative",
       delta_sd=delta_sd_by_channel,
   )

``fixed_by_channel`` and ``boundary_by_channel`` are mappings keyed exactly
by ``co2`` and ``o2``. Fixed contributions use the same native row axis as
their tracer sensitivity. ``delta_sd_by_channel`` contains positive numbers
in the declared channel concentration units; there are no production widths.
``outer`` and ``sampled_outer`` are labelled boolean arrays on the original
joint retained-state axis. The latter selects adjustable outer coefficients
and excludes outer ocean. Removing a state with covariance against sampled
states is explicitly unsupported. Nonzero covariance among retained inner
and sampled outer coefficients is preserved.

The returned mapping contains one Dataset per channel, preserving its native
row dimension. It freezes equal total weight per retained site, uniform weight
within each site's rows, and the whole-support column means. BC period gating
precedes centring. Subset the frozen channel rows afterwards; do not call
preparation again to predict a subset.

For ``multiplicative_shrunk``, also pass ``calibrated_prior``. Its original
prior must match exactly, its positive family must be available, and its
biosphere selector must lie inside ``sampled_outer``. Saved resolved moments
are reused rather than calibrated against the current rows. This is a
retained-state intervention with the coherent transport, affine term and
unresolved aggregation covariance kept frozen.

Native O2 precision
-------------------

With native anomaly :math:`A`, mass response :math:`g=M\mathbf1` and scalar
reference :math:`c`, the prescribed contribution to the centred likelihood is

.. math::

   A\mathbf1+c(g-1).

The recipe computes that expression directly in float64. It does not recover
an anomaly by subtracting a roughly 200,000 ppm O2 background. The reference
never multiplies correction widths. Absolute predictions restore the scalar
once. Fresh artifacts retain the native centred operands; the linked builder
rejects older shared artifacts lacking them and asks for preparation from the
original BC correction. The previous CO2-only saved route remains available.

Build, save and reconstruct
---------------------------

Both model builders are exported from ``openghg_inversions.rhime.co2``:

.. code-block:: python

   from openghg_inversions.rhime.co2 import (
       build_shared_co2_o2_cached_sigma_model,
   )

   cached = build_shared_co2_o2_cached_sigma_model(
       frozen,
       observations=linked.observations,
       independent_error_sd=linked.independent_error_sd,
       aggregation_error=linked.aggregation_error,
       tau_hours=tau_hours,
       site_amplitude_prior=site_amplitude_prior,
   )

``build_shared_co2_o2_model`` accepts the same scientific arrays and the
existing ordinary linked noise choices. Both builders construct one
``flux_scaling`` state. The two ``co2_bc_level``/``o2_bc_level`` variables have
independent physical level priors. Joint aggregation-error cross-channel
entries are preserved. Cached inference requires the existing matched
sigma-then-state sampler; an arbitrary sampler must not consume its quadratic
cache without that update protocol.

Save each prepared Dataset using the existing MultiIndex codec. After normal
linked sample-coordinate and absolute-predictive restoration, call
``annotate_linked_shared_baseline_trace(trace, model)``. It retains the complete
frozen preparations in ``shared_baseline_co2`` and ``shared_baseline_o2``
groups. Use ``save_trace``/``load_trace`` for the resulting artifact.

``reconstruct_linked_shared_baseline(frozen, posterior)`` consumes restored
labelled saved coefficients and returns native-channel Datasets. It builds no
model, retrieves no data and recalibrates nothing. Outputs include absolute
``modelled_concentration``, ``shared_baseline_contribution``, physical outer
contribution, raw and centred BC/outer corrections, their removed means and
``baseline_delta``. Mismatched pattern labels fail rather than being silently
relabelled. In-graph row diagnostics use the gathered joint axis and are
zero on the other tracer's rows; pure reconstruction uses native channel rows.

Why linked additive net biosphere is rejected
---------------------------------------------

The previous CO2-only additive construction collapses GPP and TER on common
unit-density support to one net correction. Different GPP/TER O2 responses
can retain information about their gross allocation even when their CO2 net
is unchanged. A universal net-biosphere oxidative ratio would lose that
direction. Linked ``additive`` therefore raises an explicit error.

A future additive recipe needs either the necessary gross degree of freedom
and a declared joint prior, or an explicit fixed gross-allocation convention
with its transported responses. The :download:`input evidence and linear
oracle <../validation/linked_shared_baseline_inputs.md>` show the condition
for an exact collapse and actual differing campaign responses.

Scope and acceptance evidence
-----------------------------

This page documents the explicit Python preparation/model/reconstruction seam.
It does not claim a complete acquisition-to-products runner or generic linked
product manifests. The :download:`design and acceptance record
<../validation/linked_shared_baseline.md>` separates those workflow acceptance
conditions from completed model checks. Corrected endpoint deliveries and fresh
convolutions have different identities and timing policies; selecting one must
be explicit. Historical inner-only enhancement observations must not be fitted
to an absolute delivered baseline without a coherent observation preparation.
