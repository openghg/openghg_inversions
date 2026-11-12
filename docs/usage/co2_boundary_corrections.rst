.. _co2-boundary-corrections:

Configure CO₂ boundary corrections
==================================

This page configures the boundary-condition (BC) term of the CO₂ and linked
CO₂/O₂ recipes: which boundary family to fit, the inputs it needs, how to
select it from TOML or Python, and which components a fit saves. The
transport operators and the additive formulation are defined in
:doc:`centered_boundary_conditions`; the alternative additive structures and
anomaly scaling are compared in :doc:`boundary_correction_structures`.

Prerequisites
-------------

* A prepared CO₂ or linked CO₂/O₂ artifact. Preparation stores either the
  absolute ``H_bc`` or native centered operands (signed ``H_bc_anomaly``,
  dimensionless ``H_bc_mass`` and scalar ``bc_reference``); see
  :doc:`surface_preparation`.
* For centered operands, a PyTensor process configured with
  ``floatX=float64``. Binding checks this once.
* For weighted additive corrections, the complete averaging weights
  ``bc_centering_weights``, stored when the artifact is prepared (see
  :ref:`co2-boundary-weights`).

Choose a boundary
-----------------

A boundary is one *choice*: the prior specification before binding, holding
no arrays. Two choices are configured here, both from
:mod:`openghg_inversions.models.boundary`:

* ``MultiplicativeChoice(prior, activity=None, frequency=None,
  anchor_time=None)`` scales the boundary sensitivity by dimensionless
  states. ``activity`` marks boundary states as active or fixed. It accepts
  absolute ``H_bc`` or centered operands.
* ``AdditiveChoice(mean_prior, anomaly_scale, frequency=None,
  anchor_time=None)`` keeps the projected baseline :math:`H_{bc}\mathbf1`
  fixed and infers a constant correction :math:`\delta` plus weighted
  region--period anomalies, in concentration units. It replaces the
  multiplicative scaling prior, has no scaling prior or state activity, and
  requires centered operands.

``frequency`` and ``anchor_time`` select the boundary periods with the syntax
and calendar of ``bc_freq``: ``None`` gives one period over the window,
``"monthly"`` calendar months, and a fixed duration such as ``"7D"`` periods
anchored at ``anchor_time``. A ``None`` anchor uses the fit's earliest
selected observation; the RHIME parameter adapter instead defaults
``bc_freq_anchor`` to the configured start date.

The additive formulation is

.. math::

   d_j=\delta+u_j,\qquad \sum_jw_ju_j=0,\qquad \sum_jw_j=1,

with one :math:`\delta` over the entire inversion window and one weighted
constraint over all region--period states :math:`j`. Individual periods can
have nonzero mean anomalies. The caller supplies positive weights; the model
normalizes them to sum to one. The correction enters the observations through
the native dimensionless transport ``H_bc_mass`` (:math:`M`):

.. math::

   g &= M\mathbf1, \\
   \mu_{bc} &= H_{bc}\mathbf1+g\delta+Mu.

``anomaly_scale`` is the fixed scale
:math:`s_{bc}` of the anomaly prior in concentration units; it controls
pairwise departures rather than the marginal standard deviation of each
anomaly. Its covariance and interpretation are derived in
:ref:`boundary-anomaly-prior-uncertainty`. Set ``anomaly_scale=0`` to infer
only the common level from ``mean_prior``: all anomalies are exactly zero
and no contrast variables are sampled; with a single boundary state the
anomaly is also exactly zero. Negative and nonfinite scales are
rejected. Choose the common-mean prior and anomaly scale for uncertainty in
the supplied boundary product.

.. _co2-boundary-weights:

Store the averaging weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Store the complete averaging weights on ``(bc_curtain, bc_period)`` with
units ``"1"``. Period labels are the actual interval start timestamps returned
by the same calendar that constructs the transport mask. For example::

    weights = xr.DataArray(
        values,
        dims=("bc_curtain", "bc_period"),
        coords={"bc_curtain": curtains, "bc_period": period_starts},
        attrs={"units": "1"},
    )

Keep every curtain/period weight, including unobserved states; unequal,
nonseparable weights are supported. Uniform weights specify an equal-state
average. For a native-field average, choose weights using the
:ref:`declared measure and basis <boundary-averaging-weights>`.

Pass the weights to preparation as ``bc_centering_weights``. The CO₂
artifact retains this unstacked array, separate from reusable spatial
transport, and raw spatial preparation stays independent of fit periods.
Selecting another frequency requires matching explicit weights for that
state partition. Binding stacks the weights once per channel and fit; exact
labelled alignment rejects a different calendar before evaluating numerical
payloads.

A configured CO₂ preparation produces centered operands for an additive
choice automatically and reads the weights from
``prepared_inputs.bc_centering_weights_path`` (a netCDF file with one
``bc_centering_weights`` variable); see :ref:`co2-configured-preparation`.

Configure a CO₂ fit
-------------------

``[model.boundary]`` without a ``type`` is multiplicative and uses the
default scaling prior unless ``prior`` is configured. ``frequency`` and
``anchor_time`` select its periods. For additive corrections, use an artifact
containing native centered operands and ``bc_centering_weights`` and
configure:

.. code-block:: toml

   [model.boundary]
   type = "additive"
   anomaly_scale = 0.2

   [model.boundary.mean_shift_prior]
   pdf = "normal"
   mu = 0.0
   sigma = 0.5

The mean prior uses the ordinary scalar prior dictionary. Its location and
scale, and ``anomaly_scale``, use the observations' concentration units. For
a common-level-only configuration, use ``anomaly_scale = 0.0`` with the same
``mean_shift_prior`` table.

In Python, pass the choice as ``boundary=`` to
:func:`~openghg_inversions.rhime.co2.co2_runner.materialize_co2_inputs` or
:func:`~openghg_inversions.rhime.co2.co2_cached_sigma_runner.build_rhime_co2_cached_sigma`,
or under ``model["boundary"]`` to ``build_co2_inference``. Staged
``prepare`` and :func:`~openghg_inversions.rhime.co2.run_co2` forward
``bc_centering_weights`` to preparation when they prepare from canonical
inputs:

.. code-block:: python

   prepared = prepare_co2_inputs(
       canonical_inputs,
       reduction,
       bc_centering_weights=boundary_weights,
   )
   from openghg_inversions.rhime.co2 import (
       materialize_co2_inputs, build_co2_from_materialized, annotate_co2_trace,
   )
   from openghg_inversions.models.boundary import AdditiveChoice
   from openghg_inversions.rhime.sampling import RhimeSampler, sample_rhime_model

   choice = AdditiveChoice({"pdf": "normal", "mu": 0.0, "sigma": 0.5}, anomaly_scale=0.2)
   inputs, priors, boundary = materialize_co2_inputs(prepared_inputs=prepared, boundary=choice)
   built = build_co2_from_materialized(
       inputs, boundary=boundary, prepared_inputs=prepared, priors=priors,
   )
   trace = sample_rhime_model(built, RhimeSampler())
   trace = annotate_co2_trace(trace, built, concentration_units=prepared.inv_inputs.mf.attrs["units"])

The 0.5 ppm common-mean scale and 0.2 ppm anomaly scale are illustrative.
The ordinary and cached fixed-OU CO₂ recipes use the same formulation. The
legacy satellite column scaling path is separate from this centered
surface-boundary representation.

Configure a linked CO₂/O₂ fit
-----------------------------

The linked ordinary and cached fixed-OU recipes apply each family
independently to each channel. In Python, pass
``boundary={"co2": co2_choice, "o2": o2_choice}``; ``None`` disables a
channel's boundary explicitly. A channel with prepared transport but no entry
has no boundary term, and the runner warns that its transport is unused. In
TOML, configure ``[channels.<name>.boundary]`` with the same keys as
``[model.boundary]``; ``enabled = false`` disables that channel.

Preparation stores each channel's ``bc_centering_weights`` argument in that
channel's native centered ``boundary_operands``. Each channel has its own
scalar mean, weighted anomalies and explicit averaging weights. No site or
channel is an identifiability reference for the other. Each channel
independently supports a zero anomaly scale, so one can use a
common-level-only correction while the other retains sampled contrasts or a
multiplicative boundary prior.

For O₂ near 200,000 ppm, a multiplicative standard deviation of 0.01 gives
about 2,000 ppm uncertainty in the boundary concentration before transport.
The observation-space uncertainty depends on transport responses. Additive
scales instead express uncertainty directly in ppm and do not grow with the
absolute O₂ background. This boundary formulation supports the existing linked
transport-equivalent ppm convention; it adds no per-meg or APO conversion.
For example, the following TOML uses a 2 ppm common
O₂ correction scale and a 1 ppm pre-centring anomaly scale:

.. code-block:: toml

   [channels.o2.boundary]
   type = "additive"
   anomaly_scale = 1.0

   [channels.o2.boundary.mean_shift_prior]
   pdf = "normal"
   mu = 0.0
   sigma = 2.0

These widths and the CO₂ widths above are illustrative scientific choices,
not defaults or recommendations. Select them from uncertainty in the supplied
boundary product and assess their sensitivity for the observing network.

Saved boundary components
-------------------------

Boundary and offset contributions remain separate from
``co2_flux_contribution`` and are included in ``modelled_concentration``, the
likelihood and sampled outputs. Fitted metadata, roles and physical
coefficient units are recorded with the trace; scientific replay preserves
those units independently of contribution units.

The additive CO₂ recipes save ``bc_mean_shift``, the labelled ``bc_anomaly``
and the total ``bc_correction`` (:math:`\delta\mathbf{1}+u`). Observation-space
outputs are ``mu_bc_reference`` for :math:`H_{bc}\mathbf{1}`, stored once in
``constant_data``, and sampled ``mu_bc_mean_shift`` for :math:`g\delta` and
``mu_bc_anomaly`` for :math:`Mu`, where :math:`g=M\mathbf1`. Their sum is
``mu_bc``, the complete boundary contribution included once in
``modelled_concentration`` and the likelihood. Public concentration products
broadcast the fixed reference over requested draws. The additive model has no
sampled boundary-scaling output. With zero anomaly scale or one boundary
state, ``bc_anomaly`` and ``mu_bc_anomaly`` are stored once as zeros in
``constant_data`` and broadcast over the draws in public products; the total
correction remains the common mean in every state.

The linked recipes prefix saved outputs with the channel: for example,
``o2_bc_mean_shift``, ``o2_bc_anomaly``, ``o2_bc_correction``, and the
``o2_mu_bc_reference``, ``o2_mu_bc_mean_shift`` and ``o2_mu_bc_anomaly``
components. ``o2_mu_bc`` is the complete O₂ boundary prediction; CO₂ uses
the corresponding ``co2_`` names.

Offsets and what centring identifies
------------------------------------

The weighted constraint uniquely decomposes a given boundary correction into
its mean and anomalies, removing the redundant direction within that
decomposition. It does not guarantee identifiability against site offsets,
boundary patterns or flux contributions: transport can leave boundary states
poorly observed or make their predictions difficult to distinguish. The `Stan User's
Guide on collinearity
<https://mc-stan.org/docs/stan-users-guide/problematic-posteriors.html#collinearity>`_
explains why redundant or nearly redundant prediction directions remain a
problem for inference.

A constant observation offset describes persistent observation bias, while
:math:`\delta` describes a constant correction to the supplied boundary
product. If :math:`g=\mathbf1`, these distinct components have the same
observation-space direction. The likelihood then identifies their sum;
separate prior information or external constraints are needed to distinguish
them. Finer boundary periods allow anomalies to vary over time while
:math:`\delta` remains constant over the whole inversion. Assess transport
sensitivity, BC--flux confounding and prior sensitivity when interpreting
these corrections.

.. warning::

   For complete normalized transport and the default NESW basis,
   :math:`g=\mathbf1` is guaranteed. The builders warn when additive
   boundaries are combined with unrestricted observation offsets. Offsets
   default to ``per_site=True, drop_first=False``. Setting
   ``drop_first=True`` removes the first encountered site's offsets; it does
   not select a named calibration reference. This removes the duplicate
   constant direction, but does not guarantee separation from boundary
   anomalies or fluxes. Centered site offsets are not currently implemented.
   See :doc:`../development/boundary_offset_identifiability` for the
   investigation.

.. _co2-boundary-binding:

Advanced: binding and subsetting in Python
------------------------------------------

The runners take the choice and own binding. One boundary follows this
sequence:

1. **Choice.** The configuration adapters produce the same choices from
   ``[model.boundary]`` and ``[channels.<name>.boundary]`` as Python callers
   construct directly.
2. **Runner binding.** The runner calls
   :func:`~openghg_inversions.models.boundary.bind_prepared` once per
   channel and fit with whichever sensitivity representation preparation
   recorded. For centered operands, validated as
   :class:`~openghg_inversions.models.boundary.RawCenteredSensitivity`, it
   calls ``choice.bind(observations, sensitivity)``. For absolute ``H_bc`` it
   calls ``choice.bind_absolute(observations, H_bc)``, which only
   ``MultiplicativeChoice`` provides; additive choices reject absolute
   transport and ask for centered preparation.
3. **Bound component.** Binding returns a bound record
   (:class:`~openghg_inversions.models.boundary.Bound`) that pairs the choice
   with borrowed labelled sensitivity on the fit's observation rows and
   boundary periods. Its ``.data`` carries the sensitivity contract:
   :class:`~openghg_inversions.models.boundary.BoundCenteredSensitivity` for
   centered families, extended to
   :class:`~openghg_inversions.models.boundary.BoundAdditiveSensitivity` for
   weighted additive anomalies, and a plain ``H_bc`` Dataset for absolute
   multiplicative transport. These contracts are static annotations on
   ordinary Datasets carrying structural guarantees; numerical requirements
   are checked once after joint materialization.
4. **Numerical builder.** The runner materializes the bound record jointly
   with the selected core arrays and prior parameters and passes it to the
   builder as ``boundary`` (a channel mapping for linked recipes). The builder
   calls ``boundary.add(...)`` inside its model and records
   ``boundary.outputs()``. Builders do not bind periods again.

To subset a complete fitted period support, keep the bound record and select
only its observation rows. The full boundary-state support and resolved prior
remain on that record::

    from dataclasses import replace
    from openghg_inversions.models.boundary import AdditiveChoice, RawCenteredSensitivity, validate_centering_weights

    choice = AdditiveChoice(mean_prior, anomaly_scale, frequency="monthly")
    sensitivity = RawCenteredSensitivity.validate(raw_operands)
    validate_centering_weights(weights)
    sensitivity = RawCenteredSensitivity.with_centering_weights(sensitivity, weights)
    bound = choice.bind(observations, sensitivity)
    subset = replace(bound, data=bound.data.isel(nmeasure=selected_rows))

Pass ``subset`` as the builder's ``boundary`` alongside the same observation
row selection. ``bind`` accepts raw operands only; do not bind already
expanded transport again. Rebinding raw inputs on a smaller window can remove
complete periods and requires matching weights.
