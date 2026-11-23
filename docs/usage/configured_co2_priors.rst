Configured CO2 priors
=====================

Use this workflow to configure :doc:`boundary_correction_structures` or the
outer-flux option of :doc:`atmospheric_baseline`. Both are choices bound to
the prepared inputs at model construction; nothing is prepared or saved
separately. Existing model defaults are unchanged. These APIs require the
package's supported Python 3.12 or 3.13 runtime.

.. _co2-configured-boundary-structures:

Boundary structures in configuration
------------------------------------

``[model.boundary]`` (and ``[channels.<name>.boundary]`` for linked CO2/O2)
selects the boundary family with ``type``; options of another family are
rejected:

* ``"multiplicative"`` (the default) uses
  :class:`~openghg_inversions.models.boundary.MultiplicativeChoice`, with
  optional ``prior``;
* ``"additive"`` uses :class:`~openghg_inversions.models.boundary.AdditiveChoice`,
  with required ``anomaly_scale`` and optional ``anomalies`` (``"weighted"``
  (default), ``"period_contrasts"`` or ``"independent"``);
* ``"anomaly_scaling"`` uses
  :class:`~openghg_inversions.models.boundary.AnomalyScalingChoice`, with
  required ``scale_sd`` and optional ``reference`` (``"stored"`` (default),
  ``"whole_window_transport_mean"`` or a number in observation concentration
  units).

``mean_shift_prior`` and ``mean_by_period`` apply to both corrections;
omitting ``mean_shift_prior`` omits the mean shift where the structure allows
it. ``frequency`` and ``anchor_time`` select the boundary period calendar.
Widths are plain numbers in observation concentration units, except the
dimensionless ``scale_sd``. Unknown or inapplicable options fail when the
configuration is parsed, before any file is opened:

.. code-block:: toml

   [model.boundary]
   type = "additive"
   anomaly_scale = 0.3
   anomalies = "period_contrasts"
   mean_shift_prior = { pdf = "normal", mu = 0.0, sigma = 0.5 }
   frequency = "monthly"

The boundary is bound to the prepared centered operands of each run; nothing
is prepared or saved separately, and the trace records the options and the
arrays the model used.

Protected inputs and outputs
----------------------------

``Co2ModelConfig.from_params`` validates choices without reading files.
``sampler_arguments()`` resolves sampling policy and ``model_arguments()``
binds the model choices. Before inference, the full ``run_co2`` and saved
``handoffs.sample`` routes check resolved trace and product targets against the
preparation and reconstruction files the run loads, including active
likelihood eigenbasis and country-reporting files and the consumed request.
Product writers check all destinations before writing, including consumed
stage manifests, prepared inputs, posterior and affine companion during saved
replay. This protection covers aliases and overlapping directory stores.

Configure the outer-flux option
-------------------------------

``[model.outer_flux]`` adds the outer-flux option of :doc:`atmospheric_baseline`
to the ordinary and cached fixed-OU CO2 recipes. Linked CO2/O2 configurations
use a top-level ``[outer_flux]`` table, because the flux state is shared by
both channels:

.. code-block:: toml

   [model.boundary]
   type = "anomaly_scaling"
   scale_sd = 0.8
   mean_shift_prior = { pdf = "normal", mu = 0.0, sigma = 2.0 }

   [model.outer_flux]
   mode = "multiplicative"   # or "fixed"
   outer_groups = ["outer"]
   fixed_sources = ["ocean"]

``outer_groups`` selects outer states by the ``basis_group`` coordinate of the
prepared retained prior, and ``fixed_sources`` holds those with a matching
``source`` coordinate at their prior mean. Model construction rejects the
option when the boundary has no ``mean_shift_prior`` (in each channel for
linked runs), or when the prepared state lacks those coordinates.

Preparation and saved preparations keep these coordinates but do not create
them. For ordinary CO2 they come from the retained-state labels of the
coherent reduction's restriction; for linked CO2/O2, from the retained prior
passed to :func:`~openghg_inversions.rhime.co2.prepare_co2_o2_inputs`, which
always carries ``source``. Configured CO2 preparation does not yet label
inner and outer states with ``basis_group``
(`#1094 <https://github.com/openghg/openghg_inversions/issues/1094>`_). Additive
outer patterns are available from Python only. Run metadata records the
outer-flux mode and selectors beside the boundary metadata.
