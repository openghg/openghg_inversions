Configured CO2 priors and saved calibration
===========================================

Use this workflow to configure :doc:`boundary_correction_structures`, a saved
:doc:`biosphere_prior` calibration or the outer-flux option of
:doc:`atmospheric_baseline`. Boundary structures and the outer-flux option are
choices bound to the prepared inputs at model construction; only the biosphere
calibration is resolved once and saved. Existing model defaults are unchanged. These APIs require the
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

Calibrate a biosphere prior once
--------------------------------

A biosphere calibration is a sensitivity intervention, prepared and saved once
and then loaded by path. ``BiospherePriorConfig.from_params(table)`` requires
exactly one policy: positive ``residual_sd_multiplier``, ``target_residual_sd`` or
``target_gross_sd``. Absolute targets are value/units tables in compatible
flux units. The gross target is marginal, including explained variance;
infeasible targets fail at calibration. There is no default shrinkage.

Pass the original retained prior and the labelled boolean ``biosphere``
selector, ``weights`` and ``reference`` to ``prepare``. Labels have the
:doc:`biosphere_prior` contracts: no source-name or positional pairing is
inferred. Calibration is an explicit eager step for the small retained
covariance.

``save(prepared, new_path)`` refuses an existing destination and uses the
trace MultiIndex codec with a ``biosphere_prior`` group; ``load_biosphere_prior``
reopens it. The saved calibration includes original moments, exact state
labels, selector, reporting functionals, resulting covariance, residual weights
and the resolved policy. Reload validates them and does not resolve the policy
again or use current inputs or selected subsets. The model samples the
calibrated covariance with the retained prior's marginal families.

Select a saved calibration for ordinary or linked inference
-----------------------------------------------------------

The CO2 TOML/JSON/Python model owner accepts:

.. code-block:: toml

   [model.biosphere]
   path = "/path/to/biosphere.nc"

``Co2ModelConfig.from_params`` validates choices without reading files.
``sampler_arguments()`` resolves sampling policy without opening artifacts;
``model_arguments()`` loads the calibration after output preflight. The full
``run_co2`` and saved ``handoffs.sample`` routes use those arguments for
ordinary and cached inference. Before inference, resolved trace and product
targets are checked against the selected calibration and the
preparation/reconstruction files the run loads, including active likelihood
eigenbasis and country-reporting files. Product writers check all destinations
before writing, including consumed stage manifests, prepared inputs, posterior
and affine companion during saved replay. This protection covers aliases and
overlapping directory stores. Saved traces carry fitted boundary facts and the
calibration group, so output replay needs neither the input paths nor a newly
calibrated prior. A different original retained prior fails at model
construction.

For linked CO2/O2, use a shared top-level ``[biosphere] path=...``.
``Co2O2ModelConfig.bind`` loads it and retains one shared calibrated flux
state; each channel keeps its own boundary choice. The shipped ``co2.toml``,
``co2_fixed_ou.toml`` and ``co2_o2.toml`` templates include commented examples.

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

Preparation and saved preparations keep these coordinates. For ordinary CO2
they come from the retained-state labels of the coherent reduction's
restriction: a configured basis built with ``fix_outer_regions`` labels its
states ``"outer"`` and ``"inner"`` (see :doc:`grouped_basis_layout`), and
per-source native priors label each state's ``source``. For linked CO2/O2 they
come from the retained prior passed to
:func:`~openghg_inversions.rhime.co2.prepare_co2_o2_inputs`, which always
carries ``source``; preparation does not add ``basis_group`` there. Additive
outer patterns are available from Python only. A configured biosphere
calibration is used as the flux prior as usual. Run metadata records the
outer-flux mode and selectors beside the boundary metadata.

Runnable saved-calibration example
----------------------------------

This offline example constructs labelled synthetic inputs, applies a marginal
gross-SD target, saves and reopens the calibration, builds ordinary CO2 with a
configured anomaly-scaling boundary,
and verifies saved absolute prior predictions. The widths demonstrate syntax
and have no calibration authority for a real inversion.

.. code-block:: bash

   PYTENSOR_FLAGS=floatX=float64,base_compiledir=/tmp/ogi-config-example \
     uv run python -m examples.biosphere_prior.saved_configuration /path/to/new-output

.. literalinclude:: ../../examples/biosphere_prior/saved_configuration.py
   :language: python
   :start-after: # BEGIN EXAMPLE
   :end-before: # END EXAMPLE
