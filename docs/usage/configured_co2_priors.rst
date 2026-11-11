Configured CO2 priors and saved calibration
===========================================

Use this workflow to configure :doc:`boundary_correction_structures`, a saved
:doc:`biosphere_prior` calibration or the CO2-only :doc:`shared_baseline`.
Boundary structures and shared baselines are choices bound to the prepared
inputs at model construction; only the biosphere calibration is resolved once
and saved. Existing model defaults are unchanged. These APIs require the
package's supported Python 3.12 or 3.13 runtime.

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

``BiospherePriorConfig.from_params(table)`` accepts ``family="lognormal"``
(default) or the signed ``gaussian`` diagnostic. Exactly one policy is
required: positive ``residual_sd_multiplier``, ``target_residual_sd`` or
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
labels, selector, reporting functionals, resulting covariance and the resolved
calibration report. Reload validates them and does not resolve the policy
again or use current inputs or selected subsets.

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

Configure the shared-baseline recipe
------------------------------------

``shared_baseline_choice(table)`` parses ``mode`` (``fixed``, ``additive``,
``multiplicative`` or ``multiplicative_shrunk``), a positive ``delta_sd`` in
observation concentration units and an optional ``boundary`` table using the
``[model.boundary]`` keys above, without ``mean_shift_prior``. Bind the
returned :class:`~openghg_inversions.models.shared_baseline.SharedBaselineChoice`
to the labelled scientific inputs as in :doc:`shared_baseline`.

``build_shared_co2_from_config({"likelihood": table}, baseline=baseline,
observations=..., observation_error=..., aggregation_error=...)`` reuses the
existing CO2 likelihood syntax. It returns the ordinary PyMC model, or the
existing cached-model result for fixed OU with ``site_amplitude_prior``.
It also supports additive sigma, fixed OU amplitudes, site sigma and the saved
scalar-sigma eigenbasis. Additive sigma uses the existing CO2 site alignment
and prior defaults; ``no_model_error=true`` disables inferred mismatch while
preserving any explicitly supplied fixed mismatch.

Restore coordinates and absolute predictive concentrations, then call
``annotate_shared_baseline_trace`` before ``save_trace``; the bound baseline is
recorded beside the draws. Generic CO2 preparation/sample/product manifests and
linked shared-baseline inference are not supported by this entry point. The
physical source of the atmospheric delta remains unidentified.

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
