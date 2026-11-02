Configured CO2 priors and saved calibration
==========================================

Use this workflow to choose the :doc:`boundary_pattern_priors`,
:doc:`biosphere_prior` or CO2-only :doc:`shared_baseline` options, freeze their
resolved scientific values and build again from the saved artifacts. Existing
model defaults are unchanged. These APIs require the package's supported Python
3.12 or 3.13 runtime; Python 3.10 is not supported by the current package metadata.

Prepare scientific choices once
-------------------------------

The preparation owners are exported from ``openghg_inversions.rhime.co2``:

* ``BoundaryCorrectionConfig.from_params(table)`` accepts ``family`` equal to
  ``level``, ``A``, ``curtain_contrasts`` or ``independent_curtains``. Supply
  ``level_sd`` for families with a level and ``pattern_sd`` for families with
  patterns, each as ``{value = ..., units = "..."}``. All widths are explicitly
  positive. Level/curtain widths are concentrations; A-pattern widths are
  dimensionless. Pint converts compatible units into the stored BC units.
* ``reference`` is ``stored_scalar`` (default),
  ``whole_window_transport_mean`` or an explicit scalar value/units table.
  ``frequency`` and ``anchor_time`` select receptor-time coefficient periods.
  ``level_by_period=true`` requires periods and a family with a level;
  otherwise the level is global. Unknown and inapplicable options fail.
* ``BiospherePriorConfig.from_params(table)`` accepts ``family="lognormal"``
  (default) or the signed ``gaussian`` diagnostic. Exactly one policy is
  required: positive ``residual_sd_multiplier``, ``target_residual_sd`` or
  ``target_gross_sd``. Absolute targets are value/units tables in compatible
  flux units. The gross target is marginal, including explained variance;
  infeasible targets fail at calibration. There is no default shrinkage.

Pass raw labelled operands and observations to ``boundary.prepare``. Pass the
original retained prior and the labelled boolean ``biosphere`` selector,
``weights`` and ``reference`` to ``biosphere.prepare``. Labels have the
:doc:`biosphere_prior` contracts: no source-name or positional pairing is
inferred. Numerical inputs remain separate from configuration choices.
Boundary operators remain lazy; biosphere calibration retains its existing
explicit eager boundary for the small retained covariance.

Call each owner's ``save(prepared, new_path)`` before subsetting observations.
Writers refuse an existing destination. Files use the existing trace MultiIndex
codec, with groups named ``boundary_correction``, ``biosphere_prior`` and
``shared_baseline``. The corresponding public ``load_boundary_correction``,
``load_biosphere_prior`` and ``load_shared_baseline`` functions reopen them.
The saved calibration includes original moments, exact state labels, selector,
reporting functionals, resulting covariance and the resolved calibration report.
Reload verifies the policy against the saved baseline and retains the saved
covariance; current inputs or subsets never enter that calculation.

Select saved priors for ordinary or linked inference
---------------------------------------------------

The existing CO2 TOML/JSON/Python model owner accepts:

.. code-block:: toml

   [model.boundary]
   correction_path = "/path/to/boundary.nc"
   [model.biosphere]
   path = "/path/to/biosphere.nc"

``Co2ModelConfig.from_params`` validates choices without reading files.
``model_arguments()`` opens the selected artifacts at explicit binding.
The full ``run_co2`` and saved ``handoffs.sample`` routes use those same
arguments, for ordinary and cached inference. Saved traces carry the existing
boundary/calibration groups, so output replay needs neither those input paths
nor a newly calibrated prior. A different original retained prior fails at model
construction. Omit the tables to preserve existing priors.

For linked CO2/O2, use a shared top-level ``[biosphere] path=...`` and independent
``[channels.co2.boundary] correction_path=...`` and
``[channels.o2.boundary] correction_path=...`` tables. ``Co2O2ModelConfig.bind``
loads them and retains one shared calibrated flux state. Each correction keeps
its own species reference and widths; no cross-species BC covariance is added.
Either channel may retain its previous boundary formulation. Omit all other
boundary options, including ``enabled``, ``prior`` and ``activity``, on channels
using a frozen correction. A frozen correction owns its periods; the linked
builders reject global period overrides when a correction is selected.

Freeze and configure the shared-baseline recipe
-----------------------------------------------

``SharedBaselineConfig.from_params(table)`` requires ``mode`` equal to ``fixed``,
``additive``, ``multiplicative`` or ``multiplicative_shrunk`` and an explicit
positive ``delta_sd`` value/units table. Its ``prepare`` method accepts the
same labelled scientific inputs as ``prepare_shared_baseline``: complete flux
sensitivity, original retained prior, fixed contribution, prepared BC correction,
``outer`` and ``sampled_outer`` boolean selectors. Supply the additive design
and physical density widths only for additive mode, and an already resolved
positive calibration only for multiplicative shrinkage.

The saved artifact freezes post-gating BC and outer column means over the full
support, equal-site weights, scalar reference, selectors, original and selected
priors, and any calibration/density widths. Subset rows only after preparation;
loading or building never recentres or recalibrates them.

``build_shared_co2_from_config({"likelihood": table}, prepared=baseline,
observations=..., observation_error=..., aggregation_error=...)`` reuses the
existing CO2 likelihood syntax. It returns the ordinary PyMC model, or the
existing cached-model result for fixed OU with ``site_amplitude_prior``.
It also supports additive sigma, fixed OU amplitudes, site sigma and the saved
scalar-sigma eigenbasis. Additive sigma uses the existing CO2 site alignment
and prior defaults; ``no_model_error=true`` disables inferred mismatch while
preserving any explicitly supplied fixed mismatch.

Restore coordinates and absolute predictive concentrations, then call
``annotate_shared_baseline_trace`` before ``save_trace``. The complete frozen
preparation is stored beside the draws. This is the shared recipe's explicit
saved-artifact workflow; generic CO2 preparation/sample/product manifests and
linked shared-baseline inference are not supported by this entry point.
The physical source of atmospheric delta remains unidentified. No production
widths or new shared-baseline mathematics are introduced.

Runnable saved-calibration example
----------------------------------

This offline example constructs labelled synthetic inputs, applies a marginal
gross-SD target, saves and reopens both priors, builds configured ordinary CO2,
and verifies saved absolute prior predictions. The widths demonstrate syntax
and have no calibration authority for a real inversion.

.. code-block:: bash

   PYTENSOR_FLAGS=floatX=float64,base_compiledir=/tmp/ogi-config-example \
     uv run python -m examples.biosphere_prior.saved_configuration /path/to/new-output

.. literalinclude:: ../../examples/biosphere_prior/saved_configuration.py
   :language: python
   :start-after: # BEGIN EXAMPLE
   :end-before: # END EXAMPLE
