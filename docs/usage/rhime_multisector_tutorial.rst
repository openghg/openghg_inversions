Multisector RHIME tutorial
==========================

This tutorial runs the supported
:func:`openghg_inversions.rhime.run_multisector` recipe and inspects its
sector-labelled state and persisted sector/source diagnostics. It uses the
current production multisector runner.

The code and recorded outputs form one stateful session. You can
:jupyter-download-notebook:`download it as a Jupyter notebook <rhime_multisector_tutorial>`
to rerun or modify locally.

.. jupyter-kernel:: python3
   :id: rhime_multisector_tutorial

Prerequisites and sector inputs
-------------------------------

Complete the environment and OpenGHG setup in :doc:`rhime_standard_tutorial`.
The observation, boundary-condition, and footprint requirements are the same.
The companion store also supplies two distinct flux products on compatible
latitude, longitude, and time coordinates.

The example models two scientific sectors, ``anthropogenic`` and ``wetlands``,
using ``edgar-v80-anthropogenic`` and ``wetcharts-v131-wetlands`` respectively.
``flux_sources`` lists what OpenGHG retrieves. ``sector_sources`` maps stable,
human-readable sector labels to those source values. Every source is used once
because independently inferred sectors may not share one source. Do not change
the sector keys without also changing ``sector_priors``.

Configuration
-------------

.. literalinclude:: ../../openghg_inversions/rhime/config/multisector_tutorial.ini
   :language: ini

Each prior describes a multiplicative scale factor for one sector. The explicit
mapping prevents sector meaning from depending on source-coordinate order.
The short sampler is a controlled smoke test only; use convergence-checked
sampling sized for the real scientific question before interpreting results.

Run it
------

The supported CLI route differs from the standard tutorial only by subcommand:

.. code-block:: console

   $ pixi run -e dev rhime run --model multisector \
       --config openghg_inversions/rhime/config/multisector_tutorial.ini \
       --output-path outputs

The :doc:`cli` page documents the equivalent ``uv run`` command for
an existing compatible uv environment.

The supported Python route is:

.. jupyter-input::

   from importlib.resources import as_file, files
   import os
   from pathlib import Path

   from openghg_inversions.rhime import run

   tutorial_output_path = Path(os.environ.get("OPENGHG_TUTORIAL_OUTPUT_PATH", "outputs"))
   resource = files("openghg_inversions.rhime").joinpath("config/multisector_tutorial.ini")
   with as_file(resource) as config:
       result = run(model="multisector", config_file=config, output_path=tutorial_output_path)

   {
       "OpenGHG Inversions commit": os.environ.get(
           "OPENGHG_TUTORIAL_CODE_REF", "local checkout"
       ),
       "tutorial data": os.environ.get("OPENGHG_TUTORIAL_DATA_TAG", "v1.0.0"),
       "sites": result.run_metadata["sites"],
       "observations": result.inv_inputs.sizes["nmeasure"],
       "posterior samples": {
           name: result.trace_group("posterior").sizes[name] for name in ("chain", "draw")
       },
   }

.. jupyter-output::

   {'OpenGHG Inversions commit': '9202222950c3ef2eb7bc83d040df3bcf6c26f94c',
    'tutorial data': 'v1.0.0',
    'sites': ['MHD', 'TAC'],
    'observations': 84,
    'posterior samples': {'chain': 2, 'draw': 50}}

Inspect labelled state and model context
----------------------------------------

.. jupyter-input::

   {
       "sectors": [
           (sector["name"], sector["flux_source"])
           for sector in result.model_metadata["sectors"]
       ],
       "H dimensions": result.inv_inputs["H"].dims,
       "source labels": result.inv_inputs["H"].source.values.tolist(),
       "flux state variable": result.variable_name("flux_scale"),
       "variable roles": result.model_metadata["variable_roles"],
   }

.. jupyter-output::

   {'sectors': [('anthropogenic', 'edgar-v80-anthropogenic'),
     ('wetlands', 'wetcharts-v131-wetlands')],
    'H dimensions': ('region', 'nmeasure', 'source'),
    'source labels': ['edgar-v80-anthropogenic', 'wetcharts-v131-wetlands'],
    'flux state variable': 'x',
    'variable roles': {'observation': 'mf',
     'observation_error': 'mf_error',
     'minimum_error': 'min_error',
     'concentration': 'y',
     'model_error': 'epsilon',
     'observation_repeatability': 'mf_repeatability',
     'observation_variability': 'mf_variability',
     'flux_scale': 'x',
     'flux_contribution': 'mu',
     'emissions_sensitivity': 'hx',
     'boundary': 'mu_bc',
     'baseline_scale': 'bc',
     'baseline_sensitivity': 'hbc',
     'baseline': 'mu_bc'}}

Shared-basis preparation represents sensitivity as
``H(region, nmeasure, source)``. Source-specific bases may instead use one
gathered state dimension labelled by ``(source, region_in_source)``. In either
case, construction uses one full physical state ``x`` and one two-dimensional
sensitivity ``hx``. Rectangular inputs are packed into a gathered source/local
state index for the model; the prepared ``H`` keeps its original layout.
Select a posterior block by its source label, for example
``result.trace_group("posterior")["x"].sel(source="wetcharts-v131-wetlands")``;
never infer sector identity from axis position. The blocks scale their own
prior fluxes, and ``mu`` gives their combined pollution concentration.
See :doc:`rhime` for both prepared layouts and :doc:`concrete_rhime_model` for
the scientific-role mapping.

Use ``az.summary`` as in the standard tutorial for ``x`` and inspect its joint
posterior across source blocks. Strong cross-sector correlation or posterior
behaviour dominated by the priors indicates that the observations do not
separately identify those sources, even when every array label is correct.

Inspect the sector/source output
--------------------------------

The configured ``inv_out`` output writes labelled sector diagnostics alongside
the self-contained inversion output. To inspect the same scientific reduction in
memory, call the product operation explicitly:

.. jupyter-input::

   from openghg_inversions.postprocessing.make_outputs import make_sector_flux_outputs

   diagnostics = make_sector_flux_outputs(
       result,
       stats=["mean"],
       include_scale_factors=True,
       report_flux_on_inversion_grid=False,
   )
   product_stem = "multisector_tutorial" + result.start_date
   diagnostic_path = tutorial_output_path / f"{product_stem}_sector_flux_diagnostics.nc"
   inversion_path = tutorial_output_path / f"{product_stem}_inversion_output.nc"
   assert diagnostic_path.exists() and inversion_path.exists()
   {
       "diagnostic variables": sorted(diagnostics.data_vars),
       "diagnostic sizes": dict(diagnostics.sizes),
       "variable dimensions": {
           name: diagnostics[name].dims for name in sorted(diagnostics.data_vars)
       },
       "sector diagnostics file": diagnostic_path.name,
       "inversion output file": inversion_path.name,
   }

.. jupyter-output::

   {'diagnostic variables': ['flux_anthropogenic_posterior_mean',
     'flux_anthropogenic_prior_mean',
     'flux_total_posterior_mean',
     'flux_total_prior_mean',
     'flux_wetlands_posterior_mean',
     'flux_wetlands_prior_mean',
     'scaling_anthropogenic_posterior_mean',
     'scaling_anthropogenic_prior_mean',
     'scaling_wetlands_posterior_mean',
     'scaling_wetlands_prior_mean'],
    'diagnostic sizes': {'flux_time': 1, 'lat': 293, 'lon': 391},
    'variable dimensions': {'flux_anthropogenic_posterior_mean': ('flux_time',
      'lat',
      'lon'),
     'flux_anthropogenic_prior_mean': ('flux_time', 'lat', 'lon'),
     'flux_total_posterior_mean': ('flux_time', 'lat', 'lon'),
     'flux_total_prior_mean': ('flux_time', 'lat', 'lon'),
     'flux_wetlands_posterior_mean': ('flux_time', 'lat', 'lon'),
     'flux_wetlands_prior_mean': ('flux_time', 'lat', 'lon'),
     'scaling_anthropogenic_posterior_mean': ('lat', 'lon'),
     'scaling_anthropogenic_prior_mean': ('lat', 'lon'),
     'scaling_wetlands_posterior_mean': ('lat', 'lon'),
     'scaling_wetlands_prior_mean': ('lat', 'lon')},
    'sector diagnostics file': 'multisector_tutorial2020-01-01_sector_flux_diagnostics.nc',
    'inversion output file': 'multisector_tutorial2020-01-01_inversion_output.nc'}

Sector posterior flux is the retained prior flux multiplied by that sector's
posterior scaling. ``flux_total_posterior_mean`` is the sum of sector fluxes,
not another independently inferred state. Keep the sector variables labelled
when comparing source contributions. The inversion output's provenance has the
same limits described by the standard tutorial: it retains prepared arrays,
basis, trace and resolved metadata, but is not an object-store query snapshot.

For ordinary finite traces, sector diagnostics reduce scaling samples before
reconstructing mean fluxes.
Requesting total quantiles or standard deviations needs draw-wise fluxes and
can use much more memory; see :doc:`retained_state_reconstruction` before
building those products for a large grid.

For a clean-checkout mechanics test, run
``uv run pytest tests/test_rhime_tutorials.py -q``. Its multisector case uses
two maintained fixture products, one a dimension-shuffled numerical duplicate,
and deterministic sampling. That verifies labelled routing and persistence,
not scientific sector separation. Normal CI does not clone or download the
companion data repository.

Reuse preparation and saved samples
------------------------------------

For the same resolved ``RhimeConfig``,
``run_multisector(prepared=prepared, config=config)`` starts at the model-input
boundary and returns ``InversionOutput``, as the full runner does.
``rhime.multisector_stages.prepare`` saves one preparation manifest;
``sample`` accepts it with current compatible sector-prior, likelihood and sampler
choices; ``postprocess`` accepts one sample manifest with current output choices.
Keep each model's samples in a separate destination. Replayed products retain the
original requested window, sector/source associations and prior flux without
building a graph or repeating inference.

See :doc:`staged_workflow` for phase-only configuration and checkpoint lifetime,
and :doc:`multisector_handoffs` for an executable notebook comparing full, staged
and replayed execution using synthetic data. Its :download:`companion script
<../../examples/multisector_handoffs.py>` also checks independent model equations.

Common failures
---------------

Missing or duplicate sources
   Supply at least two unique ``flux_sources``. Each ``sector_sources`` value
   must exist in that list and may be selected by only one sector.

Missing or unused sector prior
   When ``sector_priors`` is present it must have exactly the same keys as
   ``sector_sources``. A typo is rejected rather than silently using the shared
   prior.

Incompatible source coordinates
   Sector fluxes must have compatible spatial and time coordinates before
   shared-basis construction. Resample explicitly; do not rely on implicit
   xarray padding or positional alignment.

Unexpected output selection
   ``inv_out`` and sector diagnostics are the default tutorial products.
   ``output_format="paris"`` instead adds latest-template sector-resolved and
   total flux output plus total concentration output. See :doc:`rhime` before
   changing ``output_format``.
