Multisector RHIME tutorial
==========================

This tutorial runs the supported
:func:`openghg_inversions.rhime.run_rhime_multisector` recipe and inspects its
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

   $ pixi run -e dev openghg-inversions run-rhime-multisector \
       --config openghg_inversions/rhime/config/multisector_tutorial.ini \
       --output-path outputs

The :doc:`installation` page documents the equivalent ``uv run`` command for
an existing compatible uv environment.

The supported Python route is:

.. jupyter-input::

   from importlib.resources import as_file, files
   import os
   from pathlib import Path

   from openghg_inversions.rhime import run_rhime_multisector

   tutorial_output_path = Path(os.environ.get("OPENGHG_TUTORIAL_OUTPUT_PATH", "outputs"))
   resource = files("openghg_inversions.rhime").joinpath("config/multisector_tutorial.ini")
   with as_file(resource) as config:
       result = run_rhime_multisector(config_file=config, output_path=tutorial_output_path)

   {
       "OpenGHG Inversions commit": os.environ.get(
           "OPENGHG_TUTORIAL_CODE_REF", "local checkout"
       ),
       "tutorial data": os.environ.get("OPENGHG_TUTORIAL_DATA_TAG", "v1.0.0"),
       "sites": list(result.run_spec.sites),
       "observations": result.inv_inputs.sizes["nmeasure"],
       "posterior samples": {
           name: result.idata.posterior.sizes[name] for name in ("chain", "draw")
       },
   }

.. jupyter-output::

   Run ``docs-tutorials-record`` to refresh this output.

Inspect labelled state and provenance
-------------------------------------

.. jupyter-input::

   inversion_output = result.outputs["inversion_output"]
   {
       "sectors": [
           (sector.name, sector.flux_source) for sector in result.model_spec.sectors
       ],
       "H dimensions": result.inv_inputs["H"].dims,
       "source labels": result.inv_inputs["H"].source.values.tolist(),
       "scale variables": sorted(
           name for name in result.idata.posterior.data_vars if name.startswith("x_")
       ),
       "variable roles": result.model_build_result.variable_roles,
       "provenance contract": inversion_output.provenance["contract"],
   }

.. jupyter-output::

   Run ``docs-tutorials-record`` to refresh this output.

Shared-basis preparation represents sensitivity as
``H(region, nmeasure, source)``. Source-specific bases may instead use one
gathered state dimension labelled by ``(source, region_in_source)``. In either
case, select by the ``source`` label or gathered source level; never infer
sector identity from axis position. The posterior variables
``x_anthropogenic`` and ``x_wetlands`` scale their own prior fluxes. Their
forward contributions add to the modelled pollution concentration, while the
observation likelihood is shared. See :doc:`rhime` for both supported layouts
and :doc:`concrete_rhime_model` for the scientific-role mapping.

Use ``az.summary`` as in the standard tutorial for both scale variables and
inspect their joint posterior. Strong cross-sector correlation or posterior
behaviour dominated by the priors indicates that the observations do not
separately identify those sources, even when every array label is correct.

Inspect the sector/source output
--------------------------------

Every multisector run constructs labelled sector diagnostics. With the
configured ``inv_out`` output they are also persisted:

.. jupyter-input::

   diagnostics = result.outputs["sector_flux_diagnostics"]
   {
       "diagnostic variables": sorted(diagnostics.data_vars),
       "diagnostic sizes": dict(diagnostics.sizes),
       "variable dimensions": {
           name: diagnostics[name].dims for name in sorted(diagnostics.data_vars)
       },
       "sector diagnostics file": Path(
           result.output_metadata["sector_flux_diagnostics_path"]
       ).name,
       "inversion output file": Path(
           result.output_metadata["inversion_output_path"]
       ).name,
   }

.. jupyter-output::

   Run ``docs-tutorials-record`` to refresh this output.

Sector posterior flux is the retained prior flux multiplied by that sector's
posterior scaling. ``flux_total_posterior_mean`` is the sum of sector fluxes,
not another independently inferred state. Keep the sector variables labelled
when comparing source contributions. The inversion output's provenance has the
same limits described by the standard tutorial: it retains prepared arrays,
basis, trace and resolved metadata, but is not an object-store query snapshot.

For a clean-checkout mechanics test, run
``uv run pytest tests/test_rhime_tutorials.py -q``. Its multisector case uses
two maintained fixture products, one a dimension-shuffled numerical duplicate,
and deterministic sampling. That verifies labelled routing and persistence,
not scientific sector separation. Normal CI does not clone or download the
Git LFS repository.

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
   ``inv_out`` and sector diagnostics are supported here. Latest PARIS flux
   output is a separate, explicitly configured route; multisector PARIS
   concentration output is not currently implemented. See :doc:`rhime` before
   changing ``output_format``.
