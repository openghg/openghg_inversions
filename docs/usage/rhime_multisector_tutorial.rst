Multisector RHIME tutorial
==========================

This tutorial runs the supported
:func:`openghg_inversions.rhime.run_rhime_multisector` recipe and inspects its
sector-labelled state and persisted sector/source diagnostics. It uses the
current production multisector runner, not a transitional Verification Games
adapter.

Prerequisites and sector inputs
-------------------------------

Complete the environment and OpenGHG setup in :doc:`rhime_standard_tutorial`.
The observation, boundary-condition, boundary-basis, and footprint
requirements are the same. A multisector run also requires at least two
distinct flux products on compatible latitude, longitude, and time
coordinates. Use ``search_flux`` to find their exact OpenGHG metadata
``source`` values.

The example models two scientific sectors, ``anthropogenic`` and ``wetlands``.
``flux_sources`` lists what OpenGHG retrieves. ``sector_sources`` maps stable,
human-readable sector labels to those source values. Every source is used once
because independently inferred sectors may not share one source. Replace the
two illustrative source values with distinct products in your store; do not
change the sector keys without also changing ``sector_priors``.

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

.. code-block:: python

   from importlib.resources import as_file, files

   from openghg_inversions.rhime import run_rhime_multisector

   resource = files("openghg_inversions.rhime").joinpath("config/multisector_tutorial.ini")
   with as_file(resource) as config:
       result = run_rhime_multisector(config_file=config, output_path="outputs")

Inspect labelled state and provenance
-------------------------------------

.. code-block:: python

   print([(sector.name, sector.flux_source) for sector in result.model_spec.sectors])
   print(result.inv_inputs["H"].dims)
   print(result.inv_inputs["H"].coords)
   print(result.inv_inputs["H"].sel(source="anthropogenic-ch4"))
   print(result.model_build_result.variable_roles)
   print(result.idata.posterior[["x_anthropogenic", "x_wetlands"]])
   print(result.inv_out.provenance)

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

.. code-block:: python

   diagnostics = result.outputs["sector_flux_diagnostics"]
   print(diagnostics[[
       "flux_anthropogenic_posterior_mean",
       "flux_wetlands_posterior_mean",
       "flux_total_posterior_mean",
   ]])
   print(result.output_metadata["sector_flux_diagnostics_path"])
   print(result.output_metadata["inversion_output_path"])

Sector posterior flux is the retained prior flux multiplied by that sector's
posterior scaling. ``flux_total_posterior_mean`` is the sum of sector fluxes,
not another independently inferred state. Keep the sector variables labelled
when comparing source contributions. The inversion output's provenance has the
same limits described by the standard tutorial: it retains prepared arrays,
basis, trace and resolved metadata, but is not an object-store query snapshot.

For a clean-checkout mechanics test, run
``uv run pytest tests/test_rhime_tutorials.py -q``. Its multisector case
replaces the illustrative production source names with two maintained fixture
products, one a dimension-shuffled numerical duplicate, and uses deterministic
sampling. That verifies labelled routing and persistence, not scientific
sector separation.

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
