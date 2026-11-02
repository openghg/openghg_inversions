Site and satellite multisector inversions
=========================================

This tutorial shows the common RHIME workflow for estimating two or more
flux components independently. Surface-site and satellite inversions use the
same public runner and model. They differ only in their observation and
footprint acquisition options.

Use :func:`openghg_inversions.rhime.run_multisector` when every flux
component should have its own scaling prior within the full state. The
``flux_sources`` values are the OpenGHG ``source`` metadata used for retrieval. Optional
``sector_sources`` entries give those sources shorter model-facing names.

Before running
--------------

Check that the selected stores contain observations, footprints, boundary
conditions, and every requested flux source for the complete inversion
period. A time-resolved footprint can require flux data before the first
observation because its backward lags are convolved with earlier emissions.

Start with a short period, a small basis, and a small sampler configuration.
After inspecting the prepared dimensions and posterior diagnostics, increase
the basis and sampling sizes for a scientific run.

Surface-site multisector example
--------------------------------

The following example estimates fossil-fuel and biospheric methane
components from one surface site. Replace store names, source names, and
paths with values available in your OpenGHG installation.

.. code-block:: python

   from openghg_inversions.rhime import run_multisector

   site_result = run_multisector(
       species="ch4",
       sites=["TAC"],
       averaging_period=["1h"],
       inlet=["185m"],
       fp_height=["185m"],
       platform=["surface"],
       domain="EUROPE",
       start_date="2019-01-01",
       end_date="2019-01-08",
       obs_store="user",
       footprint_store="user",
       flux_store="user",
       bc_store="user",
       flux_sources=["ff-inventory", "biosphere-inventory"],
       sector_sources={
           "FF": "ff-inventory",
           "biosphere": "biosphere-inventory",
       },
       sector_priors={
           "FF": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
           "biosphere": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       },
       basis_algorithm="weighted",
       nbasis=20,
       output_path="outputs/site_multisector",
       output_name="tac_ch4_multisector",
       output_format="inv_out",
       draws=500,
       tune=500,
       chains=2,
   )

Every sector must map to a distinct flux source. Use a prior with support for
negative scaling when that is scientifically required; a log-normal prior is
not suitable for a component whose scaling may be negative.

Satellite multisector example
-----------------------------

Satellite column data use the same runner. Set the platform and column
options explicitly. This short OCO2 example separates anthropogenic,
respiration, and gross-primary-productivity fluxes and selects
high-time-resolution footprints.

.. code-block:: python

   from openghg_inversions.rhime import run_multisector

   satellite_result = run_multisector(
       species="co2",
       sites=["OCO2-EASTASIA"],
       averaging_period=["1H"],
       inlet=["column"],
       fp_height=["column"],
       fp_species="co2",
       instrument=[None],
       platform=["satellite"],
       max_level=[3],
       time_resolved=[True],
       domain="EASTASIA",
       # Include the backward footprint window before the soundings.
       start_date="2022-03-31 04:00:00",
       end_date="2022-04-01 04:08:10",
       obs_store="/path/to/oco2_data_store",
       flux_store="/path/to/oco2_data_store",
       bc_store="/path/to/oco2_data_store",
       footprint_store="/path/to/oco2_footprint_store",
       flux_sources=["anth", "resp", "gpp_atm"],
       sector_priors={
           "anth": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
           "resp": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
           "gpp_atm": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       },
       use_bc=True,
       bc_input="cams",
       basis_algorithm="weighted",
       landsea_path="/path/to/maps/eastasia-landsea.nc",
       basis_output_path="outputs/satellite_multisector/basis.nc",
       nbasis=8,
       averaging_error=True,
       min_error=0.0,
       output_path="outputs/satellite_multisector",
       output_name="oco2_eastasia_multisector",
       output_format="inv_out",
       draws=500,
       tune=500,
       chains=2,
   )

``landsea_path`` selects the exact class map used for weighted basis
construction, and ``basis_output_path`` is the exact retained-artifact destination.
To reuse a saved basis instead of generating one, select its exact ``basis_path``.
See :doc:`rhime_configuration` for packaged support-map selection and migration.
The BC case/directory options above retain their existing compatibility convention.

Set ``time_resolved=False`` to request integrated footprints, or ``None`` to
leave this metadata unconstrained during the OpenGHG search. For several
sites, ``platform``, ``max_level``, and ``time_resolved`` may be lists aligned
with ``sites``.

RHIME retains the column prior-factor fields, aligns small timestamp
differences between satellite observations and footprints, and rescales
boundary-condition sensitivity to the corrected column signal before model
construction. The resulting emissions sensitivity has a source-resolved
layout such as ``H(region, nmeasure, source)`` and is consumed by the same
multisector model used for surface sites.

Inspecting and processing results
---------------------------------

Both calls return an
:class:`~openghg_inversions.postprocessing.inversion_output.InversionOutput`.
Inspect the
canonical inputs before interpreting the posterior:

.. code-block:: python

   result = satellite_result  # or site_result

   print(result.inv_inputs["H"].dims)
   print(result.inv_inputs["H"].coords["source"].values)
   print(result.trace_group("posterior"))
   print(result.output_metadata)

For a shared basis, expect a distinct ``source`` dimension and one sensitivity
slice per requested flux source. Satellite inputs should also contain finite
``H_bc`` when boundary conditions are enabled. Its
``satellite_column_bc_scale`` attribute records that column scaling was
applied. Column runs retain ``mf_prior_factor`` and, when supplied,
``mf_prior_upper_level_factor`` in ``inv_inputs``.

The posterior contains one full scaling variable, ``x``, on a gathered state
coordinate carrying the source and local state labels. Its blocks retain their
sector-specific priors. Use the scientific role to find the variable and the
recorded source association to select a sector:

.. code-block:: python

   scales = result.trace_group("posterior")[result.variable_name("flux_scale")]
   for sector in result.model_metadata["sectors"]:
       block = scales.sel(source=sector["flux_source"], drop=True)
       print(sector["name"], block.mean().item())

The model uses one two-dimensional sensitivity ``hx`` and one total flux
concentration contribution ``mu``. Rectangular prepared ``H`` is packed only
for model construction; the prepared inputs and retained basis remain unchanged.
The first requested ``flux_sources`` entry supplies automatic basis weighting.
Changing the order of ``sector_sources`` does not change that preparation choice.

The returned value is the modern, serializable scientific result, including with
``output_format="none"``. ``output_format="inv_out"`` requests its file export;
``save_inversion_output`` and ``save_trace`` control optional exports. Current
output policy is separate from the result's scientific context and inference
metadata. Compute in-memory sector diagnostics explicitly:

.. code-block:: python

   from openghg_inversions.postprocessing.make_outputs import make_sector_flux_outputs

   diagnostics = make_sector_flux_outputs(
       result, stats=["mean"], include_scale_factors=True,
       report_flux_on_inversion_grid=False,
   )

Copied runners call ``postprocess_multisector_outputs(result, output_spec)`` for
configured products, then ``save_inversion_output(result, output_spec)`` for
exports. Use ``inv_out``, ``paris``, or ``none`` for multisector runs;
``basic`` and legacy HBMCMC output remain single-sector-only. See :doc:`staged_workflow` for
reusing prepared inputs and saved samples with current product choices.

Configuration and batch runs
----------------------------

The same arguments are available in
``openghg_inversions/config/templates/rhime_template.ini``. Put measurement,
store, prior, basis, sampler, and output values in a copied configuration, then
run it with the multisector command:

.. code-block:: console

   rhime run --model multisector 2022-03-31T04:00:00 2022-04-01T04:08:10 \
       --config /path/to/oco2_multisector.ini

See :doc:`cli` for scheduler and batch-script examples. Use
:doc:`rhime` for the complete preparation and model contracts.
