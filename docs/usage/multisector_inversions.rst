Site and satellite multisector inversions
========================================

This tutorial shows the common RHIME workflow for estimating two or more
flux components independently. Surface-site and satellite inversions use the
same public runner and model. They differ only in their observation and
footprint acquisition options.

Use :func:`openghg_inversions.rhime.run_rhime_multisector` when every flux
component should have its own scaling state. The ``flux_sources`` values are
the OpenGHG ``source`` metadata used for retrieval. Optional
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

   from openghg_inversions.rhime import run_rhime_multisector

   site_result = run_rhime_multisector(
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
       emissions_store="user",
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
       bc_basis_case="NESW",
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

   from openghg_inversions.rhime import run_rhime_multisector

   satellite_result = run_rhime_multisector(
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
       emissions_store="/path/to/oco2_data_store",
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
       bc_basis_case="NESW",
       bc_basis_directory="/path/to/bc_basis_functions",
       basis_algorithm="weighted",
       basis_directory="/path/to/basis_functions",
       country_directory="/path/to/countries",
       basis_output_path="outputs/satellite_multisector/basis",
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

Both calls return a :class:`openghg_inversions.rhime.RhimeResult`. Inspect the
canonical inputs before interpreting the posterior:

.. code-block:: python

   result = satellite_result  # or site_result

   print(result.inv_inputs["H"].dims)
   print(result.inv_inputs["H"].coords["source"].values)
   print(result.idata.posterior)
   print(result.output_metadata)

For a shared basis, expect a distinct ``source`` dimension and one sensitivity
slice per requested flux source. Satellite inputs should also contain finite
``H_bc`` when boundary conditions are enabled. Its
``satellite_column_bc_scale`` attribute records that column scaling was
applied. Column runs retain ``mf_prior_factor`` and, when supplied,
``mf_prior_upper_level_factor`` in ``inv_inputs``.

The posterior contains one sector scaling variable per sanitized sector name,
for example ``x_anth``, ``x_resp``, and ``x_gpp_atm``. Use the semantic mapping
instead of constructing names in downstream code:

.. code-block:: python

   roles = result.model_build_result.variable_roles
   for sector in result.model_spec.sectors:
       role = f"flux_scale:{sector.name}"
       variable = roles[role]
       print(sector.name, result.idata.posterior[variable].mean().item())

With ``output_format="inv_out"``, ``result.inv_out`` is the modern,
serializable inversion product and is also saved when
``save_inversion_output=True``. Multisector runs additionally populate
``result.outputs`` and ``result.output_metadata`` with any requested sector
diagnostics. Legacy output is intentionally unavailable for multisector runs;
use ``inv_out``, ``basic``, ``paris``, or ``none`` according to the required
downstream product.

Configuration and batch runs
----------------------------

The same arguments are available in
``openghg_inversions/config/templates/rhime_template.ini``. Put measurement,
store, prior, basis, sampler, and output values in a copied configuration, then
run it with the multisector command:

.. code-block:: console

   openghg-inversions run-rhime-multisector 2022-03-31T04:00:00 2022-04-01T04:08:10 \
       --config /path/to/oco2_multisector.ini

See :doc:`cli` for scheduler and batch-script examples. Use
:doc:`rhime` for the complete preparation and model contracts.
