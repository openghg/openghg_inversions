RHIME Terminology And Quickstart
================================

RHIME runners use the modern spec vocabulary below. New Python examples and
new config files should use these names.

Terminology
-----------

``species``
   Primary gas or tracer name used for object-store lookup and output naming.

``source``
   OpenGHG metadata key used to retrieve flux data. In sector-resolved RHIME
   inputs, ``source`` is also the xarray coordinate on flux and sensitivity
   data.

``flux_sources``
   RHIME config/API field containing the requested OpenGHG flux ``source``
   values.

``sector``
   Model component optimized separately in a multi-sector RHIME run. A sector
   is usually backed by one flux ``source``.

``sector_sources``
   Optional mapping from sector names to OpenGHG ``source`` values. Use this
   when sector labels such as ``FF`` or ``ocean`` differ from the source names
   used to retrieve flux data.

``tracer``
   Additional species used to constrain the primary species, normally with
   linked forward models. The current RHIME preparation path does not support
   tracer inversions.

``emissions_name``
   Legacy compatibility spelling accepted only when ``flux_sources`` is absent.

Python API
----------

.. code-block:: python

   from openghg_inversions.rhime import run_rhime, run_rhime_multisector

   result = run_rhime(
       species="ch4",
       sites=["TAC"],
       averaging_period=["1h"],
       domain="EUROPE",
       start_date="2019-01-01",
       end_date="2019-01-02",
       output_path="outputs",
       output_name="example",
       flux_sources=["total-ukghg-edgar7"],
   )

   multi_sector_result = run_rhime_multisector(
       species="ch4",
       sites=["TAC"],
       averaging_period=["1h"],
       domain="EUROPE",
       start_date="2019-01-01",
       end_date="2019-01-02",
       output_path="outputs",
       output_name="example_multisector",
       flux_sources=["ff-inventory", "gpp-inventory", "ter-inventory", "ocean-inventory"],
       sector_sources={
           "FF": "ff-inventory",
           "GPP": "gpp-inventory",
           "TER": "ter-inventory",
           "ocean": "ocean-inventory",
       },
       sector_priors={
           "FF": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
           "GPP": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
           "TER": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
           "ocean": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
       },
   )

Config Files
------------

New RHIME config files should use ``flux_sources``:

.. code-block:: ini

   [INPUT.MEASUREMENTS]
   species = "ch4"
   sites = ["TAC"]
   averaging_period = ["1h"]
   start_date = "2019-01-01"
   end_date = "2019-01-02"

   [INPUT.PRIORS]
   domain = "EUROPE"
   flux_sources = ["total-ukghg-edgar7"]

   [RHIME.OUTPUT]
   output_path = "outputs"
   output_name = "example"

For multi-sector RHIME, use ``sector_sources`` when the optimized sector names
are not the same strings as the OpenGHG source values.

.. code-block:: ini

   [INPUT.PRIORS]
   domain = "EUROPE"
   flux_sources = ["ff-inventory", "gpp-inventory", "ter-inventory", "ocean-inventory"]
   sector_sources = {
       "FF": "ff-inventory",
       "GPP": "gpp-inventory",
       "TER": "ter-inventory",
       "ocean": "ocean-inventory"
   }

   [RHIME.PDF]
   sector_priors = {
       "FF": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
       "GPP": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       "TER": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       "ocean": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0}
   }

Output Formats
--------------

Standard single-sector RHIME supports ``inv_out``, ``basic``, ``paris``, and
``legacy`` output formats. ``legacy`` writes the old HBMCMC-compatible NetCDF
product from the modern ``InversionOutput``. The deprecated names ``hbmcmc``
and ``hbmcmc_postprocessing`` are accepted as aliases for ``legacy``.

``run_hbmcmc.py`` is now a compatibility wrapper for old fixedbasis-style INI
files. It translates legacy option names to the modern ``run_rhime`` API and
uses the legacy filename convention. New scripts and new configs should use
``openghg-inversions run-rhime`` or ``run_rhime(...)`` directly.
