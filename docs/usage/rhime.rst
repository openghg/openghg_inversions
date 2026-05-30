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
       flux_sources=["FF", "GPP", "TER", "Ocean"],
       sector_priors={
           "FF": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
           "GPP": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
           "TER": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
           "Ocean": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
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

For multi-sector RHIME, each ``flux_sources`` entry becomes a separately
optimized sector unless a future API adds an explicit sector-to-source mapping.

.. code-block:: ini

   [INPUT.PRIORS]
   domain = "EUROPE"
   flux_sources = ["FF", "GPP", "TER", "Ocean"]

   [RHIME.PDF]
   sector_priors = {
       "FF": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
       "GPP": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       "TER": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       "Ocean": {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0}
   }
