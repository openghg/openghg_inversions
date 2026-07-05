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

Generated Basis Functions
-------------------------

RHIME can load saved flux basis functions with ``fp_basis_case`` or generate a
basis with ``basis_algorithm`` and ``nbasis``. The legacy generated-basis
choices exposed through RHIME config are still ``"quadtree"`` and
``"weighted"``.

The lower-level Python basis API also supports ``"region_constrained"`` when a
caller supplies an already loaded ``region_classes`` ``DataArray``. That path
keeps generated labels from crossing land/sea, country, inner/outer, or other
class boundaries, but RHIME config and ``run_hbmcmc.py`` do not yet load
``region_classes`` from files. For RHIME runs that need these masks today,
build and save the basis through the Python basis API, then load it as a saved
basis case.

Region-constrained algorithms have split-stopping policies at the lower-level
strategy boundary. ``MinChildWeightShare`` is a parent-relative balance guard:
it compares the lightest proposed child with the current parent partition.
``MinChildTargetWeightShare`` is an equal-target low-weight guard: it compares
the lightest proposed child with ``weights.sum() / target_regions`` for the
class/source-local field being partitioned. Thresholds such as ``0.1`` mean
different things for those policies, and generated region counts become upper
targets when split stopping rejects remaining candidates. These child-share
policies are not currently routed through RHIME config options.

Output Formats
--------------

Standard single-sector RHIME supports ``inv_out``, ``basic``, ``paris``, and
``legacy`` output formats. ``legacy`` writes the old HBMCMC-compatible NetCDF
product from the modern ``InversionOutput``. The deprecated names ``hbmcmc``
and ``hbmcmc_postprocessing`` are accepted as aliases for ``legacy``.

Modern RHIME preparation, ``InversionOutput`` artifacts, and postprocessing use
retained ``BasisFunctions`` / ``BasisOperator`` objects as the primary basis
representation. Derived flux, country, PARIS, and legacy-format products record
``basis_reconstruction_path="operator-backed"`` plus the retained basis artifact
source/path when known. Legacy flat basis NetCDF files remain readable as an
explicit compatibility fallback, and flat basis maps may still be emitted by
compatibility output formats, but new workflows should save and load DataTree
``BasisFunctions`` artifacts instead of relying on flat-basis reconstruction.

``run_hbmcmc.py`` is now a compatibility wrapper for old fixedbasis-style INI
files. It translates legacy option names to the modern ``run_rhime`` API and
uses the legacy filename convention. New scripts and new configs should use
``openghg-inversions run-rhime`` or ``run_rhime(...)`` directly.
This compatibility route no longer preserves the exact historical
``fixedbasisMCMC`` / ``inferpymc`` passthrough behaviour. Use release ``0.6`` or
earlier if you need the old fixedbasis implementation.
Direct ``fixedbasisMCMC(...)`` calls are a temporary legacy Python path, not a
wrapper around ``run_rhime(...)``.
