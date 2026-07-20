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
   is currently backed by one unique flux ``source``.

``sector_sources``
   Optional mapping from sector names to OpenGHG ``source`` values. Use this
   when sector labels such as ``FF`` or ``ocean`` differ from the source names
   used to retrieve flux data. The current multi-sector model requires a
   one-to-one mapping: two independently optimized sectors cannot select the
   same source.

``sector_priors``
   Optional mapping containing one flux-scaling prior for every sector. When
   omitted, all sectors use the shared ``x_prior``. When supplied, missing and
   unused sector keys are errors.

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

Shared-basis preparation uses ``H(region, nmeasure, source)``. When sources
have different basis indexes, preparation instead keeps one gathered state
dimension whose MultiIndex levels are ``(source, region_in_source)``. This is
the same concat-gather representation used for ``nmeasure`` with
``(site, time)`` levels: ragged values are concatenated rather than padded.
Modern preparation preserves the state-dimension name supplied by the basis
operator; it does not rename arbitrary state axes to ``region``.

``source`` remains the OpenGHG retrieval identity; sector names and priors live
in the model specification and select ``H`` by source label. Source-coordinate
order therefore does not determine sector routing. The current builder
supports one distinct source and one independent state vector per sector.
Rectangular legacy inputs may carry ``source_region_count(source)`` so padded
layouts can be rejected; modern preparation does not create that compatibility
metadata.

Running Prepared Inputs
-----------------------

Advanced workflows can prepare canonical RHIME inputs separately from data
acquisition and run them without repeating OpenGHG-backed preparation. Supply
the retained ``RhimePreparedInputs`` together with the existing public run,
model, output, and sampler specifications:

.. code-block:: python

   from openghg_inversions.models import RhimeModelSpec, SectorSpec
   from openghg_inversions.inversion_data import RhimePreparedInputs
   from openghg_inversions.rhime import (
       RhimeOutputSpec,
       RhimeRunSpec,
       RhimeSampler,
       run_rhime_from_prepared_inputs,
   )

   # Produced by prepare_rhime_inputs or by another source adapter that
   # satisfies the same canonical contract.
   prepared = prepare_inputs_elsewhere()
   prepared.save("prepared-inputs.nc")
   prepared = RhimePreparedInputs.load("prepared-inputs.nc")
   model_spec = RhimeModelSpec(
       species="ch4",
       domain="EUROPE",
       sectors=(
           SectorSpec(
               name="total",
               flux_source="total-ukghg-edgar7",
               x_prior={"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
               variable_suffix="total",
           ),
       ),
   )
   run_spec = RhimeRunSpec(
       start_date="2019-01-01",
       end_date="2019-01-02",
       sites=prepared.sites,
       averaging_period=prepared.averaging_period,
       model=model_spec,
       output=RhimeOutputSpec(output_format="none"),
       split_by_sectors=False,
   )
   result = run_rhime_from_prepared_inputs(
       prepared_inputs=prepared,
       run_spec=run_spec,
       sampler=RhimeSampler(draws=1000, tune=1000, chains=4),
   )

The prepared object is trusted canonical input: it must already contain the
observation, error, sensitivity, and optional boundary-condition variables
required by the selected model. A multisector run requires
``run_spec.split_by_sectors=True`` and a source-resolved layout on
``prepared.inv_inputs["H"]``. Shared-basis inputs may use a rectangular
``source`` dimension. Source-specific, ragged state blocks use one gathered
state dimension with a ``(source, region_in_source)`` MultiIndex. A scalar
``source`` coordinate is provenance for a single-sector input, not a
multisector layout. The sector count, prepared ``H`` layout, layout flag, and
output settings are validated before model construction or sampling. Output
side effects are still controlled by ``RhimeOutputSpec``.

Model Construction
------------------

The concrete PyMC graph, its variable names, an equivalent construction from
public model components, and links to tracked extension work are described in
:doc:`concrete_rhime_model`.

RHIME uses direct composition of the concrete standard or multisector model by
default. The private semantic-plan compiler remains available for development
and parity testing by setting ``builder_strategy="compiled"`` on
``RhimeModelSpec`` or in ``[RHIME.OPTIONS]``. There is no automatic fallback:
an error in the selected strategy stops the run. The concrete model is the
readable reference implementation; the compiled strategy is the opt-in
extension path and must preserve the externally meaningful graph contract for
components it does not intentionally change. See
:ref:`the concrete model stability contract <rhime-builder-stability>` for the
full contract.

``RhimePreparedInputs.save`` accepts NetCDF paths ending in ``.nc`` and Zarr
paths ending in ``.zarr``. Each artifact contains the canonical inversion
inputs and the retained operator-backed basis, including its reference flux;
``basis_artifact_path`` is recorded only as provenance and is not needed to
reload or run the prepared inputs. Prepared-input artifacts use a versioned
schema, CF compression-by-gathering for MultiIndexes, and a labeled
``site_metadata(site)`` dataset. Integer ``site_indicator`` values are derived
from the ``nmeasure`` site level as zero-based positions into that separate
site coordinate:
``site_metadata.site[site_indicator]`` must equal the ``site`` level of
``nmeasure``. The indicator itself is not the CF gathering coordinate, and
callers do not need to keep a second positional decoder synchronized. Site
metadata also aligns averaging periods, with exactly one value per site.
Metadata that is genuinely constant per site may be stored there too. Release
locations that vary within a site, as they do for satellite or aircraft
observations, are instead arrays aligned with the observations and must not be
reduced to site scalars. These metadata arrays may be carried alongside the
inversion arrays without implying that the model consumes them. Static
multisource bases store their order on an xarray ``source`` coordinate. Saving
to an existing artifact path replaces that artifact.

Source-specific retained fluxes must already share an exact time index before
they are stacked on ``source``. Inputs with different native frequencies, such
as hourly and monthly fluxes, require an explicit resampling policy rather than
implicit xarray alignment.

Source-neutral xarray inputs can be adapted at this boundary without OpenGHG
data acquisition. The supported container boundary is deliberately narrow:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Container
     - Support
     - Contract
   * - Ordered ``Mapping[str, Dataset]``
     - Supported
     - One active-observation dataset per site; mapping insertion order is
       retained.
   * - ``DataTree``
     - Supported
     - One direct child dataset per site; child order is retained.
   * - Direct ``Dataset``
     - Unsupported
     - Wrap one or more site-local datasets in an ordered mapping or a
       direct-child ``DataTree``. Load an already canonical artifact with
       ``RhimePreparedInputs.load``.
   * - Dense ``Dataset(site, time)``
     - Unsupported
     - No trimming or interpretation of padded rows is attempted. Split it
       into per-site datasets first.
   * - Padded dense arrays
     - Unsupported
     - Every row in a site-local dataset is active; padding has no MVP
       semantics.
   * - Pre-stacked ``nmeasure`` data
     - Unsupported
     - Pass per-site data to the adapter. Load an already canonical prepared
       artifact with ``RhimePreparedInputs.load`` instead.

This matches the cached verification-games workflow: sites may have unequal or
disjoint time axes, and the existing ragged gather path creates ``nmeasure``
only after per-site validation and basis projection. For example:

.. code-block:: python

   from collections import OrderedDict

   import numpy as np
   import xarray as xr

   from openghg_inversions.basis.basis_functions import BasisFunctions
   from openghg_inversions.inversion_data import (
       RhimePreparedInputs,
       prepare_rhime_inputs_from_xarray,
   )
   from openghg_inversions.models import RhimeModelSpec, SectorSpec
   from openghg_inversions.rhime import (
       RhimeOutputSpec,
       RhimeRunSpec,
       RhimeSampler,
       run_rhime_from_prepared_inputs,
   )

   # This retained artifact owns the basis operator, source-resolved reference
   # flux, state labels, and source order used for later reconstruction.
   basis_functions = BasisFunctions.load("base-country-basis.zarr")
   sources = ["ocean", "FF", "GPP", "TER"]  # meaningful, non-alphabetical order

   site_times = OrderedDict(
       TAC=np.array(
           ["2021-01-01T12:00", "2021-01-02T12:00"],
           dtype="datetime64[ns]",
       ),
       MHD=np.array(
           ["2021-01-01T15:00", "2021-01-03T15:00", "2021-01-04T15:00"],
           dtype="datetime64[ns]",
       ),
   )
   site_data = OrderedDict()
   for site, times in site_times.items():
       cache = xr.open_zarr(f"{site.lower()}-base-fp-x-flux.zarr")[
           "fp_x_flux_sectoral"
       ].sel(source=sources, time=times)
       # Current verification-games caches store mole fractions with units
       # "1". Numerically convert them in that consumer before adaptation;
       # changing metadata alone would be scientifically incorrect.
       if cache.attrs.get("units") == "1":
           cache = (cache * 1e6).assign_attrs({**cache.attrs, "units": "ppm"})
       if cache.attrs.get("units") != "ppm":
           raise ValueError("Normalize this cache numerically to ppm first.")
       observation = xr.DataArray(
           np.zeros(times.size),
           dims="time",
           coords={"time": times},
           attrs={"units": "ppm"},
       )
       site_data[site] = xr.Dataset(
           {
               "mf": observation,
               "mf_error": xr.ones_like(observation).assign_attrs(units="ppm"),
               "mf_repeatability": xr.ones_like(observation).assign_attrs(units="ppm"),
               "mf_variability": xr.zeros_like(observation).assign_attrs(units="ppm"),
               "fp_x_flux_sectoral": cache,
           }
       )

   prepared = prepare_rhime_inputs_from_xarray(
       site_data,
       basis_functions=basis_functions,
       averaging_period="1h",
       min_error=0.0,
       start_date="2021-01-01",
   )

   # Optional durable checkpoint: no OpenGHG retrieval or adapter projection is
   # repeated when the prepared artifact is loaded.
   prepared.save("base-prepared-inputs.zarr")
   prepared = RhimePreparedInputs.load("base-prepared-inputs.zarr")

   sectors = tuple(
       SectorSpec(
           name=source,
           flux_source=source,
           x_prior={"pdf": "lognormal", "mean": 1.0, "stdev": 1.0},
           variable_suffix=source.lower(),
       )
       for source in sources
   )
   run_spec = RhimeRunSpec(
       start_date="2021-01-01",
       end_date="2021-01-05",
       sites=prepared.sites,
       averaging_period=prepared.averaging_period,
       model=RhimeModelSpec(
           species="co2",
           domain="EUROPE",
           sectors=sectors,
           sigma_freq="monthly",
       ),
       output=RhimeOutputSpec(output_format="none"),
       split_by_sectors=True,
   )
   result = run_rhime_from_prepared_inputs(
       prepared_inputs=prepared,
       run_spec=run_spec,
       sampler=RhimeSampler(draws=1000, tune=1000, chains=4),
   )

Each site dataset must contain ``mf``, ``mf_error``, ``mf_repeatability``,
``mf_variability``, and either canonical ``H`` or cached ``fp_x_flux`` /
``fp_x_flux_sectoral``. Cached fields are projected through the retained basis
before ragged sites are gathered and are not copied into canonical inputs.
Other time-aligned extension variables are retained. Non-time-dependent data
variables are removed at this boundary rather than copied into every
observation. The retained operator and prior/reference flux are carried into
postprocessing; artifact provenance may be added to the returned basis value.

The adapter treats every supplied row as an active observation. Each site
therefore needs a nonempty, explicit, unique ``datetime64`` ``time`` coordinate
with no ``NaT`` values. Valid non-monotonic input ordering is retained.
Required observations, projected ``H``, and optional baseline contributions
must be finite. A site emptied by downstream input preparation is reported as
an error instead of being silently removed. ``min_error_per_site`` defaults to
``False``, matching the OpenGHG-backed RHIME preparation route; pass ``True``
to request per-site minimum errors.

``mf.units`` must be a nonempty string. Every present concentration-valued
field---observation-error components, canonical ``H`` or the selected cache,
``H_bc``, and ``fixed_baseline``---must declare the same exact unit string at
every site. The adapter does not perform unit conversion; normalize equivalent
spellings before calling it. Projected ``H`` inherits the selected cache units.

Canonical ``H`` may use ``(region, time)`` for one source,
``(region, time, source)`` for multisource data with one shared basis, or a
gathered state dimension with a ``(source, region_in_source)`` MultiIndex for
source-specific ragged bases. Multisector inputs must carry source-resolved
retained prior flux with the same source names and order as ``H``;
source-specific basis operators must match that order too. The adapter rejects
a total retained flux for multisector inputs instead of broadcasting it across
sectors. Labels on explicit ``source`` dimensions must be unique, nonempty
Python or NumPy strings: bytes, numbers, duplicates, and implicit coercion are
rejected. Repeated source values within the gathered
``(source, region_in_source)`` state MultiIndex are valid. Exact source order is
preserved across the cache, canonical ``H``, retained flux, and operator
metadata. Ragged state blocks are concatenated with the modern gather helpers;
the adapter never introduces rectangular zero padding. ``H_bc`` must contain
exactly ``time`` and ``bc_region`` dimensions in either order.

Release coordinates must be supplied as a pair. Stationary surface or column
sites may use scalar or singleton ``release_lat`` and ``release_lon`` values;
the adapter broadcasts them over that site's observations for the PARIS
template. Mobile platforms must supply both coordinates on exactly the site's
``time`` dimension, in matching order. Partial, non-finite, or other coordinate
layouts are rejected rather than flattened or inferred. If any retained site
supplies release coordinates, every retained site must supply its pair.

Dense padding, pre-stacked layouts, unit conversion, and coercion of source
labels are intentionally deferred generalizations. They should be added only
with a concrete consumer and explicit semantics rather than inferred by this
adapter.

Two independent baseline modes are supported at this prepared-input boundary.
Include ``H_bc`` for the existing sampled boundary-condition contribution.
Include ``fixed_baseline(time)`` for a supplied observation-aligned contribution
in the same units as ``mf`` that is added directly to the model mean. Both
variables may be present. If any retained site supplies ``fixed_baseline``, all
retained sites must supply it. The fixed term is not represented as a flux
sector. The existing INI/OpenGHG configuration wrappers do not yet synthesize
``fixed_baseline``; callers using that mode should use this xarray adapter and
``run_rhime_from_prepared_inputs``.

Generic concentration statistics and PARIS concentration products treat the
effective baseline as fixed plus sampled contributions. The sampled ``mu_bc``
trace remains unchanged; the fixed contribution is composed only in the model
mean and output views. The temporary HBMCMC-compatible ``legacy`` output does
not accept fixed baselines, and neither does the legacy HBMCMC model builder;
new prepared-input features do not extend those compatibility paths.

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

   [RHIME.OPTIONS]
   builder_strategy = "concrete"

   [RHIME.OUTPUT]
   output_path = "outputs"
   output_name = "example"

For multi-sector RHIME, use ``sector_sources`` when the optimized sector names
are not the same strings as the OpenGHG source values. Its values must match
``flux_sources`` exactly and must be unique. If ``sector_priors`` is supplied,
it must contain exactly the same sector keys as ``sector_sources``; otherwise
omit it and use one shared ``x_prior``.

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

``MaxChildPCAEccentricity`` strictly checks every proposed child by default.
Its optional ``min_child_target_weight_share`` parameter can exempt only
children below a configured share of the same class/source-local equal-target
weight. This lets a low-weight, topology-forced fragment avoid vetoing
well-shaped material children produced by ``ConnectedComponentPartitionStep``.
The exception changes split acceptance only: it does not reconnect, freeze,
prune, or marginalize the resulting small region. A direct three-argument
policy call has no target-region context and therefore remains strict.

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
