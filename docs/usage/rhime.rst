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

Configuring PyTensor precision
------------------------------

A fresh OpenGHG Inversions PyMC or RHIME process defaults PyTensor to
``float32``. This keeps float32 footprint sensitivities from being promoted
when observations or errors arrive from preparation as float64. Because
PyTensor configuration is process-wide, graph precision is not a
``RhimeModelSpec`` option and must be selected before importing PyTensor,
PyMC, or OpenGHG Inversions.

Set PyTensor's native environment flag to opt into a float64 graph::

   PYTENSOR_FLAGS="floatX=float64,warn_float64=ignore" python inversion.py

An explicitly configured or already-imported PyTensor runtime is never
overwritten. In a notebook, set the environment flag before library imports
and restart the kernel when changing precision. Storage precision, graph
precision, and numerically sensitive accumulation precision are separate
concerns; components may promote specific calculations without requiring the
whole graph or prepared-data cache to use float64.

Python API
----------

The stable package imports below are unchanged. Scientists who want to inspect
or copy a complete implementation can read
``openghg_inversions.rhime.standard`` or
``openghg_inversions.rhime.multisector`` directly; each module shows its whole
scientific process from option resolution through output construction.

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

   from openghg_inversions.rhime import (
       PollutionEventSettings,
       RhimeModelSpec,
       SectorSpec,
   )
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
       likelihood=PollutionEventSettings(),
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

Aggregation-error covariance
----------------------------

Aggregation error is disabled by default. An advanced run may explicitly add
fixed aggregation error to the observation covariance without changing the
raw ``mf_error`` values. Attach these arrays only after observation filtering,
averaging, global site gathering, and basis projection, so both covariance
axes match the final ``nmeasure`` ordering. Prepared inputs may contain one of:

* ``aggregation_error_covariance(nmeasure, nmeasure_cov)`` for the exact dense
  covariance;
* ``low_rank_factor(nmeasure, agg_rank)`` together with
  ``diagonal_residual_variance(nmeasure)`` for a low-rank-plus-diagonal
  approximation;
* ``aggregation_error_sd(nmeasure)`` for an independent diagonal
  approximation or diagnostic.

Dense and low-rank forms are the primary representations. If a matching
``aggregation_error_sd`` diagnostic is present beside either one, RHIME
validates it against the structured covariance diagonal but does not replace
the structured likelihood with it. Select ``aggregation_error_mode="dense"``,
``"low_rank"``, or ``"diagonal"`` to opt into a representation. ``"auto"``
is also an explicit opt-in that selects an available structured form first;
merely placing an array in prepared inputs does not enable it. The default
``"none"`` ignores prepared aggregation-error arrays.

This likelihood option does not perform or certify a coherent prior
transformation. If aggregation covariance represents states removed by a
native-to-reduced transformation, the caller must also supply the matching
transformed prior and forward operator as one coherent preparation product.

The minimum-error setting remains a floor on total marginal standard
deviation, including aggregation error, while off-diagonal covariance is left
unchanged. Legacy HBMCMC model and replay paths are not extended. Until the
derived-error reconstruction follow-up lands, aggregation-error runs support
``output_format="inv_out"`` and ``"none"``; ``"basic"``, ``"paris"``, and
``"legacy"`` are rejected rather than producing lossy error fields.

Model Construction
------------------

The concrete PyMC graph, its variable names, an equivalent construction from
public model components, and links to tracked extension work are described in
:doc:`concrete_rhime_model`.

RHIME directly composes the concrete standard or multisector model in its
corresponding recipe module. See
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

Source-neutral xarray preparation
---------------------------------

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
only after per-site validation and basis projection. The following first block
is illustrative application code adapted from verification-games. OpenGHG
Inversions does not define its cache paths, acquire its observations, or choose
its numerical unit conversion.

Application-specific input assembly:

.. code-block:: python

   from collections import OrderedDict

   import numpy as np
   import xarray as xr

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
       # Schematic placeholder for application-owned observation loading.
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

The public OpenGHG Inversions adapter begins here. It validates and projects
the supplied site datasets; it does not acquire or normalize the upstream data.

Adapt the application data with OpenGHG Inversions:

.. code-block:: python

   from openghg_inversions.basis.basis_functions import BasisFunctions
   from openghg_inversions.inversion_data import prepare_rhime_inputs_from_xarray

   # This retained artifact owns the basis operator, source-resolved reference
   # flux, state labels, and source order used for later reconstruction.
   basis_functions = BasisFunctions.load("base-country-basis.zarr")
   prepared = prepare_rhime_inputs_from_xarray(
       site_data,
       basis_functions=basis_functions,
       averaging_period="1h",
       min_error=0.0,
       start_date="2021-01-01",
   )

Optionally save a durable checkpoint so projection is not repeated when the
prepared artifact is reopened:

.. code-block:: python

   from openghg_inversions.inversion_data import RhimePreparedInputs

   prepared.save("base-prepared-inputs.zarr")
   prepared = RhimePreparedInputs.load("base-prepared-inputs.zarr")

Finally, construct and execute the OpenGHG Inversions run from the canonical
prepared object:

.. code-block:: python

   from openghg_inversions.rhime import RhimeModelSpec, SectorSpec
   from openghg_inversions.rhime import (
       RhimeOutputSpec,
       RhimeRunSpec,
       RhimeSampler,
       run_rhime_from_prepared_inputs,
   )

   sources = ["ocean", "FF", "GPP", "TER"]
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

This adapter is an extension boundary for data acquisition and assembly, not a
plugin interface for alternative RHIME models or likelihood components.

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
Required observations, projected ``H``, and optional sampled ``H_bc`` values
must be finite. A site emptied by downstream input preparation is reported as
an error instead of being silently removed. ``min_error_per_site`` defaults to
``False``, matching the OpenGHG-backed RHIME preparation route; pass ``True``
to request per-site minimum errors.

``mf.units`` must be a nonempty string. Every present concentration-valued
field---observation-error components, canonical ``H`` or the selected cache,
and ``H_bc``---must declare the same exact unit string at every site. The
adapter does not perform unit conversion; normalize equivalent spellings before
calling it. Projected ``H`` inherits the selected cache units.

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

Include ``H_bc`` for the existing sampled boundary-condition contribution. A
deterministic observation-aligned ``fixed_baseline`` is deliberately rejected
until a reusable semantic Baseline component defines common likelihood and
output behavior; this follow-up is tracked in `issue #550
<https://github.com/openghg/openghg_inversions/issues/550>`_. New prepared-input
features do not extend legacy HBMCMC model or output paths.

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

As weights-first Python groundwork for issue #452, built on the class
composition merged in PR #520, the lower-level API can build a
region-constrained fixed-outer basis from packaged InTEM and raw
country/land-sea class maps. The largest InTEM label defines the bounded inner
region, ``nbasis`` is allocated only across the selected inner classes, and
every distinct outer label receives one target:

.. code-block:: python

   from openghg_inversions.basis import (
       load_country_region_classes,
       load_intem_outer_regions,
       region_constrained_fixed_outer_basis_from_weights,
   )

   outer_regions = load_intem_outer_regions("EUROPE")
   region_classes = load_country_region_classes("EUROPE")
   basis = region_constrained_fixed_outer_basis_from_weights(
       weights,
       start_date="2020-01-01",
       domain="EUROPE",
       nbasis=150,
       outer_regions=outer_regions,
       region_classes=region_classes,
   )

Pass a ``SplitStrategy`` instance through ``split_strategy`` to choose a
different generator for the bounded inner classes without changing the
fixed-outer layout or its target allocation. Outer IDs remain fixed maps even
when an ID is disconnected. Strategy results are validated at the class-local
boundary before global relabelling.

``weights`` must cover the whole fixed-outer grid because the result includes
the outer states; the general constrained weights-first adapter separately
supports cropped weights. When neither loaded maps nor path arguments are
supplied, the adapter loads both packaged fields. When ``outer_regions`` is
omitted, ``outer_regions_path`` selects a direct NetCDF file; when
``region_classes`` is omitted, ``country_directory`` selects the country or
land/sea class-map directory. Already-loaded custom fields are also accepted
and normalized to the authoritative weights grid. Distinct
non-null values in a custom country map remain distinct classes. Small float
storage differences are accepted; incompatible coordinate values, units, or
CRS definitions are rejected, and null outer cells remain label ``0``. This
adapter is separate from the legacy
``fixed_outer_regions_basis`` weighted route, whose historical output remains
unchanged, and it is not yet routed through RHIME configuration.

This groundwork composes a transient class field and still passes one weight
field to the downstream algorithm. Separate source weights, sensitivities, and
``basis_group`` metadata remain distinct inputs and are not created by this
adapter.

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

Active And Fixed States
-----------------------

The modern model builders sample only active flux-scaling states. Inactive
columns are removed from the sampled vector and restored into the full ordered
``x`` (or ``x_<sector>``) model variable before the forward calculation and
postprocessing; the corresponding columns of ``H`` are not physically removed.
By default, a sensitivity column is inactive only when every value in that
column is exactly zero. No tolerance is applied, so a near-zero nonzero column
remains active. Inactive multiplicative scaling states default to one. For an
exactly-zero column the fixed value cannot affect the forward calculation.

``StateActivity`` can also freeze labelled states, ``basis_group`` values, or a
complete sector. Labelled masks and fixed values are aligned to the state
coordinate rather than interpreted as numeric state ranges. Prior distribution
parameters may likewise be scalars, full one-dimensional arrays, or labelled
``DataArray`` objects. Fixing a nonzero state treats that parameter as known
exactly; it does not integrate over its prior uncertainty. This represents an
experiment such as asking what could be recovered if one emissions sector were
known without uncertainty. The fixed state's uncertainty is deliberately
removed rather than transferred into aggregation error.

Given canonical ``inv_inputs``, the following example assumes that
``inv_inputs["H"]`` carries a state-aligned ``basis_group`` coordinate
containing the value ``"outer"``:

.. code-block:: python

   from openghg_inversions.models import StateActivity
   from openghg_inversions.observation_error import resolve_aggregation_error
   from openghg_inversions.rhime import PollutionEventSettings
   from openghg_inversions.rhime.standard import build_standard_rhime_model

   state_policy = StateActivity(
       fixed_groups=("outer",),
       fixed_value=1.0,
   )
   model = build_standard_rhime_model(
       inv_inputs["H"],
       observations=inv_inputs["mf"],
       observation_error=inv_inputs["mf_error"],
       aggregation_error=resolve_aggregation_error(inv_inputs, "none"),
       minimum_error=inv_inputs["min_error"],
       likelihood_settings=PollutionEventSettings(
           sigma_prior={"pdf": "uniform", "lower": 0.0, "upper": 0.1},
       ),
       boundary_sensitivity=inv_inputs.get("H_bc"),
       x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       state_activity=state_policy,
   )

The direct multisector builder accepts one ordered sequence of ``SectorSpec``
objects. Each sector carries its source, backend suffix, prior, and optional
``state_activity`` override; the builder's ``state_activity`` argument supplies
the shared policy. ``StateActivity(active=False)`` freezes a complete sector.
RHIME config-file syntax and persisted activity-reason tables remain follow-up
work. Each ``SectorSpec.x_prior`` supports the same scalar, full-state array,
and labelled ``DataArray`` parameter forms; labelled values must match the
selected sector's state coordinate exactly.

For low-level model construction, first inspect a labelled sensitivity matrix with
``detect_zero_sensitivity``, then combine the returned mask with a
``StateActivity`` using ``resolve_state_activity``. State-vector graph helpers
consume the resulting ``ResolvedStateActivity``; they do not inspect ``H`` or
infer its output dimension.

Boundary-condition scaling states can be handled the same way. Set
``RhimeModelSpec(bc_state_activity=StateActivity(active=False))`` (or the
equivalent low-level builder argument) to retain ``mu_bc = H_bc @ bc`` with a
fixed ``bc`` vector and no boundary-condition random variable. This is distinct
from supplying a standalone baseline time series, which belongs to a separate
baseline component.

The legacy single-sector ``inferpymc`` / ``fixedbasisMCMC`` compatibility path
does not gain multisector behavior. It now removes exact-zero ``H`` columns in
the same way as the standard model: observation-space predictions are unchanged,
but formerly unidentified entries in the full posterior ``x`` are reconstructed
at their fixed values instead of being prior draws. Derived flux products that
use those entries may therefore differ from earlier releases.

Correlated Positive Reduced States
----------------------------------

``CorrelatedLognormalPrior`` is the low-level contract for one labelled joint
positive state. It accepts an arithmetic mean vector ``m`` and covariance
``C`` and constructs the latent Gaussian moments

.. math::

   \Sigma_{ij} = \log\left(1 + \frac{C_{ij}}{m_i m_j}\right),
   \qquad
   \mu_i = \log(m_i) - \frac{1}{2}\Sigma_{ii}.

For a mean-one scale state this reduces elementwise to
:math:`\Sigma_{ij} = \log(1 + C_{ij})`; this is not a matrix logarithm. The
implementation uses ``np.log1p`` for numerical stability. The contract
validates labelled state order, arithmetic covariance, and positive
definiteness of the derived latent covariance. ``add_correlated_lognormal_state``
then creates a whitened standard-normal ``<name>_latent`` and the positive
public state ``<name>``.

.. code-block:: python

   import numpy as np
   import xarray as xr

   from openghg_inversions.models import (
       CorrelatedLognormalPrior,
       add_correlated_lognormal_state,
       registered_model,
   )

   mean = xr.DataArray(
       [1.0, 1.0],
       dims="state",
       coords={"state": ["region-a", "region-b"]},
   )
   prior = CorrelatedLognormalPrior(
       mean,
       np.array([[0.16, 0.03], [0.03, 0.09]]),
   )

   with registered_model():
       state_result = add_correlated_lognormal_state(prior, var_name="x")

``C`` must be the dense covariance of an already-reduced sampled state, such as
a basis-state covariance ``C_alpha``. It is not the covariance of a native grid.
For ``p`` sampled states this contract retains several dense ``p`` by ``p``
matrices, uses additional dense temporaries, and performs a Cholesky
factorization: persistent matrix storage scales as :math:`O(p^2)` and the
factorization as :math:`O(p^3)`. Current inversions typically have fewer than
500 sampled states. Construction emits an operational warning above 1,000
states; this threshold is not a mathematical limit. Native grids with more
than 100,000 cells require a structured covariance representation, such as a
Kronecker product, whose projected products can be evaluated without realizing
the full native covariance.

The low-level :doc:`native_covariance` API supplies this structured native
covariance action and its covariance-compatible projected product blocks. It
is a preparation interface and is not yet connected to the RHIME likelihood.

This component does **not** perform the native-to-reduced uncertainty
transformation or remove state coordinates. That work must transform the prior
uncertainty, forward operator, and aggregation error together. The coherent
covariance, transformed-forward-model, and aggregation-error identities are
exact only for a jointly Gaussian state. Reusing the resulting first two
moments while representing the retained state as LogNormal and the unresolved
contribution as Gaussian is a moment-matched closure, not exact marginalization
of a LogNormal state. `Issue #566 <https://github.com/openghg/openghg_inversions/issues/566>`_
tracks the coherent preparation contract.

The covariance matrix uses a distinct second dimension, named
``<state_dim>_covariance`` by default. If an xarray covariance supplies column
labels, they must exactly equal the primary state labels in the same order.
An unlabelled NumPy covariance is interpreted in the arithmetic mean's state
order. Reordered inputs are rejected rather than interpreted positionally.

The built-in CO2 recipe uses this component for one gathered correlated state.
Promotion into standard and multisector ``RhimeModelSpec`` recipes remains
follow-up work in `OPE-78
<https://linear.app/openghg-inversions/issue/OPE-78/promote-shared-covg-components-into-standard-and-multisector-rhime>`_;
reusable source, sector, and state selection is tracked in `OPE-80
<https://linear.app/openghg-inversions/issue/OPE-80/extract-source-sector-and-state-selection-from-compiler-plans>`_.
Current ``SectorSpec`` priors remain independent.

Basis-Aware Prior Standard Deviations
-------------------------------------

The lower-level basis API can project independent grid-cell scale-factor
uncertainty onto the labelled states of a retained ``BasisFunctions`` object.
For basis membership ``A`` and cell-total weights ``w = flux * area``,
``project_basis_prior_stdev`` computes

.. math::

   \sigma_{x,k} =
   \frac{\sqrt{\sum_i A_{ik}(w_i\,\sigma_i)^2}}
        {\left|\sum_i A_{ik}w_i\right|}.

The grid-cell standard deviation may be scalar, source-labelled, or gridded.
The retained flux is used unless an explicit replacement is supplied. State
and source labels come from the retained operator; source-specific ragged bases
retain their gathered ``(source, region_in_source)`` state coordinate.

.. code-block:: python

   from openghg_inversions.basis import (
       calibrate_basis_prior_stdev,
       project_basis_prior_stdev,
   )
   from openghg_inversions.observation_error import resolve_aggregation_error
   from openghg_inversions.rhime import PollutionEventSettings
   from openghg_inversions.rhime.standard import build_standard_rhime_model

   x_prior_stdev = project_basis_prior_stdev(
       basis_functions,
       area_grid=cell_area,
       grid_cell_prior_stdev=grid_prior_sd,
   )
   model = build_standard_rhime_model(
       inv_inputs["H"],
       observations=inv_inputs["mf"],
       observation_error=inv_inputs["mf_error"],
       aggregation_error=resolve_aggregation_error(inv_inputs, "none"),
       minimum_error=inv_inputs["min_error"],
       likelihood_settings=PollutionEventSettings(
           sigma_prior={"pdf": "uniform", "lower": 0.0, "upper": 0.1},
       ),
       boundary_sensitivity=inv_inputs.get("H_bc"),
       x_prior={"pdf": "normal", "mu": 1.0, "sigma": x_prior_stdev},
   )

``calibrate_basis_prior_stdev`` accepts a caller-defined target matrix and
requested relative standard deviation. It projects a unit cell standard
deviation, then uses linearity independently for each source. The default
``target_statistic="median-relative"`` matches the median valid target-relative
SD. ``target_statistic="mean-total"`` instead matches mean target SD divided by
mean absolute target total. Both reductions work with dask-backed target rows,
and achieved values for every target are returned for inspection.

Pass the labelled Boolean ``state_is_active`` mask used by the model when some
states will be fixed, including exact zero-sensitivity states. Calibration then
sets those state widths to zero and excludes them from achieved target
uncertainty. The positive widths for active states can be passed directly to
the prior API; fixed-state zero widths are removed by active-state prior
slicing. Requested calibration widths must be strictly positive, and negative
grid-cell standard deviations are rejected.

The result contains ``grid_cell_prior_stdev``,
``x_prior_stdev``, state and target totals, achieved target SD and relative SD,
and explicit status variables. ``zero`` identifies a target with no absolute
weighted flux, while ``cancellation`` identifies nonzero signed weights whose
target total is zero. No target masks, countries, or relative-SD defaults are
built into these helpers.

For ragged multisource bases, ``x_prior_stdev`` remains on the operator's
gathered state coordinate. Source-level calibration diagnostics use the
``calibration_source`` dimension because xarray cannot store both a gathered
MultiIndex level named ``source`` and a separate dimension with the same name
in one ``Dataset``.

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
For temporary reproduction of historical products, pass
``--legacy-fixedbasis`` to ``run_hbmcmc.py``. This explicit opt-in sends the
untranslated INI parameters to ``fixedbasisMCMC``; a missing output format or
the old ``hbmcmc`` / ``hbmcmc_postprocessing`` names select the historical
``inferpymc_postprocessouts`` product and filename. An explicit
``output_format="legacy"`` still selects the modern legacy-format adapter.
The command prints a prominent warning, raises on unsupported options, and
does not fall back to RHIME if the legacy run fails.
Direct ``fixedbasisMCMC(...)`` calls are a temporary legacy Python path, not a
wrapper around ``run_rhime(...)``.
