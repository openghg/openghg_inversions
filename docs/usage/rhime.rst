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

Aggregation-error covariance
-----------------------------

Modern RHIME can add fixed aggregation error to the observation covariance
without changing the raw ``mf_error`` values. Attach these arrays only after
observation filtering, averaging, global site gathering, and basis projection,
so both covariance axes match the final ``nmeasure`` ordering. Prepared inputs
may contain one of:

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
the structured likelihood with it. ``aggregation_error_mode="auto"`` selects
the available structured form first; set ``"dense"``, ``"low_rank"``,
``"diagonal"``, or ``"none"`` for an explicit comparison run.

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

   from openghg_inversions.models import RhimeModelSpec, SectorSpec
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
postprocessing. By default, a sensitivity column is inactive only when every
value in that column is exactly zero. No tolerance is applied, so a near-zero
nonzero column remains active. Inactive multiplicative scaling states default
to one.

``StateActivity`` can also freeze labelled states, ``basis_group`` values, or a
complete sector. Labelled masks and fixed values are aligned to the state
coordinate rather than interpreted as numeric state ranges. Prior distribution
parameters may likewise be scalars, full one-dimensional arrays, or labelled
``DataArray`` objects. Given canonical ``inv_inputs``, the following example
assumes that ``inv_inputs["H"]`` carries a state-aligned ``basis_group``
coordinate containing the value ``"outer"``:

.. code-block:: python

   from openghg_inversions.models import StateActivity, build_rhime_model
   from openghg_inversions.sigma import SigmaAlignment

   state_policy = StateActivity(
       fixed_groups=("outer",),
       fixed_value=1.0,
   )
   sigma_alignment = SigmaAlignment.from_frequency(
       inv_inputs["site_indicator"],
   )
   model = build_rhime_model(
       inv_inputs,
       sigma_alignment=sigma_alignment,
       x_prior={"pdf": "normal", "mu": 1.0, "sigma": 0.5},
       state_activity=state_policy,
   )

For multisector builders, use ``state_activity`` as a shared policy or
``sector_state_activities`` for sector-name overrides;
``StateActivity(active=False)`` freezes a complete sector. Programmatic
prepared-input runs may set a shared policy on ``RhimeModelSpec`` and use its
``sector_state_activities`` mapping for overrides. The canonical per-sector
policy is ``SectorSpec(state_activity=...)`` and takes precedence over that
mapping. RHIME config-file syntax and persisted activity-reason tables remain
follow-up work.
Each prior dictionary in ``sector_priors`` supports the same scalar,
full-state array, and labelled ``DataArray`` parameter forms; labelled values
must match the selected sector's state coordinate exactly.

For low-level model construction, first inspect a labelled design with
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
continues to sample its full flux state and does not gain multisector behavior.

Correlated Positive States And Marginalization
----------------------------------------------

``CorrelatedLognormalPrior`` is the low-level contract for one labelled joint
positive state. It accepts an arithmetic mean vector ``m`` and covariance
``C`` and constructs the latent Gaussian moments

.. math::

   \Sigma_{ij} = \log\left(1 + \frac{C_{ij}}{m_i m_j}\right),
   \qquad
   \mu_i = \log(m_i) - \frac{1}{2}\Sigma_{ii}.

For a mean-one scale state this reduces to ``Sigma = log1p(C)``. The contract
validates labelled state order, arithmetic covariance, and positive
definiteness of the derived latent covariance. ``add_correlated_lognormal_state``
then creates a whitened standard-normal ``<name>_latent`` and the positive
public state ``<name>``.

.. code-block:: python

   import numpy as np
   import pymc as pm
   import xarray as xr

   from openghg_inversions.models import (
       CorrelatedLognormalPrior,
       add_correlated_lognormal_state,
   )

   mean = xr.DataArray(
       [1.0, 1.0],
       dims="state",
       coords={"state": ["region-a", "region-b"]},
   )
   prior = CorrelatedLognormalPrior.from_moments(
       mean,
       np.array([[0.16, 0.03], [0.03, 0.09]]),
   )

   with pm.Model():
       state_result = add_correlated_lognormal_state(prior, var_name="x")

This component is intentionally separate from ``StateActivity``. An inactive
``StateActivity`` entry is conditioned on a fixed value and restored into the
full public state. By contrast, ``prior.select_marginal(retained)`` selects the
principal marginal distribution and returns only retained states; it has no
fixed-value reconstruction.

Selecting a prior marginal does **not** reduce ``H`` or construct an unresolved
aggregation covariance. A coherent reduced model must provide a matching
retained design and aggregation covariance from the same preparation ledger.
Do not apply ``select_marginal`` independently to an ordinary full-state RHIME
design. `Issue #566 <https://github.com/openghg/openghg_inversions/issues/566>`_
tracks that coherent preparation contract.

The covariance matrix uses a distinct second dimension, named
``<state_dim>_covariance`` by default. If an xarray covariance supplies column
labels, they must exactly equal the primary state labels in the same order.
An unlabelled NumPy covariance is interpreted in the arithmetic mean's state
order. Reordered inputs are rejected rather than interpreted positionally.

The initial public component is intended for custom model builders. Built-in
``RhimeModelSpec`` integration for a gathered joint state and compatibility
per-sector aliases is follow-up work; current ``SectorSpec`` priors remain
independent.

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

   x_prior_stdev = project_basis_prior_stdev(
       basis_functions,
       area_grid=cell_area,
       grid_cell_prior_stdev=grid_prior_sd,
   )
   model = build_rhime_model(
       inv_inputs,
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
This compatibility route no longer preserves the exact historical
``fixedbasisMCMC`` / ``inferpymc`` passthrough behaviour. Use release ``0.6`` or
earlier if you need the old fixedbasis implementation.
Direct ``fixedbasisMCMC(...)`` calls are a temporary legacy Python path, not a
wrapper around ``run_rhime(...)``.
