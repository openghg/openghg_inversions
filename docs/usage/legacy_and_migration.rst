Migrate from HBMCMC to RHIME
============================

RHIME is the only inversion implementation in current OpenGHG Inversions.
The direct ``fixedbasisMCMC`` and ``inferpymc`` implementation was removed in
0.8 after its remaining known active user agreed to migrate. The 0.7.x release
line is the last line containing that implementation and the
``--legacy-fixedbasis`` option. The ``openghg_inversions.hbmcmc`` namespace
hosts the transitional ``run_hbmcmc`` wrapper and explicit legacy data/basis
utilities during 0.8. The complete subpackage is scheduled for removal in 0.9,
including its imports, adapters and executable entry points.

This page is for users with fixedbasis-style Python calls, INI files, batch
scripts, or HBMCMC outputs. For a new inversion, start with the
:doc:`model recipe chooser <model_recipes>` and :doc:`RHIME guide <rhime>`.

0.8 resource migration
-----------------------

Resource selection now ends at the runner boundary. Update existing calls and
configuration as follows:

* Remove ``output_path`` from ``prepare_basis``. Save its returned object with
  ``basis.save(selected_path)``, or set the configured runner's
  ``basis_output_path`` to an exact artifact filename. Checkpoint files and this
  optional export have separate destinations.
* Remove ``country_directory`` from run configuration. Use ``country_file`` for
  reporting, ``landsea_path`` for preparation land/sea classes, and
  ``outer_regions_path`` for fixed outer regions. Required preparation maps
  default to vendored files for the exact requested domain; explicit paths win.
  Missing explicit files or required domain defaults fail before acquisition.
  Country reporting requires an explicit file, independently of these defaults.
* Select file-backed BC bases using ``bc_basis_paths``, for example
  ``("/data/bc/january.nc", "/data/bc/february.nc")``. Runners load these files
  before acquisition and preserve their time combination and alignment.
  Custom scientific sensitivity calls receive the loaded ``bc_basis`` dataset.
  NESW uses acquired directional contributions and ``bc_basis=None``; it needs
  no file. On the HBMCMC route, an old ``bc_basis_directory`` with NESW is
  unused and warns, even if that directory does not exist. Disabled BCs load no BC resources.
* Modern runners reject ``fp_basis_case``, ``basis_directory``, ``bc_basis_case``
  and ``bc_basis_directory``. Replace them with exact basis/BC selections, or use
  the temporary HBMCMC CLI or normalized dictionary adapter during 0.8. Only that
  legacy route translates historical selections after winning overrides.
  Scientific functions receive explicit selections and never discover files.
* Replace the removed RHIME ``params_from_config`` re-export with
  ``read_rhime_ini``, apply overrides, then call ``RhimeConfig.from_params``.
  The dictionary adapter remains only in ``hbmcmc.compatibility`` during 0.8.
  Runner frontends, including direct ``run_standard`` calls, retain supported
  aliases through 0.8 with deprecation warnings. ``RhimeConfig.from_params``
  accepts canonical names only and has no alias translation or warnings.
  Modern output formats and explicit save paths remain.
* Standard ``run_rhime_from_prepared_inputs`` now uses the same model operations
  as full and staged execution and rejects ``model_builder``. Use
  ``likelihood_builder`` for a custom likelihood, or construct and sample a
  custom PyMC graph directly. The complete-builder hook remains available for
  multisector compatibility. Standard builders have been consolidated into
  ``build_standard_model``, which accepts eager inputs and materialized priors.
  Call ``materialize_standard_inputs`` explicitly before construction when
  inputs or prior parameters are lazy; related arrays compute together.
  Output choices belong to the runner. Select acquisition/preparation arguments with
  ``config.data.select(...)`` and use ``config.model`` / ``config.output``
  directly; the global ``RhimeConfig.select`` API has been removed.
* Use invocation-relative paths for Python/CLI calls, INI/JSON resource and
  output values, and explicit overrides in both full and staged execution.
  Staged file values previously resolved beside their configuration file; use
  absolute paths or invoke from the intended resource directory when migrating.
  Manifest dependency references still resolve beside the declaring manifest.
  Absolute paths keep their destinations. Frontends convert declared ``Path``
  fields, including path collections, without treating scientific strings as
  paths or loading scientific data during configuration construction.

See :doc:`rhime_configuration` for complete resource selection examples and
:doc:`staged_workflow` for the separate 0.7-to-0.8 staged artifact migration.

Additional 0.8 Python API changes
---------------------------------

Replace ``build_standard``, ``build_standard_rhime_model`` and
``build_standard_rhime_model_result`` with the single eager
``build_standard_model`` contract. Model inputs and enabled prior parameters must
be materialized before calling it; ``materialize_standard_inputs`` is the shared
convenience operation. It is not another CLI phase.

Configuration helpers introduced during this development cycle are named
``load_params``, ``group_params`` and ``option_groups``; ``read_rhime_ini`` retains
its established name. ``RHIME_PREPARATION_OPTION_NAMES`` is removed; select named
arguments from their explicit configuration owner instead.

Optional basis exports save the existing scientific value without replacing it
to stamp an export-path receipt. Original basis source provenance is retained;
the requested destination belongs to configuration or run/checkpoint metadata.
Self-contained ``InversionOutput`` and trace exports retain their defaults and
formats. The proposed deprecation of the self-contained file export has not been
implemented.

Planned 0.9 deprecations and removals
-------------------------------------

**Deprecations:** historical directory/name basis output conventions and legacy
basis output are scheduled for deprecation in 0.9. Explicit legacy output has no
automatic removal date; a deprecation version is not a removal commitment.

**Removals:** ``emissions_store``, ``emissions_domain``,
``inner_emissions_store`` and ``inner_emissions_domain`` are already deprecated
and scheduled for removal in 0.9. Replace them with their ``flux_*`` spellings.
The entire ``openghg_inversions.hbmcmc`` subpackage is removed in 0.9: its
wrapper, compatibility adapters and legacy data/basis utilities will have no
forwarding imports or executable shims. Use RHIME runners, ``RhimeMergedData``
for in-memory acquisition, ``RhimePreparedInputs`` for saved preparation, and
``BasisFunctions`` for retained basis artifacts where applicable. For legacy-only
utilities or unsupported historical artifacts, keep a compatible older release;
this migration does not add readers for those artifacts. Modern
``output_format="legacy"`` and ``save_inversion_output`` remain independently
supported and are not removed with the package.
Established public RHIME imports and CLI aliases retained *through* the 0.9
compatibility cycle are not promised to disappear at the start of 0.9.

Chain handling in ``run_hbmcmc.py``
-----------------------------------

In 0.8, the compatibility wrapper uses every retained chain for derived products,
including modern ``basic`` and PARIS products and legacy-format summary fields.
The chain-0-only compatibility policy and ``--all-chains`` flag have been removed;
all retained chains are now always used. Use OpenGHG Inversions
0.7 when reproducing that historical policy. The wrapper does not save the
modern inversion-output artifact by default; request it with
``save_inversion_output``. Pooling all chains for a summary does not by itself
establish that the chains converged.

Choose a migration route
------------------------

Existing INI and batch workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing supported fixedbasis-style INI files may continue to use the
transitional wrapper::

   python -m openghg_inversions.hbmcmc.run_hbmcmc \
       2019-01-01 2019-02-01 -c my_inversion.ini

The wrapper translates the old vocabulary, copies the effective configuration
for provenance, and calls ``run_standard``. It does not run the removed HBMCMC
implementation. ``--legacy-fixedbasis`` and ``--generate`` are no longer
accepted. Keep existing configuration files while migrating, but create new
ones from ``openghg_inversions/config/templates/rhime_template.ini`` and run
them with ``openghg-inversions run-rhime``.

Batch scripts need only replace a direct source-tree path with the module form
above. Start and end dates, ``-c``, ``--output-path`` and ``--kwargs`` remain
available. The JSON supplied to ``--kwargs`` must be
quoted as one shell argument, for example::

   python -m openghg_inversions.hbmcmc.run_hbmcmc \
       2019-01-01 2019-02-01 -c my_inversion.ini \
       --kwargs '{"min_error": 20.0, "nuts_sampler": "numpyro"}'

Python workflows
~~~~~~~~~~~~~~~~

Replace ``fixedbasisMCMC(...)`` with ``run_standard(...)`` and use the names in
the table below::

   from openghg_inversions.rhime import run_standard

   result = run_standard(
       species="ch4",
       sites=["TAC"],
       averaging_period=["1h"],
       domain="EUROPE",
       start_date="2019-01-01",
       end_date="2019-02-01",
       flux_sources=["total-ukghg-edgar7"],
       draws=5_000,
       chains=4,
       output_path="outputs",
       output_name="ch4_TAC",
   )

The Python ``run_rhime`` API is removed in 0.8; importing it raises an
``ImportError`` directing callers to ``run_standard``. This supersedes the earlier
proposal to retain a deprecated wrapper. The ``run-rhime`` CLI command retains its
name, written files and logging and calls ``run_standard`` directly.

``run_standard``, standard prepared-input execution and ``stages.postprocess``
return an :class:`~openghg_inversions.postprocessing.inversion_output.InversionOutput`,
including with ``output_format="none"``. Use ``result.trace`` instead of
``result.idata``, and use ``result.inv_inputs``, ``result.basis_functions``,
``result.variable_roles``, ``result.run_metadata`` and ``result.model_metadata``
for the scientific result. Output-policy choices, written paths, live PyMC models
and samplers are not stored on it; ``output_metadata`` contains the inference
description. Call product functions such as ``basic_output(result)`` to produce
in-memory summaries. For graph inspection, call the eager model builder directly.
Deferred multisector, nested and CO2 families retain their existing result types.

Parameter mapping
-----------------

Use canonical RHIME names in new Python and configuration files. HBMCMC
translates the fixedbasis vocabulary; RHIME runner frontends retain supported
spelling aliases until 0.9, warning on devel and in 0.8. Direct
``RhimeConfig.from_params`` calls require the canonical names below. They neither
translate aliases nor emit alias deprecation warnings.

.. list-table:: Fixedbasis-to-RHIME names
   :header-rows: 1
   :widths: 28 24 48

   * - Fixedbasis name
     - RHIME name
     - Migration note
   * - ``nit``
     - ``draws``
     - Number of retained draws
   * - ``nchain``
     - ``chains``
     - Number of sampler chains
   * - ``verbose``
     - ``progressbar``
     - Whether to show sampler progress
   * - ``sampler_kwargs``
     - ``sample_kwargs``
     - Keyword mapping passed to the sampler
   * - ``outputpath``
     - ``output_path``
     - Output directory
   * - ``outputname``
     - ``output_name``
     - Run/output stem
   * - ``xprior``
     - ``x_prior``
     - Flux-scaling prior mapping
   * - ``bcprior``
     - ``bc_prior``
     - Boundary-scaling prior mapping
   * - ``sigprior``
     - ``sigma_prior``
     - Model-error prior mapping
   * - ``offsetprior``
     - ``offset_prior``
     - Additive-offset prior mapping
   * - ``emissions_name``
     - ``flux_sources``
     - OpenGHG flux ``source`` values
   * - ``hbmcmc`` or ``hbmcmc_postprocessing``
     - ``legacy``
     - Modern HBMCMC-compatible NetCDF output
   * - ``emissions_store`` / ``emissions_domain``
     - ``flux_store`` / ``flux_domain``
     - Also rename the corresponding ``inner_`` options; do not supply both spellings
   * - ``outer_region_definition_file``
     - ``outer_regions_path``
     - Exact preparation map file
   * - ``fp_basis_case`` + ``basis_directory``
     - ``basis_path``, ``basis_partition_path`` or ``basis_partition_paths``
     - Select the retained artifact or exact labelled partition file(s); remove both discovery options
   * - ``bc_basis_case="NESW"`` + ``bc_basis_directory``
     - Omit both options
     - NESW is built in and needs no BC basis files
   * - Other ``bc_basis_case`` + ``bc_basis_directory``
     - ``bc_basis_paths``
     - Select the exact BC basis files, including the months needed by the run
   * - ``country_directory``
     - Explicit map file paths
     - ``country_file`` for reporting; ``landsea_path`` / ``outer_regions_path`` for preparation
   * - ``calculate_min_error``
     - ``min_error``
     - Use ``"residual"`` or ``"percentile"``
   * - ``reparameterise_log_normal``
     - prior ``reparameterise``
     - Put ``reparameterise = true`` in each relevant lognormal prior mapping

The wrapper retains the historical output filename convention. Direct
``run_standard`` calls use the RHIME filename convention unless explicitly
configured otherwise.

Migrating an old configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, an old fixedbasis INI might contain::

   [RUN]
   nit = 5000
   nchain = 4
   emissions_name = ["total-ukghg-edgar7"]
   emissions_store = "user"
   emissions_domain = "EUROPE"
   outputpath = "outputs"
   outputname = "ch4_TAC"
   output_format = "hbmcmc"
   fp_basis_case = "monthly"
   basis_directory = "/data/bases"
   bc_basis_case = "NESW"
   bc_basis_directory = "/data/bc"

Replace that section with canonical options, retaining the rest of the run's
species, site, dates, priors, ``burn`` and other scientific settings::

   [RUN]
   draws = 5000
   chains = 4
   flux_sources = ["total-ukghg-edgar7"]
   flux_store = "user"
   flux_domain = "EUROPE"
   output_path = "outputs"
   output_name = "ch4_TAC"
   output_format = "legacy"
   basis_partition_paths = ["/data/bases/monthly/EUROPE/january.nc",
                            "/data/bases/monthly/EUROPE/february.nc"]

The selected partition filenames are examples: inspect your old case directory
and select the files actually needed. Use ``basis_path`` instead for one retained
``BasisFunctions`` artifact, or ``basis_partition_path`` for one labelled
partition file. These three selections are mutually exclusive. The plural
collection must be nonempty and is sorted by path before loading; the first
sorted file supplies reference coordinates. Nested runs have the same choices
with the ``inner_`` prefix. No NESW directory is required.

If the old run used ``country_directory``, remove it and set ``country_file``
to the reporting file. Set ``landsea_path`` or ``outer_regions_path`` separately
if overriding the packaged preparation maps. A reporting map never selects a
preparation map. Relative output and resource destinations, including ``outputs``
and an explicit ``save_trace`` filename, resolve from the invocation working
directory in both full and staged execution.

Check scientific settings independently of renaming. In particular, moving from
OpenGHG Inversions 0.7 to 0.8 can change likelihood behaviour and the chain-summary
policy described here; renaming parameters alone does not reproduce historical
scientific behaviour.

The temporary dictionary adapter is available from
``openghg_inversions.hbmcmc.compatibility``. With ``normalise=False``,
``params_from_config`` returns raw decoded options plus winning overrides. With
``normalise=True``, it translates aliases and historical selections and returns
canonical keys and resolved ``Path`` values, including selected resource files.
It does not construct configuration defaults or expand site shorthand. Direct
callers of ``fixedbasis_params_to_rhime`` must additionally call
``adapt_resource_params`` before ``RhimeConfig.from_params``; the first function
alone does not finish resource translation. These adapters are removed with
HBMCMC in 0.9. For canonical INIs, use ``read_rhime_ini`` and apply overrides
before constructing ``RhimeConfig``.

Legacy-format output
--------------------

``output_format="legacy"`` remains supported for standard single-sector
RHIME. It creates the HBMCMC-compatible NetCDF product from the modern
``InversionOutput``; it does not invoke ``fixedbasisMCMC`` or ``inferpymc``.
The deprecated output names ``hbmcmc`` and ``hbmcmc_postprocessing`` remain
aliases for ``legacy`` at runner frontends until 0.9, with
``DeprecationWarning`` on devel and in 0.8. Runners translate after combining
file options and Python overrides. ``RhimeConfig.from_params`` and direct output
configuration factories require ``legacy``; they do not translate these aliases
or issue their deprecation warnings.
Prefer ``legacy`` in new configurations. The former ``resolve_rhime_config``
wrapper is removed; use the classmethod directly.

The compatibility product uses variables such as ``Yobs``, ``Yerror``,
``Ymodmean``, ``Ymodmedian``, ``Ymodmode``, ``xtrace``, ``sigtrace``,
``fluxmode``, ``scalingmean``, ``scalingmode``, and country totals. When
boundary conditions are enabled, it also includes ``bctrace`` and the
``YmodmeanBC``, ``YmodmedianBC``, and ``YmodmodeBC`` summaries. This is a
formatting compatibility promise, not a promise to reproduce the removed
executor's exact trace or all its historical attributes.

Removed Python helper APIs
--------------------------

Code that imported fixed-basis preparation or model helpers should move to
the retained RHIME abstractions:

.. list-table:: Removed helper mappings
   :header-rows: 1
   :widths: 38 30 32

   * - Removed API
     - Current API
     - Migration note
   * - ``prepare_fixedbasis_inversion_data``
     - Named RHIME preparation stages
     - Load data, then filter, build basis and sensitivities, and assemble inputs;
       the former ``prepare_rhime_inputs`` adapter has been removed
   * - ``FixedBasisPreparedData``
     - ``RhimePreparedInputs``
     - Backend-neutral prepared observations and sensitivities
   * - ``basis_functions_wrapper``
     - ``prepare_basis``
     - Call ``BasisFunctions.sensitivity`` when applying the retained basis;
       complete inversions should normally use ``run_standard``
   * - ``add_inferpymc_likelihood_component``
     - RHIME likelihood selection
     - Configure a built-in likelihood, or pass a ``likelihood_builder`` for a
       custom Python model; there is no direct inferpymc-component replacement

See :doc:`rhime` for preparation and recipe APIs and
:doc:`customising_rhime` for the custom-likelihood boundary.

The named RHIME preparation stages accept ordinary keyword arguments. Replace
former positional ``data_args`` mappings with named arguments or an explicit
``**config.data.select("name", ...)`` selection. See :doc:`rhime_configuration` for
selection ownership and the acquisition-to-preparation sequence. Use
``SiteOptions.from_inputs`` to normalize and align the complete selector record.
Acquisition/preparation fields formerly on ``RhimeConfig`` now belong to
``config.data: DataConfig``; output fields belong to ``config.output``. Grouped
parameter mappings use ``data``. The unshipped ``preparation`` field/key is not
a compatibility alias. Runners acquire data separately and pass that handoff to
scientific preparation with its selected named arguments.
Import ``SiteOptions`` from ``openghg_inversions.inversion_data``. Selector
implementation types and the low-level ``convert_to_list`` helper live in
``inversion_data._site_options``; their former ``acquisition`` and
``get_data.convert_to_list`` import aliases are removed.

During 0.8, legacy callers import their option translators and dictionary adapter
from ``hbmcmc.compatibility``. Modern callers do not use this package. Import ``read_rhime_ini`` from
``rhime.ini`` or the public ``rhime`` package instead of ``rhime.params``.

Legacy merged-data metadata
---------------------------

Fresh acquisition is named ``retrieve_inversion_data`` for both surface and
column observations. ``data_processing_surface_notracer`` remains a deprecated
wrapper preserving its existing signature, shorthand and six-tuple return.

The merged-data mapping returned by ``retrieve_inversion_data`` and
``load_merged_data`` no longer includes the unused ``.species``, ``.units``,
or ``.scales`` entries.  The species remains an explicit run option,
observation units remain on each site's ``mf.attrs["units"]``, and each
site's calibration scale remains on its dataset ``attrs["scale"]``.  Existing
netCDF and Zarr artifacts containing the old entries remain loadable; the loader
discards those redundant copies. Pickle merged-data files can no longer be
saved or loaded. To migrate one, use an older environment to reload it and
save it as Zarr before upgrading.

Merged-data acquisition and prepared-input reuse
-----------------------------------------------

Modern RHIME acquisition returns an in-memory ``RhimeMergedData`` handoff.
The proposed acquisition-artifact ``save``, ``load`` and ``load_legacy`` methods
are removed, along with ``close`` and the processing-stage marker. The
``prepare_rhime_inputs`` convenience adapter is also removed. Use
``RhimeMergedData.from_options`` and the named scientific stages in
``rhime.preparation``, or use a complete RHIME runner.

Remove ``reload_merged_data``, ``save_merged_data``, ``merged_data_dir`` and
``merged_data_name`` from modern RHIME configurations, including their
``inner_`` counterparts in nested runs. These options raise as unknown
arguments. For repeat runs, save and load ``RhimePreparedInputs`` and use the
prepared-input runner. The staged ``prepare`` command persists only those
prepared inputs and their manifest; subsequent stages read that artifact.

Historical tuple retrieval and its NetCDF/Zarr cache helpers remain available
for existing workflows. They do not provide a modern acquisition replay
contract. Do not relabel an old filtered cache as freshly acquired data.

Removed interfaces
------------------

The following interfaces have no supported current equivalent:

- direct ``fixedbasisMCMC`` and ``inferpymc`` execution;
- ``--legacy-fixedbasis`` and legacy config-template generation;
- legacy debug-return dictionaries and ``rerun_output``;
- direct ``inferpymc_postprocessouts`` calls; and
- plotting and file-analysis helpers from ``hbmcmc_post_process``.

If an archived analysis must execute those interfaces exactly, use a pinned
OpenGHG Inversions 0.7.x environment. Do not use that release line to start a
new workflow. Current RHIME preserves old INI translation and the modern legacy
output adapter, but does not promise bit-for-bit reproduction of the removed
sampling path.

Scientific compatibility
------------------------

The 0.8 wrapper translates supported historical options into the ordinary RHIME
model. It no longer preserves the historical boundary-only pollution-event
baseline, disconnected sigma-prior variable, or additive fixed-error minimum
error floor. Affected historical configurations fail at the HBMCMC boundary with
guidance to use OpenGHG Inversions 0.7. Use that release when those likelihood or
prior quirks are required for reproduction.

The runner switches ``preserve_legacy_likelihood``,
``_compatibility_unused_sigma_settings``, ``_compatibility_minimum_error_floor``
and ``compatibility_output_chain`` have been removed, along with the lower-level
``retain_unused_sigma`` and ``derived_output_chain`` compatibility options.
The HBMCMC boundary rejects ``no_model_error=True`` for both pollution-event and
additive likelihoods. It also rejects the default pollution-event selection
(omitted ``likelihood``) when both ``add_offset=True`` and
``pollution_events_from_obs=True``. Use version 0.7 for the historical behaviour,
or configure ``run_standard`` directly to select the current model; modern
standard offsets remain supported. Other accepted wrapper configurations use
the modern full baseline and output roles. Modern fixed-error and
pollution-event models retain their documented scientific definitions; modern
legacy-format output remains supported and uses all retained chains. Review
changes to priors, likelihood options, basis construction and outputs explicitly
when migrating an analysis to 0.8.

Explicit legacy merged-data utilities
-------------------------------------

Scripts that still create or reload ``fp_all`` mappings must import
``retrieve_inversion_data``, ``_save_merged_data`` and ``load_merged_data`` from
``openghg_inversions.hbmcmc.legacy_data``. The historical netCDF and Zarr
codecs retain their existing behaviour. Dictionary-based basis adapters,
including ``basis_weights_from_fp_all`` and ``make_basis_functions_from_fp_all``,
now live in ``openghg_inversions.hbmcmc.legacy_basis``. The previous canonical
module paths and exports have been removed.

Modern acquisition uses ``RhimeMergedData.from_options`` or direct dataset
construction. Modern runners resolve native prior flux once with
``native_flux.resolve_runtime_flux(merged.flux_data, split_by_sectors=...)``.
Modern ``prepare_basis`` accepts explicit ``site_data``, resolved ``runtime_flux``
and singular ``weighting_flux`` arrays, an exact ``basis_path`` input,
and resolved scientific map arrays. The ``output_path`` argument has been
removed: call ``basis.save(selected_path)`` on the returned object to export
it. Configured runners own ``basis_output_path`` and preserve export metadata.
Historical run selectors end at the HBMCMC input compatibility boundary.
Flat-file adaptation and explicit legacy writing remain in
``basis.compatibility`` under their own deprecation schedule; removing HBMCMC
does not remove these adapters. See :doc:`rhime_configuration` for the 0.8
modern basis artifact default, explicit legacy opt-in, and migration examples. For reusable
RHIME runs, save ``RhimePreparedInputs``;
legacy merged-data files are a separate explicit utility workflow.

Dataset and ``fp_all`` basis adapters resolve native flux once at their
compatibility boundary. ``basis_functions_from_flat_basis`` now accepts
``flux=`` rather than source datasets and a combination flag. Weight builders
accept one ``weighting_flux`` array rather than a source dictionary and selector.

For new preparation, ``BasisFunctions.reuse_operator(path, flux=runtime_flux)``
reuses the saved partition and binds the current prior without reading the saved
flux payload. ``prepare_basis(basis_path=..., ...)`` performs that operation.
For reconstruction, ``BasisFunctions.load(path)`` restores the complete saved
basis and its retained flux without acquiring data. Acquisition replay remains
deferred to #829.
