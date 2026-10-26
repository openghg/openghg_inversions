Migrate from HBMCMC to RHIME
============================

RHIME is the only inversion implementation in current OpenGHG Inversions.
The direct ``fixedbasisMCMC`` and ``inferpymc`` implementation was removed on
devel after its remaining known active user agreed to migrate. The 0.7.x release
line is the last line containing that implementation and the
``--legacy-fixedbasis`` option. The complete ``openghg_inversions.hbmcmc`` namespace, transitional wrapper,
dictionary adapters and legacy data/basis utilities are removed. No forwarding
shim recreates them. Modern execution accepts canonical options and uses the
single installed ``rhime`` executable. The next modern release number remains
undecided.

This page is for users with fixedbasis-style Python calls, INI files, batch
scripts, or HBMCMC outputs. For a new inversion, start with the
:doc:`model recipe chooser <model_recipes>` and :doc:`RHIME guide <rhime>`.

Observation selection and uncertainty
-------------------------------------

Observation-selection requests accept supported registered filters. Unsupported
active names fail ordinary registry validation before any filter or lazy data
executes. Global scalar and list requests validate even without observation
sites; site-specific choices are deferred for absent sites.

Make any resampling an explicit scientific choice before observation selection,
with uncertainty propagated for the resulting observations. Acquisition's
``averaging_period`` may be appropriate for a supported provider mean, but it is
not an automatic replacement: a median differs from a mean, and weighting,
uncertainty propagation and footprint alignment can also differ. Selection never
silently substitutes an aggregation statistic.

Selection filters now run before missing observation-error standard deviations
are derived or imputed. The empirical fallback uses only the retained
observations' concentration spread and nonzero-error median. Rejected outliers
can therefore no longer change accepted observations' fallback errors. Supplied
custom ``mf_error`` values remain authoritative; missing-both-component and NaN
rules retain their existing meanings. See
:doc:`/development/observation_uncertainty` for those distinctions.

Saved preparations retain their supplied errors, and saved sample replay uses
the fitted operands. Regenerate preparation to adopt the new imputation
population; loading an existing artifact does not recompute its errors.

Development resource migration
-------------------------------

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
  ``bc_basis_algorithm="NESW"`` is the default built-in selection. It uses
  acquired directional contributions and ``bc_basis=None`` without a file.
  Exact ``bc_basis_paths`` win over the inactive algorithm; ``use_bc=False``
  skips BC basis resolution and loading.
* Modern runners reject ``fp_basis_case``, ``basis_directory``, ``bc_basis_case``
  and ``bc_basis_directory``. Replace them with exact basis/BC selections.
  Scientific functions receive explicit selections and never discover files.
* Replace the removed RHIME ``params_from_config`` re-export with
  ``read_rhime_ini``, apply overrides, then call ``RhimeConfig.from_params``.
  The dictionary adapter is removed. Runner frontends, including direct
  ``run_standard`` calls, reject historical aliases before scientific work.
  ``RhimeConfig.from_params`` accepts canonical names only and has no alias translation or warnings.
  Modern output formats and explicit save paths remain.
* Replace ``run_rhime_from_prepared_inputs(prepared_inputs=..., run_spec=...)``
  calls with ``run_standard(prepared=prepared, config=config)`` or
  ``run_multisector(prepared=prepared, config=config)``. The old entry point and
  its multisector result/configuration adapters are removed. The ordinary
  resolved ``RhimeConfig`` supplies separate data, model, output and sampler
  choices; supplied prepared inputs skip acquisition/preparation. Custom
  likelihoods use ``likelihood_builder``. Replacing the complete graph requires
  direct construction and sampling; see :ref:`rhime-complete-model-composition`.
  Both recipes expose ``materialize_*_inputs`` and ``build_*_model`` operations
  so related arrays and enabled priors are computed together before PyMC
  construction. Select acquisition/preparation arguments with
  ``config.data.select(...)`` and use ``config.model`` / ``config.output`` directly;
  the global ``RhimeConfig.select`` API has been removed.
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

Additional development Python API changes
------------------------------------------

.. _prepared-input-factory-retirement:

Replace ``PreparedInputs.from_legacy_inputs(...)`` with
``PreparedInputs(inv_inputs=..., basis_functions=..., site_metadata=...)``.
Supply a dataset indexed by ``site`` with one aligned ``averaging_period`` value
per site, retaining the intended site order. Keep release coordinates aligned
to observations in ``inv_inputs``; for raw stationary or moving xarray inputs,
use ``prepare_rhime_inputs_from_xarray`` so its existing boundary normalizes them.
The xarray path preserves source/path metadata and its ``supplied`` provenance
fallback without mutating the supplied basis. The old per-site coordinate
arguments are not added to the constructor. Prepared-input formats, versions and
saved replay follow their persistence contracts independently of constructor
names. The Python constants are now ``PREPARED_INPUTS_SCHEMA`` and
``PREPARED_INPUTS_SCHEMA_VERSION``; their stored values and codec layout are
unchanged in this change. Persisted identifier migration is tracked separately in
`#976 <https://github.com/openghg/openghg_inversions/issues/976>`_.
This exact factory removal was
approved on 24 October 2026 under #945/#663; other constructors and independent
basis capabilities retain their own contracts.


Replace ``build_standard``, ``build_standard_rhime_model`` and
``build_standard_rhime_model_result`` with the single eager
``build_standard_model`` contract. Model inputs and enabled prior parameters must
be materialized before calling it; ``materialize_standard_inputs`` is the shared
convenience operation. It is not another CLI phase.

Replace ``build_multisector_rhime_model`` and
``build_multisector_rhime_model_result`` with ``build_multisector_model``.
Pass the canonical Dataset and resolved ``RhimeModelSpec``, together with eager
``priors`` from ``materialize_multisector_inputs``. The single builder constructs
the scientific graph and returns ``RhimeModelBuildResult``; use ``built.model``
when a caller needs the PyMC graph. Its optional ``sigma_alignment`` argument
accepts an already resolved mismatch alignment, as in the standard builder.

Nested Python callers now use ``run_nested(config=config, prepared=nested)``
in place of ``run_rhime_nested_from_prepared_inputs``. The returned
``NestedOutput`` contains ``outer`` and ``inner`` scientific ``InversionOutput``
views of one trace; use ``result.outer.trace`` to inspect samples and each view's
``basis_functions`` for its native grid. ``NestedRhimeResult`` and its
``RhimeResult`` forwarding envelope are removed. Use ``rhime run --model nested`` with canonical nested INI options.

The nested CLI now saves both native ``InversionOutput`` artifacts through the
standard writer. Their ``output_metadata`` no longer copies ``output_format``,
``output_path``, ``output_name``, ``save_trace`` or ``save_inversion_output``;
their ``provenance`` records package ``version``, ``revision`` and ``dirty``
status instead of the former ``contract``, ``compatibility_issue`` and
``basis_representation`` markers. Saving emits
``TIMING rhime.output.inversion_output_save`` log lines and validates
``output_name`` and ``start_date`` as filename components. The scientific
arrays, native grids and nested overlap metadata retain their meanings.
Requested basis exports retain their destination receipts in both native views.

Use ``materialize_nested_inputs(nested.combined, config.model)`` and pass the
returned Dataset and ``priors`` to the single ``build_nested_rhime_model``.
It now returns ``RhimeModelBuildResult``; ``build_nested_rhime_model_result``
and ``RhimeModelBuilderContext`` are removed. Scientific preparation takes
acquired outer/inner data and explicit resolved choices; the runner owns
resource loading and optional basis exports. Explicit product construction uses
``make_nested_paris_products(result.outer, result.inner, ...)`` instead of the
removed ``make_nested_paris_outputs`` result adapter. Configured PARIS runs now
require ``output_path`` before sampling; use that direct product function for
in-memory products. Runs that write PARIS products retain those datasets in
``result.products`` and all written destinations in ``result.paths``.
``inner_data_config`` exposes the same inner configuration derivation used by
the full runner for callers preparing their own handoff. See :doc:`nested_domains`.

``RhimeModelBuildResult`` now takes only ``model`` and ``output_contract``.
Move its former ``variable_roles``, ``supported_output_formats`` and ``metadata``
arguments into an ``OutputContract`` from
``openghg_inversions.postprocessing.contracts``. The legacy
``make_standard_rhime_result`` facade is removed. The former
``make_standard_output`` and ``make_multisector_output`` constructors are
replaced by ``make_model_output(..., split_by_sectors=...)`` for both recipes.
Pass ``False`` for standard and ``True`` for multisector scientific results.
The former ``save_standard_artifacts`` and ``save_multisector_artifacts``
exporters are replaced by the shared ``save_inversion_output`` function.

Configuration helpers introduced during this development cycle are named
``load_params``, ``group_params`` and ``option_groups``; ``read_rhime_ini`` retains
its established name. ``RHIME_PREPARATION_OPTION_NAMES`` is removed; select named
arguments from their explicit configuration owner instead.

Standard ``output_format="basic"`` does not compute a discarded in-memory summary
or write a product file. Call ``basic_output(result, country_file=...)`` explicitly
on the returned ``InversionOutput``. PARIS and legacy outputs retain their file
writers, invoked through ``postprocess_standard_outputs(result, output_spec)``.
Call ``validate_standard_output_resources(output_spec)`` before execution;
standard PARIS and legacy requests require an ``output_path``. The separate
``save_inversion_output(result, output_spec)`` exports inversion outputs and
traces without computing derived products. Copied runners should call the
postprocessor before the exporter.

``acquire_merged_data(data_config)`` now only acquires data. Validate a supplied
handoff with ``merged_data.validate_for_preparation(...)`` in the runner, then
call ``prepare_standard`` with the selected resources and resolved choices.
The combined ``standard.prepare_inputs`` convenience function is removed.

Optional standard basis exports save the existing scientific value without replacing it
to stamp an export-path receipt. Original basis source provenance is retained;
the requested destination belongs to configuration or run/checkpoint metadata.
Deferred-family basis exports retain their existing metadata contract.
Self-contained ``InversionOutput`` and trace exports retain their defaults and
formats. The proposed deprecation of the self-contained file export has not been
implemented. Checkpoints retain dependencies between phases; a self-contained
inversion-output export supports reporting without those manifests, while a
trace or basis export alone is not a complete inversion result. See
:doc:`staged_workflow` for artifact choices.

Agreed breaking-development boundary
-------------------------------------

The maintainer-approved `#944 treatment table
<https://github.com/openghg/openghg_inversions/issues/944>`_ and bounded
`#961 contract amendment <https://github.com/openghg/openghg_inversions/issues/961>`_
supersede the earlier 0.8/0.9 compatibility schedules. Modern development has
no HBMCMC namespace, wrappers, forwarding shims or dictionary adapters. Modern
runner/acquisition boundaries accept canonical names; the enumerated deprecated
parameter/output-format/flux-selector spellings are unsupported. Modern
configuration has no ``use_tracer`` field: every explicit occurrence, regardless
of value, receives ordinary unsupported-option validation before scientific
work; omission remains valid. Migration M1 work in
`#719 <https://github.com/openghg/openghg_inversions/issues/719>`_ delivers
this retirement with preserved evidence and canonical regression coverage. This
grants no general exception for public signatures or result wrappers. The explicitly named API migrations
documented on this page retain their own contracts; other current model selectors
and shared Python names retain their contracts except for the
:ref:`five named aliases <rhime-python-alias-retirement>` and the separately
approved :ref:`prepared-input factory <prepared-input-factory-retirement>`.
The factory is a bounded constructor removal, not a sixth alias. These changes preserve canonical behavior
and scientific artifact contents.
Supported formats/versions and explicit conversion or rejection of unsupported
artifacts are specified independently; it promises no automatic 0.7 compatibility.
The separately accepted
:ref:`CLI command map <rhime-cli-command-retirement>` requires
``rhime`` as the sole installed executable, without ``openghg-inversions``
or the four redundant full-run commands. Positional dates, ``--kwargs``,
effective configuration/date/output overrides, scientific execution, outputs
and artifacts are preserved during this command-name migration. Positional dates
and ``--kwargs`` remain open for discussion and revision; their replacement
syntax has not been chosen. No package/distribution/install rename is agreed.
Broader Python names/imports need their own exact replacement
maps. Removal PRs must reconcile
obsolete diagnostics with the behavior actually removed and its migration route.
Finished converters, launchers and a final release number are not prerequisites.
See the dated :doc:`retirement evidence <../development/migration_960>` for
affected source comparisons, historical cases and follow-up owners.

Unknown option names are rejected through ordinary supported-name validation,
including names in inactive groups. Values are validated only when applicable to
the selected phase: a preparation-only request defers a known output format until
an output consumer resolves it. The consumer rejects invalid formats against its
supported values. Modern validation has no special historical-name or token
checks; replacement advice belongs in this guide and migration tooling.

The current promises are `canonical request validation
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-configuration/spec.md#requirement-canonical-requests-and-phase-owned-validation>`_,
`independent INI decoding
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-configuration/spec.md#requirement-ini-decoding-is-independent-of-semantic-construction>`_,
`configuration identity
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-configuration/spec.md#requirement-configuration-is-not-a-historical-identity-schema>`_,
`modern flux selectors
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-configuration/spec.md#requirement-canonical-modern-flux-selectors>`_,
`canonical Python interfaces
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-model-selection/spec.md#requirement-canonical-public-python-interfaces>`_,
and `retained modern behavior
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/hbmcmc-isolation-retirement/spec.md#requirement-retained-modern-behavior-has-modern-owners>`_.
The `affected-evidence requirement
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence>`_
protects preserved observations and scientific expectations. The concrete
`retirement delivery checklist
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/changes/isolate-and-retire-hbmcmc/tasks.md#migration-evidence-checklist-and-delivery-order>`_
enumerates the removal surfaces and records the baseline, current owners and
delivery order. It requires each removal or deliberate behavior-change PR to:

* Identify affected :doc:`#960 ledger entries <../development/migration_960>`
  and plain cases; add bounded evidence for newly discovered behavior before removal.
* Compare the PR base and affected source with pinned revision
  ``0a32d73a5c4a72be0505a4ed682472e1613e49e9``, accounting for intervening changes.
* Record disposition and follow-up owner. Preserve historical inputs and
  expectations; add dated comparisons instead of replacing them with modern outputs.
* Preserve or port relevant scientific coverage before retiring adapter/test
  plumbing. Check unchanged behavior on canonical inputs and agreed scientific
  changes against their new semantics; link executed validation to its revision.

Migration M2 still owns final public API/alias decisions and release numbering
(`#945 <https://github.com/openghg/openghg_inversions/issues/945>`_), source
coverage (#948), converter (#949), launcher (#950), acceptance (#952) and
migration documentation (#432). Unresolved public aliases and independent basis
reader/writer, discovery and reconstruction decisions remain with #945 and
`#951 <https://github.com/openghg/openghg_inversions/issues/951>`_; they are not
blanket removal permission.

Modern INI decoding, canonical configuration, site shorthand, override behavior,
scientific contracts and
:doc:`borrowed-array/execution guarantees <../development/numerical_array_ownership>`
remain. Modern ``output_format="legacy"``, ``InversionOutput`` codecs and
``save_inversion_output`` remain independently supported. Useful basis algorithms,
labelled partitions and exact artifact capabilities retain their owning contracts.
For unsupported historical execution or artifacts, preserve a compatible older
environment; this boundary supplies no new artifact converter or release-readiness claim.

Choose a migration route
------------------------

Existing INI and batch workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Update fixedbasis-style settings to canonical names before invoking modern
execution. Keep the original configuration and matching 0.7 environment when
historical equations or artifacts must be reproduced. No automatic converter or
replacement launcher is supplied by this retirement.

For a canonical INI, use the packaged ``rhime_template.ini`` and run::

   rhime run 2019-01-01 2019-02-01 -c my_inversion.ini \
       --kwargs '{"min_error": 20.0, "nuts_sampler": "numpyro"}'

Dates, ``-c``, ``--output-path`` and JSON ``--kwargs`` retain their meanings.
See :doc:`cli` for the exact command and five Python alias replacement maps.

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
       mismatch_model="pollution_event",
       draws=5_000,
       chains=4,
       output_path="outputs",
       output_name="ch4_TAC",
   )

Choose a supported ``mismatch_model`` explicitly, or supply a custom likelihood.
This example uses the former ordinary wrapper's ``pollution_event`` model choice
with the modern scientific semantics described below.

The Python ``run_rhime`` API is unavailable; use ``run_standard``. Removed names
have ordinary import/attribute absence, without a migration-specific import hook.
Replacement advice belongs here rather than in attribute access. The ``run-rhime`` CLI alias is also
removed; use ``rhime run``. Effective inputs, written files and logging retain
their canonical meanings.

``run_standard``, ``run_multisector``, their ``prepared=...`` execution and their
saved ``postprocess`` operations return an
:class:`~openghg_inversions.postprocessing.inversion_output.InversionOutput`,
including with ``output_format="none"``. Use ``result.trace`` instead of
``result.idata``, and use ``result.inv_inputs``, ``result.basis_functions``,
``result.variable_roles``, ``result.run_metadata`` and ``result.model_metadata``
for the scientific result. Output-policy choices, written paths, live PyMC models
and samplers are not stored on it; ``output_metadata`` contains the inference
description. Call product functions such as ``basic_output(result)`` to produce
in-memory summaries. For graph inspection, call the eager model builder directly.
Nested execution returns ``NestedOutput`` with two native ``InversionOutput``
views; CO2 families retain their existing result types.

For multisector prepared execution with ordinary configuration, use
``run_multisector(prepared=prepared, config=config)`` and a ``RhimeConfig``
constructed with ``multisector=True``. Keep sector/source mappings and ordering
unchanged. The result is already the inversion output; remove
``result.inv_out`` and ``result.outputs["inversion_output"]`` access. Read sector
records from ``result.model_metadata["sectors"]`` and their scientific roles from
``result.variable_roles``. Compute in-memory sector diagnostics with
``postprocessing.make_outputs.make_sector_flux_outputs``. Copied runners call
``postprocess_multisector_outputs(result, output_spec)`` for requested products,
then ``save_inversion_output(result, output_spec)`` for exports.

New standard and multisector models expose one full physical ``x``, sensitivity
``hx`` and combined flux concentration contribution ``mu``. Multisector callers
select a source-labelled block from ``result.trace_group("posterior")["x"]``
rather than read ``x_<sector>`` variables; use the saved sector/source mapping
when scientific sector names differ from inventory names. Existing current-format
built-in sample handoffs with ``x_<suffix>`` state names remain readable. This
does not restore the retired historical staged-envelope format.

``RhimeRunSpec`` and ``RhimeResult`` are removed. All configured CO2 likelihoods
use ``run_co2`` (or ``rhime.run(model="co2")``) and return ``Co2Output``:
prepared inputs on ``result.prepared``, the complete trace on ``result.idata``,
calculated products on ``result.outputs`` and conditional affine reconstruction
on ``result.bound``. Replay retains no live model or sampler.

Replace ``run_rhime_co2_cached_sigma`` and ``model="co2_cached_sigma"`` with
``run_co2`` and a ``fixed_ou`` likelihood containing ``site_amplitude_prior``.
Remove CO2 ``variant`` and ``sampling.nuts_sampler`` settings. The likelihood
selects the matched cached sampler automatically. Use a supported positive-prior
table rather than ``site_amplitude_prior_scale``. The packaged example is now
``co2_fixed_ou.toml``.

Replace linked ``Co2O2RunSetup`` and ``resolve_co2_family_config`` with
``Co2O2ModelConfig.from_params(config)`` and pass ``model.bind(prepared)`` as
scientific keywords to ``run_rhime_co2_o2_from_prepared_inputs``. Remove linked
``variant`` settings. Concrete linked configuration supports fixed independent
error and fixed-amplitude OU; the direct cached scientific runner remains
available with explicit Python arguments. Ordinary and linked preparation use
``Co2PreparationConfig`` and ``Co2O2PreparationConfig``. Saved sampling/prior
phases accept their concrete model owners without reparsing them.

All CO2 stages use the shared preparation/sample manifests, original requested
dates/domain, and shared product naming. Sampling consumes one preparation
manifest; postprocessing consumes one sample manifest plus current output choices.
Regenerate retired cached-stage envelopes; there is no compatibility adapter.
Output settings use ``Co2OutputOptions`` fields, without the retired
``save_inversion_output`` and ``output_filename_convention`` keys.
The ``make_co2_rhime_result`` and ``make_co2_rhime_outputs`` adapters,
``retained_run_spec`` and ``with_prepared_sites`` helpers, and
``with_prepared_rhime_sites`` alias are removed. Copied CO2 runners use
``make_co2_output``, ``validate_co2_product_request``, ``calculate_co2_products``
and ``save_co2_products`` with separate ``Co2OutputOptions``.

Replay ``output_metadata["sampler"]`` retains available ``draws``, ``burn``,
``tune``, ``chains`` and ``nuts_sampler`` values from completed inference.
Missing or invalid descriptions are ``{}``; partial descriptions stay partial
without selecting current defaults. ``output_metadata["provenance"]``
comes from the sample manifest. The former ``build_and_sample_seconds`` replay
placeholder is removed: replay does not measure the earlier inference duration.

Remove multisector
calls to ``make_multisector_rhime_result`` and ``make_multisector_rhime_outputs``.
Use ``build_multisector_model``, ``make_model_output``, and the explicit
product/export calls above. For a custom complete model, pass selected eager
scientific inputs to an ordinary function and sample its returned graph;
there is no multisector model-builder context or ``model_builder`` hook.

The saved multisector envelope and its complete-configuration identity and
output-binding gates are removed in 0.8. Replace multisector
``prepare_rhime_stage``/``sample_rhime_stage``/``postprocess_rhime_stage`` calls
with ``rhime.multisector_stages.prepare``/``sample``/``postprocess``. Sampling
requires one preparation manifest and current model/sampler choices; products
require one sample manifest and current output choices. Use :doc:`staged_workflow`
to regenerate reusable preparation and sample manifests. Numerical prepared-input
codecs and final product formats are preserved; this migration does not convert
0.7 staged manifests.

Parameter mapping
-----------------

Use canonical RHIME names in Python and configuration files. Historical names
below are unsupported in modern runners and acquisition. Convert the intended
settings explicitly; runners and ``RhimeConfig.from_params`` do not translate
aliases or emit alias deprecation warnings.

.. list-table:: Fixedbasis-to-RHIME names
   :header-rows: 1
   :widths: 28 24 48

   * - Fixedbasis name
     - RHIME name
     - Migration note
   * - ``nit``
     - ``draws``
     - Post-tuning draws before separately discarding ``burn``
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
     - ``bc_basis_algorithm="NESW"`` (also the default)
     - ``bc_basis_algorithm="NESW"`` is the default; no BC files are needed
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

``run_standard`` uses the RHIME filename convention unless explicitly configured
otherwise. Select ``output_filename_convention="legacy"`` explicitly when needed.

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
   mismatch_model = "pollution_event"
   basis_partition_paths = ["/data/bases/monthly/EUROPE/january.nc",
                            "/data/bases/monthly/EUROPE/february.nc"]

The modern likelihood choice is explicit: the former ordinary wrapper selected
``pollution_event`` when omitted, while true historical fixedbasis execution had
different defaults and equations. Review this choice against the original route
and the scientific compatibility guidance below.

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
OpenGHG Inversions 0.7 to modern development can change likelihood behaviour and
the chain-summary policy described here; renaming parameters alone does not reproduce historical
scientific behaviour.

The historical dictionary adapter is removed. For canonical INIs, use
``read_rhime_ini``, apply overrides and construct ``RhimeConfig``. Every explicit
``use_tracer`` option is unsupported, including ``False`` and ``None``; omit it.
No tracer execution is introduced.

Legacy-format output
--------------------

``output_format="legacy"`` remains supported for standard single-sector
RHIME. It creates the HBMCMC-compatible NetCDF product from the modern
``InversionOutput``; it does not invoke ``fixedbasisMCMC`` or ``inferpymc``.
The output aliases ``hbmcmc`` and ``hbmcmc_postprocessing`` are unsupported.
Use ``legacy`` directly. ``InversionOutput`` codecs, ``save_inversion_output``
and exact explicit save paths remain independently supported.

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
     - ``PreparedInputs``
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

Import ``read_rhime_ini`` from ``rhime.ini`` or the public ``rhime`` package
instead of ``rhime.params``.

Merged-data acquisition and prepared-input reuse
------------------------------------------------

Modern RHIME acquisition returns an in-memory ``MergedData`` handoff.
The proposed acquisition-artifact ``save``, ``load`` and ``load_legacy`` methods
are removed, along with ``close`` and the processing-stage marker. The
``prepare_rhime_inputs`` convenience adapter is also removed. Use
``MergedData.from_options`` and the named scientific stages in
``rhime.preparation``, or use a complete RHIME runner.

Remove ``reload_merged_data``, ``save_merged_data``, ``merged_data_dir`` and
``merged_data_name`` from modern RHIME configurations, including their
``inner_`` counterparts in nested runs. These options raise as unknown
arguments. For repeat runs, save and load ``PreparedInputs`` and use the
prepared-input runner. The staged ``prepare`` command persists only those
prepared inputs and their manifest; subsequent stages read that artifact.

Historical six-tuple retrieval and its NetCDF/Zarr cache helpers are removed
with HBMCMC. No modern six-tuple replacement or historical-cache converter is
promised. Use the matching older software for those artifacts.
Do not relabel an old filtered cache as freshly acquired data.

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
new workflow. Modern legacy-format output remains supported independently;
modern RHIME does not promise bit-for-bit historical sampling reproduction.

Scientific compatibility
------------------------

Modern construction uses its documented full-baseline pollution-event model,
fixed-error model and all retained chains for derived summaries. It does not
restore historical boundary-only baselines, disconnected sigma variables or
additive fixed-error minimum-error floors. Modern standard offsets remain
supported. Review priors, likelihood, basis and output settings independently
of spelling changes; use matching 0.7 software for historical scientific quirks.
The original observations and modern comparisons are preserved separately in
:doc:`../development/migration_960`.

Retained basis and prepared-input capabilities
-----------------------------------------------

Modern acquisition uses ``MergedData.from_options`` or direct dataset
construction. Modern runners resolve native prior flux once with
``native_flux.resolve_runtime_flux(merged.flux_data, split_by_sectors=...)``.
Modern ``prepare_basis`` accepts explicit ``site_data``, resolved ``runtime_flux``
and singular ``weighting_flux`` arrays, an exact ``basis_path`` input,
and resolved scientific map arrays. The ``output_path`` argument has been
removed: call ``basis.save(selected_path)`` on the returned object to export
it. Configured runners own ``basis_output_path`` and preserve export metadata.
Historical run selectors are unsupported in modern runners.
Flat-file adaptation and explicit legacy writing remain in
``basis.compatibility`` pending the independent #951 dispositions; removing HBMCMC
does not authorize removing these adapters. See :doc:`rhime_configuration` for the
modern basis artifact default, explicit legacy opt-in, and migration examples. For reusable
RHIME runs, save ``PreparedInputs``; historical merged-data files require
matching older software.

Independent flat-partition basis adapters resolve native flux once at their
compatibility boundary. ``basis_functions_from_flat_basis`` now accepts
``flux=`` rather than source datasets and a combination flag. Weight builders
accept one ``weighting_flux`` array rather than a source dictionary and selector.

For new preparation, ``BasisFunctions.reuse_operator(path, flux=runtime_flux)``
reuses the saved partition and binds the current prior without reading the saved
flux payload. ``prepare_basis(basis_path=..., ...)`` performs that operation.
For reconstruction, ``BasisFunctions.load(path)`` restores the complete saved
basis and its retained flux without acquiring data. Acquisition replay remains
deferred to #829.
