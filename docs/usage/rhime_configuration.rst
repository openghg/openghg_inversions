Advanced RHIME configuration
============================

Use this reference when inspecting a resolved request, applying overrides, or
forwarding configuration values in a custom Python runner. For a first complete
inversion, start with the :doc:`standard tutorial <rhime_standard_tutorial>` or
:doc:`multisector tutorial <rhime_multisector_tutorial>`. The
:doc:`RHIME reference <rhime>` describes vocabulary and prepared inputs;
:doc:`customising_rhime` explains how to adapt the scientific preparation stages.

Inspecting the resolved request
-------------------------------

Standard and multisector runners resolve their effective options before
acquisition. Use ``RhimeConfig.from_params`` to inspect the same choices without
retrieving observations, building a model or sampling.
The returned ``RhimeConfig`` is the complete requested configuration:

* ``data`` is one ``DataConfig`` containing
  ``species``, ``domain``, dates, stores, basis, filtering and error settings.
  ``data.site_options`` is one public ``SiteOptions`` record containing complete
  aligned tuples of requested sites and their selectors.
* ``model`` is the existing ``RhimeModelSpec`` containing sectors, priors,
  likelihood and other scientific model choices.
* ``output`` is the existing ``RhimeOutputSpec`` describing final products,
  destinations, naming and save choices.
* ``sampler`` is the existing ``RhimeSampler`` with resolved sampling settings.
  It executes inference only when supplied a completed model. Its implementation
  lives in ``inference.sampling``; the established RHIME imports remain valid.

Sampler options cannot override model-owned coordinates: ``coords`` and ``dims``
inside ``sample_kwargs["idata_kwargs"]`` or
``posterior_predictive_kwargs["idata_kwargs"]`` are rejected, even if empty. Register
coordinates and dimensions when constructing the model instead. Posterior
``sample_kwargs["var_names"]`` overrides are also rejected at the sampler boundary;
posterior-predictive variable selection remains supported. This rule also
applies when constructing or directly invoking ``RhimeSampler``.

For example, a scalar period broadcasts across the effective requested sites:

.. code-block:: python

   from openghg_inversions.rhime import RhimeConfig

   options = {
       "species": "ch4",
       "sites": ["tac", "MHD"],
       "averaging_period": "1h",
       "domain": "EUROPE",
       "start_date": "2019-01-01",
       "end_date": "2019-01-02",
       "flux_sources": ["total-ukghg-edgar7"],
       "output_name": "inspect-request",
       "output_format": "none",
       "mismatch_model": "fixed_error",
   }
   config = RhimeConfig.from_params(options, multisector=False)
   assert config.data.site_options.sites == ("TAC", "MHD")
   assert config.data.site_options.averaging_period == ("1h", "1h")

The constructor accepts canonical option names and applies established defaults,
validation and site expansion beside the configuration it creates. It does not
translate deprecated spellings or emit their deprecation warnings. Pass an
inspected configuration to the matching runner to execute it without resolving
its options again:

.. code-block:: python

   from openghg_inversions.rhime import run_standard

   result = run_standard(config=config)

For multisector requests, construct with ``multisector=True`` and use
``run_multisector(config=config)``. Supply ``prepared=prepared`` to reuse
canonical inputs with the same resolved configuration, bypassing acquisition and
preparation. An explicit ``data.split_by_sectors`` value must match the selected
recipe; omitted values inherit it. Saved standard and multisector preparations
with a conflicting recorded layout are rejected before numerical inputs are
loaded. A resolved ``config`` cannot be
combined with ``config_file`` or raw configuration keyword arguments. Python
execution inputs such as ``merged_data`` and a custom ``likelihood_builder``
remain separate arguments; a custom likelihood still requires the resolved
request to have ``mismatch_model=None``.

``averaging_period=["1h", "1h"]`` resolves to the same period tuple.
An explicit ``["1h"]`` sequence for these two sites raises ``ValueError``
before acquisition. Optional selectors retain their existing meanings:
``time_resolved=None`` remains unspecified for each site, and supported inlet
slices are preserved. Empty or case-insensitively duplicate site requests and
incorrect selector lengths are rejected at resolution.

``DataConfig.from_params`` resolves acquisition/preparation choices independently.
Grouped mappings use ``data``, ``model``, ``sampler`` and ``output``; the unshipped
``preparation`` group is not an alias. Runners select acquisition arguments from
``config.data`` and pass the acquired handoff with separately selected named
arguments to scientific preparation. Preparation never retrieves data or exports
a basis itself.

INI files and overrides
-----------------------

For an INI file, read its options and resolve the effective request explicitly:

.. code-block:: python

   from openghg_inversions.rhime import RhimeConfig, read_rhime_ini

   options = read_rhime_ini("rhime.ini")
   options.update(sites=["TAC", "MHD"], averaging_period="1h")
   config = RhimeConfig.from_params(options, multisector=False)

``read_rhime_ini`` returns decoded options. It does not require a complete run,
choose a recipe, apply overrides, translate deprecated names or expand shorthand.
The current reader flattens section options into bare names and uses the first
occurrence when a name repeats across sections. Other frontends need not adopt
those INI conventions.

Runners combine file options and winning keyword overrides, extract their own
recipe-specific choices, then resolve the remaining request once. Shorthand
stays available until that last step: changing sites or averaging periods before
resolution expands the final values together. The resulting ``config`` can be
passed to a runner without resolving again.

``RhimeConfig.from_params`` is the single construction entry point; the former
``resolve_rhime_config`` wrapper has been removed. Runner frontends, including
``run_standard`` and staged commands, translate supported deprecated names before
calling this constructor. They emit ``DeprecationWarning`` on devel and in 0.8;
these aliases are removed in 0.9. For names such as ``xprior`` and ``outputname``,
canonical names take precedence. Flux keyword deprecations follow the stricter
rules below. Canonical inputs emit no warning. Configuration construction has no
alias vocabulary; deprecated names passed directly to ``RhimeConfig.from_params``
are ordinary unknown options. Modern configuration and runners do not import or
call HBMCMC.

The deprecated dictionary-returning ``params_from_config`` adapter is available
only from ``openghg_inversions.hbmcmc.compatibility`` during 0.8; its RHIME
re-export is removed. ``normalise=False`` returns decoded values plus winning
overrides. ``normalise=True`` translates aliases and historical resource
selections after overrides and returns canonical keys and resolved ``Path``
values. It does not construct configuration defaults or expand site shorthand.
Modern callers use ``read_rhime_ini``, apply overrides, then call
``RhimeConfig.from_params`` as above. Prefer ``x_prior``, ``output_name`` and
``flux_sources`` when migrating code or files.

Basis artifact defaults and the 0.8 transition are described below; a redesign
of the INI sections remains a separate decision.

Flux selectors and deprecations
-------------------------------

``flux_store`` selects the OpenGHG flux store (default ``"user"``).
``flux_domain`` selects the source flux dataset's domain, defaulting to the
footprint ``domain`` when omitted or ``None``. It does not inherently select
a nested inner domain. Nested preparation uses ``inner_flux_store`` (default:
outer flux store) and ``inner_flux_domain`` (default: inner footprint domain,
independent of the outer flux-domain selector).

At runner frontends, the deprecated spellings ``emissions_store``,
``emissions_domain``,
``inner_emissions_store`` and ``inner_emissions_domain`` remain accepted with
``DeprecationWarning`` until removal in **0.9**. Supply only one spelling of
each selector: old and new names together raise ``ValueError``, including
equal values and explicit ``None``. These modern deprecations are resolved
before configuration construction. ``RhimeConfig.from_params`` accepts only
the ``flux_*`` names, with no translation or deprecation warnings. Resolved
configuration fields and canonical option inventories contain only the new names.

Basis paths, support maps and compatibility
-------------------------------------------

``basis_path`` selects one exact retained ``BasisFunctions`` artifact, with an
arbitrary filename ending in ``.nc`` or ``.zarr``. Omit it to generate a basis
with ``basis_algorithm`` and ``nbasis``. Alternatively,
``basis_partition_path`` selects an exact file containing a labelled ``basis``
partition; this valid scientific representation does not require historical
case/glob selection. For multiple labelled partition files, use the nonempty
``basis_partition_paths`` collection. Files are sorted by path before loading;
the first sorted file supplies the reference coordinates. Select only one of
``basis_path``, ``basis_partition_path`` and ``basis_partition_paths``. Nested
inner preparation has the equivalent mutually exclusive ``inner_basis_path``,
``inner_basis_partition_path`` and ``inner_basis_partition_paths`` options.

``basis_output_path`` is an optional
exact destination, including the filename; it is not an output directory or an
enable flag. Full and staged runners preserve this destination, including paths
outside a preparation checkpoint directory.

A selected retained artifact supplies its operator and metadata to new
preparation. The runner resolves current native prior flux once after filtering
and binds it with ``BasisFunctions.reuse_operator``; the saved flux payload is
not read. Generated-basis weights use only the first requested source (or the
first source mapping entry), while retained flux includes all runtime sources.
These resolved arrays are local scientific inputs, not configuration fields.
``BasisFunctions.load`` has a separate reconstruction contract: it restores the
complete saved basis with its saved prior, without acquisition. See the
:doc:`modern preparation example <customising_rhime>`.

``landsea_path`` and ``outer_regions_path`` select exact scientific support
files. ``RhimeConfig.from_params`` resolves options without opening scientific
data. The application preparation boundary loads selected maps before
acquisition, then passes labelled ``landsea_classes`` and ``outer_regions``
arrays to ``prepare_basis``. This function returns the scientific basis without
writing an export and no longer accepts ``output_path``. Runners explicitly
save it through ``BasisFunctions.save`` when ``basis_output_path`` is requested,
then retain the selected export path in metadata. The application helper
``rhime.preparation.resolve_basis_inputs(config.data)`` loads these scientific inputs.
Historical case/directory selections are accepted only by the temporary HBMCMC
entry points in 0.8, which translate winning selections before calling modern
APIs. Modern runners reject ``fp_basis_case``, ``basis_directory``,
``bc_basis_case`` and ``bc_basis_directory`` before acquisition; resolved requests
contain explicit selections. The numerical functions do no file discovery.
When the selected algorithm needs a packaged support map, the application
resolves omitted support paths to that exact domain's packaged resources.
Explicit paths take precedence; missing files do not fall back to defaults.
Unavailable domain resources raise before acquisition, without substituting
another domain or creating directories. Algorithms that do not need a map
do not require its packaged resource.

To select a packaged resource explicitly from Python:

.. code-block:: python

   from openghg_inversions.basis import packaged_basis_support_path

   options["landsea_path"] = packaged_basis_support_path("landsea", "EUROPE")
   options["outer_regions_path"] = packaged_basis_support_path("outer_regions", "EUROPE")

These resources support basis construction. Country reporting is a separate
operation: supply ``country_file`` for ``basic`` or ``paris`` country reporting.
Standard full runs call
``rhime.validate_standard_output_resources(config.output)`` before acquisition or
inference. This checks country resources and requires ``output_path`` for PARIS
and legacy products. Staged standard postprocessing applies the same check before
loading samples. Direct callers of ``postprocess_standard_outputs`` must perform
this check first. Multisector full and saved runs call
``rhime.validate_multisector_output_resources`` before acquisition/inference or
sample loading; PARIS products require ``output_path``. Nested routes retain
``rhime.validate_output_resources``, which checks country resources without
requiring a written-product destination. Preparation-only workflows do not
require reporting resources. Packaged tutorial fixtures remain usable when
explicitly selected.

For devel/0.8, saving a basis uses the retained modern artifact by default.
For historical callers, ``basis.compatibility.make_basis_functions`` accepts
``basis_output_format="legacy"`` as explicit opt-in. This is an adapter option,
not a canonical ``RhimeConfig`` field.
Legacy basis output is deprecated in 0.9; that deprecation does not schedule
automatic removal in 0.9. This changes basis artifacts, independently of the
final inversion product's ``output_format``.

Historical ``fp_basis_case`` / ``basis_directory`` discovery, directory-style
basis outputs, ``emissions_name`` and legacy artifact
adaptation belong to the compatibility boundary. Deprecated public
``make_basis_functions`` and ``load_basis_functions`` adapters live in
``basis.compatibility`` and invoke canonical modern operations. Their independent
deprecation schedule is unaffected by removing HBMCMC in 0.9. Migrate a
case/directory pair to an explicit artifact and output destination::

   # Before: fp_basis_case="monthly", basis_directory="/data/bases"
   #         basis_output_path="/data/generated" (historical directory)
   options["basis_path"] = "/data/bases/selected-month.nc"
   options["basis_output_path"] = "/data/generated/my-retained-basis.zarr"
   options["landsea_path"] = "/data/maps/classes.nc"

The first requested flux source still determines generated-basis weighting;
all runtime sources are retained. A selected saved basis takes precedence over
generation.

For file-backed boundary conditions, select exact files with
``bc_basis_paths=("/data/bc/january.nc", "/data/bc/february.nc")`` (a one-file
collection is also supported). The runner loads every selected file before
acquisition and passes a labelled ``bc_basis`` dataset to sensitivity
construction, preserving time combination and alignment. Missing or unreadable
selected files fail before acquisition. Scientific sensitivity functions accept
resolved data, not ``bc_basis_case`` or ``bc_basis_directory`` discovery options.

The default NESW calculation uses acquired directional boundary contributions
and needs no basis file. Leave ``bc_basis_paths`` unset for NESW; scientific
calls use ``bc_basis=None``. On the HBMCMC route only, a legacy
``bc_basis_directory`` supplied with NESW warns that it is unused and deprecated;
it triggers neither discovery nor a missing-directory error. ``use_bc=False``
skips BC resource loading.

``country_directory`` is removed from run configuration in 0.8 and is rejected
before acquisition. Replace it with ``country_file`` for reporting,
``landsea_path`` for land/sea classes, or ``outer_regions_path`` for fixed outer
regions. Reporting maps remain explicit: ``domain`` never selects a reporting
``country_file``. See :doc:`legacy_and_migration` for the migration schedule.

Site selectors and retained observations
----------------------------------------

``SiteOptions``, exported from ``openghg_inversions.inversion_data``, also works
without a complete configured run. Its ``from_inputs`` factory expands external
shorthand and normalizes selector labels; direct construction accepts complete
aligned values and checks their structural alignment. For example:

.. code-block:: python

   from openghg_inversions.inversion_data import SiteOptions

   selectors = SiteOptions.from_inputs(
       sites=["tac", "MHD"], averaging_period="1h"
   )
   assert selectors.sites == ("TAC", "MHD")
   assert selectors.averaging_period == ("1h", "1h")

Direct retrieval APIs retain their scalar shorthand without requiring model,
output or sampler settings. The acquisition-and-preparation convenience function
``prepare_rhime_inputs`` has been removed. For a complete inversion, use
``run_standard``; for custom preparation, call ``acquire_merged_data`` (or
``RhimeMergedData.from_options``) followed by ``prepare_observation_errors``,
``filter_observations``, ``openghg_inversions.basis.prepare_basis``,
``build_sensitivities`` and ``assemble_rhime_inputs``. These steps take the
acquired per-site datasets and ``SiteOptions`` and return new mappings; the
acquired ``MergedData`` never carries derived errors or filtered data. The explicit legacy function
``openghg_inversions.hbmcmc.legacy_data.retrieve_inversion_data`` performs
fresh surface or column acquisition and returns its established six-tuple of
merged data and retained metadata lists. The former
``data_processing_surface_notracer`` name is a deprecated wrapper with the same
signature and return. ``RhimeMergedData.from_options`` performs fresh acquisition
from resolved selectors. ``averaging_error`` is a preparation choice: missing ``mf_error`` is derived
from acquired repeatability/variability before temporal filtering.
Existing custom errors remain authoritative. ``averaging_period`` still selects
OpenGHG acquisition/resampling. Acquisition stays in memory; persist the completed
``RhimePreparedInputs`` for reuse. Runners reuse a supplied ``merged_data``
handoff before acquisition, bypassing store access. Before preparation, the merged
owner checks known species, domain, date bounds and source layout against the
request. Conflicts, including changed acquisition windows, raise without
relabeling or selecting observations. Recorded selectors remain authoritative;
unknown historical facts stay unknown. See :doc:`customising_rhime` for
in-memory acquisition and :doc:`rhime` for saving and loading prepared inputs.

The requested configuration contains no acquired/prepared handoff.
Merged and prepared handoffs describe the sites retained by acquisition and
filtering. If only TAC remains from a TAC/MHD request, the configuration
continues to name both sites while the prepared handoff names TAC. Runners bind
retained sites and periods into scientific outputs and pass model and output
choices separately. A supplied compatible merged handoff keeps its own
authoritative site options. The :doc:`prepared-input API <rhime>` remains independent of ``RhimeConfig``.

Forwarding resolved values in custom runners
--------------------------------------------

Given an acquired ``merged`` handoff, copied runners can select explicitly named
values for an ordinary keyword call:

.. code-block:: python

   from openghg_inversions.rhime import filter_observations, prepare_observation_errors

   merged.validate_for_preparation(
       **config.data.select("species", "domain", "start_date", "end_date", "split_by_sectors")
   )
   with_errors = prepare_observation_errors(merged.site_data, averaging_error=config.data.averaging_error)
   filtered, site_options = filter_observations(
       with_errors, merged.site_options, **config.data.select("filters")
   )

``config.data.select(*names)`` returns a fresh dictionary of the named attributes.
Use ``config.model`` and ``config.output`` for their separate owners;
``RhimeConfig`` has no global ``select`` API.
Values are borrowed: mutable mappings and objects are shared with ``config``.
Selection does not normalize, validate, copy nested values or serialize settings;
an unknown attribute raises ``AttributeError``. Scientific stages still accept
named keyword arguments and can be called without configuration. Their former
positional ``data_args`` mappings have been removed. Supply the required basis
identity/source arguments, sensitivity source arguments and resolved BC basis,
and assembly domain/start date by keyword; the API reference lists each function's signature.

Changing and persisting configuration
-------------------------------------

Apply external overrides to the raw options before ``RhimeConfig.from_params``,
after reading the INI file when applicable. ``dataclasses.replace`` is suitable
only for already coherent resolved changes: it does not recompute dependent
defaults or reconcile model/output fields when dates, sources or other shared
choices change.

The configuration record is frozen; its existing mapping and sampler members
remain mutable. Resolution leaves caller options unchanged, and retained-site
selection leaves the requested configuration unchanged. Configuration
serialization and an INI writer are deferred to
`Issue 814 <https://github.com/openghg/openghg_inversions/issues/814>`_.
Standard saved sampling checks scientific compatibility with preparation,
allowing compatible model choices to change without full-configuration equality.
For persisted workflows, see :doc:`staged_workflow`: version 0.8 does not
support staged artifacts produced by 0.7; consume them with 0.7 or regenerate
them with 0.8. Configuration resource and output paths resolve from the invocation
working directory in both full and staged execution, including INI/JSON values
and explicit overrides. Runners forward absolute resource and export paths
downstream; absolute inputs keep their destinations. Manifest dependency references
remain relative to the declaring manifest. Filesystem fields are declared as ``Path`` types,
including optional paths and collections such as ``bc_basis_paths``. Frontends
accept path strings and convert only these annotated fields; domain names and
scientific selectors remain ordinary strings. Each configuration owner supplies
its own field annotations to the independent converter. External-input factories
resolve paths once without loading resources; configured runners and identity
encoding forward canonical values. Direct dataclass construction expects resolved
values. ``RhimeOutputSpec.from_params`` accepts external output options on its
own, without constructing a full run. Explicit ``save_trace`` and
``save_inversion_output`` paths remain supported alongside their Boolean switches.
