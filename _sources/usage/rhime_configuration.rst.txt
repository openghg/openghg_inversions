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

* Acquisition and preparation choices are direct fields, including
  ``species``, ``domain``, dates, stores, basis, filtering and error settings.
  ``site_options`` is one public ``SiteOptions`` record containing complete
  aligned tuples of requested sites and their selectors.
* ``model`` is the existing ``RhimeModelSpec`` containing sectors, priors,
  likelihood and other scientific model choices.
* ``output`` is the existing ``RhimeOutputSpec`` describing final products,
  destinations, naming and save choices.
* ``sampler`` is the existing ``RhimeSampler`` with resolved sampling settings.
  It executes inference only when supplied a completed model. Its implementation
  lives in ``inference.sampling``; the established RHIME imports remain valid.

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
   assert config.site_options.sites == ("TAC", "MHD")
   assert config.site_options.averaging_period == ("1h", "1h")

The constructor translates supported deprecated aliases, then applies established
defaults, validation and site expansion beside the configuration it creates. Pass an
inspected configuration to the matching runner to execute it without resolving
its options again:

.. code-block:: python

   from openghg_inversions.rhime import run_rhime

   result = run_rhime(config=config)

For multisector requests, construct with ``multisector=True`` and use
``run_rhime_multisector(config=config)``. A resolved ``config`` cannot be
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
``resolve_rhime_config`` wrapper has been removed. The constructor calls the
translator in ``hbmcmc.compatibility`` and emits ``DeprecationWarning`` when a
deprecated name or output-format value is replaced or removed. For historical HBMCMC aliases, canonical names take precedence. Modern flux
keyword deprecations follow the stricter rules below. Canonical inputs emit no
deprecation warning. The deprecated
``params_from_config`` adapter retains its dictionary return, overrides and
``normalise`` control there; modern readers and runners do not depend on that
adapter. Prefer modern names such as ``x_prior``, ``output_name`` and
``flux_sources`` when migrating code or files.

Existing option defaults remain unchanged by this separation. Scientific default
policy and a redesign of the INI sections are separate decisions.

Flux selectors and deprecations
-------------------------------

``flux_store`` selects the OpenGHG flux store (default ``"user"``).
``flux_domain`` selects the source flux dataset's domain, defaulting to the
footprint ``domain`` when omitted or ``None``. It does not inherently select
a nested inner domain. Nested preparation uses ``inner_flux_store`` (default:
outer flux store) and ``inner_flux_domain`` (default: inner footprint domain,
independent of the outer flux-domain selector).

The modern spellings ``emissions_store``, ``emissions_domain``,
``inner_emissions_store`` and ``inner_emissions_domain`` remain accepted with
``DeprecationWarning`` until removal in **0.9**. Supply only one spelling of
each selector: old and new names together raise ``ValueError``, including
equal values and explicit ``None``. These modern deprecations are resolved
separately from historical HBMCMC aliases. Resolved configuration fields and
canonical option inventories contain only the new names.

``basis.make_basis_functions`` now accepts ``flux_sources`` for source
weighting. Its shipped ``emissions_name`` keyword follows the same warning,
conflict and 0.9 removal policy. The first requested source still determines
basis weights, all runtime sources remain available, and saved bases retain
precedence over generation. Newly introduced dataset-native basis helpers use
``flux_sources`` directly. Historical HBMCMC basis and retrieval APIs retain
their compatibility signatures.

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
``run_rhime``; for custom preparation, call ``RhimeMergedData.from_options`` followed by
``prepare_observation_errors``, ``filter_observations``,
``openghg_inversions.basis.make_basis_functions``,
``build_sensitivities`` and ``assemble_rhime_inputs``. The explicit legacy function
``openghg_inversions.hbmcmc.legacy_data.retrieve_inversion_data`` performs
fresh surface or column acquisition and returns its established six-tuple of
merged data and retained metadata lists. The former
``data_processing_surface_notracer`` name is a deprecated wrapper with the same
signature and return. ``RhimeMergedData.from_options`` performs fresh acquisition
from resolved selectors. ``averaging_error`` is a preparation choice: missing ``mf_error`` is derived
from acquired repeatability/variability before temporal filtering or aggregation.
Existing custom errors remain authoritative. ``averaging_period`` still selects
OpenGHG acquisition/resampling. Acquisition stays in memory; persist the completed
``RhimePreparedInputs`` for reuse. Runners reuse a supplied ``merged_data``
handoff before acquisition, bypassing store access. Before preparation, the merged
owner checks known species, domain, date bounds and source layout against the
request. Conflicts, including changed acquisition windows, raise without
relabeling or selecting observations. Recorded selectors remain authoritative;
unknown historical facts stay unknown. See :doc:`customising_rhime` for
in-memory acquisition and :doc:`rhime` for saving and loading prepared inputs.

The requested configuration contains no acquired/prepared handoff or
``RhimeRunSpec``.
Merged and prepared handoffs describe the sites retained by acquisition and
filtering. Ordinary runners create the execution ``RhimeRunSpec`` after
preparation: it combines those retained sites and periods with the requested
date bounds and resolved model/output choices. If only TAC remains from a
TAC/MHD request, the configuration continues to name both sites while the run
specification names TAC. A supplied compatible merged handoff keeps its own
authoritative site options. The :doc:`prepared-input API <rhime>` remains independent of ``RhimeConfig``.

Forwarding resolved values in custom runners
--------------------------------------------

Given an acquired ``merged`` handoff, copied runners can select explicitly named
values for an ordinary keyword call:

.. code-block:: python

   from openghg_inversions.rhime import filter_observations, prepare_observation_errors

   merged.validate_for_preparation(
       **config.select("species", "domain", "start_date", "end_date", "split_by_sectors")
   )
   merged = prepare_observation_errors(merged, averaging_error=config.averaging_error)
   filtered = filter_observations(merged, **config.select("filters"))

``config.select(*names)`` returns a fresh dictionary of the named attributes.
Values are borrowed: mutable mappings and objects are shared with ``config``.
Selection does not normalize, validate, copy nested values or serialize settings;
an unknown attribute raises ``AttributeError``. Scientific stages still accept
named keyword arguments and can be called without configuration. Their former
positional ``data_args`` mappings have been removed. Supply the required basis
identity/source arguments, sensitivity domain/source arguments, and assembly
domain/start date by keyword; the API reference lists each function's signature.

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
Equal resolved scalar/list choices do not guarantee equal historical
configuration hashes.
For persisted workflows, see :doc:`staged_workflow`: version 0.8 does not
support staged artifacts produced by 0.7; consume them with 0.7 or regenerate
them with 0.8. Ordinary paths remain relative to the working directory and
staged paths retain their configuration-file-relative convention.
