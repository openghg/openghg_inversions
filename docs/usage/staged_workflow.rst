Staged RHIME workflows
======================

The standard and multisector recipes can prepare once, sample compatible models
into separate locations, and produce different outputs from saved samples.
``run_standard`` and ``run_multisector``, with or without ``prepared=...``,
use the same scientific operations as their saved phases: prepare inputs, explicitly materialize selected arrays and
enabled priors where needed, build one eager model, sample, construct an
``InversionOutput`` view, then write requested products and exports. Checkpoints
add loading and saving at those boundaries; full execution needs no intermediate
saves. Model construction accepts no output configuration and performs no hidden
materialization.

Standard saved phases
---------------------

``prepare`` returns one absolute path to ``prepare-manifest.json``. It references
prepared numerical inputs and records the ``DataConfig`` that produced them in
its external option form; ``load_preparation`` returns the prepared inputs and
that ``DataConfig`` and its generation identifier. The recorded configuration
is validated exactly as a full-run request is. Retained sites and their averaging
periods come from the
numerical artifact, including when acquisition or filtering drops a site.
Observation errors are established before filtering, then basis construction,
sensitivities and assembly run in scientific order. No filtered checkpoint is
written.

``sample`` takes that manifest and current model and sampler choices, returning
one absolute path to ``sample-manifest.json``. A Python ``likelihood_builder``
(with ``likelihood_kwargs``) may replace the built-in likelihood, as in
``run_standard``. Compatible graph-time priors and
likelihoods reuse the preparation. Required BC and minimum-error arrays must be
available. Omitted model ``use_bc`` inherits the preparation choice; explicit
``False`` may ignore available BC inputs. A repeated ``bc_freq`` must agree with
the baked policy, even when the selected model ignores BCs. Changing acquired
data, filters, basis, native flux or baked numerical policy needs new preparation.

``load_samples`` decodes the sample manifest to ``InversionOutput``.
``postprocess`` uses that reader with current output choices, calls
``postprocess_standard_outputs`` to compute and write requested PARIS or legacy
products, then calls ``save_inversion_output`` for optional exports. It loads
the recorded model meaning, output contract, inference metadata, prepared inputs
and samples without building a graph or running inference. It returns
``InversionOutput`` without a live model or sampler. Missing descriptive metadata stays
unavailable. Output naming, destination and supported formats can change;
scientific choices cannot relabel saved samples. Default sampling retains
scientific ``x``, including when it is a deterministic transform of ``x_latent``.
Posterior ``sample_kwargs["var_names"]`` overrides are rejected by the sampler;
intentional posterior-predictive variable selection remains supported.

Prior parameters in saved samples describe the priors actually constructed.
Numerical scalars are retained inline in model metadata. Array-valued parameters
are registered as model data and saved in the trace's ``constant_data`` group,
with their scientific dimensions and coordinates; metadata references those
arrays rather than embedding numerical arrays in JSON. Disabled components
contribute no prior metadata or references. Modern inversion outputs retain
records captured by shared prior construction in
``model_metadata["constructed_priors"]``, keyed by their concrete model variable
names. Directly constructed PyMC priors are not automatically captured, and
``Interpolated`` persistence remains excluded. Sector associations use the output
contract's variable roles, including separate outer and inner state variables
in nested outputs.

Related lazy prior inputs, including supported zero-dimensional numerical
inputs and auxiliary coordinates, are materialized together with selected model
inputs at the explicit caller materialization boundary before construction. An enabled prior may be computed before sensitivity
evaluation finds that its states have no effect on observations and fixes them
instead of sampling them; such a prior contributes no constructed-prior record.
Disabled BC and offset components are excluded before this computation.
Metadata writing uses the captured scalar values and registered arrays, without
executing the original caller's lazy prior graph again. Scalar-shaped symbolic
parameters retain their graph dependencies. Ordinary and reparameterized
lognormal construction share the same symbolic moment conversion, preserving
the supplied arithmetic or log-space parameterization in metadata.

Ordinary standard sampling routes attach constructed-prior records and callable
provenance to the trace once after sampling. When composing these operations
directly, call the public ``annotate_sample_trace`` with the build
result and posterior before ``save_samples``; pass the generation returned by
``load_preparation`` as ``preparation_generation``. The generic sampler does not
own RHIME scientific metadata. Required array references are checked when saving
and reloading samples;
missing values fail before completing a sample handoff or writing products.

A selected saved basis supplies its operator while new preparation binds current
native flux. Saved preparation and replay retain that flux and do not reacquire
it. Species-dependent mass conversion belongs to postprocessing.

Relative filesystem values inside an INI or JSON parameter file resolve from
the invocation working directory, in both full and staged execution. This includes
resource inputs, ``output_path`` and explicit ``save_trace`` /
``save_inversion_output`` filenames. Explicit ``--kwargs`` path overrides follow
the same rule; absolute paths keep their destination.
Filesystem annotations identify the fields to convert: resolved configuration
uses ``Path`` values, including tuples of paths, while parameter JSON and stage
manifests store paths as strings and path collections as JSON arrays. Scientific
strings such as domain names are unchanged; configuration construction does not
load scientific resources. Each owner converts its declared fields once; later
configured execution and identity encoding forward canonical paths.

For file-backed boundary conditions, specify exact files, for example
``"bc_basis_paths": ["bc/january.nc", "bc/february.nc"]`` in the parameter JSON.
``prepare`` loads all selected files before acquisition and passes their labelled
data to sensitivity construction, preserving time combination and alignment.
Missing or unreadable selected files fail before acquisition. Omit
``bc_basis_paths`` for NESW: it uses acquired directional contributions without
reading a BC basis file. Disabled BCs load no BC basis resources. Staged commands
reject historical ``fp_basis_case``, ``basis_directory``, ``bc_basis_case`` and
``bc_basis_directory`` options. Use explicit modern paths. While retained on devel,
only the HBMCMC entry route translates those historical selectors after winning overrides;
on that route, an NESW directory warns that it is unused without any lookup.


``country_directory`` is removed in 0.8. Use ``country_file`` for reporting,
``landsea_path`` for preparation land/sea classes, or ``outer_regions_path`` for
fixed outer regions. Required preparation maps default to packaged files for the
exact requested domain; explicit paths win and missing resources fail without
fallback. Reporting country files remain explicit and independent of these
preparation defaults. See :doc:`legacy_and_migration` for the migration schedule.

Keep each manifest and its referenced files available for later phases. References
resolve relative to the declaring manifest; absolute shared-cache references are
also accepted. No environment variable locates replacement artifacts. Numerical
files use existing codecs, and manifests use JSON without mandatory content
hashes or full-configuration equality gates. Missing or malformed files fail at
loading; incompatible scientific inputs fail at their owning operation.

The public ``load_preparation`` returns ``(prepared, data_config, generation)``.
External ``save_samples`` calls must supply that exact ``preparation_generation``;
the writer does not silently bind an old posterior to whichever preparation is
currently stored at a path.

Each preparation save assigns a generation identifier. A sample manifest records
that identifier and rejects replay against a replacement preparation at the same
path. This binds checkpoint generations, not arbitrary payload contents; it does
not provide content-integrity or tamper detection.

Standard ``output_format="basic"`` returns the scientific view without computing
an unused summary. For in-memory reporting, call ``basic_output(inv_out,
country_file=...)`` yourself. PARIS and legacy products require ``output_path``;
missing destinations fail before sampling in a full run or before loading samples
in staged postprocessing. ``save_inversion_output`` only exports the scientific
result and trace; it does not compute derived products.

Choose saved artifacts by what you need to reuse:

* Preparation and sample checkpoints preserve phase inputs and execution records.
  A sample checkpoint references its preparation, so keep that dependency with it.
* A self-contained ``InversionOutput`` export carries the prepared scientific
  values, basis and trace needed for later reporting without the checkpoint manifests.
* A trace export carries sampled variables and diagnostics; it does not replace
  the prepared inputs and output contract required for scientific reconstruction.
* An optional basis export preserves the basis for inspection or reuse. It does
  not contain the observations and posterior of an inversion.

Full execution needs no checkpoints, so these optional exports remain useful.
Their defaults and formats are unchanged.

Callers choose checkpoint locations and own collision avoidance and run isolation.
A simple layout is ``run/prepare``, ``run/sample`` and ``run/products``; explicit
locations are supported, including existing directories. Numerical files are saved
first and the manifest is written last as the completion marker. A failed payload save
can leave partial files but no completed manifest. Checkpoint-owned payload and
manifest targets must stay within the chosen destination and must not alias
referenced preparation artifacts. Postprocessing uses the same
product writer and overwrite behaviour as full execution; in-memory products
require no destination. Normal filesystem errors propagate. Concrete payload and manifest targets are
checked for redirects outside their destination or aliases to prepared inputs;
prior-predictive saves use the same target checks. This does not require fresh
directories or recursive directory scanning.
``basis_output_path`` is an exact destination and may be outside the checkpoint
directory. It is resolved at the frontend and forwarded unchanged to the runner's
optional basis export.

Legacy fixedbasis-style PARIS INIs contain unsupported options such as
``nit``, ``nchain``, ``xprior`` and ``mcmc_type``. Use canonical settings before
invoking modern staged commands; no runtime alias translation or HBMCMC wrapper
remains. See :doc:`legacy_and_migration` for explicit setting mappings and
historical execution guidance.

Configuration and CLI
---------------------

Preparation accepts ``--config run.ini`` or ``--params-file choices.json``.
Sampling and postprocessing optionally accept either source; a shared full
configuration contributes only the active phase's choices. JSON can contain
flat options or ``data``, ``model``, ``sampler`` and ``output`` mappings.
Unknown names remain errors, while known inactive values are not resolved.
Explicit ``--kwargs`` overrides for another phase and unsupported names are
reported together in one error. Raw file options and overrides are combined
before canonical validation. Historical spellings are unsupported. Relative input
paths resolve from the invocation working directory. ``CONFIG_FILE`` is not an automatic
source; ``OUTPUT_DIR`` is an optional destination default.

For example, with preparation settings in ``preparation.json``:

.. code-block:: bash

   rhime prepare --params-file preparation.json --output-dir run/prepare

   rhime sample --preparation-manifest run/prepare/prepare-manifest.json \
     --kwargs '{"x_prior":{"pdf":"normal","mu":1,"sigma":0.2}}' \
     --output-dir run/model-a

   rhime sample --preparation-manifest run/prepare/prepare-manifest.json \
     --kwargs '{"x_prior":{"pdf":"normal","mu":1,"sigma":0.5}}' \
     --output-dir run/model-b

   rhime postprocess --sample-manifest run/model-a/sample-manifest.json \
     --kwargs '{"output_format":"inv_out"}' --output-dir run/inversion-output

   rhime postprocess --sample-manifest run/model-a/sample-manifest.json \
     --kwargs '{"output_format":"paris","country_file":"/data/countries.nc"}' \
     --output-dir run/paris

``--model standard`` is the default for these commands. A manifest's family must
match the selected route. CO₂ configured full execution is available
through ``rhime run --model co2``; see :doc:`co2_model_family` for its
prepared-input configuration.

Python callers can use the same handoffs:

.. code-block:: python

   from openghg_inversions.rhime import DataConfig, RhimeOutputSpec
   from openghg_inversions.rhime.stages import prepare, sample, postprocess

   preparation = DataConfig.from_params(preparation_options)
   prepared_manifest = prepare(preparation, output_dir="run/prepare")
   sample_manifest = sample(
       prepared_manifest,
       model={"x_prior": {"pdf": "normal", "mu": 1, "sigma": 0.2}},
       output_dir="run/model-a",
   )
   result = postprocess(
       sample_manifest,
       output=RhimeOutputSpec(output_path="run/products"),
   )

For a runnable synthetic scientific parity example, see
:download:`standard_handoffs.py <../../examples/standard_handoffs.py>`.

Multisector saved phases
------------------------

Use ``openghg_inversions.rhime.multisector_stages`` for multisector ``prepare``,
``sample``, ``load_preparation``, ``load_samples`` and ``postprocess``. These
operations use the same manifest lifetime, generation binding and numerical
codecs as standard handoffs. They call the multisector preparation, materialization,
construction, sampling and product operations directly. Full
``run_multisector`` execution calls those same operations in memory; no
checkpoint is required. Acquisition or supplied-merged validation happens before
``prepare_multisector``; preparation itself neither retrieves data nor writes
exports. Caller-owned xarray and Dask arrays remain borrowed until explicit
materialization or serialization boundaries.

Resolve preparation with ``DataConfig.from_params(options, multisector=True)``.
Keep ``flux_sources`` ordered. At sampling, supply ``sector_sources`` to map
scientific sector names to those source labels. Each sector uses a distinct
prepared source and keeps its own prior block in the full sampled ``x`` state.
The saved roles are ``flux_scale: x``, ``emissions_sensitivity: hx`` and
``flux_contribution: mu``; source labels and recorded sector associations select
blocks for reporting.
Both rectangular shared-basis and source-labelled gathered layouts retain their
scientific interpretation. Preparation retains all source fluxes; the first
requested source supplies basis weighting, independently of sector declaration
order. Retained sites and their averaging
periods come from the prepared numerical inputs, even when a requested site is
dropped. Original requested dates remain the inversion window for later products.

Sampling can change ``sector_priors`` or a compatible Python
``likelihood_builder`` without preparing again. The custom likelihood requires
``mismatch_model=None`` and receives the combined sector observation mean through
the existing likelihood interface. Missing required arrays, incompatible source
layouts, or a conflicting baked ``bc_freq`` require corrected inputs or a new
preparation. Current model choices cannot change acquisition, filtering, basis
construction or the saved native flux.

Saved sampling follows the standard recordable-model limits: explicit
``state_activity``, ``bc_state_activity`` and sector activity overrides are not
encoded in sample records and fail before inference. Full and direct prepared
execution can still use those policies. Whole-model customization uses direct
scientific composition; see :ref:`rhime-complete-model-composition`. Saved
sampling supports the ordinary custom-likelihood seam.

For example, after placing multisector preparation choices in ``sectors.json``:

.. code-block:: bash

   rhime prepare --model multisector --params-file sectors.json \
     --output-dir run/prepare
   rhime sample --model multisector \
     --preparation-manifest run/prepare/prepare-manifest.json \
     --kwargs '{"sector_sources":{"anth":"anth-inventory","wetlands":"wetlands-inventory"},"sector_priors":{"anth":{"pdf":"normal","mu":1,"sigma":0.2},"wetlands":{"pdf":"normal","mu":1,"sigma":0.5}}}' \
     --output-dir run/model-a
   rhime postprocess --model multisector \
     --sample-manifest run/model-a/sample-manifest.json \
     --kwargs '{"output_format":"inv_out"}' --output-dir run/products

The keys in ``sector_sources`` and ``sector_priors`` must agree, and the mapped
sources must match those acquired by ``sectors.json``. Use separate sample
destinations for each model. Sampling accepts model and sampler choices alone;
postprocessing accepts output choices alone, including a current output name,
destination or country file.

``load_samples`` and ``postprocess`` return ``InversionOutput`` with the original
scientific context, retained source flux, output roles and actual inference
metadata. The result holds no live model or sampler; replay builds no PyMC model
and runs no inference. It does not replace unknown original metadata with current
defaults. ``make_model_output`` binds samples to scientific context;
``postprocess_multisector_outputs(result, output_spec)`` computes and writes
requested products, then ``save_inversion_output(result, output_spec)``
exports the scientific result and trace. Full and saved routes call those same
operations. Legacy HBMCMC products remain unsupported for multisector.

Use ``result.trace`` for samples, ``result.variable_roles`` for scientific roles,
and ``result.run_metadata`` / ``result.model_metadata`` for recorded context.
Product choices and written-file receipts are separate from this scientific view.
For in-memory sector diagnostics, call
``postprocessing.make_outputs.make_sector_flux_outputs`` explicitly.
``output_format="none"`` still returns the scientific result. Multisector supports
``none``, ``inv_out`` and ``paris``; ``basic`` and ``legacy`` are single-sector formats.

Follow :doc:`multisector_handoffs` for an executable notebook demonstrating
preparation reuse, separate model samples, multiple output choices and comparison
with full execution. Its :download:`companion script
<../../examples/multisector_handoffs.py>` also checks independent model equations.

Existing current-format built-in samples declaring ``x_<suffix>`` per-sector
state variables remain readable at the output boundary. A saved canonical
``flux_scale`` role is authoritative: missing or incompatible ``x`` cannot fall back to old variables.
This compatibility does not restore historical staged envelopes or add alias
variables to new graphs.

CO₂ handoffs
------------

CO₂ uses its own coherent preparation and conditional affine output
meaning. ``run_co2`` and its saved phases call the same preparation,
materialization, construction, sampling and product operations; full execution
requires no manifests. ``rhime run --model co2`` consumes the configured model,
sampler and output request. Python
``rhime.run(model="co2", config_file="co2.toml")`` selects the same ``run_co2``
entry point and returns the scientific ``Co2Output`` view. This route requires
prepared inputs or caller-supplied canonical inputs with a coherent reduction;
configured acquisition and native-reduction construction are separate work.

``rhime sample --model co2 --preparation-manifest ...`` consumes one preparation
handoff and current inference choices. Compatible BC/offset priors or supported
mismatch choices reuse preparation; changing defining native priors, covariance,
operators or projection requires a coherent new preparation. Requested dates
remain separate from retained observation extrema, including for imported pairs.

``rhime postprocess --model co2 --sample-manifest ...`` loads saved scientific
meaning and provenance without constructing a graph or sampling. It returns a
``Co2Output`` scientific view without a live sampler and supports the same
concentration and conditional native/country products as full execution.
Missing imported request facts must be supplied explicitly. See
:doc:`co2_model_family` for configuration, readiness/diagnosis, affine
compatibility, and
:download:`co2_handoffs.py <../../examples/co2_handoffs.py>` for an executable
full/saved scientific comparison. Linked CO₂/O₂ has a separate joint handoff
interface described below; nested workflows remain separate.

All supported CO2 likelihoods, including fixed OU with inferred amplitudes,
use these handoffs. Replay uses the completed inference description and saved
provenance; it does not select current sampler defaults. See
:doc:`legacy_and_migration` for retired entry points.

Readiness and independent diagnosis
-----------------------------------

``rhime prior-predictive --preparation-manifest ... --output-dir ...`` builds
the selected model and draws prior predictions without posterior inference.
``--draws`` controls its sample count. Loading and serialization errors
propagate; for standard and multisector, supported model/input failures produce
a readiness ``fail`` result. CO₂ construction and inference errors
propagate; evaluated missing, empty or non-finite predictive evidence produces
a failed check.
Select ``--model multisector`` for prior predictions from a multisector preparation
manifest. Python callers use ``multisector_stages.prior_predictive`` with current
model choices and the same likelihood seam as sampling.
Readiness checks finite evidence, not scientific plausibility. See
:doc:`prior_predictive_checking` for interpreting predictions.

.. _staged-convergence-check:

``rhime diagnose --posterior posterior.nc --output-dir ...`` independently
reports maximum R-hat, minimum bulk/tail ESS, divergences and unassessable
variables. ``--sample-manifest`` optionally supplies descriptive run context.
For a current sample manifest, omit ``--posterior`` to use its recorded posterior
reference; an explicit posterior path takes precedence.
Defaults are maximum R-hat 1.01, minimum bulk/tail ESS 400 and zero divergences.
Available metrics exceeding a threshold yield ``fail``; missing metrics yield
``unknown`` when none fails; otherwise the result is ``pass``. Neither check
repairs missing evidence or performs posterior inference. ``--strict`` makes
``fail`` exit nonzero. Both write schema-version-1 check JSON with producer
``openghg_inversions``; ``--check-output`` and ``--check-stage`` select its path
and stage label.

Other families and migration
----------------------------

Linked CO₂/O₂ saved phases use ``--model co2_o2`` and one preparation or sample
manifest, with the complete joint covariance and unequal channel axes retained.
See :ref:`linked-staged-commands` for configuration, the five installed commands,
common-unit restrictions and supported scientific products. The runnable
:download:`co2_o2_handoffs.py <../../examples/co2_o2_handoffs.py>` example uses
synthetic scientific arrays and no private campaign inputs.

The new standard and multisector handoff formats replace their historical staged
envelopes and multi-artifact argument lists. For multisector, replace
``prepare_rhime_stage``, ``sample_rhime_stage``, ``postprocess_rhime_stage`` and
stage-setup/configuration-identity calls with ``multisector_stages.prepare``,
``sample`` and ``postprocess``. CLI sampling takes only
``--preparation-manifest``; postprocessing takes only ``--sample-manifest``.
Remove multisector ``--prepared-inputs`` and ``--posterior`` phase arguments.
Regenerate handoffs with 0.8, or use the original package version to consume them;
there is no historical-envelope adapter. Existing numerical codecs and final
product formats retain their contracts. See :doc:`legacy_and_migration` for the
multisector prepared-input entry point and retained custom-builder boundary.
