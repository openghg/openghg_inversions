Staged RHIME workflows
======================

``openghg-inversions`` exposes file-backed stages for schedulers such as
``openghg-run``. The interface composes the same configuration resolver,
preparation functions, model recipes, sampler and output functions used by
the corresponding Python runners. It is not a second scientific configuration
system, and OpenGHG Inversions does not depend on ``openghg-run``.

The supported recipes are ``standard``, ``multisector`` and ``co2``. Recipe
selection is always explicit with ``--model``.  It is not inferred from
``species``, ``fp_model`` or ``met_model``: CH4 and CO2 need not use the same
model, while several F-gases may use the same standard recipe.  The distinct
CO2 recipe accepts an existing coherent prepared-input artifact and supports
ordinary and cached fixed-OU execution. Linked CO2/O2 staging remains separate
follow-up work. See :doc:`co2_model_family` for CO2 prerequisites, commands and
output capabilities.

Configuration inputs
--------------------

For ``--model standard`` and ``--model multisector``, every scientific stage
except ``diagnose`` accepts exactly one of:

* ``--config /absolute/path/run.ini``, using the existing RHIME INI vocabulary;
* ``--params-file /absolute/path/ogi-params.json``, containing one JSON object
  with the same keyword names accepted by ``run_rhime``.

``--kwargs '{...}'`` can explicitly override either source. The JSON form
supports ``openghg-run`` cases that do not declare an INI ``CONFIG_FILE``; it
does not add new scientific keys. The existing ``resolve_rhime_options`` boundary still
normalizes and validates every value.  The staged commands deliberately do not
read ambient ``CONFIG_FILE``.  ``OUTPUT_DIR`` is the only automatic path
default, and the effective ``openghg-run`` ``STAGE`` is the default check-stage
label.
Relative filesystem values inside an INI or JSON parameter file are resolved
against that file's directory, not the process working directory.

``openghg-run``'s ``EFFECTIVE_CONFIG`` is a stage-specific TOML envelope, not
an OpenGHG Inversions scientific parameter file, so do not pass it to
``--params-file``. It remains orchestration provenance. A no-INI campaign
should explicitly copy or provide a canonical OpenGHG Inversions JSON
parameter file and pass that path (for example beneath ``SOURCE_DIR`` or
``RUN_ROOT/scripts``).

Legacy fixedbasis-style PARIS INIs contain options such as ``nit``, ``nchain``,
``xprior`` and ``mcmc_type``.  Their translation remains owned by the existing
``run_hbmcmc`` compatibility path.  Do not copy those aliases into a staged
JSON file; use canonical RHIME names or first migrate the configuration through
that existing compatibility boundary.

For ``--model co2``, pass ``--config co2.toml`` using the CO2-family TOML
vocabulary, or a ``--params-file`` JSON object with the same nested structure.
The CO2 resolver validates the recipe; the optional ``[outputs]`` table selects
staged products. Relative artifact paths resolve from the configuration file.
CO2 stage manifests require an identifiable installed Git revision from a VCS
installation or source checkout; see :doc:`co2_model_family`.

Configuration flow
------------------

For standard and multisector recipes, the existing RHIME configuration supports
gas-, site-, transport-, model-, year-, month- and prior-specific choices
directly:

* ``species``, ``start_date`` and ``end_date`` select gas and period;
* ``sites`` and the aligned ``averaging_period``, ``inlet``, ``instrument``,
  ``fp_height``, ``obs_data_level``, ``met_model``, ``platform`` and
  ``max_level`` values select site-specific observations and footprints;
* ``fp_model`` selects footprint transport, independently of ``--model``;
* ``flux_sources`` selects OpenGHG prior flux identities.  Multisector runs may
  additionally use ``sector_sources`` and ``sector_priors``;
* ``x_prior``, ``bc_prior``, ``sigma_prior`` and the selected mismatch model
  define mathematical priors and likelihood policy.

Sampling keywords may configure ``sample_kwargs.idata_kwargs`` options such as
``log_likelihood``, but cannot supply ``coords`` or ``dims``. Define labelled
coordinates through ``registered_model()`` and its ``CoordRegistry``; see
:doc:`concrete_rhime_model` for custom-builder guidance. Other supported sampler
keywords remain available.

All requested choices and the exact stage-contained preparation choices that
were executed are written to ``prepare-manifest.json``.  Its
``configuration_identity`` is a SHA-256 digest of the resolved preparation,
period, model and prior settings.  Sampling and output-only changes do not
invalidate reusable prepared inputs; cache locations and preparation artifact
destinations are likewise excluded because the artifact digest identifies the
resulting input content.  Consequently a gas, period, site, source, transport
or prior change produces a different identity.  Downstream scientific commands
require ``--preparation-manifest`` and reject a mismatched configuration or
prepared-input content digest before model construction. Preparation retains
usable requested sites after acquisition, compatible cache reload, and filtering,
and aligns all per-site choices to that retained subset. An empty retained set
fails before basis construction or inference. Requested choices remain in the
manifest for provenance; sampling uses the prepared handoff's retained labels.

The manifest also records a content SHA-256 for ``prepared-inputs.nc``.
``openghg-run``
independently verifies the declared stage-output directory and records its own
filesystem identity; the OpenGHG Inversions digest gives scientific consumers
a compact, direct identity for the prepared handoff.

Commands and artifacts
----------------------

Stage checkpoints, reports and products are written below ``--output-dir``
(or ``openghg-run``'s ``OUTPUT_DIR``). Commands do not depend on the caller's
working directory and reject pre-existing symlinks beneath a stage output
directory. The separately configured optional pre-filter acquisition cache
retains its existing ``merged_data_dir`` location, which may be shared with
full Python runs outside the stage destination.

``prepare``
  Retrieves/reloads merged data, filters observations, constructs basis and
  sensitivities, assembles canonical inputs, and writes ``prepared-inputs.nc``
  and ``prepare-manifest.json``. It never builds a PyMC graph or samples a
  posterior. The optional ``save_merged_data`` cache stores acquisition output
  before filtering, using the same naming, formats, validation and reload
  fallback as full Python execution. No filtered merged snapshot is written
  or required by later stages. The prepared NetCDF is a versioned
  ``RhimePreparedInputs`` artifact and is independently inspectable/loadable.
  For ``co2``, preparation validates and copies the configured
  ``Co2PreparedInputs`` artifact and optional bound affine reconstruction,
  preserving their content identities; it does not acquire observation data.

``prior-predictive``
  Loads ``--prepared-inputs``, validates its layout against the explicit
  configuration, builds the selected model, draws from its prior predictive,
  and writes ``prior-predictive.nc`` plus a readiness CheckResult.  It never
  runs posterior sampling.  ``--strict`` converts a readiness ``fail`` to a
  nonzero process exit when a scheduler policy wants that behavior. The
  readiness result checks construction and finite values; it does not assess
  scientific plausibility. Follow :doc:`prior_predictive_checking` to inspect
  the artifact and then sample the posterior from the same prepared data.

``sample``
  Standard and multisector workflows load the prepared artifact, build the
  selected model, sample it with the resolved ``RhimeSampler``, and write
  ``posterior.nc``, ``output-binding.json``, and ``sample-manifest.json``. The sample manifest
  uses schema version 3 and records the effective sampling configuration and
  content identities for the posterior, prepared input, and output binding.
  The binding stores variable roles, supported formats, provenance, and any
  explicit state-dimension mapping, together with the two numerical artifact
  identities. Keep the binding beside its sample manifest when moving a run.
  CO2 writes ``posterior.nc`` and a schema-version-3 sample manifest, retaining
  saved trace roles and any authenticated affine-reconstruction identity.
  No family silently invokes preparation.

``diagnose``
  Loads ``--posterior``, writes ``posterior-diagnostics.nc`` and emits the
  ``sampler-convergence`` CheckResult.  Scientific failure exits zero by
  default, keeping scheduler status separate from scientific health.
  ``--sample-manifest`` authenticates a posterior produced by the staged
  sampler. It is optional so persisted posteriors from earlier OpenGHG
  Inversions runs can
  still be diagnosed.  ``--strict`` is available only for an explicitly
  chosen process policy.

``postprocess``
  Standard and multisector workflows load matched prepared inputs, posterior,
  and the saved output binding, then invoke the existing RHIME output
  implementation without constructing a PyMC model. A missing, altered, or
  mismatched binding fails validation; it does not cause a model rebuild.
  Each recipe explicitly validates its supported manifest and scientific
  identity versions before posterior loading or product writes. CO2 replay
  uses roles in the authenticated saved trace and any separately authenticated
  affine artifact. Unsupported or retired contracts fail without graph recovery.
  New posterior predictive calculations still require a separate explicit
  model-building route. The standard/multisector configuration's
  ``output_format`` controls ``inv_out``, ``basic``, ``paris`` or ``legacy``
  products; explicit save paths in configuration are replaced so every product
  remains beneath the stage output directory. For the same
  reason, staged postprocessing requires safe filename components and both the
  preparation and sample manifests.  The latter binds the posterior to its
  prepared-input digest and scientific configuration.  The sampler settings
  restored into result metadata come from that sample manifest; sampler values
  in the postprocessing configuration cannot relabel an existing posterior.
  Modern derived ``basic`` and PARIS products use every retained posterior
  chain. Derived trace quantities preserve their ``chain`` and ``draw``
  identity until an explicitly requested statistic or covariance combines the
  samples. ``postprocess-manifest.json`` is always written; the otherwise
  in-memory ``basic`` product is written as ``basic.nc``.
  CO2 supports ``none``, ``basic`` and constrained ``paris`` products. Its
  native-flux and country outputs require an exact bound affine reconstruction
  and prior draws, and retain conditional uncertainty scope. The complete
  supported CO2 contract is documented in :doc:`co2_model_family`.

Staged metadata compatibility
-----------------------------

The next minor release establishes a breaking boundary for staged setup APIs,
workflow manifest envelopes and scientific identities. Standard, multisector and
CO2 preparation, prior-predictive, sampling and postprocessing workflows currently
write and accept manifest schema version 3, scientific identity version 1, and an
explicit recipe label. Each family owns its supported-version policy;
a shared envelope loader recognizing a version does not authorize its use by
another family. Standard/multisector saved-output bindings remain required;
CO2 affine companions are authenticated independently.

Pre-refactor schema versions 1 and 2 are retired for these workflow handoffs.
Rerun preparation and sampling to create the supported staged contract; no
artifact migration or historical graph recovery is provided. The filtered ``merged-data/merged-data.nc``
checkpoint and its ``merged_data`` manifest entries are also removed. Do not
reuse that historical snapshot as a pre-filter acquisition cache.

Independent diagnosis still accepts sample envelopes using schema versions 1,
2 or 3 and authenticates the posterior content digest. It requires no family
configuration, output binding or affine companion, and does not authorize
scientific replay. Diagnostic CheckResults remain schema version 1; CO2 diagnosis
also retains its schema-version-1 diagnostic manifest.

This boundary preserves standalone numerical prepared-input and posterior
formats, scientific Python runner/builder interfaces, product names and schemas,
and the optional pre-filter acquisition cache. Full Python runs continue to
sequence scientific operations in memory with intermediate saves disabled.

CheckResult contract
--------------------

Both checks use the schema version 1 understood by ``openghg-run`` and producer
``openghg_inversions``.  ``--check-output`` chooses the JSON path and
``--check-stage`` chooses the producing ``openghg-run`` stage name.

``prior-predictive-readiness``
  Status is ``pass`` when model construction succeeds, prior and
  prior-predictive variables are produced, and all sampled values are finite.
  It reports ``draws`` and ``non_finite_values`` against
  ``max_non_finite_values = 0``. For standard and multisector recipes,
  construction or prediction ``KeyError``/``ValueError`` within the readiness
  catch produce a readable ``fail`` without predictive artifacts. External
  loading/authentication and artifact serialization errors propagate. CO2
  construction and prediction errors propagate before destination creation.
  Returned empty or non-finite evidence produces a ``fail`` check while retaining
  the existing predictive/report writes, including CO2's prior manifest.

.. _staged-convergence-check:

``sampler-convergence``
  Reports retained chain and draw counts, maximum R-hat and its variable,
  minimum bulk ESS and its variable, minimum tail ESS and its variable, total
  divergences, divergences per chain, and labels for unassessable R-hat/ESS
  elements.  Defaults are maximum R-hat 1.01,
  minimum bulk ESS 400, minimum tail ESS 400, and maximum divergences 0.
  Thresholds have CLI options.  The result is ``unknown`` when a signal is not
  assessable (for example R-hat from one chain), ``fail`` when an available
  signal exceeds policy, and ``pass`` otherwise.  This is the machine-visible
  convergence outcome requested by issues #656/#667; ``openghg-run`` does not
  import ArviZ or PyMC.

Minimal command sequence
------------------------

The following is a complete run using a canonical JSON parameter file. An INI
can be substituted with ``--config``. Set ``OGI_PARAMS_FILE`` to the path of
that OpenGHG Inversions JSON parameter file before running the commands.

.. code-block:: bash

   openghg-inversions prepare \
     --params-file "$OGI_PARAMS_FILE" --model standard \
     --output-dir "$OUTPUT_DIR/prepare"

   openghg-inversions prior-predictive \
     --params-file "$OGI_PARAMS_FILE" --model standard \
     --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" \
     --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" \
     --output-dir "$OUTPUT_DIR/prior-predictive" \
     --check-output "$OUTPUT_DIR/prior-predictive/prior-ready.json"

   openghg-run record-check "$RUN_ROOT" \
     "$OUTPUT_DIR/prior-predictive/prior-ready.json" --strict

   openghg-inversions sample \
     --params-file "$OGI_PARAMS_FILE" --model standard \
     --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" \
     --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" \
     --output-dir "$OUTPUT_DIR/sample"

   openghg-inversions diagnose \
     --posterior "$OUTPUT_DIR/sample/posterior.nc" \
     --sample-manifest "$OUTPUT_DIR/sample/sample-manifest.json" \
     --output-dir "$OUTPUT_DIR/diagnose" \
     --check-output "$OUTPUT_DIR/diagnose/convergence.json"

   openghg-run record-check "$RUN_ROOT" \
     "$OUTPUT_DIR/diagnose/convergence.json"

   openghg-inversions postprocess \
     --params-file "$OGI_PARAMS_FILE" --model standard \
     --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" \
     --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" \
     --posterior "$OUTPUT_DIR/sample/posterior.nc" \
     --sample-manifest "$OUTPUT_DIR/sample/sample-manifest.json" \
     --output-dir "$OUTPUT_DIR/postprocess"

``openghg-run`` campaign stages
-------------------------------

A campaign can place the same commands directly in named stages and declare
their output directories.  No hand-authored SLURM script is required:

.. code-block:: toml

   [[stage]]
   name = "prepare"
   command = '''openghg-inversions prepare --config "$CONFIG_FILE" --model standard --output-dir "$OUTPUT_DIR/prepare"'''
   outputs = ["outputs/prepare"]

   [[stage]]
   name = "prior-predictive"
   depends_on = ["prepare"]
   command = '''openghg-inversions prior-predictive --config "$CONFIG_FILE" --model standard --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" --output-dir "$OUTPUT_DIR/prior-predictive" --check-output "$OUTPUT_DIR/prior-predictive/prior-ready.json" && openghg-run record-check "$RUN_ROOT" "$OUTPUT_DIR/prior-predictive/prior-ready.json"'''
   outputs = ["outputs/prior-predictive"]

   [[stage]]
   name = "sample"
   depends_on = ["prior-predictive"]
   required_checks = ["prior-predictive-readiness"]
   command = '''openghg-inversions sample --config "$CONFIG_FILE" --model standard --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" --output-dir "$OUTPUT_DIR/sample"'''
   outputs = ["outputs/sample"]

   [[stage]]
   name = "diagnose"
   depends_on = ["sample"]
   command = '''openghg-inversions diagnose --posterior "$OUTPUT_DIR/sample/posterior.nc" --sample-manifest "$OUTPUT_DIR/sample/sample-manifest.json" --output-dir "$OUTPUT_DIR/diagnose" --check-output "$OUTPUT_DIR/diagnose/convergence.json" && openghg-run record-check "$RUN_ROOT" "$OUTPUT_DIR/diagnose/convergence.json"'''
   outputs = ["outputs/diagnose"]

   [[stage]]
   name = "postprocess"
   depends_on = ["prepare", "sample", "diagnose"]
   command = '''openghg-inversions postprocess --config "$CONFIG_FILE" --model standard --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" --posterior "$OUTPUT_DIR/sample/posterior.nc" --sample-manifest "$OUTPUT_DIR/sample/sample-manifest.json" --output-dir "$OUTPUT_DIR/postprocess"'''
   outputs = ["outputs/postprocess"]

``openghg-run`` owns campaign matrices, selected tasks, dependencies, scheduler
state, declared-output hashing, continuation manifests, check recording and
gates. OpenGHG Inversions owns data preparation, model selection, sampling,
scientific products, check calculations and their threshold policy.
``STAGE_INPUTS_MANIFEST`` may be used
by application code to discover continued parent outputs, but it is not needed
for full-graph submission and is not implicitly interpreted by OpenGHG
Inversions. The
predictable explicit paths above work in both modes.
