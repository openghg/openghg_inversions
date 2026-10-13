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
does not add new scientific keys. Both routes construct the same complete
``RhimeConfig`` through the RHIME resolver, applying overrides before defaults
and site shorthand.  The staged commands deliberately do not
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

All requested choices and the exact stage-contained preparation choices that
were executed are written to ``prepare-manifest.json``.  Its
``configuration_identity`` is a SHA-256 digest of the resolved preparation,
period, model and prior settings.  Sampling and output-only changes do not
invalidate reusable prepared inputs; cache locations and preparation artifact
destinations are likewise excluded. Consequently a gas, period, site, source,
transport or prior change produces a different identity. Downstream scientific commands
require ``--preparation-manifest`` and reject a mismatched configuration
before model construction. Preparation also treats every configured site as
required and fails with the gas and period
named if the existing acquisition layer could not produce it.

.. important::

   Staged artifacts produced by 0.7.x are not guaranteed to be readable by
   0.8.x; use 0.7.x to consume those artifacts. Removing the unused
   ``use_tracer=False`` field from resolved preparation options changes the
   configuration hash even when scientific settings are unchanged.
   Resolved configuration also expands site shorthand and applies defaults;
   staged identity uses those canonical choices rather than reconstructing raw
   input forms to preserve an older hash.

The staged effective-configuration mapping records preparation choices,
``model_spec``, output policy and sampler settings. It contains no requested-site
execution run specification: ``RhimeRunSpec`` is created from retained prepared
metadata. This mapping is owned by staged manifests, not a serialization API
for ``RhimeConfig``. The resolved representation changes staged configuration
identity; historical hash equality is not a compatibility guarantee. Existing
configuration-equality checks still apply.

Stages consume trusted, matched run files and configuration. Stage records do
not write or require content digests for their dependencies. Readers check
record formats, required contents and file references; numerical consumers
check scientific compatibility, including CO2 affine associations.
Keep the original scientific configuration with the run and use current output
settings only to select products and destinations. The configuration-equality
restrictions described above remain until the corresponding workflow changes
provide their replacement context routing.

Commands and artifacts
----------------------

Stage-owned artifacts are written below ``--output-dir`` (or
``openghg-run``'s ``OUTPUT_DIR``). An explicitly configured
``basis_output_path`` keeps its exact destination; preparation does not replace
it with a stage directory. Relative configured paths resolve from the INI or
JSON file as described above. Commands do not depend on the caller's
working directory and reject pre-existing symlinks beneath a stage output
directory.

Preparation resolves selected basis and support resources. Country reporting
resources are checked by ``postprocess`` before creating its output directory;
``prepare`` and ``sample`` do not require them.

``prepare``
  Retrieves merged data, filters observations, constructs basis and
  sensitivities, assembles canonical inputs, and writes ``prepared-inputs.nc``
  and ``prepare-manifest.json``. It never builds a PyMC graph or samples a
  posterior. The NetCDF is a versioned ``RhimePreparedInputs`` artifact and is
  independently inspectable/loadable. Later stages consume this artifact;
  preparation does not write a filtered merged-data snapshot.
  Native runtime flux is resolved once from the filtered source datasets; basis
  design weights use the first requested source. A selected saved basis reuses
  its operator with current runtime flux. The prepared artifact retains that
  bound prior for later reconstruction, which needs no reacquisition.
  For ``co2``, preparation validates and copies the configured
  ``Co2PreparedInputs`` artifact and optional bound affine reconstruction,
  without re-encoding them; it does not acquire observation data. The numerical
  affine binder still checks its association with the prepared artifact.

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
  uses schema version 2 and records the effective sampling configuration and
  file references.
  The binding stores variable roles, supported formats, provenance, and any
  explicit state-dimension mapping. Keep the binding beside its sample
  manifest when moving a run.
  CO2 writes ``posterior.nc`` and a schema-version-1 sample manifest, retaining
  saved trace roles. Affine reconstruction compatibility is checked by its
  existing numerical owner.
  No family silently invokes preparation.

``diagnose``
  Loads ``--posterior``, writes ``posterior-diagnostics.nc`` and emits the
  ``sampler-convergence`` CheckResult.  Scientific failure exits zero by
  default, keeping scheduler status separate from scientific health.
  ``--sample-manifest`` optionally supplies descriptive run context; it does
  not authenticate the supplied posterior. Persisted posteriors can also be
  diagnosed without a manifest. ``--strict`` is available only for an explicitly
  chosen process policy.

``postprocess``
  Standard and multisector workflows load matched prepared inputs, posterior,
  and the saved output binding, then invoke the existing RHIME output
  implementation without constructing a PyMC model. A missing, malformed or
  unsupported binding fails loading; it does not cause a model rebuild.
  Genuine standard/multisector schema-version-1 sample manifests retain the
  older graph-building compatibility route because they did not store output
  bindings. CO2 schema-version-1 replay remains graph-free, using roles in the
  saved trace and any scientifically compatible bound affine artifact.
  CO2 postprocessing rejects other sample-manifest versions before replay.
  New posterior predictive calculations still require a separate explicit
  model-building route. The standard/multisector configuration's
  ``output_format`` controls ``inv_out``, ``basic``, ``paris`` or ``legacy``
  products; explicit save paths in configuration are replaced so every product
  remains beneath the stage output directory. For the same
  reason, staged postprocessing requires safe filename components and both the
  preparation and sample manifests. Their scientific configuration-equality
  checks remain in this workflow. The sampler settings
  retained in result metadata come from that sample manifest; sampler values
  in the postprocessing configuration cannot relabel an existing posterior.
  Replay returns ``result.sampler is None`` and forwards descriptive sampler
  metadata without running inference. Existing sampler settings may be
  constructed to describe recorded values. Missing or unparseable descriptions
  remain unavailable, without blocking products or selecting current defaults.
  Modern derived ``basic`` and PARIS products use every retained posterior
  chain. Derived trace quantities preserve their ``chain`` and ``draw``
  identity until an explicitly requested statistic or covariance combines the
  samples. ``postprocess-manifest.json`` is always written; the otherwise
  in-memory ``basic`` product is written as ``basic.nc``.
  CO2 supports ``none``, ``basic`` and constrained ``paris`` products. Its
  native-flux and country outputs require an exact bound affine reconstruction
  and saved prior/posterior state draws, and retain conditional uncertainty
  scope. The output consumer checks those draws; the current sampler's
  ``sample_prior_predictive`` setting does not gate replay. The complete
  supported CO2 contract is documented in :doc:`co2_model_family`.

CheckResult contract
--------------------

Both checks use the schema version 1 understood by ``openghg-run`` and producer
``openghg_inversions``.  ``--check-output`` chooses the JSON path and
``--check-stage`` chooses the producing ``openghg-run`` stage name.

``prior-predictive-readiness``
  Status is ``pass`` when model construction succeeds, prior and
  prior-predictive variables are produced, and all sampled values are finite.
  It reports ``draws`` and ``non_finite_values`` against
  ``max_non_finite_values = 0``.  Construction, input and sampling exceptions
  become a readable ``fail`` result for standard and multisector recipes. CO2
  handoff, construction and sampling errors reject the command before a
  readiness check is written; non-finite predictive values produce a ``fail``
  check.

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
