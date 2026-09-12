Staged RHIME workflows
======================

``openghg-inversions`` exposes file-backed stages for schedulers such as
``openghg-run`` (OGR).  The interface composes the same configuration resolver,
preparation functions, model recipes, sampler and output functions used by
``run_rhime``.  It is not a second scientific configuration system and OGI does
not depend on OGR.

The first supported recipes are ``standard`` and ``multisector``.  Recipe
selection is always explicit with ``--model``.  It is not inferred from
``species``, ``fp_model`` or ``met_model``: CH4 and CO2 need not use the same
model, while several F-gases may use the same standard recipe.  The distinct
CO2 and CO2/O2 prepared-input recipes are not yet routed through these commands.

Configuration inputs
--------------------

Every scientific stage except ``diagnose`` accepts exactly one of:

* ``--config /absolute/path/run.ini``, using the existing RHIME INI vocabulary;
* ``--params-file /absolute/path/ogi-params.json``, containing one JSON object
  with the same keyword names accepted by ``run_rhime``.

``--kwargs '{...}'`` can explicitly override either source.  The JSON form
supports OGR cases that do not declare an INI ``CONFIG_FILE``; it does not add
new scientific keys.  The existing ``resolve_rhime_options`` boundary still
normalizes and validates every value.  The staged commands deliberately do not
read ambient ``CONFIG_FILE``.  ``OUTPUT_DIR`` is the only automatic path
default, and the effective OGR ``STAGE`` is the default check-stage label.

OGR's ``EFFECTIVE_CONFIG`` is a stage-specific OGR TOML envelope, not an OGI
scientific parameter file, so do not pass it to ``--params-file``.  It remains
orchestration provenance.  A no-INI campaign should explicitly copy or provide
a canonical OGI JSON parameter file and pass that path (for example beneath
``SOURCE_DIR`` or ``RUN_ROOT/scripts``).

Legacy fixedbasis-style PARIS INIs contain options such as ``nit``, ``nchain``,
``xprior`` and ``mcmc_type``.  Their translation remains owned by the existing
``run_hbmcmc`` compatibility path.  Do not copy those aliases into a staged
JSON file; use canonical RHIME names or first migrate the configuration through
that existing compatibility boundary.

Configuration flow
------------------

The existing RHIME configuration supports gas-, site-, transport-, model-,
year-, month- and prior-specific choices directly:

* ``species``, ``start_date`` and ``end_date`` select gas and period;
* ``sites`` and the aligned ``averaging_period``, ``inlet``, ``instrument``,
  ``fp_height``, ``obs_data_level``, ``met_model``, ``platform`` and
  ``max_level`` values select site-specific observations and footprints;
* ``fp_model`` selects footprint transport, independently of ``--model``;
* ``flux_sources`` selects OpenGHG prior flux identities.  Multisector runs may
  additionally use ``sector_sources`` and ``sector_priors``;
* ``x_prior``, ``bc_prior``, ``sigma_prior`` and the selected mismatch model
  define mathematical priors and likelihood policy.

All resolved choices are written to ``prepare-manifest.json``.  Its
``configuration_identity`` is a SHA-256 digest of the resolved preparation,
period, model and prior settings.  Sampling and output-only changes do not
invalidate reusable prepared inputs.  Consequently a gas, period, site, source,
transport or prior change produces a different identity.  Downstream commands
can pass ``--preparation-manifest`` to reject a mismatched configuration or
prepared-input content digest before model construction.  Preparation also
treats every configured site as required and fails with the gas and period
named if the existing acquisition layer could not produce it.

The manifest also records a content SHA-256 for ``prepared-inputs.nc``.  OGR
independently verifies the declared stage-output directory and records its own
filesystem identity; the OGI digest gives scientific consumers a compact,
direct identity for the prepared handoff.

Commands and artifacts
----------------------

All paths are explicit and outputs are written below ``--output-dir`` (or OGR's
``OUTPUT_DIR``).  Commands do not depend on the caller's working directory.

``prepare``
  Retrieves/reloads merged data, filters observations, constructs basis and
  sensitivities, assembles canonical inputs, and writes an inspectable
  ``merged-data/merged-data.nc``, ``prepared-inputs.nc`` and
  ``prepare-manifest.json``.  It never builds a
  PyMC graph or samples a posterior.  The NetCDF is a versioned
  ``RhimePreparedInputs`` artifact and is independently inspectable/loadable.

``prior-predictive``
  Loads ``--prepared-inputs``, validates its layout against the explicit
  configuration, builds the selected model, draws from its prior predictive,
  and writes ``prior-predictive.nc`` plus a readiness CheckResult.  It never
  runs posterior sampling.  ``--strict`` converts a readiness ``fail`` to a
  nonzero process exit when a scheduler policy wants that behavior.

``sample``
  Loads the prepared artifact, builds the selected model, samples it with the
  resolved ``RhimeSampler``, and writes ``posterior.nc`` plus
  ``sample-manifest.json``.  It never silently invokes preparation.

``diagnose``
  Loads ``--posterior``, writes ``posterior-diagnostics.nc`` and emits the
  ``sampler-convergence`` CheckResult.  Scientific failure exits zero by
  default, keeping scheduler status separate from scientific health.
  ``--strict`` is available only for an explicitly chosen process policy.

``postprocess``
  Loads both prepared inputs and posterior, reconstructs the selected model's
  output contract, and invokes the existing RHIME output implementation.  The
  configuration's ``output_format`` controls ``inv_out``, ``basic``, ``paris``
  or ``legacy`` products; explicit save paths in configuration are replaced so
  every product remains beneath the stage output directory.  For the same
  reason, staged postprocessing requires ``output_name`` to be a filename stem,
  not an absolute path or a name containing directories.  Current all-chain
  limitations of derived basic and
  PARIS products remain tracked separately; this stage does not change their
  scientific calculation.  ``postprocess-manifest.json`` is always written;
  the otherwise in-memory ``basic`` product is written as ``basic.nc``.

CheckResult contract
--------------------

Both checks use OGR-compatible schema version 1 and producer
``openghg_inversions``.  ``--check-output`` chooses the JSON path and
``--check-stage`` chooses the producing OGR stage name.

``prior-predictive-readiness``
  Status is ``pass`` when model construction succeeds, prior and
  prior-predictive variables are produced, and all sampled values are finite.
  It reports ``draws`` and ``non_finite_values`` against
  ``max_non_finite_values = 0``.  Construction, input and sampling exceptions
  become a readable ``fail`` result.

``sampler-convergence``
  Reports retained chain and draw counts, maximum R-hat and its variable,
  minimum bulk ESS and its variable, minimum tail ESS and its variable, total
  divergences, and divergences per chain.  Defaults are maximum R-hat 1.01,
  minimum bulk ESS 400, minimum tail ESS 400, and maximum divergences 0.
  Thresholds have CLI options.  The result is ``unknown`` when a signal is not
  assessable (for example R-hat from one chain), ``fail`` when an available
  signal exceeds policy, and ``pass`` otherwise.  This is the machine-visible
  convergence outcome requested by issues #656/#667; OGR does not import
  ArviZ or PyMC.

Minimal command sequence
------------------------

The following is a complete run using a canonical JSON parameter file.  An INI
can be substituted with ``--config``.

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
     --output-dir "$OUTPUT_DIR/diagnose" \
     --check-output "$OUTPUT_DIR/diagnose/convergence.json"

   openghg-run record-check "$RUN_ROOT" \
     "$OUTPUT_DIR/diagnose/convergence.json"

   openghg-inversions postprocess \
     --params-file "$OGI_PARAMS_FILE" --model standard \
     --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" \
     --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" \
     --posterior "$OUTPUT_DIR/sample/posterior.nc" \
     --output-dir "$OUTPUT_DIR/postprocess"

OGR campaign stages
-------------------

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
   command = '''openghg-inversions diagnose --posterior "$OUTPUT_DIR/sample/posterior.nc" --output-dir "$OUTPUT_DIR/diagnose" --check-output "$OUTPUT_DIR/diagnose/convergence.json" && openghg-run record-check "$RUN_ROOT" "$OUTPUT_DIR/diagnose/convergence.json"'''
   outputs = ["outputs/diagnose"]

   [[stage]]
   name = "postprocess"
   depends_on = ["diagnose"]
   command = '''openghg-inversions postprocess --config "$CONFIG_FILE" --model standard --prepared-inputs "$OUTPUT_DIR/prepare/prepared-inputs.nc" --preparation-manifest "$OUTPUT_DIR/prepare/prepare-manifest.json" --posterior "$OUTPUT_DIR/sample/posterior.nc" --output-dir "$OUTPUT_DIR/postprocess"'''
   outputs = ["outputs/postprocess"]

OGR owns campaign matrices, selected tasks, dependencies, scheduler state,
declared-output hashing, continuation manifests, check recording and gates. OGI
owns data preparation, model selection, sampling, scientific products, check
calculations and their threshold policy.  ``STAGE_INPUTS_MANIFEST`` may be used
by application code to discover continued parent outputs, but it is not needed
for full-graph submission and is not implicitly interpreted by OGI.  The
predictable explicit paths above work in both modes.
