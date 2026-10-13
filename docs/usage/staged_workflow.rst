Staged RHIME workflows
======================

The standard recipe can prepare once, sample compatible models into separate
locations, and produce different outputs from saved samples. Full execution
uses the same scientific operations in memory and needs no intermediate saves.

Standard saved phases
---------------------

``prepare`` returns one absolute path to ``prepare-manifest.json``. It references
prepared numerical inputs and retains the original requested window and baked
preparation policy. Retained sites and their averaging periods come from the
numerical artifact, including when acquisition or filtering drops a site.
Observation errors are established before filtering, then basis construction,
sensitivities and assembly run in scientific order. No filtered checkpoint is
written.

``sample`` takes that manifest and current model and sampler choices, returning
one absolute path to ``sample-manifest.json``. Compatible graph-time priors and
likelihoods reuse the preparation. Required BC and minimum-error arrays must be
available. Omitted model ``use_bc`` inherits the preparation choice; explicit
``False`` may ignore available BC inputs. A repeated ``bc_freq`` must agree with
the baked policy, even when the selected model ignores BCs. Changing acquired
data, filters, basis, native flux or baked numerical policy needs new preparation.

``postprocess`` takes the sample manifest and current output choices. It loads
the recorded model meaning, output contract, inference metadata, prepared inputs
and samples without building a graph or running inference. Its returned
``RhimeResult`` has ``sampler=None``. Missing descriptive metadata stays
unavailable. Output naming, destination and supported formats can change;
scientific choices cannot relabel saved samples.

A selected saved basis supplies its operator while new preparation binds current
native flux. Saved preparation and replay retain that flux and do not reacquire
it. Species-dependent mass conversion belongs to postprocessing.

Keep each manifest and its referenced files available for later phases. References
resolve relative to the declaring manifest; absolute shared-cache references are
also accepted. No environment variable locates replacement artifacts. Numerical
files use existing codecs, and manifests use JSON without mandatory content
hashes or full-configuration equality gates. Missing or malformed files fail at
loading; incompatible scientific inputs fail at their owning operation.

Preparation and sample saves need fresh destinations. Numerical files are saved first and the
manifest is written last as the completion marker. A failed save can leave
partial files but no completed manifest. Postprocessing uses the same product
writer as full execution; in-memory products require no destination. Standard staged ``basis_output_path``
must be within the preparation destination; use full execution for a separately
located basis export.

Configuration and CLI
---------------------

Preparation accepts ``--config run.ini`` or ``--params-file choices.json``.
Sampling and postprocessing optionally accept either source; a shared full
configuration contributes only the active phase's choices. JSON can contain
flat options or ``preparation``, ``model``, ``sampler`` and ``output`` mappings.
Unknown names remain errors, while known inactive values are not resolved.
Explicit ``--kwargs`` overrides for another phase are errors. Relative input
paths resolve from the configuration file. ``CONFIG_FILE`` is not an automatic
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
     --kwargs '{"output_format":"basic","country_file":"/data/countries.nc"}' \
     --output-dir run/basic

``--model standard`` is the default for these commands. A manifest's family must
match the selected route. Full-run ``rhime run`` choices and aliases are
unchanged; saved-phase selection does not add CO2 configuration-file support to
the full-run command.

Python callers can use the same handoffs:

.. code-block:: python

   from openghg_inversions.rhime import PreparationConfig, RhimeOutputSpec
   from openghg_inversions.rhime.stages import prepare, sample, postprocess

   preparation = PreparationConfig.from_params(preparation_options)
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

Readiness and independent diagnosis
-----------------------------------

``rhime prior-predictive --preparation-manifest ... --output-dir ...`` builds
the selected model and draws prior predictions without posterior inference.
``--draws`` controls its sample count. Loading and serialization errors
propagate; supported model/input failures produce a readiness ``fail`` result.
Readiness checks finite evidence, not scientific plausibility. See
:doc:`prior_predictive_checking` for interpreting predictions.

.. _staged-convergence-check:

``rhime diagnose --posterior posterior.nc --output-dir ...`` independently
reports maximum R-hat, minimum bulk/tail ESS, divergences and unassessable
variables. ``--sample-manifest`` optionally supplies descriptive run context.
Defaults are maximum R-hat 1.01, minimum bulk/tail ESS 400 and zero divergences.
Available metrics exceeding a threshold yield ``fail``; missing metrics yield
``unknown`` when none fails; otherwise the result is ``pass``. Neither check
repairs missing evidence or performs posterior inference. ``--strict`` makes
``fail`` exit nonzero. Both write schema-version-1 check JSON with producer
``openghg_inversions``; ``--check-output`` and ``--check-stage`` select its path
and stage label.

Other families and migration
----------------------------

Multisector and CO2 retain their existing staged interfaces pending separate
handoff adoption. They require explicit ``--model``, a full configuration and
``--prepared-inputs``/``--preparation-manifest`` for sampling. Postprocessing also
requires ``--posterior`` and ``--sample-manifest``. Multisector retains its
configuration restriction and output-binding record. CO2 takes an existing
coherent preparation; see :doc:`co2_model_family` for its supported outputs.
Linked CO2/O2 staging remains separate work.

The new standard handoff format replaces historical standard staged envelopes
and multi-artifact argument lists. Regenerate those handoffs with 0.8, or use the
original package version to consume them. Existing numerical codecs, scientific
Python APIs and final product formats retain their contracts.
