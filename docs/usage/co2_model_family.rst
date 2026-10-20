CO₂ model family
================

The advanced CO₂ model family contains the **CO₂-only** and **linked CO₂/O₂**
recipes. O₂ is a tracer in the linked recipe, not a standalone supported model.
Both recipes begin from scientific arrays that have already been prepared; they
are not alternative species settings for the complete standard runner.

Current support
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Capability
     - CO₂-only
     - Linked CO₂/O₂
   * - Intended use
     - One CO₂ channel with a correlated positive retained state and coherent
       aggregation covariance.
     - CO₂ and O₂ channels with shared and tracer-specific retained states and
       cross-channel covariance.
   * - Public entry point
     - :func:`openghg_inversions.rhime.co2.run_co2` consumes a
       configured product request in Python or through model selection; see the
       :ref:`package-supported cached-sigma specialization
       <co2-cached-sigma-recipe>` for its prepared-input runner.
     - :func:`openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs`
       and the matched fixed-OU
       :func:`~openghg_inversions.rhime.co2.run_rhime_co2_o2_cached_sigma_from_prepared_inputs`.
       A complete ``run_rhime_co2_o2`` entry point is not available.
   * - Acquisition and preparation
     - :func:`openghg_inversions.rhime.co2.prepare_co2_inputs` combines
       canonical RHIME observations and metadata with one coherent reduction
       in a dedicated
       :class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact.
     - :func:`openghg_inversions.rhime.co2.prepare_co2_o2_inputs` gathers
       caller-supplied, channel-native prepared arrays; it does not acquire
       OpenGHG data. :ref:`Save and replay linked prepared inputs
       <linked-prepared-replay>` describes NetCDF/Zarr persistence, including
       optional labelled independent errors.
   * - Configuration
     - A packaged TOML template and strict resolver produce explicit arguments
       for the ordinary or cached fixed-OU prepared-input Python runner.
       Ordinary full and saved operations resolve their own applicable choices.
     - A packaged TOML template and strict resolver produce explicit arguments
       for the linked ordinary and cached fixed-OU prepared-input Python runners.
   * - Fixed-OU mismatch
     - Fixed tau and fixed or inferred site amplitudes; optional matched
       cached quadratic sampler.
     - Fixed tau and fixed or inferred amplitudes by species/site, preserving
       cross-channel aggregation covariance. The matched cached sampler
       updates all active affine coefficients. Requires the same channel units.
   * - Boundary conditions and offsets
     - The ordinary and cached-sigma runners can select prepared ``H_bc``
       boundary sensitivity and add global, site, or site-by-period offsets.
     - Channel-labelled boundary sensitivities, priors and activity, plus
       independent global, site, or site-by-period offsets (same-unit channels).
   * - Prediction
     - Native PyMC prior/posterior observation draws; fixed-OU recipes expose
       a joint log likelihood, including the matched cached-sigma recipe.
     - Native PyMC prior/posterior observation draws; fixed-OU recipes expose
       a joint log likelihood, preserving cross-channel covariance with the
       matched cached sampler.
   * - Staged workflow
     - Ordinary ``--model co2`` phases consume one preparation or sample
       manifest. Cached fixed-OU retains its separate staged interface; see
       `Staged CO₂ commands`_ below.
     - Not supported by the staged CLI.
   * - Outputs and postprocessing
     - Scientific operation composition and the cached runner retain an
       annotated xarray ``DataTree``. Configured ordinary execution returns
       ``Co2Output`` with its trace,
       in-memory products and requested file paths.
       :ref:`Conditional native-flux and country summaries
       <co2-affine-flux-summaries>` are available from a bound affine artifact;
       staged basic, concentration, conditional native-flux and country products
       are available, with supported PARIS exports.
     - Returns an annotated xarray ``DataTree``. A bounded Python adapter produces
       separate CO2/O2 PARIS concentration products and supported native flux
       products; staged output integration remains future work.
   * - Validation and acceptance
     - Coherent preparation, dedicated serialization, model construction,
       configuration resolution, replay, provenance, and cached-sampler
       behavior have automated regression tests, including installed ordinary
       and cached stage sequences. Scientist acceptance remains future work.
     - Preparation, graph construction, mixed-unit metadata, replay, and
       provenance have automated regression tests. Configuration currently
       limits linked replay to one shared concentration-unit label. Production
       scientist acceptance remains future work.

Prerequisites
-------------

Readers should already understand labelled xarray state and observation
dimensions, the distinction between native and retained state, and covariance
in the observation likelihood. The shared pages on :doc:`grouped basis layouts
<grouped_basis_layout>`, :doc:`native covariance <native_covariance>`, and
:doc:`coherent reduction <coherent_reduction>` provide that background without
making those concepts specific to CO₂.

The family page records present software support, not evidence that a selected
recipe is scientifically suitable for a particular inversion. Linked
staged integration and scientist acceptance are tracked in `OPE-79
<https://linear.app/openghg-inversions/issue/OPE-79>`_. The CO₂ commands reuse
the TOML configuration boundary and prepared-input scientific operations.

The CO2-only handoff is intentionally separate from both the generic
``RhimePreparedInputs`` boundary and the linked CO2/O2 preparation contract.
It does not add coherent-reduction or aggregation-error options to the
standard and multisector runners or to ``run_hbmcmc.py``.

.. toctree::
   :maxdepth: 1

   co2_models

Staged CO₂ commands
-------------------

Ordinary CO₂ starts from canonical scientific inputs and a coherent reduction,
or an existing :class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact.
It does not acquire OpenGHG data or infer a coherent reduction from a gas name.
``run_co2`` prepares or loads inputs, explicitly materializes the
selected scientific arrays, constructs the model, samples, then
calculates and writes requested products. Full execution uses no manifests.
Direct scientific callers can compose materialization, model construction,
sampling and trace annotation explicitly; see :doc:`co2_models`.

Saved phases call those same scientific operations. ``prepare`` references an
existing numerical artifact and its optional affine companion in place; keep
those files available for later operations. Python callers
can also prepare from canonical inputs and a coherent reduction. A new
prepared/affine pair is saved with the existing codecs and checked by the affine
loader before its manifest is written. Imported pairs retain their numerical
association. Both routes retain source labels, grids and projection meaning.

Record original requested dates and domain explicitly. Filtering can remove
observations at both edges, so retained observation extrema cannot supply the
original request. Missing imported facts fail clearly; supply
``prepared_inputs.start_date``, ``end_date`` and ``domain``, or retain those facts
in preparation provenance. Sites and averaging periods come from the retained
numerical preparation.

An ordinary TOML configuration can contain:

.. code-block:: toml

   format_version = 1
   recipe = "co2"
   variant = "ordinary"

   [prepared_inputs]
   path = "prepared-co2.nc"
   start_date = "2020-01-01"
   end_date = "2020-02-01"
   domain = "EUROPE"

   [likelihood]
   kind = "additive_sigma"
   sigma_prior = { pdf = "halfnormal", sigma = 1.0 }

   [sampling]
   draws = 1000
   tune = 1000
   chains = 4
   nuts_sampler = "pymc"
   random_seed = 12345
   sample_prior_predictive = 100
   sample_posterior_predictive = true

   [outputs]
   output_format = "basic"
   output_name = "co2"

Configured execution from prepared inputs writes the selected products:

.. code-block:: bash

   rhime run --model co2 -c co2.toml --output-path run/full

``rhime run-co2`` remains an alias. This configured CLI route uses
``run_co2`` and writes requested products. Python
``rhime.run(model="co2", config_file="co2.toml")`` selects the same configured
runner and returns ``Co2Output``. Acquisition and construction of the defining
native-resolution reduction remain separate from this prepared-input route.
That extension is tracked in
`CO₂ native preparation #890 <https://github.com/openghg/openghg_inversions/issues/890>`_.

Alternatively, save preparation and inference separately:

.. code-block:: bash

   rhime prepare --model co2 -c co2.toml --output-dir run/prepare
   rhime prior-predictive --model co2 -c co2.toml \
     --preparation-manifest run/prepare/prepare-manifest.json \
     --draws 100 --strict --output-dir run/prior
   rhime sample --model co2 -c co2.toml \
     --preparation-manifest run/prepare/prepare-manifest.json \
     --output-dir run/sample
   rhime diagnose --posterior run/sample/posterior.nc \
     --sample-manifest run/sample/sample-manifest.json \
     --strict --output-dir run/diagnose
   rhime postprocess --model co2 -c co2.toml \
     --sample-manifest run/sample/sample-manifest.json \
     --output-dir run/postprocess

``prepare`` and ``sample`` each print one absolute manifest path. Ordinary
sampling needs only that preparation manifest and current inference choices;
postprocessing needs only the sample manifest and current output choices.
Separate ``--prepared-inputs`` or ``--posterior`` arguments are not accepted on
these ordinary scientific phases. Each phase resolves its own configuration
sections; prior readiness requires no posterior sampler or final-product choices.
Relative configuration paths resolve from the invocation working directory.
Manifest references resolve relative to their declaring manifest; absolute
shared-cache references are also supported. Keep manifests and their dependencies.

Compatible BC/offset priors and supported mismatch changes can sample the same
preparation into another destination. Per-site model mappings validate every
supplied value, including values for dropped sites, require retained-site
coverage, then select retained labels. Ordinary fixed sigma is nonnegative;
OU time scales are positive. Defining native priors, covariance, operators and
projections determine coherent preparation and require a coherent new preparation
when changed. Full-configuration equality and callable representations do not
decide scientific compatibility.

Replay loads saved trace roles, inference provenance and affine associations.
It performs no acquisition, preparation, model construction or inference, and
returns ``Co2Output`` without a live sampler. Missing descriptive facts remain
unavailable. Records include package version and Git revision/checkout status
when available; unavailable VCS facts are ``null`` and do not prevent execution.

``prior-predictive --strict`` exits unsuccessfully for a failed readiness check;
unknown checks retain their existing meaning. Invalid returned predictive
evidence produces a failed report and retains applicable predictive artifacts.
CO₂ execution errors propagate before creating the readiness destination.
``diagnose`` assesses the supplied posterior independently of construction;
``--strict`` exits unsuccessfully when an available convergence threshold fails.
A short smoke test is not evidence of convergence.

Conditional native-flux and country products require prior and posterior
``flux_scale`` draws and a companion bound to the preparation used for inference.
Enable ``sampling.sample_prior_predictive`` to retain prior draws. Missing
required draws or incompatible affine inputs fail before product writing. The
companion belongs to preparation; country masks are current reporting choices:

.. code-block:: toml

   [prepared_inputs]
   # Alongside the prepared path, requested dates and domain above:
   reconstruction_path = "co2-affine.nc"

   [outputs]
   output_format = "basic"
   country_file = "countries.nc"

The affine artifact APIs are described under :ref:`co2-affine-flux-summaries`.
Sample replay uses the saved companion; later output choices cannot replace its
scientific meaning. Native sources retain their identities; when reporting
sectors differ, supply an explicit ``outputs.source_to_sector`` mapping.
Source sums and sector transforms precede uncertainty statistics. Native-flux
and country products carry ``retained_state_conditional`` uncertainty scope;
they exclude unresolved native-state posterior uncertainty tracked in OPE-68.

``Co2Output`` borrows the preparation and complete annotated trace DataTree,
retaining separate prior, posterior, predictive and diagnostic groups with
their own coordinates. Ordinary CO2 retains the original requested window, optional
bound affine reconstruction, in-memory products and descriptive provenance.
It carries no writing policy or live sampler field. Concentration products
deliberately use a derived Dataset with separate prior sample axes; this does not
flatten the original trace.

Pass :class:`openghg_inversions.rhime.co2.co2_outputs.Co2OutputOptions` to
calculation and export operations. Its ordinary CO₂ fields are
``output_format`` (``none``, ``basic`` or ``paris``), ``output_path``,
``output_name``, ``save_trace``, ``country_file`` and
``paris_postprocessing_kwargs``. Naming uses the RHIME convention. Reporting
``source_to_sector`` mappings are current product choices, separate from saved
inference. Standard inversion-output flags and naming options are not accepted.
Direct product callers first use
:func:`~openghg_inversions.rhime.co2.co2_outputs.validate_co2_product_request`,
which checks the selected products and loads compatible reporting masks once;
calculation reuses those resolved reporting inputs.
Supported products are basic summaries, concentration components
and conditional native/country summaries. A later ``output_format = "paris"``
request writes supported concentration and flux products from the same samples;
it requires the bound companion and an explicit country file. PARIS supports the
latest templates, means, native grids and midpoint flux timestamps. ``inv_out``,
legacy formats and unsupported PARIS options fail before product writing.
See :download:`co2_handoffs.py <../../examples/co2_handoffs.py>` for real
coherent preparation, filtering at both edges, full/saved comparison and signed
affine products. ``Co2Output`` retains the conditional affine meaning; a standard
multiplicative ``InversionOutput`` cannot represent that reconstruction.

Cached and linked boundaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cached CO₂ handoff adoption remains separate. Its existing
``variant = "cached_fixed_ou"`` staged interface continues to use explicit
``--prepared-inputs``, ``--preparation-manifest``, ``--posterior`` and
``--sample-manifest`` arguments where applicable. It does not consume the new
ordinary handoff format. Use ``likelihood.kind = "fixed_ou"``, a positive
``tau_hours``, ``site_amplitude_prior_scale`` and ``sampling.nuts_sampler = "pymc"``
for that existing route. Preserve its original configuration and matched files.
Cached postprocessing also returns ``Co2Output``, with its matched preparation,
trace and optional affine reconstruction. Its reporting window remains the
first observed day through the day after the final observation. Cached stage
products retain their literal filenames such as ``basic.nc`` and
``concentration_components.nc`` and their postprocess manifest. Recorded sampler
settings are descriptive provenance and do not construct a sampler during replay.
Linked CO₂/O₂ staged routing remains separate in OPE-165; nested workflows are
outside these CO₂ commands.
