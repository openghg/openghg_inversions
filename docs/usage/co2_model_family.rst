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
     - :func:`openghg_inversions.rhime.run_rhime_co2`; see the
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
   * - Staged workflow
     - Installed ``--model co2`` stages support ordinary and cached fixed-OU
       replay from a saved coherent prepared-input artifact; see
       `Staged CO₂ commands`_ below.
     - Not supported by the staged CLI.
   * - Outputs and postprocessing
     - Returns an annotated xarray ``DataTree``. Use the documented serialization
       boundary. :ref:`Conditional native-flux and country summaries
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
<https://linear.app/openghg-inversions/issue/OPE-79>`_. The CO₂ staged commands
reuse the existing TOML configuration boundary and prepared-input Python runners.

The CO2-only handoff is intentionally separate from both the generic
``RhimePreparedInputs`` boundary and the linked CO2/O2 preparation contract.
It does not add coherent-reduction or aggregation-error options to the
standard and multisector runners or to ``run_hbmcmc.py``.

.. toctree::
   :maxdepth: 1

   co2_models

.. _co2-staged-commands:

Staged CO₂ commands
-------------------

Use the installed ``openghg-inversions`` command with ``--model co2`` and a
CO₂ TOML configuration. First construct and save a
:class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact with
:func:`~openghg_inversions.rhime.co2.prepare_co2_inputs`; this is the scientific
handoff from your observation preparation and coherent reduction. The staged
``prepare`` command validates and copies that artifact and writes a manifest.
It does not acquire OpenGHG data or infer a coherent reduction from a gas name.

Direct prepared-input and staged execution use the same resolved CO2 recipe
configuration and ordinary/cached scientific operations. Sampler choices are
immutable; each invocation gets its own runtime sampler. Prepared observations
select and align retained site metadata in both routes. Staged manifest schema
version 3 and identity version 1 are the currently supported CO2 preparation,
prior-predictive, sampling and postprocessing contract; pre-refactor workflow
handoffs are rejected. Independent diagnosis retains historical sample-envelope
support. See :doc:`staged_workflow` for the next-minor compatibility boundary,
diagnosis support and graph-free replay authentication.

CO₂ workflow manifests record the installed package version, exact Git revision and
checkout modification status when available. Run from a source checkout or a
VCS installation
whose package metadata records the commit. An installation without an
identifiable Git revision, such as a release wheel without VCS metadata or an
associated checkout, is rejected by these workflow stages.

For example, an ordinary configuration can contain:

.. code-block:: toml

   format_version = 1
   recipe = "co2"
   variant = "ordinary"

   [prepared_inputs]
   path = "prepared-co2.nc"

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

Run each stage with explicit handoffs:

.. code-block:: bash

   openghg-inversions prepare --model co2 -c co2.toml \
     --output-dir run/prepare
   openghg-inversions prior-predictive --model co2 -c co2.toml \
     --prepared-inputs run/prepare/prepared-inputs.nc \
     --preparation-manifest run/prepare/prepare-manifest.json \
     --draws 100 --strict --output-dir run/prior
   openghg-inversions sample --model co2 -c co2.toml \
     --prepared-inputs run/prepare/prepared-inputs.nc \
     --preparation-manifest run/prepare/prepare-manifest.json \
     --output-dir run/sample
   openghg-inversions diagnose \
     --posterior run/sample/posterior.nc \
     --sample-manifest run/sample/sample-manifest.json \
     --strict --output-dir run/diagnose
   openghg-inversions postprocess --model co2 -c co2.toml \
     --prepared-inputs run/prepare/prepared-inputs.nc \
     --preparation-manifest run/prepare/prepare-manifest.json \
     --posterior run/sample/posterior.nc \
     --sample-manifest run/sample/sample-manifest.json \
     --output-dir run/postprocess

``prior-predictive --strict`` exits unsuccessfully when its readiness check
fails. ``diagnose --strict`` exits unsuccessfully when a convergence threshold
fails; inspect ``sampler-convergence.json`` and the diagnostic summary before
using posterior summaries. A short smoke test is not evidence of convergence.

For cached fixed-OU replay, use ``variant = "cached_fixed_ou"`` and replace the
likelihood table with:

.. code-block:: toml

   [likelihood]
   kind = "fixed_ou"
   tau_hours = 24.0
   site_amplitude_prior_scale = 1.0

Keep ``sampling.nuts_sampler = "pymc"``. The same staged commands apply. The
ordinary variant also supports its documented fixed-OU and specialized
likelihood choices; their prepared data and cache prerequisites still apply.

The preparation and sample manifests authenticate the configured recipe,
prepared artifact and posterior. Preserve these manifests with the NetCDF
files. Replacing an artifact with another having the same labels does not make
it compatible with an existing posterior. Paths inside the TOML file resolve
relative to that file.

Conditional native-flux and country outputs require prior draws
(``sampling.sample_prior_predictive``) and a saved affine reconstruction bound
to the exact prepared artifact. Add the following to
``[outputs]`` when those artifacts are available:

.. code-block:: toml

   reconstruction_path = "co2-affine.nc"
   country_file = "countries.nc"

The affine artifact is produced with the APIs described under
:ref:`co2-affine-flux-summaries`. Source identities are preserved through
reconstruction. If reporting sectors differ from native sources, supply an
explicit ``[outputs.source_to_sector]`` mapping. Source sums and sector
transforms precede uncertainty statistics. Native-flux and country products
carry ``retained_state_conditional`` uncertainty scope; they do not include the
unresolved native-state posterior uncertainty tracked separately in OPE-68.

``postprocess`` writes ``basic.nc``, concentration components and requested
conditional products, plus ``postprocess-manifest.json`` listing the product
paths. ``output_format = "paris"`` requests supported PARIS concentration and
flux products and requires the bound reconstruction and an explicit
``outputs.country_file``. Supported PARIS exports use the latest templates,
mean summaries, native grids and midpoint flux
timestamps. Legacy inversion-output formats and unsupported PARIS options are
rejected before product writing.
Linked CO₂/O₂ staged routing is tracked separately in OPE-165.
