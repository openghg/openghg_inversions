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
     - :func:`openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs`.
       A complete ``run_rhime_co2_o2`` entry point is not available.
   * - Acquisition and preparation
     - :func:`openghg_inversions.rhime.co2.prepare_co2_inputs` combines
       canonical RHIME observations and metadata with one coherent reduction
       in a dedicated
       :class:`~openghg_inversions.rhime.co2.Co2PreparedInputs` artifact.
     - :func:`openghg_inversions.rhime.co2.prepare_co2_o2_inputs` gathers
       caller-supplied, channel-native prepared arrays; it does not acquire
       OpenGHG data.
   * - Configuration
     - A packaged TOML template and strict resolver produce explicit arguments
       for the ordinary or cached fixed-OU prepared-input Python runner.
     - A packaged TOML template and strict resolver produce explicit arguments
       for the linked prepared-input Python runner.
   * - Boundary conditions and offsets
     - The ordinary and cached-sigma runners can select prepared ``H_bc``
       boundary sensitivity and add global, site, or site-by-period offsets.
     - Not exposed by the linked prepared-input runner.
   * - Staged workflow
     - Not supported by the staged CLI.
     - Not supported by the staged CLI.
   * - Outputs and postprocessing
     - Returns annotated ``InferenceData``. Use the documented serialization
       boundary; the complete RHIME output pipeline is not integrated.
     - Returns annotated ``InferenceData``. Use the documented serialization
       boundary; family-specific output and postprocessing are not integrated.
   * - Validation and acceptance
     - Coherent preparation, dedicated serialization, model construction,
       configuration resolution, replay, provenance, and cached-sampler
       behavior have automated regression tests. Complete outputs, staged
       integration, and scientist acceptance remain future work.
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
recipe is scientifically suitable for a particular inversion. Complete
outputs, staged integration, and scientist acceptance are tracked in `OPE-79
<https://linear.app/openghg-inversions/issue/OPE-79>`_. The TOML resolver is a
configuration boundary for the existing Python runners; it does not add a
CO₂ command to the staged CLI.

The CO2-only handoff is intentionally separate from both the generic
``RhimePreparedInputs`` boundary and the linked CO2/O2 preparation contract.
It does not add coherent-reduction or aggregation-error options to the
standard and multisector runners or to ``run_hbmcmc.py``.

.. toctree::
   :maxdepth: 1

   co2_models
