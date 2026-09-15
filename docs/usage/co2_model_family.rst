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
     - Consumes a prepared coherent-reduction ``RhimePreparedInputs`` artifact.
       The public handoff that assembles that artifact is incomplete.
     - :func:`openghg_inversions.rhime.co2.prepare_co2_o2_inputs` gathers
       caller-supplied, channel-native prepared arrays; it does not acquire
       OpenGHG data.
   * - Configuration
     - Python arguments at the prepared-input boundary; no complete built-in
       CO₂ configuration workflow.
     - Python arguments at the prepared-input boundary; no complete built-in
       linked configuration workflow.
   * - Staged workflow
     - Not supported by the staged CLI.
     - Not supported by the staged CLI.
   * - Outputs and postprocessing
     - Returns annotated ``InferenceData``. Use the documented serialization
       boundary; the complete RHIME output pipeline is not integrated.
     - Returns annotated ``InferenceData``. Use the documented serialization
       boundary; family-specific output and postprocessing are not integrated.
   * - Validation and acceptance
     - Model construction, replay, provenance, and cached-sampler behavior have
       automated regression tests. Complete configuration, outputs, staged
       integration, and scientist acceptance remain future work.
     - Preparation, graph construction, mixed-unit metadata, replay, and
       provenance have automated regression tests. Production scientist
       acceptance remains future work.

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
configuration, outputs, staged integration, and scientist acceptance are
tracked in `OPE-79 <https://linear.app/openghg-inversions/issue/OPE-79>`_.

.. toctree::
   :maxdepth: 1

   co2_models
