API reference
=============

This reference lists the public Python interfaces for running inversions and
composing OpenGHG Inversions workflows. Start with the
:doc:`model recipe chooser </usage/model_recipes>` if you need to decide which
workflow fits a scientific question. The tutorials and guides explain how the
objects fit together; the tables below link to signatures and detailed API
documentation.

Run an inversion
----------------

The standard, multisector, and nested-domain runners cover complete
acquisition-to-output workflows. Prepared-input and CO₂-family runners are
advanced entry points for workflows that already satisfy their input
contracts. See the
:doc:`standard tutorial </usage/rhime_standard_tutorial>`,
:doc:`multisector tutorial </usage/rhime_multisector_tutorial>`, and
:doc:`nested-domain tutorial </usage/nested_domains>`, or the
:doc:`CO₂ model family guide </usage/co2_model_family>` before selecting a
runner.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.run
   openghg_inversions.rhime.run_standard
   openghg_inversions.rhime.run_multisector
   openghg_inversions.rhime.run_nested
   openghg_inversions.rhime.run_multisector
   openghg_inversions.rhime.run_nested
   openghg_inversions.rhime.co2.run_co2
   openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs
   openghg_inversions.rhime.co2.run_rhime_co2_o2_cached_sigma_from_prepared_inputs

Resolve a requested RHIME configuration
---------------------------------------

Inspect standard/multisector choices before data access. Decode an INI with
``read_rhime_ini``, or start with Python options. Apply overrides and extract
recipe-specific options, then construct canonical ``RhimeConfig.from_params``.
Pass the result to the corresponding
runner's ``config`` argument to reuse the resolved choices. The
:doc:`configuration guide </usage/rhime_configuration>` explains requested site selectors,
retained execution metadata and compatibility helpers. For artifact-only CO2
and linked CO2/O2 requests, load the file and apply overrides, then construct the
preparation and model owners below; see :doc:`CO2 configuration </usage/co2_models>`.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.read_rhime_ini
   openghg_inversions.rhime.RhimeConfig
   openghg_inversions.rhime.RhimeConfig.from_params
   openghg_inversions.rhime.DataConfig
   openghg_inversions.rhime.DataConfig.from_params
   openghg_inversions.rhime.DataConfig.select
   openghg_inversions.rhime.co2.Co2PreparationConfig
   openghg_inversions.rhime.co2.Co2ModelConfig
   openghg_inversions.rhime.co2.Co2O2PreparationConfig
   openghg_inversions.rhime.co2.Co2O2ModelConfig
   openghg_inversions.inversion_data.SiteOptions

Reuse standard and multisector preparation and samples
------------------------------------------------------

These operations share their recipe's full runner preparation, inference and
output owners. See the :doc:`saved-phase guide </usage/staged_workflow>` for
manifest lifetime, compatible model changes and graph-free product replay.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.stages.prepare
   openghg_inversions.rhime.stages.save_preparation
   openghg_inversions.rhime.stages.load_preparation
   openghg_inversions.rhime.stages.sample
   openghg_inversions.rhime.stages.save_samples
   openghg_inversions.rhime.stages.load_samples
   openghg_inversions.rhime.stages.postprocess
   openghg_inversions.rhime.stages.prior_predictive
   openghg_inversions.rhime.multisector.prepare_multisector
   openghg_inversions.rhime.multisector.materialize_multisector_inputs
   openghg_inversions.rhime.multisector.build_multisector_model
   openghg_inversions.rhime.multisector_stages.prepare
   openghg_inversions.rhime.multisector_stages.save_preparation
   openghg_inversions.rhime.multisector_stages.load_preparation
   openghg_inversions.rhime.multisector_stages.sample
   openghg_inversions.rhime.multisector_stages.save_samples
   openghg_inversions.rhime.multisector_stages.load_samples
   openghg_inversions.rhime.multisector_stages.postprocess
   openghg_inversions.rhime.multisector_stages.prior_predictive
   openghg_inversions.rhime.outputs.make_model_output
   openghg_inversions.rhime.outputs.postprocess_multisector_outputs
   openghg_inversions.rhime.outputs.save_inversion_output

Run specifications and results
------------------------------

Standard and multisector runners return ``InversionOutput``, including prepared
execution and graph-free saved replay. CO2 postprocessing uses
``Co2Output``; linked saved replay uses ``Co2O2Output``. Nested execution returns ``NestedOutput``
with two native ``InversionOutput`` views borrowing one trace. Model choices,
inference and output policy have separate owners.
The sampler's former ``rhime`` imports remain compatible; its implementation is
owned by ``inference``.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.RhimeModelSpec
   openghg_inversions.rhime.RhimeOutputSpec
   openghg_inversions.inference.RhimeSampler
   openghg_inversions.rhime.NestedOutput
   openghg_inversions.rhime.materialize_nested_inputs
   openghg_inversions.rhime.inner_data_config
   openghg_inversions.rhime.build_nested_rhime_model
   openghg_inversions.rhime.make_nested_inversion_outputs
   openghg_inversions.rhime.SectorSpec
   openghg_inversions.rhime.AdditiveSigmaSettings
   openghg_inversions.rhime.FixedErrorSettings
   openghg_inversions.rhime.PollutionEventSettings

Prepared inversion data
-----------------------

Use these interfaces to prepare, save, reload, or adapt canonical inputs
before a separate model run. The
:doc:`RHIME terminology and prepared-input reference </usage/rhime>`
documents the expected variables, dimensions, and coordinates. Acquisition
mechanics live in ``inversion_data.acquisition``; the prepared
value and its version-2 raw-input schema live in
``inversion_data.prepared_inputs``. Import both values from ``openghg_inversions.inversion_data``.

.. autosummary::
   :nosignatures:

   openghg_inversions.inversion_data.SiteOptions
   openghg_inversions.inversion_data.MergedData
   openghg_inversions.inversion_data.AcquisitionFacts
   openghg_inversions.inversion_data.InputProvenance
   openghg_inversions.inversion_data.MergedDataProvenance
   openghg_inversions.inversion_data.PreparedInputs
   openghg_inversions.inversion_data.prepare_rhime_inputs_from_xarray
   openghg_inversions.inversion_data.prepare_surface_site
   openghg_inversions.inversion_data.MergedData.from_options
   openghg_inversions.inversion_data.MergedData.validate_for_preparation
   openghg_inversions.inversion_data.MergedData.with_site_data
   openghg_inversions.rhime.preparation.resolve_basis_inputs
   openghg_inversions.rhime.preparation.resolve_bc_basis
   openghg_inversions.rhime.preparation.acquire_merged_data
   openghg_inversions.rhime.preparation.export_basis
   openghg_inversions.rhime.standard.build_standard_model
   openghg_inversions.rhime.standard.materialize_standard_inputs
   openghg_inversions.rhime.resolve_model_inputs
   openghg_inversions.rhime.standard.prepare_standard
   openghg_inversions.rhime.postprocess_standard_outputs
   openghg_inversions.rhime.annotate_sample_trace
   openghg_inversions.rhime.requests.load_params
   openghg_inversions.rhime.filter_observations
   openghg_inversions.rhime.prepare_observation_errors
   openghg_inversions.rhime.build_sensitivities
   openghg_inversions.rhime.assemble_inputs
   openghg_inversions.rhime.NestedRhimePreparedInputs
   openghg_inversions.rhime.combine_nested_rhime_inputs
   openghg_inversions.rhime.co2.Co2PreparedInputs
   openghg_inversions.rhime.co2.co2_outputs.Co2Output
   openghg_inversions.rhime.co2.co2_outputs.Co2OutputOptions
   openghg_inversions.rhime.co2.BoundCo2AffineFluxMap
   openghg_inversions.rhime.co2.produce_bucket_affine_flux_map
   openghg_inversions.rhime.co2.import_explicit_affine_flux_map
   openghg_inversions.rhime.co2.load_and_bind_affine_flux_map
   openghg_inversions.rhime.co2.prepared_inputs_content_id
   openghg_inversions.rhime.co2.Co2O2PreparedInputs
   openghg_inversions.rhime.co2.co2_o2_outputs.Co2O2Output
   openghg_inversions.rhime.co2.co2_o2_outputs.Co2O2OutputOptions
   openghg_inversions.rhime.co2.co2_o2_handoffs.prepare
   openghg_inversions.rhime.co2.co2_o2_handoffs.save_preparation
   openghg_inversions.rhime.co2.co2_o2_handoffs.load_preparation
   openghg_inversions.rhime.co2.co2_o2_handoffs.sample
   openghg_inversions.rhime.co2.co2_o2_handoffs.save_samples
   openghg_inversions.rhime.co2.co2_o2_handoffs.load_samples
   openghg_inversions.rhime.co2.co2_o2_handoffs.prior_predictive
   openghg_inversions.rhime.co2.co2_o2_handoffs.postprocess
   openghg_inversions.rhime.co2.prepare_co2_inputs
   openghg_inversions.rhime.co2.handoffs.prepare
   openghg_inversions.rhime.co2.handoffs.save_preparation
   openghg_inversions.rhime.co2.handoffs.load_preparation
   openghg_inversions.rhime.co2.handoffs.sample
   openghg_inversions.rhime.co2.handoffs.save_samples
   openghg_inversions.rhime.co2.handoffs.load_samples
   openghg_inversions.rhime.co2.handoffs.postprocess
   openghg_inversions.rhime.co2.handoffs.prior_predictive
   openghg_inversions.rhime.co2.co2_runner.materialize_co2_inputs
   openghg_inversions.rhime.co2.co2_runner.build_co2_from_materialized
   openghg_inversions.rhime.co2.co2_outputs.validate_co2_product_request
   openghg_inversions.rhime.co2.build_rhime_co2_cached_sigma
   openghg_inversions.rhime.co2.prepare_co2_o2_inputs

Basis construction and state geometry
-------------------------------------

The high-level basis interfaces generate, load, and apply basis functions.
``AffineFluxMap`` reconstructs retained-state-conditional native scaling and
flux fields. The operator and layout classes expose the lower-level labelled
state geometry used by custom preparation workflows. See
:doc:`grouped basis and state metadata </usage/grouped_basis_layout>` for an
executed layout example and :doc:`affine native-flux reconstruction
</usage/affine_flux_map>` for the centred reconstruction contract. See
:doc:`linear retained-state reconstruction
</usage/retained_state_reconstruction>` for basis and flux actions.

.. autosummary::
   :nosignatures:

   openghg_inversions.basis.BasisFunctions
   openghg_inversions.basis.AffineFluxMap
   openghg_inversions.basis.AffineFluxMapArtifact
   openghg_inversions.basis.save_affine_flux_map
   openghg_inversions.basis.load_affine_flux_map
   openghg_inversions.native_flux.resolve_runtime_flux
   openghg_inversions.basis.prepare_basis
   openghg_inversions.basis.basis_functions.FluxWeightedBasis.reuse_operator
   openghg_inversions.basis.basis_functions.FluxWeightedBasis.load
   openghg_inversions.basis.packaged_basis_support_path
   openghg_inversions.basis.load_basis_support_map
   openghg_inversions.basis.load_basis_partition
   openghg_inversions.basis.make_basis_functions
   openghg_inversions.basis.load_basis_functions
   openghg_inversions.basis.basis_functions_from_flat_basis
   openghg_inversions.basis.basis_weights_from_data
   openghg_inversions.basis.bucket_basis_from_weights
   openghg_inversions.basis.quadtree_basis_from_weights
   openghg_inversions.basis.region_constrained_basis_from_weights
   openghg_inversions.basis.project_basis_prior_stdev
   openghg_inversions.basis.calibrate_basis_prior_stdev
   openghg_inversions.basis.operators.BasisOperator
   openghg_inversions.basis.operators.BucketBasisOperator
   openghg_inversions.basis.operators.MultiSourceBucketBasisOperator
   openghg_inversions.basis.layout.BasisPartition
   openghg_inversions.basis.layout.BasisLayout
   openghg_inversions.basis.layout.BasisLayoutResult

Covariance and coherent reduction
---------------------------------

These lower-level interfaces construct labelled native covariance actions,
project them into retained state space, and form coherent reduced Gaussian
models. Read the :doc:`native covariance guide </usage/native_covariance>` and
:doc:`coherent reduction guide </usage/coherent_reduction>` for their
mathematical contracts and eager-computation boundaries.

.. autosummary::
   :nosignatures:

   openghg_inversions.native_covariance.NativeCovarianceAction
   openghg_inversions.native_covariance.InvertibleNativeCovarianceAction
   openghg_inversions.native_covariance.SeparableExponentialCovariance
   openghg_inversions.source_covariance.IndependentSourceCovariance
   openghg_inversions.basis.NativeCovarianceProducts
   openghg_inversions.basis.RetainedProjection
   openghg_inversions.basis.PreserveBucketProlongation
   openghg_inversions.basis.project_native_covariance
   openghg_inversions.coherent_reduction.CoherentGaussianReduction
   openghg_inversions.coherent_reduction.reduce_native_gaussian
   openghg_inversions.observation_error.AggregationError
   openghg_inversions.observation_error.prepare_low_rank_aggregation_error
   openghg_inversions.observation_error.resolve_aggregation_error

Compose a RHIME model
---------------------

These public components support model recipes that need a different
combination of states, priors, coordinates, or likelihoods. The
:doc:`concrete RHIME model guide </usage/concrete_rhime_model>` explains the
component boundary and when to copy a complete recipe instead.

.. autosummary::
   :nosignatures:

   openghg_inversions.models.StateActivity
   openghg_inversions.models.PreparedLinearSensitivity
   openghg_inversions.models.prepare_linear_sensitivity
   openghg_inversions.models.CorrelatedLognormalPrior
   openghg_inversions.models.parse_prior
   openghg_inversions.models.add_model_data
   openghg_inversions.models.add_linear_component
   openghg_inversions.models.add_linked_linear_component
   openghg_inversions.models.add_correlated_lognormal_state
   openghg_inversions.models.add_offset_component
   openghg_inversions.models.add_sigma_component
   openghg_inversions.models.add_site_sigma_gaussian_likelihood

Outputs and serialisation
-------------------------

``InversionOutput`` is the canonical result container used by RHIME output
helpers. Serialisation helpers preserve the labelled indexes required by
prepared inputs and inference data.

``OutputContract`` records model roles and explicit state mappings without a
live graph. The output-view factory binds that metadata to prepared values
and samples; it does not write products. See
:doc:`staged replay </usage/staged_workflow>` for persistence.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.outputs.validate_output_resources
   openghg_inversions.rhime.validate_standard_output_resources
   openghg_inversions.postprocessing.inversion_output.InversionOutput
   openghg_inversions.postprocessing.contracts.OutputContract
   openghg_inversions.postprocessing.output_views.make_inversion_output
   openghg_inversions.postprocessing.co2_flux_outputs.co2_native_flux_outputs
   openghg_inversions.postprocessing.co2_flux_outputs.co2_country_flux_outputs
   openghg_inversions.rhime.co2.co2_outputs.make_co2_output
   openghg_inversions.rhime.co2.co2_outputs.validate_co2_product_request
   openghg_inversions.rhime.co2.co2_outputs.calculate_co2_products
   openghg_inversions.rhime.co2.co2_outputs.save_co2_products
   openghg_inversions.rhime.co2.co2_outputs.reconstruct_co2_concentrations
   openghg_inversions.postprocessing.linked_paris_outputs.make_co2_o2_paris_outputs
   openghg_inversions.postprocessing.linked_paris_outputs.reconstruct_co2_o2_concentrations
   openghg_inversions.postprocessing.countries.Countries
   openghg_inversions.postprocessing.make_outputs.basic_output
   openghg_inversions.postprocessing.make_outputs.make_flux_outputs
   openghg_inversions.postprocessing.make_outputs.make_country_outputs
   openghg_inversions.serialization.save_datatree
   openghg_inversions.serialization.open_datatree_loaded
   openghg_inversions.serialization.save_inferencedata
   openghg_inversions.serialization.load_inferencedata

Inference diagnostics and scientific metrics
--------------------------------------------

Neutral convergence summaries consume posterior samples while preserving
chain/draw structure. Scientific metrics compare observations with predictions
at their stated sampling support. The established
``postprocessing.diagnostics`` module remains an output compatibility adapter;
calculating a score does not apply a scientific acceptance threshold.

.. autosummary::
   :nosignatures:

   openghg_inversions.inference.diagnostics.posterior_summary
   openghg_inversions.postprocessing.metrics.bayes_r2_by_site
   openghg_inversions.postprocessing.metrics.bayes_r2_by_site_resample

Legacy compatibility APIs
-------------------------

The direct fixed-basis and hierarchical Bayesian Markov chain Monte Carlo
(HBMCMC) Python APIs were removed in 0.8. Existing fixedbasis-style INI files
must migrate to canonical RHIME settings or retain matching older software.
See :doc:`/usage/legacy_and_migration` for the retired namespace and aliases.

Detailed module reference
-------------------------

The generated module pages contain the complete autodoc reference, including
lower-level helpers and compatibility modules. Use the curated tables above
for the supported workflow entry points, or
:doc:`browse the complete module index <openghg_inversions>` when you need a
module-level view.

.. toctree::
   :hidden:
   :maxdepth: 4

   openghg_inversions
