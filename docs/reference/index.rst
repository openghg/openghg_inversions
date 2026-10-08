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

   openghg_inversions.rhime.run_rhime
   openghg_inversions.rhime.run_rhime_multisector
   openghg_inversions.rhime.run_rhime_nested
   openghg_inversions.rhime.run_rhime_from_prepared_inputs
   openghg_inversions.rhime.run_rhime_nested_from_prepared_inputs
   openghg_inversions.rhime.run_rhime_co2
   openghg_inversions.rhime.co2.run_rhime_co2_cached_sigma
   openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs
   openghg_inversions.rhime.co2.run_rhime_co2_o2_cached_sigma_from_prepared_inputs

Resolve a requested RHIME configuration
--------------------------------------

Inspect standard/multisector choices before data access. Decode an INI with
``read_rhime_ini``, or start with Python options. Apply overrides and extract
recipe-specific options, then construct canonical ``RhimeConfig.from_params``.
Pass the result to the corresponding
runner's ``config`` argument to reuse the resolved choices. The
:doc:`configuration guide </usage/rhime>` explains requested site selectors,
retained execution metadata and compatibility helpers.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.read_rhime_ini
   openghg_inversions.rhime.RhimeConfig
   openghg_inversions.rhime.RhimeConfig.from_params
   openghg_inversions.rhime.RhimeConfig.select
   openghg_inversions.inversion_data.SiteOptions

Run specifications and results
------------------------------

These objects describe the model, sampling, and output settings accepted by
the RHIME runners. The sampler's former ``rhime`` imports remain compatible;
its implementation is owned by ``inference``.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.RhimeRunSpec
   openghg_inversions.rhime.RhimeModelSpec
   openghg_inversions.rhime.RhimeOutputSpec
   openghg_inversions.inference.RhimeSampler
   openghg_inversions.rhime.RhimeResult
   openghg_inversions.rhime.NestedRhimeResult
   openghg_inversions.rhime.SectorSpec
   openghg_inversions.rhime.AdditiveSigmaSettings
   openghg_inversions.rhime.FixedErrorSettings
   openghg_inversions.rhime.PollutionEventSettings

Prepared inversion data
-----------------------

Use these interfaces to prepare, save, reload, or adapt canonical inputs
before a separate model run. The
:doc:`RHIME configuration and prepared-input reference </usage/rhime>`
documents the expected variables, dimensions, and coordinates. Acquisition
and reload mechanics live in ``inversion_data.acquisition``; the prepared
value and its unchanged version-1 schema live in
``inversion_data.prepared_inputs``. Existing package and preparation-module
imports identify the same classes.

.. autosummary::
   :nosignatures:

   openghg_inversions.inversion_data.SiteOptions
   openghg_inversions.inversion_data.retrieve_inversion_data
   openghg_inversions.inversion_data.RhimeMergedData
   openghg_inversions.inversion_data.RhimePreparedInputs
   openghg_inversions.inversion_data.prepare_rhime_inputs
   openghg_inversions.inversion_data.prepare_rhime_inputs_from_xarray
   openghg_inversions.inversion_data.load_merged_data
   openghg_inversions.inversion_data.RhimeMergedData.from_options
   openghg_inversions.inversion_data.RhimeMergedData.load
   openghg_inversions.inversion_data.RhimeMergedData.save
   openghg_inversions.rhime.filter_rhime_observations
   openghg_inversions.rhime.build_rhime_basis
   openghg_inversions.rhime.build_rhime_sensitivities
   openghg_inversions.rhime.assemble_rhime_inputs
   openghg_inversions.rhime.NestedRhimePreparedInputs
   openghg_inversions.rhime.combine_nested_rhime_inputs
   openghg_inversions.rhime.co2.Co2PreparedInputs
   openghg_inversions.rhime.co2.BoundCo2AffineFluxMap
   openghg_inversions.rhime.co2.produce_bucket_affine_flux_map
   openghg_inversions.rhime.co2.import_explicit_affine_flux_map
   openghg_inversions.rhime.co2.load_and_bind_affine_flux_map
   openghg_inversions.rhime.co2.prepared_inputs_content_id
   openghg_inversions.rhime.co2.Co2O2PreparedInputs
   openghg_inversions.rhime.co2.prepare_co2_inputs
   openghg_inversions.rhime.co2.stages.prepare_co2_stage
   openghg_inversions.rhime.co2.build_rhime_co2
   openghg_inversions.rhime.co2.build_rhime_co2_cached_sigma
   openghg_inversions.rhime.co2.sample_co2_cached_prior_predictive
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
   openghg_inversions.basis.make_basis_functions
   openghg_inversions.basis.load_basis_functions
   openghg_inversions.basis.basis_functions_from_fp_all_flat_basis
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

   openghg_inversions.postprocessing.inversion_output.InversionOutput
   openghg_inversions.postprocessing.contracts.OutputContract
   openghg_inversions.postprocessing.output_views.make_inversion_output
   openghg_inversions.postprocessing.co2_flux_outputs.co2_native_flux_outputs
   openghg_inversions.postprocessing.co2_flux_outputs.co2_country_flux_outputs
   openghg_inversions.rhime.co2.co2_outputs.make_co2_rhime_result
   openghg_inversions.rhime.co2.co2_outputs.make_co2_rhime_outputs
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
can use the transitional :doc:`HBMCMC compatibility wrapper
<openghg_inversions.hbmcmc.run_hbmcmc>`, which translates supported options
and calls RHIME. See :doc:`legacy interfaces and migration
</usage/legacy_and_migration>` for replacements and unsupported interfaces.

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
