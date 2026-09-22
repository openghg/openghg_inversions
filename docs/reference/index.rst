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

The standard and multisector runners cover complete acquisition-to-output
workflows. Prepared-input and CO₂-family runners are advanced entry points for
workflows that already satisfy their input contracts. See the
:doc:`standard tutorial </usage/rhime_standard_tutorial>`,
:doc:`multisector tutorial </usage/rhime_multisector_tutorial>`, and
:doc:`CO₂ model family guide </usage/co2_model_family>` before selecting a
runner.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.run_rhime
   openghg_inversions.rhime.run_rhime_multisector
   openghg_inversions.rhime.run_rhime_from_prepared_inputs
   openghg_inversions.rhime.run_rhime_co2
   openghg_inversions.rhime.co2.run_rhime_co2_cached_sigma
   openghg_inversions.rhime.co2.run_rhime_co2_o2_from_prepared_inputs

Run specifications and results
------------------------------

These objects describe the model, sampling, and output settings accepted by
the RHIME runners.

.. autosummary::
   :nosignatures:

   openghg_inversions.rhime.RhimeRunSpec
   openghg_inversions.rhime.RhimeModelSpec
   openghg_inversions.rhime.RhimeOutputSpec
   openghg_inversions.rhime.RhimeSampler
   openghg_inversions.rhime.RhimeResult
   openghg_inversions.rhime.SectorSpec
   openghg_inversions.rhime.AdditiveSigmaSettings
   openghg_inversions.rhime.FixedErrorSettings
   openghg_inversions.rhime.PollutionEventSettings

Prepared inversion data
-----------------------

Use these interfaces to prepare, save, reload, or adapt canonical inputs
before a separate model run. The
:doc:`RHIME configuration and prepared-input reference </usage/rhime>`
documents the expected variables, dimensions, and coordinates.

.. autosummary::
   :nosignatures:

   openghg_inversions.inversion_data.RhimeMergedData
   openghg_inversions.inversion_data.RhimePreparedInputs
   openghg_inversions.inversion_data.prepare_rhime_inputs
   openghg_inversions.inversion_data.prepare_rhime_inputs_from_xarray
   openghg_inversions.inversion_data.load_merged_data
   openghg_inversions.rhime.co2.Co2PreparedInputs
   openghg_inversions.rhime.co2.Co2O2PreparedInputs
   openghg_inversions.rhime.co2.prepare_co2_inputs
   openghg_inversions.rhime.co2.prepare_co2_o2_inputs

Basis construction and state geometry
-------------------------------------

The high-level basis interfaces generate, load, and apply basis functions.
The operator and layout classes expose the lower-level labelled state geometry
used by custom preparation workflows. See
:doc:`grouped basis and state metadata </usage/grouped_basis_layout>` for an
executed layout example.

.. autosummary::
   :nosignatures:

   openghg_inversions.basis.BasisFunctions
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
   openghg_inversions.models.add_inferpymc_likelihood_component

Outputs and serialisation
-------------------------

``InversionOutput`` is the canonical result container used by RHIME output
helpers. Serialisation helpers preserve the labelled indexes required by
prepared inputs and inference data.

.. autosummary::
   :nosignatures:

   openghg_inversions.postprocessing.inversion_output.InversionOutput
   openghg_inversions.postprocessing.countries.Countries
   openghg_inversions.postprocessing.make_outputs.basic_output
   openghg_inversions.postprocessing.make_outputs.make_flux_outputs
   openghg_inversions.postprocessing.make_outputs.make_country_outputs
   openghg_inversions.serialization.save_datatree
   openghg_inversions.serialization.open_datatree_loaded
   openghg_inversions.serialization.save_inferencedata
   openghg_inversions.serialization.load_inferencedata

Legacy compatibility APIs
-------------------------

The fixed-basis and hierarchical Bayesian Markov chain Monte Carlo (HBMCMC)
interfaces remain for existing scripts and historical outputs. They are not
the starting point for new workflows. See
:doc:`legacy interfaces and migration </usage/legacy_and_migration>` before
using these compatibility APIs.

.. autosummary::
   :nosignatures:

   openghg_inversions.hbmcmc.hbmcmc.fixedbasisMCMC
   openghg_inversions.hbmcmc.preparation.FixedBasisPreparedData
   openghg_inversions.hbmcmc.preparation.prepare_fixedbasis_inversion_data
   openghg_inversions.hbmcmc.inversion_pymc.inferpymc

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
