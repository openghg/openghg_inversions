Reuse a multisector preparation
===============================

This executable walkthrough prepares a small two-source inversion once, samples
it with two different priors and likelihoods, and replays the saved samples with
new output choices. It then compares replayed flux products with full and
prepared-input execution. Read :doc:`staged_workflow` for artifact lifetime and
compatibility rules, or :doc:`rhime_multisector_tutorial` for an inversion using
OpenGHG data.

You can :jupyter-download-notebook:`download this walkthrough as a notebook
<multisector_handoffs>`. Run it from an OpenGHG Inversions source checkout with
the package and Jupyter installed. The synthetic data and model declarations
come from :download:`examples/multisector_handoffs.py
<../../examples/multisector_handoffs.py>`; that script also checks the model's
log density against independent equations.

The synthetic observations replace acquisition from an OpenGHG object store.
Preparation, sampling, checkpoint serialization and flux reconstruction use the
production operations. One short chain makes execution observable; it provides
no evidence of convergence or scientifically adequate posterior uncertainty.

.. jupyter-kernel:: python3
   :id: multisector_handoffs

Set up two source fields
------------------------

The fixture supplies four concentration observations, two distinct native flux
fields, source-resolved transport responses, and a saved one-region operator.
Its scientific sector names, ``biosphere`` and ``fossil``, refer to the acquired
sources ``supplemental`` and ``inventory``. Keeping these labels explicit lets
reporting retain both associations.

.. jupyter-execute::

   from dataclasses import replace
   from pathlib import Path
   import sys
   from tempfile import TemporaryDirectory

   import xarray as xr

   # Also allow execution from the documentation's usage directory.
   repository = next(
       path for path in (Path.cwd(), *Path.cwd().parents)
       if (path / "examples" / "multisector_handoffs.py").is_file()
   )
   sys.path.insert(0, str(repository))

   from examples.multisector_handoffs import model_options, synthetic_inputs
   from examples.rhime_cookiecutter.my_inversion.likelihoods import likelihood_builder
   from openghg_inversions.postprocessing.make_outputs import make_sector_flux_outputs
   from openghg_inversions.rhime import multisector_stages as handoffs
   from openghg_inversions.rhime.multisector import run_multisector
   from openghg_inversions.rhime.params import DataConfig, RhimeConfig
   from openghg_inversions.rhime.sampling import RhimeSampler
   from openghg_inversions.rhime.specs import RhimeOutputSpec

   workspace = TemporaryDirectory(prefix="multisector-handoffs-")
   destination = Path(workspace.name)
   merged, options, flux, responses, observations = synthetic_inputs(destination)
   {"acquired sources": list(merged.flux_data), "sectors": model_options(0.2)["sector_sources"]}

Prepare once
------------

Preparation establishes observation errors, binds the operator to the current
native flux fields, constructs sensitivities and assembles labelled model
inputs. It does not select a prior or build a probability graph. The manifest
records the acquisition and preparation choices needed by later phases.

.. jupyter-execute::

   data = DataConfig.from_params(options, multisector=True)
   preparation_manifest = handoffs.prepare(
       data,
       merged_data=merged,
       output_dir=destination / "prepared",
   )
   prepared, recorded_data, generation = handoffs.load_preparation(preparation_manifest)
   prepared.inv_inputs[["mf", "mf_error", "H"]].compute()

Sample a Normal model
---------------------

Each sector has a Normal scaling prior. The first model uses prior standard
deviation 0.2 and an independent Normal likelihood with fixed observation
errors. Sampling explicitly materializes the selected prepared arrays and prior
parameters before constructing the model. Its sample manifest records the
scientific meaning alongside the posterior samples.

.. jupyter-execute::

   sampler = RhimeSampler(
       draws=8,
       tune=8,
       chains=1,
       sample_kwargs={"cores": 1, "random_seed": 42},
       sample_prior_predictive=False,
       sample_posterior_predictive=False,
   )
   normal_model = model_options(0.2)
   normal_manifest = handoffs.sample(
       preparation_manifest,
       model=normal_model,
       sampler=sampler,
       output_dir=destination / "normal",
   )

Change the prior and likelihood
-------------------------------

The second model widens both prior standard deviations to 0.6 and replaces the
observation likelihood with the example's independent Student-t likelihood
(four degrees of freedom). Both changes belong to model construction, so they
reuse the same preparation. Changing transport, native flux, filters or the
basis would require a new preparation; see :doc:`staged_workflow` for the
compatibility boundary.

.. jupyter-execute::

   student_model = model_options(0.6, student_t=True)
   student_manifest = handoffs.sample(
       preparation_manifest,
       model=student_model,
       sampler=sampler,
       likelihood_builder=likelihood_builder,
       output_dir=destination / "student",
   )

Replay with current output choices
----------------------------------

Postprocessing loads each saved posterior and its original scientific context
without constructing a graph or sampling again. ``output_format="none"``
returns the scientific output for analysis. The same samples can also be saved
as a self-contained inversion output at a new destination. These reporting
choices do not change the original source associations, prior flux or requested
time window.

.. jupyter-execute::

   normal = handoffs.postprocess(
       normal_manifest,
       output=RhimeOutputSpec(output_format="none"),
   )
   student = handoffs.postprocess(
       student_manifest,
       output=RhimeOutputSpec(output_format="none"),
   )
   exported = handoffs.postprocess(
       student_manifest,
       output=RhimeOutputSpec(
           output_format="inv_out",
           output_name="student-replay",
           output_path=str(destination / "exports"),
       ),
   )
   assert student.trace.equals(exported.trace)
   student.run_metadata["start_date"], student.run_metadata["end_date"]

Compute flux products explicitly. The source-resolved products retain the native
inventories' spatial patterns; total flux includes both sources. With so few
draws, differences between these two models are illustrative and must not be
interpreted as a sensitivity analysis.

.. jupyter-execute::

   normal_flux = make_sector_flux_outputs(
       normal,
       stats=["mean"],
       report_flux_on_inversion_grid=False,
   )
   student_flux = make_sector_flux_outputs(
       student,
       stats=["mean"],
       report_flux_on_inversion_grid=False,
   )
   xr.concat(
       [normal_flux, student_flux],
       dim=xr.IndexVariable("model", ["Normal", "Student-t"]),
   )

Compare full and prepared-input execution
-----------------------------------------

The full runner prepares supplied acquired data before constructing and
sampling the model. Passing ``prepared=prepared`` starts from the existing
scientific preparation instead. Both runners and the saved phases compose the
same scientific operations; checkpoint files are optional boundaries between
those operations.

Use the second model's choices and the same random seed to compare their flux
products with replay. This checks execution parity, not convergence.

.. jupyter-execute::

   config = RhimeConfig.from_params(
       {
           **options,
           **student_model,
           "output_name": "student-parity",
           "output_format": "none",
           "save_inversion_output": False,
       },
       multisector=True,
   )
   config = replace(config, sampler=sampler)
   full = run_multisector(
       config=config,
       merged_data=merged,
       likelihood_builder=likelihood_builder,
   )
   resumed = run_multisector(
       config=config,
       prepared=prepared,
       likelihood_builder=likelihood_builder,
   )
   for result in (full, resumed):
       xr.testing.assert_allclose(
           student_flux,
           make_sector_flux_outputs(
               result,
               stats=["mean"],
               report_flux_on_inversion_grid=False,
           ),
       )
   print("Full, prepared-input and replayed flux products agree.")

The preparation and both sample manifests remain in ``destination`` until the
following cleanup. For a real project, choose persistent directories and keep
each manifest with its referenced files so that replay remains possible.

.. jupyter-execute::

   workspace.cleanup()
