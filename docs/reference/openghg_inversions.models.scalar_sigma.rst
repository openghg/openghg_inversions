openghg\_inversions.models.scalar\_sigma
==========================================

The CO2 scalar-sigma component evaluates the exact Gaussian covariance
``A + D_obs + sigma_global**2 I``. It is split across two explicit boundaries:

* :func:`openghg_inversions.rhime.co2.prepare_co2_scalar_sigma_eigenbasis`
  aligns the labelled CO2 inputs, materializes the related error arrays
  together, constructs ``A + D_obs``, and factorizes it once.
* :func:`openghg_inversions.models.scalar_sigma.add_scalar_sigma_eigen_likelihood`
  consumes the trusted eigenbasis and an explicit positive-support prior while
  constructing the PyMC graph. It performs no file access or dense covariance
  reconstruction.

The optional cache is a small versioned xarray Dataset stored as NetCDF. It
contains labelled ``eigenvectors`` and ``eigenvalues``. Loading checks the
schema, dimensions, finite values, concentration units, and exact ordered
observation coordinate; it does not create a second content-identity system or
reconstruct the dense covariance.

Prepare and save a reusable cache::

   from openghg_inversions.models import save_scalar_sigma_eigenbasis
   from openghg_inversions.rhime.co2 import prepare_co2_scalar_sigma_eigenbasis

   eigenbasis = prepare_co2_scalar_sigma_eigenbasis(
       prepared_inputs,
       aggregation_error_mode="dense",
   )
   save_scalar_sigma_eigenbasis("scalar-sigma-eigenbasis.nc", eigenbasis)

Select it through the existing CO2 likelihood seam::

   from openghg_inversions.models import add_scalar_sigma_eigen_likelihood
   from openghg_inversions.rhime.co2 import run_rhime_co2

   trace = run_rhime_co2(
       prepared_inputs=prepared_inputs,
       aggregation_error_mode="dense",
       likelihood_builder=add_scalar_sigma_eigen_likelihood,
       likelihood_kwargs={
           "eigenbasis_path": "scalar-sigma-eigenbasis.nc",
           "sigma_prior": {"pdf": "halfnormal", "sigma": 0.75},
       },
   )

``mf``, ``mf_error``, and ``sigma_global`` must share one physical
concentration unit; aggregation variances and covariances use its square. The
component is not a mixed-unit CO2/O2 likelihood. A direct
:func:`openghg_inversions.rhime.co2.build_co2_model` caller may instead load
the cache with :func:`load_scalar_sigma_eigenbasis` before calling the builder
and pass the resulting ``eigenbasis`` in ``likelihood_kwargs``.

.. automodule:: openghg_inversions.models.scalar_sigma
   :members:
   :show-inheritance:
   :undoc-members:
