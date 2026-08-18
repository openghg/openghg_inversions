Coherent Gaussian Reduction
===========================

The coherent-reduction API turns one labelled native covariance projection
into an exact retained Gaussian model. It is a backend-neutral preparation
operation: PyMC and analytic model builders may consume its result, but the
operation does not construct either backend.

Let the native scaling state and observation model be

.. math::

   x \sim \mathcal N(m, B),
   \qquad
   y = Hx + \epsilon,
   \qquad
   \alpha = \Pi x.

The low-level :doc:`native_covariance` operation constructs
:math:`C_\alpha = \Pi B\Pi^\mathsf{T}`,
:math:`HB\Pi^\mathsf{T}`, and :math:`HBH^\mathsf{T}`. Coherent reduction
constructs those linked products locally, then solves for

.. math::

   H_\alpha = HB\Pi^\mathsf{T}C_\alpha^{-1}

without constructing an inverse, then returns the centred conditional model

.. math::

   y \mid \alpha \sim \mathcal N\!\left(
      Hm + H_\alpha(\alpha - \Pi m),
      R + A
   \right),

.. math::

   A = HBH^\mathsf{T} - H_\alpha C_\alpha H_\alpha^\mathsf{T}.

All three retained quantities---the prior, effective forward operator, and
unresolved covariance---come from the same product set and are validated
together.

Preparing the exact result
--------------------------

Pass the native covariance, canonical basis prolongation, labelled native mean,
and sensitivity through one operation:

.. code-block:: python

   from openghg_inversions.coherent_reduction import reduce_native_gaussian

   reduction = reduce_native_gaussian(
       covariance=covariance,
       basis_prolongation=basis_prolongation,
       state_dim="state",
       native_mean=native_mean,
       native_sensitivity=native_sensitivity,
       observation_dim="observation",
   )

   retained_mean = reduction.retained_mean
   retained_covariance = reduction.retained_covariance
   effective_operator = reduction.effective_observation_operator
   unresolved_covariance = reduction.unresolved_observation_covariance

The operation is a named eager boundary. Related Dask-backed mean and
sensitivity inputs are densified lazily where necessary and materialized
together. The covariance product blocks are built and reduced locally, so a
sensitivity cannot be substituted after projection. The result fields are
eager xarray snapshots with explicit retained and observation labels. The
frozen result prevents field reassignment but does not claim that contained
xarray objects are immutable.

Validation and limits
---------------------

The first exact contract requires a full-rank retained restriction, so
:math:`C_\alpha` must be positive definite. It does not silently choose a
pseudoinverse. The reducer also requires dense :math:`HBH^\mathsf{T}` products,
checks exact label order, finite-real values, declared units, Cholesky solve
residuals, the redundant :math:`H_\alpha = HU_*` identity, and positive
semidefiniteness of :math:`A` for matrices within the documented eigenvalue
diagnostic threshold. Larger dense matrices retain symmetry, conditioning,
and coherent-product checks while recording that the cubic global eigenvalue
diagnostic was skipped.

This operation does not add observation error :math:`R`, construct a
likelihood, approximate :math:`A` as low-rank plus diagonal, serialize the
result, or reconstruct arbitrary native functionals. Those are separate
scientific or persistence boundaries. In particular, durable coherent-result
identity and storage belong to OPE-40.

See :mod:`openghg_inversions.coherent_reduction` for the public interface.
