openghg\_inversions.models.rhime\_likelihood
============================================

This module defines the Python-only likelihood boundary used by the ordinary
RHIME runners. Pass a :class:`RhimeLikelihoodBuilder` to ``run_rhime`` or
``run_rhime_multisector``; the runner creates the context while the active
PyMC model is being built. Builders are not read from configuration or stored
as executable state in run or model specifications.

Builder contract
----------------

A builder receives :class:`RhimeLikelihoodContext`, adds the complete error
and observed-distribution component to the active model, and returns
:class:`RhimeLikelihoodResult`. The result must declare:

* semantic variable roles, including ``concentration`` for the observed
  variable;
* every RHIME output format that the likelihood supports; and
* JSON-compatible metadata for result and serialized-output provenance.

Roles drive posterior-predictive selection, and output compatibility is
validated before sampling. The runner separately records the builder's module
and qualified name; it never serializes callable code.

Reusable construction helpers
-----------------------------

:func:`build_rhime_observation_state` retains RHIME's current mean and error
construction while allowing a builder to replace only the observed
distribution. :class:`RhimeObservationState` exposes the observed values,
combined mean, independent variance, resolved aggregation error, and marginal
error scale.

:func:`build_gaussian_rhime_likelihood` implements the default
pollution-event-scaled Gaussian. The opt-in
:func:`build_absolute_sigma_gaussian_likelihood` instead treats inferred sigma
as an absolute observation-aligned contribution. For a complete editable
Student-t example and the minimal runner call, see
:doc:`../usage/customising_rhime`.

API reference
-------------

.. automodule:: openghg_inversions.models.rhime_likelihood
   :members:
   :show-inheritance:
   :undoc-members:
