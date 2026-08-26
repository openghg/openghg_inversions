openghg\_inversions.models.additive\_sigma
===========================================

``add_additive_sigma_gaussian_likelihood`` is the complete opt-in likelihood.
A model recipe supplies its completed forward-model mean explicitly; the
likelihood then combines reported observation uncertainty, additive
model-data-mismatch variance, and any selected fixed aggregation covariance.
An optional prepared minimum error applies the historical floor on total
marginal standard deviation.
``additive_sigma_likelihood_builder`` exposes that complete construction to
the standard and multisector RHIME customization seam without accepting
pollution-event-only inputs.

.. automodule:: openghg_inversions.models.additive_sigma
   :members:
   :show-inheritance:
   :undoc-members:
