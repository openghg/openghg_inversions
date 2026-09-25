Linear retained-state reconstruction
====================================

``BasisFunctions`` pairs a retained basis operator with its signed reference
flux. Its bucket basis defines a coarse-to-fine map (prolongation)
:math:`U_{\mathrm{bucket}}` from retained state :math:`\alpha` to native-grid
scaling :math:`x`. The directional operations reconstruct

.. math::

   x = U_{\mathrm{bucket}}\alpha, \qquad f = F x,

where :math:`F` is the flux retained with ``BasisFunctions``. Call
``basis_functions.state_to_native(state)`` for scaling or
``basis_functions.state_to_flux(state)`` for signed flux. The operator itself
owns only ``state_to_native``. Both actions require the exact ordered retained
state labels; flux reconstruction also requires exact native-grid labels.

For a source-specific basis, the retained ``state`` coordinate is a ragged
MultiIndex with a ``source`` level. Reconstructed scaling and flux carry one
ordered ``native_source`` dimension. A shared basis instead yields one scaling
field; source-resolved retained flux adds a ``source`` dimension to its flux
result. Sum the result's source dimension explicitly when a total grid is
required::

   from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator

   flux = basis_functions.state_to_flux(state)
   if isinstance(basis_functions.operator, MultiSourceBucketBasisOperator):
       total_flux = flux.sum(f"native_{basis_functions.operator.source_dim}")
   elif "source" in basis_functions.flux.dims:
       total_flux = flux.sum("source")
   else:
       total_flux = flux

Non-state chain, draw, and other sample axes remain labelled. A state sample
axis that shares a native or flux dimension name receives a ``state_`` prefix
so independent coordinates are not aligned. Source summation follows the basis
and retained flux, not a coincidentally named sample axis. Inputs are borrowed
and Dask application remains lazy. Completed postprocessing products are
converted to dense NumPy data at their materialization boundary; the NetCDF
writer also converts sparse payloads before serialization.

``interpolate`` is deprecated. Use ``state_to_native`` for unweighted scaling
and ``state_to_flux`` for the retained-flux product. During deprecation,
multisource ``interpolate`` still returns a source-summed eager result. The
native-map adapter is internal; ``basis_matrix`` remains the public basis map
ingredient, and :class:`~openghg_inversions.basis.affine_flux_map.AffineFluxMap`
retains its public ``prolongation`` ingredient.

This linear operation is distinct from :doc:`affine_flux_map`, whose centred
equation uses native mean :math:`m`, covariance-natural map :math:`U_*`, and an
explicit reference state :math:`\alpha_{ref}`. Current postprocessing products
keep their historical ``flux_time`` coordinate label. The retained flux keeps
its own native time coordinate until that output boundary; a state sample
``time`` axis remains independent.
