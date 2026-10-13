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

Multisector flux statistics and memory
--------------------------------------

:func:`~openghg_inversions.postprocessing.make_outputs.make_sector_flux_outputs`
reports sector and total flux statistics. For ``stats=["mean"]`` with ordinary
jointly sampled chain/draw traces, it reduces scaling samples before
reconstructing the grid.
Linearity makes this equivalent to averaging reconstructed draws, without
creating a latitude/longitude flux field for every sample. Shared whole-sample
NaN padding, such as different prior and posterior sample counts, is supported.
Unusual missing-data layouts, including differing sample availability across
sectors or entirely missing sample groups, use the general draw-wise path.
Partial-state NaNs, infinities, or a custom sample dimension also use that path.

RHIME's sector flux diagnostics request means. The multisector route
through :func:`~openghg_inversions.postprocessing.make_outputs.make_flux_outputs`
uses the same optimization when only the mean is requested.

.. warning::

   Other statistics, including the default mean and quantiles, and mean-only
   inputs outside the conditions above reconstruct draw-wise multisector fluxes.
   Total quantiles and standard deviations must be calculated after summing
   sectors at matching chain/draw coordinates so that posterior dependence
   between sectors is retained. Summing sector
   quantiles or standard deviations does not give the total's statistic.

   These products can require substantial memory as the spatial grid, retained
   flux times, samples, and number of sectors grow. Concurrent Dask tasks and
   temporary arrays also affect peak memory; lazy input alone does not make
   this calculation fit in memory. Request ``stats=["mean"]`` when means are
   sufficient, and size nonlinear-statistic jobs for their draw-wise workspace.

The mean-only path preserves float32 flux products when the scaling samples,
basis, and prior flux use float32; it does not impose a blanket float64 cast.
Mixed input precision and operations with an explicitly chosen accumulation
precision can still produce wider results. Storage precision and PyTensor graph
precision are separate choices; see :ref:`rhime-terminology-and-quickstart`.
