Affine native-flux reconstruction
=================================

For the linear bucket-basis equation and source summation, see
:doc:`retained_state_reconstruction`. The affine map's public ``prolongation``
ingredient is a coarse-to-fine map (prolongation); its centred reconstruction
equation remains distinct from the bucket-basis action.

``AffineFluxMap`` reconstructs the retained-state-conditional mean of a
native scaling field and its signed flux field.  For native mean :math:`m`,
covariance-natural prolongation :math:`U_*`, retained state :math:`\alpha`,
and authoritative retained reference state :math:`\alpha_{ref}`, it evaluates

.. math::

   \bar{x}(\alpha) = m + U_*(\alpha - \alpha_{ref}),

   \bar{f}(\alpha) = F\bar{x}(\alpha).

Use :meth:`~openghg_inversions.basis.affine_flux_map.AffineFluxMap.state_to_native`
for native scaling and
:meth:`~openghg_inversions.basis.affine_flux_map.AffineFluxMap.state_to_flux`
for signed flux.  Both operations preserve chain, draw, and other non-state
dimensions.  The reference state is always explicit; it is not assumed to be
one.

When a state-input axis has the same name as a native or flux axis, it is
renamed ``state_<axis>`` in the result so the two independent dimensions remain
distinct.  For example, state-input ``time`` and flux ``time`` become
``state_time`` and ``time``.  If ``state_time`` is already a dimension or
coordinate name, the state-input axis becomes ``state_time_2`` (then ``_3``).
An already prefixed axis such as ``state_time`` uses the numeric suffix rather
than another ``state_`` prefix.  Compatible dimensionless units, including
percent, are converted numerically to ``1`` during application.

The value keeps :math:`m`, :math:`F`, and :math:`U_*` separate.  A
bucket-preserving prolongation remains a ``BucketBasisOperator`` or
``MultiSourceBucketBasisOperator``.  A supplied-restriction workflow instead
passes its exact labelled, native-by-state :class:`xarray.DataArray`.  Neither
representation precomputes :math:`FU_*`.

Inputs are borrowed and may remain sparse or Dask-backed.  Constructing or
inspecting the value does not copy, compute, persist, densify, or rechunk their
payloads.  Reconstructed native-grid outputs are created only by an explicit
``state_to_native`` or ``state_to_flux`` request.
Application may add a lazy Dask rechunk layer for the contraction; it does
not execute or persist that graph.

Every reconstructed array carries the machine-readable uncertainty scope
``retained_state_conditional``.  It is the conditional mean given the retained
state, not complete observation-conditioned native-grid inference.  In
particular, this value does not add unresolved variance or residual draws.

Persistence and binding to coherent-CO2 prepared inputs are separate staged
artifact concerns.  Country and other aggregate consumers should contract
their functional with :math:`F` and :math:`U_*` before applying posterior
samples rather than first constructing native-grid values for every draw.
