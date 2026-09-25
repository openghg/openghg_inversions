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

Persist and bind coherent-CO2 reconstruction
----------------------------------------------

The versioned affine artifact stores :math:`m`, signed :math:`F`, and one tagged
bucket or explicit :math:`U_*` representation. It preserves native and retained
coordinates (including MultiIndexes), units, the intrinsic source labels and
their order, ``retained_state_conditional`` scope, prepared-input content
identity, and JSON-safe projection, reconstruction, and source provenance.
It does not store a second reference state or a precomposed :math:`FU_*`.

Save the prepared inputs first. The saved file or Zarr store supplies the
content identity used by the affine artifact::

   from openghg_inversions.basis import save_affine_flux_map
   from openghg_inversions.rhime.co2 import (
       load_and_bind_affine_flux_map,
       prepared_inputs_content_id,
       produce_bucket_affine_flux_map,
   )

   prepared.save(prepared_path)
   artifact = produce_bucket_affine_flux_map(
       prepared,
       native_mean,
       prepared_inputs_id=prepared_inputs_content_id(prepared_path),
   )
   save_affine_flux_map(artifact, reconstruction_path)  # .nc or .zarr
   bound = load_and_bind_affine_flux_map(reconstruction_path, prepared_path)
   native_draws = bound.state_to_native(alpha_draws)
   flux_draws = bound.state_to_flux(alpha_draws)

Here ``native_mean`` is the labelled native scaling mean retained while
coherent preparation has it, and ``alpha_draws`` carries the exact prepared
retained-state labels. The bound operations take their authoritative
:math:`\alpha_{ref}` from ``prepared.inv_inputs['alpha_prior_mean']``. Loading
rejects a changed prepared artifact, incompatible labels or units, or a
different declared projection strategy before applying draws. The path-based
loader computes identity from the saved prepared artifact.

For an externally calculated supplied restriction, pass exact labelled
``native_mean``, signed ``flux``, and explicit ``prolongation`` to
``import_explicit_affine_flux_map``. If the incoming bundle has a reference
state, pass it as ``reference_state``; import checks exact agreement with the
prepared mean and discards the duplicate. This path accepts :math:`U_*` that
differs from the bucket operator and does not construct the restriction
:math:`\Pi` or derive :math:`U_*` from it. Native source labels and their order
belong to the ingredients; a gathered retained state has one state axis, not
a padded source-by-state axis. Source-to-sector reporting mappings are applied
downstream.

Aggregate before applying samples
---------------------------------

For a labelled country functional :math:`A` containing membership, cell area,
physical conversion, and any source or time selection, first contract over
native dimensions:

.. math::

   q_{ref}=AFm, \qquad R_q=AFU_*.

Then apply the compact country-by-state action to posterior draws:

.. math::

   \bar q(\alpha)=q_{ref}+R_q(\alpha-\alpha_{ref}).

The contraction occurs before chain and draw axes appear. Country values can
therefore be calculated without producing a native-grid-by-sample array. The
country action is a transient consumer calculation, absent from the affine
artifact. Native-grid arrays are materialized only when ``state_to_native``
or ``state_to_flux`` is requested. Complete country uncertainty and unresolved
native covariance are outside this conditional-mean contract (OPE-68);
generic quantity maps belong to OPE-24, and staged output routing and
reporting-sector mappings belong to OPE-164.
