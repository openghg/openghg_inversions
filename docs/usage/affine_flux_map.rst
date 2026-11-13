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
generic quantity maps belong to OPE-24. The :doc:`CO₂ staged output adapter
<co2_model_family>` binds this reconstruction to posterior artifacts and accepts
explicit reporting-sector mappings.

.. _co2-affine-flux-summaries:

Summarize conditional CO2 fluxes
--------------------------------

The CO2 postprocessing adapter consumes an ordinary or cached fixed-OU CO2
trace and a bound affine reconstruction. Use the posterior from the same
preparation as the bound artifact. Binding checks the reconstruction against
the saved preparation; matching trace labels alone does not establish that
the posterior came from that preparation.

The trace must declare the ``flux_scale`` scientific role and contain the full
retained state, including restored fixed states. Prior and posterior groups
are summarized independently, so they may have different sample counts::

   from openghg_inversions.postprocessing.co2_flux_outputs import (
       co2_native_flux_outputs,
       co2_country_flux_outputs,
   )
   from openghg_inversions.postprocessing.countries import Countries

   bound = load_and_bind_affine_flux_map(reconstruction_path, prepared_path)
   native = co2_native_flux_outputs(trace, bound)
   countries = Countries.from_file(country_file=country_path)
   country = co2_country_flux_outputs(trace, bound, countries)

Here ``trace`` is the annotated CO2 ``DataTree`` returned by its runner, or
reloaded through the :doc:`documented serialization boundary
<co2_models>`. The country file must have exactly the reconstruction's latitude
and longitude coordinates. The adapter does not regrid masks implicitly.

Native outputs use ``mol m-2 s-1``. Country outputs use grams of CO2 per
fixed 365-day year, preserving any flux-time axis. This is an annualized rate
at each time, not a calendar-year integral. Country membership, regional
selection and cell area come from ``Countries``.

Both prior and posterior groups are required. Variable names have the form
``flux_total_posterior_mean`` or ``country_total_prior_stdev``. The statistics
are the mean, population standard deviation and 0.159/0.841 quantiles over
all chains and draws. Multisource products also contain ``flux_*`` or
``country_*`` variables retaining the native-source axis. Totals sum sources
on each draw before calculating statistics, preserving signed cancellation
and source covariance. Quantiles are calculated after the signed affine
reconstruction, never by interpolating retained-state quantiles.

Country output first constructs compact country-by-state actions with dense
chunk payloads, so it does not allocate native-grid-by-draw values. Its
country-by-state actions are prepared like ordinary country projections, with
output dimensions chunked within the summary point budget. Native output
explicitly requests grid reconstruction after chunking the native mean,
reference flux and prolongation within the same budget, so eager or lazy inputs
reconstruct draws in bounded tiles. Shared sample statistics complete bounded
non-sample blocks and return computed summaries, retaining complete required
samples within each block.
Neither function mutates the trace or reconstruction inputs.

Country totals as sums of whole regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When every basis region with flux in a country lies inside it, for every
source, the country total is the sum of those regions' totals. A
region-constrained basis built with the same class layer for every source,
for example :func:`~openghg_inversions.basis.algorithms.country_border_classes`
intersected with the land/sea classes, preserves the listed borders.
:func:`~openghg_inversions.postprocessing.border_countries.check_border_preserving_countries`
certifies this from the labels, a membership matrix and a support weight
(``|flux| * area``), so an ocean source with no flux on land is exempt.
Negative labels are inactive cells; country support on them, like support in
crossing regions, counts against the country. The check is strict by default;
``tolerance`` admits fractional border cells, and the opt-in
``negligible_source_share`` ignores a failing source that carries almost none
of the country's flux. Both leaks are recorded.

A border-preserving total is still retained-state conditional. In a coherent
reduction it omits ``Q x_perp`` unless the country functional ``Q`` lies in the
row space of the restriction ``Pi``. For an absolute-flux-weighted regional
mean, that holds for whole regions only when the flux keeps one sign and one
spatial pattern over time. Mixed-sign or time-varying flux leaves a residual
even when no border is crossed.
:func:`~openghg_inversions.postprocessing.border_countries.co2_region_sum_country_outputs`
labels each country with ``country_total_method``:

* ``exact_region_sum``: no border leak, and the row-space residual of every
  source and time against the supplied ``restriction=Pi`` is negligible;
* ``region_sum_retained_conditional``: a border-preserving region sum without
  that guarantee. This includes every map without ``restriction``. Bucket
  prolongations also come from a coherent reduction: for
  ``PreserveBucketProlongation``, ``Pi_U = (U^T B^-1 U)^-1 U^T B^-1``, and a
  whole-region functional lies in its row space only when ``B Q^T`` is
  constant within each region;
* ``retained_state_functional``: a country whose border is crossed, written only
  with ``country_outputs="all"``.

The default ``country_outputs="border_preserving_only"`` writes only the
border-preserving countries. Others are listed in
``non_border_preserving_countries``, with their failing sources, regions and
leak fractions in ``border_check_details``. Country coordinates
``border_leak_fraction`` and ``restriction_residual`` record each row's leak
and largest residual::

   from openghg_inversions.postprocessing.border_countries import (
       co2_region_sum_country_outputs,
   )

   totals = co2_region_sum_country_outputs(
       trace, bound, countries, labels, restriction=restriction
   )

``labels`` maps each native source to its region labels; bucket maps supply
them. Passing ``membership=region_class_membership(region_classes)`` reports
the classes the basis was built with instead of countries. In a CO2 run, set
``country_outputs = "border_preserving_only"`` in ``Co2OutputOptions`` (basic
output with a ``country_file``). Labels then come from the prepared bucket
basis. The saved reconstruction does not carry ``Pi``, so rows there are
labelled ``region_sum_retained_conditional``. Border preservation makes the
unresolved part small (in one OPE-188 study at least 99.6% of each focus
country's prior variance was resolved), but complete country totals still
need the unresolved term.

Both products retain prepared-artifact identity, reconstruction provenance and
``uncertainty_scope="retained_state_conditional"``. Their intervals exclude
unresolved native-state uncertainty and its observation-conditioned mean
update. These functions return labelled summary datasets. The
:doc:`CO₂ staged output adapter <co2_model_family>` uses them for conditional
native-flux and country products and supported PARIS exports. Linked CO2/O2
traces need a
tracer-specific reconstruction contract and are rejected here.
