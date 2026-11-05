Supplied backgrounds and an inner flux domain
============================================

The standard and independent multisector modern RHIME recipes accept
deterministic supplied backgrounds through :func:`openghg_inversions.rhime.prepare_supplied_baseline`. This Python
prepared-input API implements

.. math::

   \mu = b_{supplied} + c_{inner,fixed} + H_{inner} x_{inner} + o.

In the single-sector standard recipe, ``o`` is an optional observation offset.
A separately enabled sampled boundary component is added to the supplied
background only when explicitly declared to be a correction. Independent
multisector supplied-background runs do not support observation offsets or
sampled boundary corrections.
The supplied background already contains the boundary and exterior-flux
contributions. Adding an ordinary sampled boundary
again would double-count the background and is rejected.

Preparing the inner model
-------------------------

Start with ordinary ``PreparedInputs`` and the native footprint-times-prior-flux
response for the same retained observations. Multisector responses also carry
an indexed ``source`` dimension matching the retained source labels.
Provide a Boolean ``inner_support`` mask on the exact basis grid. For example::

    from openghg_inversions.rhime import prepare_supplied_baseline

    inner = prepare_supplied_baseline(
        prepared,
        background,  # DataArray, or Dataset with boundary and exterior_flux
        native_response=fp_x_flux,
        inner_support=inner_mask,
        inner_fixed_contribution=inner_affine,
        baseline_site_metadata=background_site_metadata,
        species="ch4",
        provenance="background product, version and processing description",
        constituents=("boundary", "exterior_flux"),
    )
    # Pass inner as prepared= to run_standard with use_bc=False, or use
    # materialize_standard_inputs followed by build_standard_model.

The helper masks the native response before basis reduction and rebuilds ``H``.
It also masks the retained native prior flux for flux outputs. The acquired
inputs and original prepared object remain unchanged. The support mask and
all baseline data remain in the returned artifact. The retained flux carries
this mask as a Boolean ``flux_support`` coordinate: inversion-grid reporting
averages only included cells, and both native and inversion-grid reconstruction
are zero outside the support. A state spanning a domain edge sees only its
included cells. Exterior-only columns are zero and are removed from sampling
by the existing state-activity machinery.

The overlap policy includes or excludes whole cells on an exactly matching
grid. Both the mask and native response must have indexed coordinates for every
native grid dimension; same-shaped arrays without indexes are rejected.
There is no area weighting, interpolation, automatic bounding box, or inference of an inner domain from the larger footprint extent. For a WUR-style
background, prepare the mask matching the background's inner domain on the
NAME grid. For an InTEM-style background, use the explicit inner/outer map and
supply the boundary and exterior-region series as separate Dataset variables.
When source grids differ, resolve cell ownership upstream and supply a
reproducible mask on the actual transport/basis grid. Fractional overlap is
outside this API.

The inner affine term is mandatory; supply a labelled zero series when absent.
It is not part of ``background``. Existing fixed inner flux states remain
ordinary inner contributions through ``H @ x`` and must not also appear in
``inner_fixed_contribution``. The helper replaces ``H`` from the supplied
native response and rejects coherently reduced operators. For a coherent CO2
preparation, use :func:`openghg_inversions.rhime.co2.prepare_co2_supplied_baseline`
as described in :ref:`supplied-baseline-coherent-co2`; masking does not commute
with coherent reduction.

Observation and uncertainty contracts
--------------------------------------

Backgrounds use a unique ``nmeasure`` MultiIndex with the same site and time
meaning as the prepared observations. Additional releases are selected away;
missing retained releases are rejected. No nearest-time alignment or
interpolation occurs. Site labels must uniquely identify the observation
stream, including inlet where relevant. ``baseline_site_metadata`` must match
the prepared metadata, including averaging periods and any inlet metadata.
The caller must average the background over the same support before ingress.

The background declares ``species`` in its attributes. All supplied series
require concentration ``units`` compatible with mol/mol; conversion to the
observation units is lazy. Nonfinite total background or inner affine values
are rejected at the coordinated PyMC materialization boundary. A Dataset
preserves each named constituent and their converted units alongside the total.
Provenance, species, constituent names and uncertainty policy survive prepared
NetCDF/Zarr and modern inversion-output serialization.

No additional uncertainty is assigned to the supplied baseline. Removing sampled
exterior states removes their explicit state uncertainty and dependence on inner
states; it does not remove exterior uncertainty already represented by a separate
aggregation-error covariance. The standard recipe uses its
usual independent inner-state priors; it does not condition a joint prior or
marginalize exterior uncertainty. Any externally prepared aggregation-error
covariance used by a custom likelihood must be appropriate for this fixed
exterior treatment. The independent-state helper does not recompute that covariance. For the
correlated CO2 route below, inherited aggregation covariance is retained
explicitly, including any unresolved exterior modes. A supplied mean does not
remove every exterior uncertainty term.

.. _supplied-baseline-coherent-co2:

Correlated CO2 preparations and cached likelihoods
-------------------------------------------------

Use :func:`openghg_inversions.rhime.co2.prepare_co2_supplied_baseline` for an
existing coherent-reduction ``Co2PreparedInputs`` artifact::

    from openghg_inversions.rhime.co2 import prepare_co2_supplied_baseline

    inner = prepare_co2_supplied_baseline(
        full_preparation,
        background,
        inner_states=inner_state_mask,
        inner_fixed_contribution=inner_affine,
        inner_support=inner_native_mask,
        baseline_site_metadata=background_site_metadata,
        species="co2",
        provenance="background product and averaging procedure",
        constituents=("boundary", "exterior_flux"),
        aggregation_error_policy="inherited",
    )

``inner_state_mask`` is an eager Boolean array with the exact complete state
index. The helper selects coherent ``H`` columns verbatim; it never reprojects
that operator through the bucket geometry. It selects the matching arithmetic
prior mean and covariance principal block, giving the lognormal inner marginal.
It does not condition the inner state on exterior coefficients fixed at one.
The explicit inner affine replaces the full affine term. It must exclude the
removed full-minus-inner fixed contribution. All boundary operands are removed.
Native geometry is restricted separately for flux reporting. Regenerated bucket
labels retain a lossless mapping through ``original_state_label`` or
``original_region_in_source`` and the source labels. Selection preserves state
order. Arrays remain borrowed and may be lazy; the labelled selection and
whole-cell masks are explicit eager inputs.

The sole covariance policy is ``inherited``: the original dense or low-rank
aggregation payload is borrowed unchanged. It may still contain unresolved
exterior modes. Do not replace it with covariance from a historical enhancement
preparation merely because the inner means agree. Do not add the removed outer
state prior covariance again. Re-deriving an inner-only unresolved covariance is
a separate preparation operation. Report inherited aggregation covariance,
reported errors and inferred mismatch terms separately.

The ordinary CO2, cached fixed-OU and scalar-sigma routes consume the same
supplied series. The cached target adds it once to its fixed intercept. These
routes require ``use_bc=False`` and no observation offsets. Independent
multisector supplied-background runs have the same restrictions; the existing
single-sector standard recipe retains its explicit correction/offset options.
Linked CO2/O2 supplied-background preparation is not supported by this API.

CO2 traces retain ``supplied_baseline_contribution`` separately from
``co2_flux_contribution``. The latter includes the inner affine term, while
``modelled_concentration`` is their absolute sum. Scientific concentration
products preserve the fixed baseline in both prior and posterior groups,
including after serialization. The background must not be added again to
these absolute predictions.

Likelihoods and absolute outputs
--------------------------------

In the standard single-sector recipe, the shared baseline component composes
supplied and sampled terms once, before either a built-in or custom likelihood
receives the complete mean.
``mu_baseline`` stores that effective background; ``mu_bc`` remains the sampled
correction when enabled. ``mu_inner`` contains inner linear flux plus the inner
fixed term, and ``mu_total`` stores the absolute mean including any offset.
Offsets remain separate in the output, but observation-derived pollution
scaling removes the complete baseline, including any offset, as in the ordinary
standard recipe. Concentration products select the effective
baseline through the output-role contract, including for fixed-only runs;
optional offset combination follows the product request. The builder declares
``baseline_includes_offset=False`` in the output contract. For other models,
a declared ``True`` prevents adding the offset again. Without this explicit
policy, the historical PARIS boundary-plus-offset convention is preserved
when a separate boundary is available. Legacy HBMCMC output is unsupported.

Under the same translation-invariant error model, fitting absolute observations
with a fixed background gives the same inference as subtracting that background
from observations. Fixed Gaussian errors satisfy this identity. Observation-
dependent error rules, floors calculated from observations, and pollution-event
scaling need additional care: subtraction can change their inputs. Keep the
same prepared error model and scaling quantities when testing equivalence;
do not assume that rerunning error preparation on residual observations leaves
the model unchanged. Absolute ``mu_total`` and predictive ``y`` already contain
the supplied background, which must not be added again during output.

.. automodule:: openghg_inversions.rhime.baseline
   :members: prepare_supplied_baseline
