Supplied backgrounds and an inner flux domain
============================================

The standard modern RHIME recipe accepts deterministic supplied backgrounds
through :func:`openghg_inversions.rhime.prepare_supplied_baseline`. This Python
prepared-input API implements

.. math::

   \mu = b_{supplied} + c_{inner,fixed} + H_{inner} x_{inner} + o.

Here ``o`` is an optional observation offset. A separately enabled sampled
boundary component is added to the supplied background only when explicitly
declared to be a correction. The supplied background already contains the
boundary and exterior-flux contributions. Adding an ordinary sampled boundary
again would double-count the background and is rejected.

Preparing the inner model
-------------------------

Start with ordinary single-sector ``PreparedInputs`` and the native
footprint-times-prior-flux response for the same retained observations.
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
all baseline data remain in the returned artifact. A state spanning a domain
edge sees only its included cells. Exterior-only columns are zero and are
removed from sampling by the existing state-activity machinery.

The overlap policy includes or excludes whole cells on an exactly matching
grid. There is no area weighting, interpolation, automatic bounding box, or
inference of an inner domain from the larger footprint extent. For a WUR-style
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
native response, so callers with a coherently reduced operator must provide
its appropriate inner response and affine term rather than assuming that
masking commutes with coherent reduction.

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

Baseline uncertainty is omitted. Removing exterior states also removes their
uncertainty and their dependence on inner states. The standard recipe uses its
usual independent inner-state priors; it does not condition a joint prior or
marginalize exterior uncertainty. Any externally prepared aggregation-error
covariance used by a custom likelihood must be appropriate for this fixed
exterior treatment. The helper neither recomputes nor claims to preserve that
covariance. Correlated-prior and CO2-family recipes are outside this bounded
API. Their materializers reject these supplied inputs instead of ignoring them.

Likelihoods and absolute outputs
--------------------------------

The shared baseline component composes the supplied and sampled terms once,
before either a built-in or custom likelihood receives the complete mean.
``mu_baseline`` stores that effective background; ``mu_bc`` remains the sampled
correction when enabled. ``mu_inner`` contains inner linear flux plus the inner
fixed term, and ``mu_total`` stores the absolute mean including any offset.
Offsets remain separate and are excluded from observation-derived pollution
scaling for supplied backgrounds. Concentration products select the effective
baseline through the output-role contract, including for fixed-only runs;
optional offset combination follows the product request. Legacy HBMCMC output
is unsupported.

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
