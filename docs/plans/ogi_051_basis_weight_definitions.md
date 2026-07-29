# Basis Weight Definition Design

Tracker: OGI-051

This note defines generated-basis weight builders separately from partition
algorithms. The split algorithms should continue to consume explicit
two-dimensional `weights` fields; richer observation-aware behavior belongs in
weight builders and preparation routing.

## Current Production Weight

The current generated-basis weight is `mean_fp_times_mean_flux`:

```text
W(x) = mean_o(fp_o(x)) * mean_t(flux_t(x))
```

where `o` ranges over the footprints passed to the basis builder and `t` ranges
over flux times. The implementation is `basis_weights_from_fp_all(...)`, which
adapts legacy `fp_all` inputs into a two-dimensional field before calling
weight-first helpers such as `quadtree_basis_from_weights`,
`bucket_basis_from_weights`, or `region_constrained_basis_from_weights`.

This default remains the only production-routed weight builder for now. Direct
caller-supplied two-dimensional weights are acceptable at the lower-level Python
algorithm boundary, and `paired_abs_response_weights(...)` is available as a
pure lower-level helper, but neither route is a public config promise.

## Design Rules

Generated-basis `weights` are prior design inputs, not posterior diagnostics.
Production weight builders may use retained footprints, prior flux, fixed
site/time metadata, and predeclared uncertainty metadata. They must not use
observed mole fractions, posterior scale factors, residuals, held-out
footprints, validation scores, or any hyperparameter selected against the same
data used to evaluate the basis.

Observation-aware builders must use only observations retained after filters.
This should be implemented on top of the OGI-041 filter-before-basis RHIME path
now available on `devel`. Saved-basis runs should keep their current behavior
because filtering after loading a fixed basis does not change how that basis was
constructed.

Every generated basis built with a non-default weight builder should record
metadata describing:

- `weight_kind`;
- flux pairing or reduction rule;
- absolute, signed, RMS, quantile, or uncertainty reduction details;
- retained site/time counts and filters;
- any fixed site weights, uncertainty source, or preset hyperparameters;
- whether weights were source-specific or aggregated across sources.

## Candidate Builders

### Production-Candidate Builders

These are suitable for small pure-builder prototypes before public routing. The
first helper, `paired_abs_response_weights(...)`, is intentionally unrouted and
expects callers/tests to pass already-retained footprints.

- `paired_abs_response`: build `W = mean_o(abs(fp_o(x) * flux_at_o(x)))`.
  This keeps footprint and flux time pairing instead of multiplying two separate
  means, so intermittent footprints and time-varying prior fluxes can matter.
- `paired_rms_response`: build
  `W = sqrt(mean_o((fp_o(x) * flux_at_o(x)) ** 2))`. This emphasizes episodic
  high-response observations more strongly than the absolute mean.
- `site_balanced_response`: reduce within each site first, normalize or weight
  sites by a fixed preset rule, then average site fields. This prevents a
  high-count site from dominating solely because it has more retained
  observations.
- `flux_only_prior_mass`: build `W = mean_t(abs(flux_t(x)))` or
  `W = abs(mean_t(flux_t(x)))`. This is a useful baseline when the basis should
  follow prior emissions mass rather than measurement sensitivity.

For these builders, `flux_at_o` alignment must be explicit. The first
implementation should use the same prior flux product period as the footprint
time when that is well-defined, or otherwise document whether it uses nearest,
forward-fill, or period-mean alignment.

### Research-Only Builders

These should stay behind planning notes or experimental scripts until their
state-vector and validation semantics are clearer.

- `paired_quantile_response`: build a high quantile of
  `abs(fp_o(x) * flux_at_o(x))` to preserve plume tails. Quantile choice must be
  preset or selected in nested prior-only validation.
- `uncertainty_weighted_response`: build an error-weighted field such as
  `sqrt(sum_o((fp_o(x) * flux_at_o(x) / sigma_o) ** 2))`. This is only safe if
  `sigma_o` is pre-observation metadata. Residual-derived or posterior-derived
  error terms would leak inversion information into the basis.
- `signed_flux_components`: build separate positive and negative support fields,
  for example `W_pos = mean_o(fp_o(x) * max(flux_at_o(x), 0))` and
  `W_neg = mean_o(fp_o(x) * abs(min(flux_at_o(x), 0)))`. This needs an explicit
  state layout decision: separate source groups, separate basis maps, or a
  nonnegative combined proxy.
- `time_blocked_weights` and true three-dimensional `(time, lat, lon)`
  partitions. Current basis operators are built around spatial aggregation
  operators; time-varying basis regions need a separate operator and output
  design.

## API Boundary

The minimal implementation seam is the weight-first helper layer:

1. Add pure weight-builder functions next to `basis_weights_from_fp_all(...)`.
   `paired_abs_response_weights(...)` is the first example.
2. Keep `basis_weights_from_fp_all(...)` as the default adapter for existing
   public behavior.
3. Add an internal `basis_weights` or weight-builder dispatch point in
   `make_basis_functions(...)` only for generated bases, then call
   `*_basis_from_weights(...)`.
4. Leave loaded `fp_basis_case` behavior unchanged.
5. Do not add `.ini`, `run_hbmcmc.py`, or RHIME config routing until the pure
   builder and retained-observation tests are in place.

For fixed outer regions, define whether supplied weights are full-domain or
inner-mask-local before implementation. The current helper delegates through
the legacy `fp_all` algorithms, so a precomputed-weight route must avoid
silently applying the wrong mask twice.

For multi-sector RHIME, define source aggregation before implementation. The
current default selects the first requested flux source for basis weights,
whereas future multi-sector bases may need a shared aggregate field, one field
per source group, or source-specific basis artifacts.

## Validation Plan

The pure-helper slice should keep validation focused on construction semantics:

- keep the current default formula unchanged;
- test that `paired_abs_response_weights(...)` uses only the retained footprint
  times supplied by the caller;
- test a time-varying-flux case where `paired_abs_response` differs from
  `mean_fp_times_mean_flux`;
- test mask handling and fail-fast behavior for unpaired footprint/flux times.

The first routing implementation task should then:

- keep `paired_abs_response_weights(...)` as the only non-default builder;
- add a `make_basis_functions(...)` or `prepare_rhime_inputs(...)` test covering
  the OGI-041 retained-observation filtering path;
- add metadata assertions for `weight_kind` and retained site/time counts when
  generated basis metadata routing exists.

Held-out evaluation can follow the OGI-048 pattern: construct basis weights
from training footprints only, then score prior-flux observation-space
compression on holdout footprints. That evaluation may guide future defaults,
but it must not select production hyperparameters on the same observations used
to report performance.
