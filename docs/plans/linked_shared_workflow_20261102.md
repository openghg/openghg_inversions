# Linked shared-baseline workflow: implementation plan, 2 November 2026

This is a dated implementation plan, not a production capability specification.
Related ownership: #550 model semantics, #661 recipe configuration, #976 saved
workflows. The existing supported routes are described in
[CO2 model families](../usage/co2_model_family.rst) and
[configured priors](../usage/configured_co2_priors.rst).

## Dependency and ownership boundary

The follow-up starts in a fresh worktree from tested #1005
`613567501527607ab80ab97badcc86020ed4af13`. Remote heads checked before work:
#1002 `4f3640d5963ebf790e05efec0ad93df5a8bf4b4d`, #1003
`6c61837f5e08ed2de9ff67bd28f3a39dfeb6e54b`; devel remains `80782dd4`.
The shared-model lead owns linked latent-state semantics, equations, the model
API and sequencing. Acquisition/configuration integration consumes its reviewed
exact commit and cannot infer a linked builder from the CO2-only API.

OGI owns reusable source validation, scientific preparation, concrete recipe
configuration and normal prepare/sample/product operations. Verification Games
owns campaign selection, archived-delivery adaptation and scoring. Broader job
orchestration stays in openghg-run when needed. No new orchestration or generic
artifact framework is proposed.

## Existing seams and bounded delivery order

| Phase | Existing owner | Integration work |
| --- | --- | --- |
| Select/acquire | `MergedData.from_options`, `acquire_surface_data`, `SiteOptions`, provider getters | Resolve each tracer's sites/inlets/window, exact input versions and transport modes explicitly. Keep independent observation targets and source lineage. |
| Native preparation | `prepare_surface_site`, `baseline_reference_sensitivities` | Preserve borrowed lazy surface responses, native endpoint requirements, explicit per-tracer scalar references and one unit conversion. Reuse #1002 for native temporal choices. |
| Coherent preparation | `reduce_native_gaussian`, `prepare_co2_o2_inputs` | Join channels only through one reviewed retained prior, signed response/ratio convention and joint aggregation covariance. A current Python caller supplies these products explicitly; an acquisition-to-reduction recipe is not yet delivered. |
| Shared preparation/model | Shared-model lead's reviewed dependency | Freeze full-support per-tracer BC/outer means and weights once. Preserve joint physical flux states and explicit outer selectors; consume supported modes only. |
| Configure/build/sample | Existing linked configuration, runner, sampler owners | Bind reviewed shared preparation explicitly. Select ordinary/cached routes only where the lead supports identical equations. Preserve existing CO2-only and linked defaults. |
| Prepare/sample artifacts | Existing linked numerical codec and `co2_o2_handoffs` | Persist concrete recipe identity, resolved source/calibration/reference/selector/weight payloads and exact numerical content bindings. Protect dependency paths against output aliases. |
| Graph-free products | Linked trace roles, `Co2O2Output`, existing linked concentration/flux products | Reconstruct absolute concentrations and physical components from saved values. Selected-state outputs must restore fixed/removed outer states using the reviewed preparation; never relabel them as the original full state without reconstruction. |

The first independent slice adds exact content bindings to current linked
prepare/sample records and offline acquisition fixtures. It does not claim
linked shared-baseline construction. After the model contract arrives, extend
these concrete owners and write one runnable maintained example rather than
introducing an experimental Dataset assembly requirement for users.

## Scientific input checks

- Keep tracer, site, inlet, time, masks and native-response identities explicit.
  Do not infer an observation source or a complete native model from a species
  string. O2 observations in absolute mole fraction and O2 response units are
  converted once into the configured scale; float64 arithmetic cannot recover
  precision already lost in source data.
- Each tracer has one independently supplied scalar reference over complete
  retained support. A BC coefficient frequency can gate corrections; it cannot
  select subperiod references or trigger width recalibration.
- Retained labels, original covariance, coherent affine intercept, outer and
  sampled-outer masks must agree exactly. Supported nonzero cross covariance
  remains part of the same joint target. Units alone cannot prove a source or
  covariance identity.
- Per-tracer levels are independent choices by default; physical flux latent
  variables remain shared. Do not collapse additive GPP/TER into a fixed
  effective net/O2 ratio when their oxidative ratios differ. The model lead
  must support a scientifically defined design or reject the mode.
- Saved calibration uses its original prior and reporting functionals. Reloads
  and observation subsets reuse resolved calibration and full-support means.
  Changing native fields, endpoints or coherent inputs requires fresh numerical
  preparation rather than a metadata migration.

## Corrected data: two distinct preparation routes

The OPE-239 r2 bundle's `usage.txt`, `catalog-registration.json`, site-file
metadata and `delivery/manifest.json` were inspected read-only. Catalog
registration is external-path discovery; it does not patch historical
OpenGHG footprint retrieval.

1. **Import a delivered normalized baseline.** Pin its record ID and SHA256;
   retain its exact tracer/site/time support and documented timing policy.
   The inspected site files use `species,time,bc_region`, float64 ppm anomaly
   and background, and dimensionless mass responses. They store reference
   values only in a descriptive string; require explicit scalar reference
   inputs and verify reconstruction instead of parsing scientific defaults.
   The delivered policy is archived CORSO midpoint-label forward fill with a
   six-hour bound. Importing these operands must not silently rerun a new
   native-field calculation.
2. **Prepare fresh native responses.** Pin the original time-resolved parent,
   the monthly endpoint overlays and the boundary fields. Apply the maintained
   endpoint adapter before calling OGI's supplied-input preparation seam.
   Preserve the original lazy `fp`/`fp_HiTRes` surface transport; the overlay
   is not a complete replacement footprint. Monthly release coverage and
   parent/grid identity must match, with no silent missing-endpoint fill.
   Use #1002's explicit native temporal contract; do not duplicate its policy.

The existing `InputProvenance` holds OpenGHG store/UUID/dataversion identities.
It cannot represent a complete OGcat/overlay lineage. Pinned catalog IDs,
paths, checksums, parent identities and timing policy need explicit acquisition
metadata carried into preparation; they must not be disguised as UUIDs.
Campaign-specific record selection remains outside OGI.

Corrected UOB deliveries and retained WUR observations are different targets.
OPE-239 repairs CO2 at 31 sites, but O2 only at the six recorded replacement
sites. Preserve each product's gaps and replacement scope.

## Bounded real-input replay prepared for the integration stage

Use one explicitly pinned site/day with both tracer supports, original surface
transport, matching monthly repaired endpoint overlay and checked native BC
fields. The campaign adapter supplies the exact source inventory and hashes;
no OGI source file contains a campaign path. Select the common observation
support explicitly rather than filling delivery gaps.

Construct a small synthetic target from declared physical flux coefficients,
separate known tracer levels and the reviewed prior/operator. Record that truth
independently of inference; do not use it to tune calibration or widths. Retain
the original coherent preparation and joint covariance for matched ordinary/
cached checks. This is a small implementation replay, not a duplicate full
campaign or evidence that concentration fit identifies gross fluxes.

Acceptance checks: actual acquisition/preparation fixture; nonzero configured
versus explicit densities/gradients and parameter identities; supported cache
parity; O2 background-independent physical widths; row/unit/source-binding
rejections; frozen subset means/calibration; and graph-free absolute component
and physical flux reconstruction after save/reopen. The maintained end-to-end
example and final configured shared entry points depend on the reviewed linked
model contract and remain to be implemented.
