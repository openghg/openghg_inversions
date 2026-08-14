# Mask-Constrained Basis Work Plan

## Scope

This tracks the branch sequence for issues #318, #449, #325, and the split-strategy
part of #340.

## Branches

- `codex/basis-prototype-examples`: pushed as a reference branch only. No PR opened.
- `codex/318-landsea-regression`: targeted regression test for current weighted
  land/sea behavior. Merged through PR #462.
- `codex/449-mask-constrained-core`: pure mask/region-constrained helper and
  split-strategy protocol. Merged through PR #462.
- `codex/325-region-mask-basis`: country/region-mask caller adapter built on the
  #449 helper. Merged through PR #462.
- `codex/449-docs-docstrings`: follow-up branch for merged-status tracking,
  user-facing region-configuration notes, and focused basis docstring cleanup.
- `codex/449-state-vector-groups-design`: design-only branch for state-vector
  grouping, inner/outer layout, grouped priors, and layered partitions.
- `codex/449-layered-region-classes`: local follow-up adding a pure helper for
  intersecting multiple aligned region-class layers into composite labels. Not
  pushed.
- `codex/449-weight-first-basis`: additive interface cleanup that exposes
  weight-first generated-basis helpers and keeps the legacy `fp_all` functions
  as adapters.
- `codex/ogi-043-fixed-outer-layout`: design note for preserving legacy InTEM
  fixed outer regions while exposing them as grouped constrained partitions.
- `codex/ogi-041-filter-before-basis`: RHIME-only bugfix branch with one
  observation-filter stage before basis loading or generation and sensitivity
  construction.

## Design Notes

- Keep file loading outside the core constrained helper. The core should accept
  a 2D weight field and a 2D mask/region-class `xarray.DataArray`.
- Partition each mapped class independently, then offset labels globally so
  region labels never collide across classes.
- Preserve unmapped cells consistently with label `0`.
- Keep greedy axis-parallel repeated-bisection as the default split strategy
  rather than the legacy recursive weighted bucket splitter. Keep the strategy
  boundary explicit so inertial, quadtree-style, recursive weighted, or other
  partitioning steps can be substituted when they have tests and routing.
- Document `nbasis` allocation. Initial target: support explicit per-class
  allocation plus automatic allocation proportional to class total weight, with a
  minimum for non-empty mapped classes.

## Concrete Region And Partition Options

Do not change accepted `run_hbmcmc.py` inputs until #448 has landed. Once that
interface is available, the same region/partition options should be exposed
through `run-rhime`, `run-rhime-multisector`, and their Python equivalents. The
pre-#448 work should stay in pure helpers and lower-level Python APIs so it can
be tested without committing to config syntax.

### Region Sources

- Existing land/sea files. This is the smallest usable option because the
  legacy weighted algorithm already uses files such as
  `country-land-sea_{domain}.nc` and `country-EUROPE-UKMO-landsea-2023.nc`.
  Add a coordinate-preserving loader that returns a `region_classes`
  `DataArray`, rather than reusing the current NumPy-only
  `_weighted.load_landsea_indices`. This gives a direct lower-level replacement
  for "weighted, but constrained by land/sea" and should be easy to test with
  the existing synthetic #318 fixture plus one package land/sea file.
- Country or region masks from `country_file`. The workflows preserved in
  `~/Documents/ipython_histories/ipython_hist_26.py` and
  `ipython_hist_27.py` used country masks, PARIS country groupings, and
  country-fraction matrices. A concrete helper could load a named variable from
  a country file, optionally group country labels into larger named regions, and
  return `region_classes`. This is useful, but needs careful handling of ocean,
  unmatched country codes, fractional cells, and whether grouping follows
  postprocessing country-region definitions.
- User-specified inner rectangle. Add a helper that converts latitude/longitude
  bounds, two corner points, or grid-index bounds into a binary
  `region_classes` field. This would be useful for inner/outer experiments and
  realistic tests before full country-region configuration is available.
  Decide whether the outer cells are a second class, left unmapped, or routed to
  a separate fixed-outer-region path.
- Fixed InTEM outer-region files. Keep `fixed_outer_regions_basis` for backwards
  compatibility because the InTEM regions are familiar to users. A new, clearer
  interface can still expose the same files as a region source, but should
  preserve the known outer-region behaviour while allowing the inner region to
  use any split strategy.
- Direct user-supplied mask file and variable. For Python APIs this already
  exists as a `DataArray`. For CLI/config routes after #448, the minimal general
  form is likely a file path plus variable name plus optional unmapped values.
  This overlaps with land/sea and country files, so keep the implementation
  common.

### Partition And Grouping Options

- Keep greedy axis-parallel splitting as the current default because it is the
  cleaned-up prototype path and does not depend on the poorly designed legacy
  weighted recursion.
- Keep the existing recursive weighted splitter as an explicit compatibility
  strategy only.
- Add small partition-step adapters for quadtree-style splits and prototype
  inertial splits when there is a clear test case. The current `PartitionStep`
  boundary can already accept steps that return more than two child partitions.
- Keep three split inputs conceptually separate. The existing `weights` field is
  the contribution/importance field used for priority, half-weight cuts,
  allocation, and split-stopping thresholds. Optional split `geometry` is only
  for physical coordinate decisions such as axis selection and inertial
  projection. Physical cell area is a separate future allocation/input policy
  and should not silently multiply the contribution weights.
- Lat/lon-aware splitting should be opt-in at the split-step boundary. Passing
  `LatLonGridGeometry.from_dataarray(...)` to `AxisParallelSplitStep` or
  `InertialSplitStep` scales grid-cell centers into local north-south and
  east-west metre offsets for the currently selected partition. Axis-parallel
  cuts remain constant-row or constant-column cuts; geometry only changes which
  axis is considered physically longer. Inertial splitting uses the same local
  physical coordinates for covariance and projection. Default split steps
  continue to use row/column index geometry.
- Keep split stopping at the greedy orchestration boundary. A `PartitionStep`
  proposes child partitions; the greedy strategy decides whether to accept
  those children before updating the active queue.
- Split-stopping policies have distinct threshold semantics. `MinChildWeightShare`
  is a parent-relative balance guard: it rejects a proposed split when the
  lightest child falls below a configured share of the current parent partition
  weight. It does not prevent a very low-weight parent from splitting again if
  that split is balanced.
- `MinChildTargetWeightShare` is the class/source-total low-weight region rule.
  It rejects a proposed split when the lightest child would fall below a
  configured share of the equal-weight target region,
  `weights.sum() / target_regions`, for the class/source-local field passed to
  greedy partitioning. In formula form, accepted splits satisfy
  `min(child_weight) / (weights.sum() / target_regions) >= threshold`. This is
  the policy to use when RHIME-style experiments should avoid creating basis
  regions that are much smaller than the requested equal-weight target.
- Thresholds are not interchangeable between these policies. A value such as
  `0.02` or `0.1` means "share of current parent" for `MinChildWeightShare`, but
  "share of the equal-weight class/source target" for
  `MinChildTargetWeightShare`. For example, with 150 target regions, a threshold
  of `0.1` on `MinChildTargetWeightShare` is about `0.00067` of class/source
  total, not `0.1` of class/source total. The two policies can be composed when
  callers need both a minimum target-weight child and parent-relative child
  balance.
- When stopping is enabled, requested region counts are upper targets: the
  strategy may return fewer labels if remaining split candidates fail the
  policy.
- Freeze is the first supported rejected-split action. A rejected parent is
  marked done so deterministic split steps do not propose the same rejected
  children repeatedly. Requeue remains a possible future policy, but it needs
  explicit loop prevention and a reason that a later attempt could succeed.
  Fallback to a different split step or a more permissive split is also deferred
  until there is evidence that the extra behavior is needed.
- Current public routing is intentionally narrower than the algorithm layer.
  Lower-level Python calls can pass split-stopping policies such as
  `MinChildWeightShare`, `MinChildTargetWeightShare`,
  `MaxChildPCAEccentricity`, or `AllSplitAcceptancePolicies` through
  `GreedySplitStrategy` with an explicit partition step. The higher-level basis wrappers currently
  route only `split_acceptance="none"` and `split_acceptance="contrast_score"`.
  `.ini`, `run_hbmcmc.py`, `run_rhime`, and `run-rhime` do not yet expose
  parent-share or equal-target child-share thresholds as config options. Public
  config routing for those policies should wait for the surrounding basis option
  schema to settle.
- Treat simulated annealing, precomputed multiscale weights, and observation-
  weighted definitions as future optimizer/weight-builder work. They are useful
  sources from `~/Documents/basis_functions`, but they should not block the
  first usable region-source options.
- See `docs/plans/ogi_051_basis_weight_definitions.md` for the OGI-051 weight
  builder design. The short version is that alternative observation-aware
  weights belong before partitioning, must use retained observations only, and
  should not be routed through public config until filter-before-basis behavior
  and focused tests have landed on `devel`.
- Start tracking partition/group metadata alongside labels. Inner/outer regions,
  land/sea, country groups, and future layered masks should be represented as
  basis groups or coordinates so priors and posterior summaries can be applied
  by group. This likely needs a non-`xarray` internal representation before
  conversion to a `BasisFunctions` artifact.
- See `docs/plans/state_vector_grouping.md` for the proposed first grouped
  layout contract.

### Test Strategy

- Before #448: test pure loaders and helper functions with small fixtures, plus
  lower-level Python calls into `region_constrained_basis_function`.
- After #448: add real-world integration tests through `run_hbmcmc.py`,
  `run-rhime`, and `run-rhime-multisector`, using the same option schema where
  possible.
- For regression fixtures, prefer tiny deterministic arrays first. Larger frozen
  basis fixtures can be added later behind a pytest mark once the interface is
  stable.

## Status

- 2026-06-01: Prototype examples pushed without PR.
- 2026-06-01: Started #318 regression branch from `origin/devel`.
- 2026-06-01: Added a synthetic #318 regression test for
  `bucket_split_landsea_basis`. Current direct weighted land/sea labels do not
  cross land/sea classes, so this branch does not need a production fix.
- 2026-06-01: Mapper review found a separate coordinate-safety risk: when
  `fixed_outer_regions_basis` calls weighted with a cropped mask,
  `_mean_fp_times_mean_flux(..., drop=True)` can shift the weight-field origin
  while `_weighted.bucket_split_landsea_basis` still slices the full land/sea
  file from index zero. The #449 helper should preserve coordinates and align
  masks before converting to arrays.
- 2026-06-01: Added #449 core helper in
  `openghg_inversions.basis.algorithms._constrained`.
  `region_constrained_basis(weights, region_classes, nbasis, ...)` aligns 2D
  `xarray.DataArray` inputs exactly, treats non-null class values as mapped,
  preserves unmapped cells as output label `0`, partitions each class
  independently, and offsets labels globally.
- 2026-06-01: Added `allocate_nbasis_by_class` with explicit allocation mapping
  support and automatic `weight`/`area` allocation. Automatic allocation enforces
  `min_regions_per_class`, raises when the requested `nbasis` is below the class
  minimum or above mapped-cell capacity, and falls back from weight to area when
  all class weights are zero.
- 2026-06-01: Added `SplitStrategy` plus the initial
  `AxisAlignedWeightedSplitStrategy`. This kept the current axis-parallel
  weighted split available while leaving a direct substitution point for future
  inertial or quadtree-style strategies. No inertial split implemented.
- 2026-06-01: Subagent review fixes for #449 core: all-zero class weights now
  split using an area surrogate, explicit allocations are checked against
  mapped-cell capacity, and `SplitStrategy` is exported for typed custom
  strategies.
- 2026-06-02: User review noted the existing recursive weighted algorithm is a
  poor default for the constrained helper. Prototype research found the clean
  non-recursive replacement in
  `~/Documents/inversions/src/inversions/basis_algorithms.py`: greedy repeated
  bisection that repeatedly splits the highest-weight current part, with
  axis-parallel and inertial split functions. The #449 core now uses a
  cleaned-up `GreedySplitStrategy` composed with `AxisParallelSplitStep` as the
  constrained default and keeps `AxisAlignedWeightedSplitStrategy` as an
  explicit compatibility strategy.
- 2026-06-01: Added #325 caller-facing adapter
  `region_constrained_basis_function` and
  `basis_algorithm="region_constrained"`.
  It uses the #449 core helper with a caller-supplied `region_classes`
  `DataArray`; country/region file loading remains outside the algorithm. The
  wrapper now passes through `region_classes`, `region_allocation`, and
  `min_regions_per_class` only for this algorithm.
- 2026-06-01: Subagent review completed on the stacked #325 branch. Fixes made:
  `region_constrained` now works through `fixed_outer_regions_basis` when
  `region_classes` is supplied; all-zero class weights use an area surrogate for
  splitting; explicit allocation is checked against mapped-cell capacity; and
  `SplitStrategy` is exported. The apparent positional API issue for
  the legacy compressed names `quadtreebasisfunction`/`bucketbasisfunction` was
  present on `origin/devel`, but new wrapper options were moved to the end of
  `basis_functions_wrapper` to avoid changing its existing positional order.
- 2026-06-02: Research in `~/Documents/basis_functions` found bucket-threshold
  optimizers, simulated annealing experiments, and numba notebook fragments.
  These are useful future research inputs but are not clean drop-in
  `SplitStrategy` candidates yet.
- 2026-06-02: Histories `~/Documents/ipython_histories/ipython_hist_26.py` and
  `ipython_hist_27.py` preserve country/region partition workflows: land/sea
  and PARIS country partitions, country-fraction matrix labeling, and
  `outer_region_definition_EUROPE.nc` with inner region `region == 6`. Follow-up
  #325 work should keep file loading outside the pure helper and add tiny
  land/sea, country, and outer-region fixtures that assert labels do not cross
  classes.
- 2026-06-02: Added PEP-style public function names
  `quadtree_basis_function`, `bucket_basis_function`, and
  `region_constrained_basis_function`. The old compressed names
  `quadtreebasisfunction` and `bucketbasisfunction` remain as deprecated aliases
  with warnings for compatibility; the draft-only compressed
  `regionconstrainedbasisfunction` alias was removed before PR.
- 2026-06-02: Extracted a `PartitionStep` protocol and `AxisParallelSplitStep`
  from the greedy constrained strategy. Greedy orchestration now uses a small
  priority-queue wrapper that pops the highest-weight partition first and can
  accept split steps that return more than two child partitions without
  overshooting the requested target count.
- 2026-08-10: Separated basis-group orchestration from group-local algorithms.
  `_constrained` now owns masks, allocation, per-group dispatch, validation,
  and relabelling. `_partition` owns geometry, steps, acceptance policies, the
  public `greedy_partitioning` engine, and its `GreedySplitStrategy` adapter;
  `_weighted` owns `AxisAlignedWeightedSplitStrategy`. Axis-parallel behavior
  is composed explicitly at higher-level defaults.
- 2026-06-02: PR #462 merged the stacked #318/#449/#325 implementation into
  `devel`. The direct weighted land/sea regression did not expose a boundary
  crossing bug, so production weighted behavior was left unchanged; the separate
  coordinate-safety risk remains tracked as future interface cleanup.
- 2026-06-02: Started `codex/449-docs-docstrings` to document the merged
  `region_constrained` path. Current note: the pure Python basis API accepts an
  already loaded `region_classes` `DataArray`, but `.ini`/`run_hbmcmc.py` users
  do not yet have a file-loading/config hook for these masks.
- 2026-06-02: Added `docs/plans/state_vector_grouping.md` on
  `codex/449-state-vector-groups-design`. The note proposes a small internal
  `BasisLayout`/`BasisPartition` concept plus xarray state coordinates
  (`basis_group`, `basis_partition`, `region_in_partition`) so inner/outer,
  land/sea, country groups, and layered masks can share one state axis while
  retaining enough metadata for grouped priors and postprocessing.
- 2026-06-02: Started `codex/449-layered-region-classes` from `origin/devel`.
  This adds a small lattice-style helper for crossing loaded masks such as
  land/sea by inner/outer while keeping file loading and runner configuration
  outside the algorithm.
- 2026-06-02: Started `codex/449-weight-first-basis` from `origin/devel`.
  Extracted `basis_weights_from_fp_all` and weight-first generated-basis helpers
  for quadtree, weighted bucket, and region-constrained bases. The existing
  public `fp_all` functions now delegate to those helpers, so callers can start
  moving toward explicit weight fields without losing the legacy adapter path.
- 2026-07-05: Added `docs/plans/fixed_outer_regions_grouping.md` for OGI-043.
  The decision keeps `fixed_outer_regions_basis` as the legacy compatibility
  route while defining the grouped RHIME path as a constrained allocation:
  fixed outer classes get one basis region each and the inner class receives the
  requested generated regions.
- 2026-07-05: Started OGI-041 on `codex/ogi-041-filter-before-basis`.
  Modern RHIME preparation now applies observation filters once, before either
  loading or generating the basis and before sensitivity construction. The same
  filtered merged data, sites, and averaging metadata flow through both stages.
  The legacy fixedbasis preparation path remains unchanged and should be handled
  by a separate task if that behavior needs to move.
- 2026-06-21: Added greedy split stopping through a lower-level
  `SplitAcceptancePolicy` hook and `MinChildWeightShare` policy. Rejected
  splits freeze the selected parent partition, so the requested class-local
  region count is treated as an upper target when stopping is configured.
- 2026-06-21: Corrected the split-stopping design distinction after PR #480.
  `MinChildWeightShare` is a balance guard, while `MinChildTargetWeightShare`
  handles class/source-total low-weight region stopping using the class-local
  `weights.sum() / target_regions` equal-region denominator.
- 2026-06-24: Started the OGI-048 implementation path for lat/lon-aware split
  geometry. The narrow API adds `LatLonGridGeometry` as an opt-in geometry
  object for axis-parallel and inertial split steps, while keeping weights,
  split stopping, allocation, and public config routing unchanged.
- 2026-07-05: Added the OGI-051 basis-weight design note. The decision is to
  keep the current `mean_fp_times_mean_flux` builder as the only
  production-routed default for now, prototype alternative builders at the
  weight-first helper layer, and defer public config routing until retained
  observation filtering and focused tests are merged.
- 2026-07-05: Documented OGI-060 split-stopping semantics in the user-facing
  basis notes. The docs now state that parent-relative and equal-target
  thresholds are not interchangeable, that requested region counts become upper
  targets when stopping is enabled, and that only `"none"` and
  `"contrast_score"` are currently routed through the higher-level wrappers.
