# Mask-Constrained Basis Work Plan

## Scope

This tracks the branch sequence for issues #318, #449, #325, and the split-strategy
part of #340.

## Branches

- `codex/basis-prototype-examples`: pushed as a reference branch only. No PR opened.
- `codex/318-landsea-regression`: targeted regression test for current weighted
  land/sea behavior. Intended to prove whether the existing algorithm crosses
  land/sea classes before changing production code.
- `codex/449-mask-constrained-core`: planned follow-up branch for a pure
  mask/region-constrained helper and split-strategy protocol.
- `codex/325-region-mask-basis`: planned follow-up branch for country/region-mask
  callers built on the #449 helper.

## Design Notes

- Keep file loading outside the core constrained helper. The core should accept
  a 2D weight field and a 2D mask/region-class `xarray.DataArray`.
- Partition each mapped class independently, then offset labels globally so
  region labels never collide across classes.
- Preserve unmapped cells consistently with label `0`.
- Start with the current axis-parallel weighted split strategy. Do not implement
  inertial partitions yet, but keep the strategy boundary explicit so an inertial
  split, quadtree-style split, or other partitioning step can be substituted.
- Document `nbasis` allocation. Initial target: support explicit per-class
  allocation plus automatic allocation proportional to class total weight, with a
  minimum for non-empty mapped classes.

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
- 2026-06-01: Added `SplitStrategy` plus
  `AxisAlignedWeightedSplitStrategy`. This keeps the current axis-parallel
  weighted split as the default while leaving a direct substitution point for
  future inertial or quadtree-style strategies. No inertial split implemented.
- 2026-06-01: Subagent review fixes for #449 core: all-zero class weights now
  split using an area surrogate, explicit allocations are checked against
  mapped-cell capacity, and `SplitStrategy` is exported for typed custom
  strategies.
