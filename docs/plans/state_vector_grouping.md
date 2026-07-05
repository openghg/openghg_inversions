# State-Vector Grouping Design Note

## Purpose

Issue #449 exposed a broader requirement than "labels must not cross a mask".
For inner/outer regions, land/sea splits, country groups, and future layered
partitions, we need to know which state-vector components came from which
partition. That grouping should support:

- separate priors by partition group;
- separate posterior summaries by partition group;
- separate optimization routes for inner and outer regions, including different
  weights;
- clear serialization in retained `BasisFunctions` artifacts;
- legacy output compatibility where the model still expects a flat `region`
  dimension.

PR #359 is the motivating inner/outer example: inner and outer components may be
optimized with different weights. Treating inner/outer as only another mask
loses the fact that those labels may need different priors and separate
diagnostics.

## Current State

`BasisOperator` already has the right low-level shape:

- `BasisMeta.state_dim` names the reduced state axis.
- `BucketBasisOperator` stores a flat label map and a dummy matrix with dims
  `(*grid_dims, state_dim)`.
- `MultiSourceBucketBasisOperator` uses a gathered MultiIndex state coordinate
  over `(source, region_in_source)`.
- Postprocessing helpers can respect `basis_functions.operator.meta.state_dim`
  and fall back to legacy `region` or `nx`.

What is missing is semantic state metadata. Today a state coordinate says
"region 0" or `(source, region_in_source)`, but not "this region is land",
"this region is inner", or "this region belongs to the InTEM outer partition".

## Proposed Internal Representation

Add an internal, non-xarray-first object for grouped basis construction:

```python
@dataclass(frozen=True)
class BasisPartition:
    name: str
    labels: xr.DataArray
    group: str
    weight_name: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class BasisLayout:
    partitions: tuple[BasisPartition, ...]
    state_dim: str = "state"
```

The layout object would be responsible for combining partition-local labels into
one global state axis. This keeps algorithm output simple while giving the
wrappers a single place to record group membership, state-label offsets, and
later, per-group prior defaults.

The first implementation does not need to support every layout. It should
support:

- one partition, one group: current behavior;
- multiple disjoint partitions, one global state axis: land/sea or country
  groups;
- inner plus outer partitions, one global state axis with group metadata:
  useful for fixed InTEM outer regions and PR #359 style workflows.

`state_dim="state"` is the generic operator default. RHIME and legacy wrapper
paths can still request `state_dim="region"` when they need the existing public
dimension name.

## Xarray Encoding

The retained `BasisFunctions` artifact should encode group metadata as state
coordinates rather than only attrs. Coordinates are easier to select in model
building and postprocessing.

For a single flat state dimension:

- dim: the final operator state dimension, usually `state` internally and
  `region` for RHIME/legacy-facing wrappers;
- coordinates:
  - state coordinate: numeric or label values used by the sampler;
  - `basis_group(state_dim)`: e.g. `"inner"`, `"outer"`, `"land"`, `"sea"`;
  - `basis_partition(state_dim)`: stable partition name, e.g. `"uk_inner"`,
    `"intem_outer"`;
  - `region_in_partition(state_dim)`: local integer region number.

These metadata coordinates must be indexed by the final operator state
coordinate order, not by raw positive labels in the flat basis map. This matters
because `BucketBasisOperator` can remap raw labels to zero-based coordinates with
`region_labels="range0"`. The layout builder should create metadata after the
operator state coordinate has been chosen, or carry an explicit raw-label to
state-coordinate mapping and validate it.

For multi-source layouts, keep the existing gathered MultiIndex approach and add
the same metadata as coordinates on the gathered state dimension. The
MultiIndex levels should remain structural, not semantic.

This avoids a special "inner/outer state vector" model. It is still one state
axis, but with coordinates that let model builders and postprocessing select
groups.

DataTree serialization must round-trip the metadata coordinates. Current bucket
operators serialize `basis_flat` and rebuild state coordinates on load; grouped
state metadata will need an explicit serialized state-metadata dataset or
equivalent attrs plus tests proving `BasisFunctions.save/load` preserves
`basis_group`, `basis_partition`, and `region_in_partition`.

## Model And Prior Interface

The first model-facing API should be simple and explicit:

```python
x_priors_by_group = {
    "inner": {"pdf": "normal", "mu": 1.0, "sigma": 0.5},
    "outer": {"pdf": "normal", "mu": 1.0, "sigma": 0.2},
}
```

Model construction can broadcast each prior to the subset of the state axis
where `basis_group == group`, then concatenate back to the full state axis.
The final model must still expose the existing full ordered `x(state_dim)` role
used by postprocessing. A grouped-prior implementation should either create one
full `x` random variable or a deterministic `x` assembled from per-group random
variables, but downstream code should continue to see a complete state vector.

Validation requirements:

- every state belongs to exactly one prior group;
- every configured group matches at least one state, unless explicitly allowed
  as unused;
- no state is covered by two grouped priors;
- final `x` ordering exactly matches the canonical state coordinate used by `H`.

## Layering And Masks

Layered mask construction should feed this layout, not bypass it. A reasonable
first version is:

1. Convert each layer to a boolean or categorical `DataArray`.
2. Compute the lattice/intersection of supplied layers.
3. Drop empty intersections and optionally flag tiny or disconnected parts.
4. Split each resulting part with a chosen partition strategy.
5. Build a `BasisLayout` with group coordinates describing where each part came
   from.

This gives one route for land/sea, countries, user rectangles, and inner/outer
masks. It also makes problematic pieces visible before optimization.

## Phased Work

1. Add a small `BasisLayout`/`BasisPartition` module and tests that combine two
   tiny disjoint partitions into one state coordinate with group metadata.
2. Teach `BasisFunctions.from_flat_basis` or a new constructor to preserve
   state metadata coordinates in `BucketBasisOperator`.
3. Extend operator DataTree serialization so state metadata coordinates survive
   `BasisFunctions.save/load`.
4. Route `region_constrained` and fixed-outer-region generation through the
   layout builder, preserving current flat labels by default.
5. Add postprocessing helpers that select posterior/prior summaries by
   `basis_group`.
6. Add grouped prior support in RHIME model construction while preserving a full
   ordered `x` state vector.

## Open Questions

- Should `basis_group` be a required coordinate for all generated bases, with a
  default value like `"emissions"`, or only present for grouped layouts?
- Should InTEM outer regions preserve their familiar fixed labels as
  `region_in_partition`, `basis_partition`, or a dedicated coordinate?
- How should disconnected pieces with the same country or group be represented:
  one semantic group with multiple partitions, or separate partitions with a
  shared group?
- Should group metadata live in `BasisOperator` only, or also in
  `FluxWeightedBasis.metadata` for fast inspection before loading the full
  operator matrix?
