# Fixed Outer Regions Grouping Note

## Purpose

`fixed_outer_regions_basis` should stay available as the compatibility route for
InTEM-style outer regions. Those files are familiar to users and current
`run_hbmcmc.py`-era configs still depend on the generated-basis names
`quadtree` and `weighted`.

The future RHIME/grouped-basis route should expose the same concept more
directly: fixed outer regions are not an opaque mask around an inner generated
basis. They are explicit basis partitions. Each fixed outer class receives one
basis region, while the inner class receives the requested generated regions
from the selected constrained-basis split strategy.

## Current Compatibility Behavior

`fixed_outer_regions_basis` currently:

1. Loads `outer_region_definition_{domain}.nc`.
2. Treats the largest region value as the inner inversion region.
3. Builds a boolean mask for that inner region.
4. Runs one registered generated-basis algorithm inside the mask.
5. Splices generated inner labels back into the fixed outer map.
6. Shifts labels because the InTEM file uses zero-based region IDs.

This behavior should not be removed or silently changed. It is the compatibility
surface for users who already understand the InTEM outer-region files.

The current route also passes through `make_basis_functions` and then into a
retained `BasisFunctions` object. Today that retained object preserves the flat
map and basic artifact metadata, but it does not have a place to serialize
semantic state metadata such as inner/outer group membership.

## Target Grouped Layout

The grouped route should model fixed outer regions as a constrained/layout
problem:

1. Load the outer-region file as a 2D `region_classes` field with coordinates
   aligned to the weight grid.
2. Identify the inner class. The compatibility default is still the largest
   InTEM region value; a future config can make this explicit.
3. Build an explicit allocation mapping:
   - every outer class maps to `1`;
   - the inner class maps to the requested inner-region count.
4. Call the constrained basis path with that explicit allocation and the chosen
   inner split strategy.
5. Build grouped state metadata from the class-local output.

The generated state axis should include at least:

| Coordinate | Outer states | Inner states |
| --- | --- | --- |
| `basis_group` | `outer` | `inner` |
| `basis_partition` | `intem_outer_<label>` | `intem_inner` |
| `region_in_partition` | original InTEM label | inner local label |

The total state count is therefore:

```text
number_of_outer_classes + requested_inner_regions
```

not just `requested_inner_regions`. This is the main behavioral distinction
that should be visible in RHIME artifacts and postprocessing.

## Label Policy

Flat label arrays may keep legacy-friendly positive integer labels, but label
numbers alone should not carry the semantics. Code that needs to select inner
or outer components should use the grouped metadata coordinates. This avoids
binding future priors, diagnostics, and postprocessing to raw label offsets.

The metadata must be attached after the final operator state coordinate order is
known. `BucketBasisOperator` can remap raw labels to a zero-based `region`
coordinate, so a raw InTEM label is not necessarily the same thing as the final
state coordinate value.

For legacy compatibility, `fixed_outer_regions_basis` can continue returning a
plain `basis(lat, lon, time)` field. The grouped route should prefer a retained
`BasisFunctions` artifact whose operator state dimension carries
`basis_group`, `basis_partition`, and `region_in_partition`.

## Implementation Boundary

Do not add grouped fixed-outer behavior by making `fixed_outer_regions_basis`
more complex. That function is the legacy flat-map adapter.

The implementation should wait for, or introduce, a grouped layout/metadata
boundary that can:

- build an explicit class allocation for fixed outer and inner classes;
- map raw class labels to the final operator state coordinate order;
- persist state metadata through `BasisFunctions.save/load`;
- leave the old flat-map compatibility route covered by its existing regression
  tests.

## RHIME Interface Policy

After the runner/config option schema is ready, RHIME should expose fixed outer
regions as a basis layout option, not as a hidden postprocessing side effect.
A minimal spec should contain:

- outer-region source: default package InTEM file or a user path;
- inner class selection: default `max(region)` compatibility behavior or an
  explicit region value;
- inner basis algorithm: initially `region_constrained` with explicit split
  strategy options;
- inner basis count: class-local requested count for the inner class;
- outer prior group and inner prior group names.

This keeps `run_hbmcmc.py` compatibility separate from the new grouped RHIME
surface and avoids changing the accepted legacy `quadtree` and `weighted`
options.

## Validation For Implementation

When this moves from design to code, use a tiny deterministic fixture that has
two fixed outer classes and one inner class. The test should assert:

- each outer class produces exactly one state;
- the inner class produces the requested number of states, subject to any split
  stopping policy;
- labels never cross the fixed outer/inner class boundaries;
- grouped metadata selects the expected outer and inner states;
- `BasisFunctions.save/load` preserves grouped metadata coordinates.

The existing `fixed_outer_regions_basis` regression should remain in place as a
compatibility test unless the user explicitly approves changing that legacy
route.
