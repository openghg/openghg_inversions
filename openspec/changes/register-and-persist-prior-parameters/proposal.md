# Proposal

## Why

Review of [#866](https://github.com/openghg/openghg_inversions/pull/866)
identified that array prior parameters bypass model-data registration, and its
new sample records lose lazy parameter values by retaining only their shapes.
Resolve [#872](https://github.com/openghg/openghg_inversions/issues/872) before
#866 merges, using the existing registered-model and inference-data boundaries.

## What Changes

- Specify the division of responsibility between `CoordRegistry` (scientific
  indexes, auxiliary coordinates, PyMC range coordinates and restoration) and
  registered model data (fixed numerical arrays). Constructed-prior metadata
  retains scalar numerical parameters inline.
- Register array prior parameters while scientific labels are available; retain
  scalar arguments and broadcasting, canonical alignment and active state
  selection. Keep stochastic parameters and derived expressions in the graph.
- Derive effective lognormal parameters through one symbolic conversion path for
  scalar, array and symbolic moments, shared by ordinary and reparameterized
  construction. Retain fixed moments inline or through array-data references.
- Materialize related lazy inputs together at the explicit model boundary,
  borrowing caller arrays. Save actual values and coordinates through existing
  inference-data serialization; model metadata references registered inputs
  and retains resolved scalars inline without recomputing caller graphs or
  embedding numerical arrays in JSON.
- Record only priors used by the constructed model. Omit unused prior metadata
  and references without materializing configured lazy parameters of disabled
  components.
- Make full and saved standard execution consume the same corrected shared
  prior path; add a cross-reference in the existing sampled-handoff proposal.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `rhime-model-composition`: Extend registered coordinate ownership and add fixed
  prior input registration, parameterization, execution and persistence contracts.

The pending `sampled-recipe-handoff` capability remains owned by
`replace-recipe-workflow-with-reusable-handoffs`; its cross-reference consumes
this amendment without duplicating it.

## Impact

Implementation affects shared prior/state alignment helpers, their direct
callers, model-record encoding/decoding and the existing trace persistence path.
No new registry, serializer framework or dependency is needed. Coordinate issue
#856, general model resizing, `Interpolated` registration/persistence and
additional saved state-activity support remain outside this change. This scope
does not remove existing runtime handling of `Interpolated` priors.

The initial spec-only push was reviewed before implementation was authorized
in the same PR. The change remains stacked on `codex/standard-reusable-handoffs`
at `7dad0b05`, with runtime checks, maintained usage documentation and
`newsfragments/872.bugfix.md` delivered together. Durable specs remain unchanged
until delivery is validated; planning approval does not establish implemented
behavior.
