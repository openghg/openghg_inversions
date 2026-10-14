# Design

## Context

See [proposal.md](proposal.md) for motivation and delivery scope. At #866 head
`7dad0b05`, `active_prior_args` aligns and materializes parameters individually,
then drops their labels; `parse_prior` passes their numerical values directly to
distributions. Lognormal moment conversion uses NumPy, and reparameterization
coerces effective parameters through `pm.floatX`. Sample model records retain
lazy array shapes instead of values and embed eager arrays in JSON.

The existing `registered_model`, `add_model_data`, coordinate restoration and
`save_trace` / `load_trace` already provide the relevant owners. This change spans
alignment, graph construction and record decoding, so these decisions need
review before implementation.

## Goals / Non-Goals

**Goals:** Preserve the supplied parameterization, scientific alignment and actual
fixed inputs of constructed priors using existing model data and trace storage.
Apply the fix at shared prior construction so full and saved standard runs agree.

**Non-Goals:** A second registry, general tensor-expression serialization,
automatic data deduplication across unrelated priors, arbitrary model resizing,
#856, or extending the saved frontend to accept currently rejected state-activity
policies. Active-state behavior is checked through its already supported model
construction interface, including default zero-sensitivity pruning.

## Decisions

### 1. Keep alignment labelled until registration

Align state-valued parameters to the canonical full-state coordinate before
stripping labels. Preserve lazy backing until related required inputs can be
materialized together. Register the canonical supplied values once, then use
active-index expressions to select distribution inputs. The full-state coordinate
and existing activity information explain the selected state's relationship to
those values without storing another active-only numerical parameter array.

Scalars remain zero-dimensional; positional arrays acquire the dimensions owned
by their component. Support grids retain a separate support axis. Do not assign
every parameter the distribution's dimensions. The alternative of registering
already stripped NumPy arrays loses scientific interpretation too early.

Keep this correction local to shared prior/alignment helpers and their callers.
Audit all `parse_prior` callers, including direct scalar priors, mismatch, offset
and experimental components. Reuse or relocate the existing registration helper
if its current import direction prevents reuse from `priors.py`; do not duplicate
it or silently create another coordinate registry.

### 2. Register supplied fixed parameters and derive expressions in the graph

Use the existing model-data boundary for supplied fixed numerical parameters.
Recognize symbolic inputs before numerical coercion and retain their graph
dependencies. Distribution selectors and controls such as `pdf`, `reparameterise`
and `dims` are metadata, not numerical prior parameters.

For arithmetic lognormal moments, register `mean` and `stdev` and derive:

```text
variance_log = log(1 + (stdev / mean)^2)
mu = log(mean) - variance_log / 2
sigma = sqrt(variance_log)
```

Both ordinary LogNormal construction and `exp(mu + sigma * latent)` consume these
expressions. Direct `mu`/`sigma` input retains its own parameterization. Preserve
the existing numerical conversion helper's public behavior where it is used
outside graph construction. Storing both arithmetic and log-space arrays adds
duplicate payloads; evaluating symbolic inputs would erase dependencies.

### 3. Record names and policy; save values through the trace

Give registered prior inputs deterministic names scoped to their scientific
variable and parameter. A reference such as `{"data": "x_prior_mean"}` is enough
to associate a model-record field with `constant_data`; exact spelling is an
implementation detail. Record distribution and parameterization names,
reparameterization policy and the association with existing activity metadata.
These are ordinary descriptive mappings, not a second data registry.

Build those associations from constructed priors and registered inputs, and pass
them explicitly to record/output owners. Preserve them through the sampled trace
and existing metadata channels. Do not walk the original lazy request again to
recover values. Replace both the shape-only lazy encoding and eager-array JSON
encoding in `RhimeModelSpec.to_record`; reuse the registered values when producing
full-run output metadata as well as saved sample metadata.

The existing trace codec owns numerical storage and MultiIndex round trips.
On replay, load that trace and resolve required prior-data references before
decoding model choices for result construction. Decode into ordinary labelled
values without building a graph or executing recorded callables. Missing required
references fail at the owning writer/reader; no JSON numerical-array fallback or
new sidecar is needed. Symbolic parameters retain descriptive associations;
this is not a promise to reconstruct arbitrary hierarchical models from metadata.

### 4. Keep the execution boundary explicit and shared

Materialize related prior parameters, required auxiliary coordinates and other
model inputs sharing their lazy graph together at the owning construction
boundary. Use `to_dense` where sparse chunk payloads require densification and
reuse the existing joint materialization pattern. Install results in locally
owned containers; do not load or persist borrowed caller arrays in place.

Full, prepared-input and saved standard routes already meet at `build_standard`
in #866. Keep that shared flow and correct the lower-level prior helpers rather
than add a saved-route registration pass. Let complete custom builders continue
to own their own numerical boundary.

## Risks / Trade-offs

- Losing labels while slicing active states: register canonical inputs first;
  check a reversed-label example with partial activity independently.
- Registering a graph expression as fixed data: check a hierarchical parameter
  still depends on its parent random variable.
- Reexecuting a shared lazy graph through metadata or coordinates: use counted
  delayed inputs with shared dependencies and verify counts across construction,
  sampling, saving and reload, with borrowed inputs unchanged.
- Accidentally broadening saved state-activity support: preserve #866's explicit
  rejection and check active construction separately from saved frontend support.

## Migration Plan

1. Review this planning-only commit in a PR targeting
   `codex/standard-reusable-handoffs`. Do not promote the durable spec yet.
2. After maintainer spec review, use the proposal workflow to create implementation
   tasks, then implement the shared fix and record integration in the same PR.
3. Check ordinary and reparameterized densities against independent calculations;
   perform a small real sample/save/reload with labelled Dask-backed moments and
   auxiliary coordinates, plus focused scalar, support-grid and symbolic checks.
   Compare full and saved standard construction without requiring identical draws.
4. Update maintained usage documentation and add `newsfragments/872.bugfix.md`,
   then merge this child PR into #866 before #866 completes. Reconcile/promote
   durable specs only with validated delivery; no compatibility reader for #866's
   unshipped lossy model-record representation is required.

The initial push contains no implementation or runnable tests. Implementation
tasks are intentionally deferred until the spec has been reviewed.
