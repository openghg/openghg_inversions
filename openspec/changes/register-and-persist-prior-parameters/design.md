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
fixed inputs of constructed priors using inline scalars and existing model data
and trace storage for arrays.
Apply the fix at shared prior construction so full and saved standard runs agree.

**Non-Goals:** A second registry, general tensor-expression serialization,
automatic data deduplication across unrelated priors, arbitrary model resizing,
#856, or extending the saved frontend to accept currently rejected state-activity
policies. Active-state behavior is checked through its already supported model
construction interface, including default zero-sensitivity pruning.
`Interpolated` registration/persistence is outside this change; its existing
runtime handling is not slated for removal.

## Decisions

### 1. Keep alignment labelled until registration

Align state-valued parameters to the canonical full-state coordinate before
stripping labels. Preserve lazy backing until related required inputs can be
materialized together. Register the canonical supplied arrays once, then use
active-index expressions to select distribution inputs. The full-state coordinate
and existing activity information explain the selected state's relationship to
those values without storing another active-only numerical parameter array.

Numerical scalars, including zero-dimensional numerical values, remain ordinary
scalar arguments and are captured inline in constructed-prior metadata.
Positional arrays acquire the dimensions owned by their component; a
one-dimensional array of length one remains an array. Do not broadcast scalars
into stored state arrays or assign every parameter the distribution's dimensions.
The alternative of registering already stripped NumPy arrays loses scientific
interpretation too early.

Keep this correction local to shared prior/alignment helpers and their callers.
Audit all `parse_prior` callers, including direct scalar priors, mismatch, offset
and experimental components. Reuse or relocate the existing registration helper
if its current import direction prevents reuse from `priors.py`; do not duplicate
it or silently create another coordinate registry.

### 2. Register arrays, retain scalar constants and derive expressions in the graph

Use the existing model-data boundary for fixed numerical arrays. Ordinary
numerical scalars need no named `pm.Data` node or lookup reference. Recognize
symbolic inputs before scalar/array coercion and retain their graph dependencies;
a scalar-shaped symbolic parameter is not a numerical constant. Distribution
selectors and controls such as `pdf`, `reparameterise` and `dims` are metadata,
not numerical prior parameters.

For arithmetic lognormal moments, use one PyTensor conversion in model
construction. Scalar moments enter as constants, array moments as registered
inputs, and symbolic moments as existing expressions. All use:

```text
variance_log = log(1 + (stdev / mean)^2)
mu = log(mean) - variance_log / 2
sigma = sqrt(variance_log)
```

Both ordinary LogNormal construction and `exp(mu + sigma * latent)` consume these
expressions, including mixed scalar/array inputs. Do not split model conversion
into NumPy and PyTensor branches based on parameter shape. Direct `mu`/`sigma`
input retains its own parameterization. Preserve the existing numerical conversion
helper's public behavior where it is used outside graph construction. Storing
both arithmetic and log-space arrays adds duplicate payloads; evaluating symbolic
inputs would erase dependencies. The latent's internal zero and one constants
need no model-data registration.

### 3. Record names and policy; save values through the trace

Give registered prior arrays deterministic names scoped to their scientific
variable and parameter. A reference such as `{"data": "x_prior_mean"}` is enough
to associate a model-record field with `constant_data`; exact spelling is an
implementation detail. Retain scalar values inline, for example:

```json
{"pdf": "normal", "mu": {"data": "x_prior_mu"}, "sigma": 0.2}
```

Record distribution and parameterization names, reparameterization policy and
the association with existing activity metadata.
These are ordinary descriptive mappings, not a second data registry.

Capture resolved scalar values, including applicable resolved defaults, and array
associations at construction, and pass them explicitly to record/output owners.
Include only priors actually used by the constructed model: `bc_prior` with
`use_bc=False`, or `offset_prior` with offsets disabled, produces no saved prior
metadata or parameter references.
Construction owns this selection; being inside a model context alone does not
prevent `RhimeModelSpec.to_record` from independently walking unused settings.
Do not materialize unused lazy parameters to record configuration, or add a
separate filtering/validation framework.

Preserve the constructed associations through the sampled trace and existing
metadata channels. Replace the configured-prior walk and both shape-only lazy
and eager-array JSON encoding in `RhimeModelSpec.to_record`; reuse captured
scalars and array references for full-run output metadata as well as saved sample
metadata.

The existing trace codec owns numerical storage and MultiIndex round trips.
On replay, load that trace and resolve required prior-array references before
decoding model choices for result construction. Scalar constants need no data
lookup. Decode arrays into ordinary labelled values without building a graph or
executing recorded callables. Missing required array references fail at the
owning writer/reader; no JSON numerical-array fallback or new sidecar is needed.
Symbolic parameters retain descriptive associations; this is not a promise to
reconstruct arbitrary hierarchical models from metadata.

### 4. Keep the execution boundary explicit and shared

Materialize related prior parameters, required auxiliary coordinates and other
model inputs sharing their lazy graph together at the owning construction
boundary. Include supported lazy zero-dimensional numerical inputs in that same
computation, then capture their resulting scalar values inline. Metadata writing
does not compute the original lazy scalars again. Use `to_dense` where sparse
chunk payloads require densification and reuse the existing joint materialization
pattern. Install results in locally owned containers; do not load or persist
borrowed caller arrays in place.

Full, prepared-input and saved standard routes already meet at `build_standard`
in #866. Keep that shared flow and correct the lower-level prior helpers rather
than add a saved-route registration pass. If the complete custom-builder hook
remains supported, those builders own their own numerical boundary; this change
does not require preserving that hook during #866's simplification.

## Risks / Trade-offs

- Losing labels while slicing active states: register canonical inputs first;
  check a reversed-label example with partial activity independently.
- Registering a graph expression as fixed data: check a hierarchical parameter
  still depends on its parent random variable, including scalar-shaped expressions.
- Losing the scalar/array distinction: check scalar-only priors add no
  prior-parameter data variables and a length-one labelled array retains its axis.
- Reexecuting a shared lazy graph through metadata or coordinates: use counted
  delayed inputs, including a supported lazy scalar, with shared dependencies
  and verify counts across construction, sampling, saving and reload, with
  borrowed inputs unchanged.
- Serializing unused request settings: configure a disabled component with a
  counted lazy prior; saving must succeed with no prior metadata or references
  for that component and zero executions of its lazy graph.
- Accidentally broadening saved state-activity support: preserve #866's explicit
  rejection and check active construction separately from saved frontend support.

## Migration Plan

1. Review this planning-only commit in a PR targeting
   `codex/standard-reusable-handoffs`. Do not promote the durable spec yet.
2. After maintainer spec review, use the proposal workflow to create implementation
   tasks, then implement the shared fix and record integration in the same PR.
3. Check ordinary and reparameterized densities against independent calculations;
   perform a small real sample/save/reload with labelled Dask-backed arrays and
   auxiliary coordinates, retaining scalar parameters inline. Include focused
   scalar-only, length-one array, symbolic and unused-lazy-prior checks.
   Compare full and saved standard construction without requiring identical draws.
4. Update maintained usage documentation and add `newsfragments/872.bugfix.md`,
   then merge this child PR into #866 before #866 completes. Reconcile/promote
   durable specs only with validated delivery; no compatibility reader for #866's
   unshipped lossy model-record representation is required.

The initial push contained only the spec for maintainer review. That review is
complete and implementation is authorized; `tasks.md` tracks delivery and
validation in the same stacked PR.

## Construction ordering clarification

Enabled prior inputs may materialize jointly with sensitivities before exact-zero
columns are detected and their states fixed rather than sampled. This preserves
shared lazy computation even when sensitivity evaluation later makes an enabled
prior unnecessary. Disabled BC and offset components are excluded before the
boundary. Prior metadata still describes only priors actually constructed.
