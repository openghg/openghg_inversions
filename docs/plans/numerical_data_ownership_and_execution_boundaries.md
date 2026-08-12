# Numerical data ownership and execution boundaries

- **Status:** Working design guidance
- **Date:** 2026-08-12
- **Scope:** NumPy-, xarray-, and Dask-backed scientific data; eager numerical
  kernels; PyMC preparation; serialization; and array-bearing Python classes
- **Motivation:** Design discussion prompted by the native covariance work in
  PR #581 and similar ownership patterns elsewhere in the codebase

## Summary

OpenGHG data has historically reached OpenGHG Inversions as labelled xarray
objects backed by Dask arrays. Footprints, fluxes, observations, and boundary
conditions therefore carry both scientific labels and a delayed computation
graph. Code should preserve that representation until there is a deliberate
reason to cross into eager NumPy data.

The desired rule is not "always use Dask." Dask introduces scheduler overhead,
graph-management costs, chunk-alignment constraints, and debugging complexity.
For small or dense numerical problems, computing once and using NumPy or SciPy
can be simpler and faster. The important requirement is that the eager boundary
is chosen at the workflow level, made visible in the API, and assessed in the
context of all downstream products that share the upstream graph.

The central principles are:

1. Scientific xarray objects are borrowed and potentially lazy by default.
2. Ordinary property access must not copy, compute, persist, load, densify, or
   rechunk substantial data implicitly.
3. Validate dimensions, indexed coordinate labels, and other already-eager
   structure without materializing data variables. Treat auxiliary coordinates
   as potentially lazy, and consolidate value-wide validation at an explicit
   eager boundary.
4. Compute related Dask collections together, or cache a shared intermediate,
   rather than issuing separate computations over substantially shared graphs.
5. Use dataclasses for genuinely value-like specifications and metadata, not
   by default for array-backed computational objects with derived caches.
6. Choose one coherent execution model for an operation: Dask-native execution
   around eager kernels, or an explicitly eager blocked algorithm. Do not
   discard Dask partitioning and then imply that a manual loop restores
   out-of-core execution.

This note records working guidance. The proposed location and chunking of a
Zarr working cache remain hypotheses to test on representative inversions.

## Why conventional defensive programming can be counterproductive

Defensive programming normally attempts to prevent callers from invalidating
an object's assumptions. With mutable NumPy and xarray objects, this can lead
to a sequence such as:

```text
copy constructor inputs
    -> store only private arrays
    -> expose deep-copy properties
    -> freeze the containing dataclass
    -> bypass the public properties inside performance-sensitive methods
    -> rebuild dependent caches through custom replacement logic
```

Each step is locally understandable. Together they often produce an object
that is harder to read than the numerical model it represents, while still not
being deeply immutable. A determined caller can mutate a private NumPy buffer,
and a frozen dataclass only blocks ordinary attribute rebinding; it does not
freeze objects reachable through its fields.

For the scientific Python stack this defensiveness can have additional costs:

- copies may allocate large eager buffers;
- xarray copy behavior depends on the wrapped array type;
- Dask-backed copies may create distinct collection wrappers or change later
  graph lineage without producing independent in-memory values;
- a property can hide an operation whose cost and semantics differ between
  NumPy, Dask, sparse, GPU, and other duck-array backends;
- internal methods may need to bypass the public API to avoid accidental
  copies, giving one attribute two different meanings;
- array-valued validation can trigger unexpectedly large computations.

The project should protect against ordinary aliasing where it threatens a real
invariant. It should not attempt to make every reachable scientific array safe
against deliberate hostile mutation. Python's normal convention remains
useful: callers must not mutate borrowed data unless an API says that mutation
is supported.

## Vocabulary

Precise language is more useful than broad claims such as "immutable" or
"read-only."

| Term | Meaning in this project |
| --- | --- |
| **Borrowed array or view** | An ordinary reference returned for inspection or computation. The caller must not mutate it in place. No independence or copy is promised. |
| **Lazy collection** | An xarray object whose data includes an unexecuted backend graph, normally Dask. Ordinary transformations should preserve laziness. |
| **Owned eager buffer** | Concrete in-memory storage controlled by one computational object. The object may rely on callers not obtaining or mutating that private buffer. |
| **Materialized value** | NumPy, scalar, or other concrete data produced by explicitly executing a lazy collection. |
| **Eager kernel** | A numerical routine that deliberately consumes concrete arrays, for example a SciPy factorization. |
| **Dask chunk** | A unit represented in Dask storage and task scheduling. It participates in graph optimization and scheduler decisions. |
| **Algorithmic block** | A slice processed by an eager numerical loop to bound workspace. It is not automatically out-of-core or scheduler-managed. |
| **Serialized artifact** | A durable representation written to storage. Reopening it may be eager or lazy depending on the format and access path. |
| **Working cache** | A disposable, reproducible intermediate intended to shorten graphs or avoid repeated retrieval and alignment. It is not necessarily a durable result artifact. |

Use "independent snapshot" only when the implementation specifies independent
eager memory or another equally strong and testable isolation guarantee.

## Copying is backend-dependent

### NumPy

For ordinary numeric dtypes, `ndarray.copy()` allocates a distinct numeric
buffer. That is still not universal deep immutability: object-dtype arrays copy
their references rather than recursively copying every referenced Python
object, and writable flags can be changed or bypassed in various ways.

Copying can be appropriate when a small retained input is used to construct a
cache that must not become stale. For example, copying latitude and longitude
coordinate vectors once before constructing covariance factors is a reasonable
ownership boundary. Returning another copy on every coordinate access is a
separate decision and usually provides little value.

### xarray

An xarray object contains data, coordinates, indexes, attributes, encodings,
and a wrapped array backend. `copy(deep=True)` delegates important behavior to
that backend; it should not be read as "recursively materialize independent
NumPy memory." The xarray documentation describes deep copying in terms of the
data and coordinates, but the exact consequences still depend on the wrapped
array implementation.

Coordinate checks need a more precise rule than "coordinates are eager."
Xarray normally creates an index for each one-dimensional dimension coordinate,
and those indexed labels are ordinarily already in memory so that selection and
alignment are fast. That covers the usual `time`, `lat`, `lon`, `site`, and
similar structural coordinates in this project. Non-indexed auxiliary
coordinates can still be Dask-backed, however, and newer xarray index types may
represent coordinates without materializing all their values. Inspect whether a
coordinate is indexed or chunked before assuming that a value-wide coordinate
check is free. Checking ordinary indexed dimension coordinates must not be
treated as materializing the associated data variables.

Use `.data` when code intends to preserve and operate on the underlying duck
array. Use an explicit conversion such as `.to_numpy()` or `.as_numpy()` when
the operation truly requires NumPy. Avoid `.values` in new numerical code when
its only purpose is implicit coercion: the intent and eager boundary are less
clear, and some duck-array types cannot be converted safely that way.

Density and eagerness are separate concerns. The project's
`array_ops.to_dense()` helper handles the important case where xarray data is
backed by sparse arrays:

- an unchunked sparse array becomes an eager, dense NumPy-backed array;
- a Dask array with sparse chunks is transformed lazily into a Dask array whose
  chunks will be dense NumPy arrays when executed; and
- a Dask array whose chunks are already dense is returned unchanged.

This conversion is needed before operations or serialization backends that
cannot consume sparse chunk payloads. For Dask-backed input, densifying the
chunks should normally remain lazy and the writer should remain the execution
boundary. Do not call `.to_numpy()` merely to make such an object serializable
when changing the chunk representation is sufficient.

### Dask-backed xarray

A Dask-backed deep copy need not execute any tasks or allocate independent
numeric buffers. Depending on library versions and subsequent operations, it
may share the original graph and keys, create a new collection wrapper, or
fork graph lineage when the returned object is modified. Thus copy-on-access
properties introduce backend- and version-dependent graph semantics without
making the eager cost visible.

This has two implications:

1. A defensive copy is not a reliable immutability abstraction for lazy data.
2. Copying, materializing, persisting, and serializing must be treated as
   different operations rather than as interchangeable forms of isolation.

## Ownership and public API guidance

### Default policy

- Retain xarray/Dask objects without copying merely to make them appear
  read-only.
- Return borrowed arrays through ordinary properties when callers genuinely
  need them. Document that in-place mutation is unsupported.
- Do not hide substantial `.copy()`, `.compute()`, `.persist()`, `.load()`,
  `.as_numpy()`, or rechunking operations behind property access.
- Prefer narrow action APIs when consumers need behavior rather than stored
  arrays. For example, an object may expose `native_dims`, `apply`, and `solve`
  without exposing its cached factors or duplicate coordinate snapshots.
- Copy a retained constructor input once only when caller aliasing could
  invalidate derived state and the copied object is suitably small.
- Reconstruct an object explicitly when a change invalidates derived factors.
  Add an explicit `with_parameters()` method only if real callers require that
  operation and its inheritance semantics can be stated unambiguously.

### What static typing can enforce

Python's standard type qualifiers cannot express a deeply read-only
`xarray.DataArray` or `numpy.ndarray`.

- `Final[xr.DataArray]` prevents rebinding the annotated name or attribute. It
  does not reject `array[...] = value`, `array.data[...] = value`, or other
  mutation through the referenced object.
- A property without a setter prevents `obj.array = replacement`, but the
  returned array remains mutable.
- `ReadOnly` applies to `TypedDict` items and is also shallow: it prevents
  replacing an item, not mutation of a mutable value stored in that item.
- A narrow `Protocol` that exposes only observational operations can make some
  mutations fail static checking. Maintaining a parallel read-only facade for
  xarray's large API would be costly, though, and aliases or access to the
  underlying data can escape it.

Use `Final` when non-rebinding is itself useful, not as evidence of array
immutability. Prefer a narrow behavioral interface over returning a cache-
sensitive array. Where returning an xarray object is useful, the borrowed-data
contract, review, and focused tests are more honest than a misleading type
annotation. A read-only protocol is worth considering only for a genuinely
small project interface, not as a shadow type for all of xarray.

### Snapshot and export methods

Names such as `to_dataset()` describe representation, not necessarily memory
independence. A Dask-backed returned dataset may still share tasks with the
source. If a caller specifically requires concrete independent values, use an
API whose name and documentation expose that cost, for example
`materialize()` or `to_numpy_copy()`.

Serialization methods should state whether they:

- retain a lazy graph;
- execute it as part of writing;
- reuse existing persisted or on-disk data;
- materialize in memory before writing; or
- return an independently mutable eager result.

## Validation without accidental execution

Validation is valuable, but it must be proportional to the failure it
prevents.

Validate lazily where possible:

- required dimensions and their order;
- sizes and non-empty axes;
- indexed dimension-coordinate presence, names, labels, and units, which are
  normally already eager in xarray;
- dtype metadata and declared backend capabilities;
- relationships expressible through indexes or already eager coordinates.

Treat non-indexed or multidimensional auxiliary coordinates as potentially
lazy. Checking their complete values can be a computation boundary even though
they appear under `.coords` rather than `.data_vars`.

Value-wide checks such as finiteness, positivity, symmetry, or positive
definiteness may require execution. Those checks should occur at a named eager
boundary and should be consolidated with the computation that already needs
the values. Avoid separate scalar `.compute()` calls for several validations;
they can repeat shared upstream work.

Some numerical objects legitimately own eager state. A dense covariance and
its Cholesky factor, for example, are natural NumPy/SciPy objects once their
size is known to be manageable. This does not imply that all upstream inputs
or every accessor should also become eager.

## Dataclass fit and common antipatterns

Dataclasses are a good fit when the declared fields are an understandable
value representation and the generated operations have sensible meanings:

- initialization from those fields;
- concise representation;
- fieldwise equality;
- replacement by changing fields; and
- serialization or introspection over those same fields.

They are usually a poor fit for an identity-bearing computational object that
contains large or mutable arrays, lazy graphs, derived numerical caches, or
resource handles.

Warning signs include:

- `@dataclass(frozen=True, init=False, eq=False)`;
- a completely custom initializer dominated by `object.__setattr__`;
- public constructor arguments that differ from stored dataclass fields;
- private bookkeeping fields added only to support `dataclasses.replace()`;
- array fields excluded from every generated operation;
- copy-returning properties layered over private dataclass fields;
- `repr`, equality, or `asdict()` traversing large arrays or lazy graphs; and
- replacement semantics that cannot distinguish an inherited default from an
  explicit equal-valued override.

`frozen=True` is shallow. Generated equality can invoke ambiguous array
comparisons. `asdict()` recursively deep-copies values. Generated replacement
can expose internal fields or reconstruct stale caches unless the dataclass
field model exactly matches the public configuration model.

Prefer a normal class for a validated computational action. Use a small frozen
configuration dataclass only if that configuration has an independent
lifecycle: it is passed around, compared, reused, serialized, or transported
without the compiled runtime state. Do not split one object into specification
and runtime classes solely to satisfy a pattern.

A generic defensive-copy descriptor is not currently recommended. It would
standardize implicit copying while leaving relational validation, cache
rebuilding, Dask semantics, serialization, and equality unresolved.

## Preferred lazy-to-eager workflow

The default workflow should be:

```text
retrieve
    -> align and structurally validate
    -> filter or derive stable shared inputs where appropriate
    -> retain one shared lazy graph
    -> jointly compute, persist, or cache shared intermediates
    -> cross an explicit eager numerical/backend boundary
    -> serialize durable outputs
```

Typical acceptable eager boundaries are:

- immediately before passing concrete values into PyMC/PyTensor;
- immediately before a serialization backend that must execute the graph;
- entry to a clearly documented NumPy/SciPy kernel;
- a workflow checkpoint where measured Dask graph overhead, memory behavior,
  or repeated scheduling is worse than materialization.

The boundary should be high enough to see related consumers and shared work.
It should not normally be buried in a low-level property or validation helper.

### Shared graphs and separate computation

A historical failure mode was to compute these products independently:

- the sensitivity matrix `H`;
- the average footprint field used to fit basis functions; and
- the boundary-condition sensitivity `H_bc`.

These products shared much of their retrieval, alignment, filtering, and
preprocessing graph. Separate calls such as `H.compute()`,
`average_fp.compute()`, and `H_bc.compute()` can execute common dependencies
more than once. Dask can merge graphs and share intermediate tasks when
collections are submitted in one `dask.compute(H, average_fp, H_bc)` call.

Joint computation is not always the complete solution. If consumers run at
different times, if the shared intermediate is larger than available memory,
or if downstream graphs become excessively large, deliberate persistence or
an on-disk working cache may be preferable. The workflow must make that choice
once rather than allowing each downstream helper to call `.compute()`
independently.

## Dask is preferred, not mandatory

Dask preserves lazy I/O, parallelism, and shared upstream work, but it adds
real complexity:

- every chunk and operation adds task-graph overhead;
- inappropriate small chunks create very large graphs;
- operations may require rechunking or aligned chunks;
- debugging scheduler and worker memory behavior is harder than debugging
  eager NumPy;
- some SciPy and PyMC entry points require concrete arrays; and
- a compact eager computation can outperform a complicated lazy graph.

Computing is therefore sometimes correct for performance as well as
compatibility. The decision should answer:

1. What complete data will become resident?
2. Which related outputs share this graph?
3. Can they be computed together?
4. Will the result be reused, and if so should it be persisted or cached?
5. Is the graph too large because chunks are too fine or operations too
   fragmented?
6. Does the next backend genuinely require eager NumPy data?
7. What peak-memory behavior does the chosen boundary imply?

When an eager boundary is chosen, make it visible in the method name,
docstring, and tests.

## Dask chunks versus eager batching

Manual batching after full densification is not a replacement for Dask
chunking. Consider an observation-by-native sensitivity matrix `H` with shape
`(O, N)` and an eager loop with block size `b`.

If the complete `H` is first computed into NumPy and the loop then applies a
covariance to `b` observation right-hand sides at a time, the block size can
bound:

- an `N x b` right-hand-side block;
- the corresponding covariance-applied temporary; and
- for dense products, an `O x b` result block.

It does not bound:

- the already materialized `O x N` sensitivity;
- other eager projection arrays;
- or a preallocated dense `O x O` observation covariance.

Such blocking may be a legitimate eager linear-algebra optimization. It must
be described precisely as workspace blocking, using a name such as
`rhs_block_size` or `observation_block_size`. A Dask-native implementation
instead retains chunks and graph scheduling around an eager per-block kernel.

Choose one model deliberately:

1. **Dask-native orchestration:** preserve the backend and use xarray/Dask
   operations, `apply_ufunc`, or block kernels; materialize at the outer
   boundary.
2. **Explicit eager kernel:** require or document eager input, convert once,
   use blocked NumPy/SciPy operations, and make no out-of-core claim.

Renaming manual chunking to batching does not resolve a duplicated execution
model if Dask was discarded immediately beforehand.

## Working-cache hypothesis: aligned data in Zarr

The following is a hypothesis to test, not an adopted architecture:

> After retrieval and alignment, materialize stable shared inputs to an
> ordinary Zarr store, then reopen them lazily for basis fitting, sensitivity
> construction, boundary-condition products, and other downstream consumers.

This could provide a useful checkpoint:

```text
OpenGHG retrieval and alignment graph
    -> one coordinated Zarr write
    -> compact lazy graphs rooted in stable, chunked on-disk arrays
    -> several downstream scientific products
```

Potential benefits include:

- avoiding repeated remote retrieval and alignment;
- preventing separate consumers from rebuilding a large common graph;
- allowing downstream products to share a stable on-disk representation;
- controlling chunks for the dominant access patterns; and
- making workflow restart and inspection easier.

Potential costs include:

- extra write/read I/O and storage;
- poor performance if Zarr chunks do not suit basis fitting, observation
  selection, and sensitivity construction simultaneously;
- cache invalidation and provenance complexity;
- partial or concurrent write failure;
- duplicated durable artifacts; and
- moving the boundary too early, before useful filtering has reduced the
  data.

The cache should be distinguished from final inversion serialization. A
working cache is reproducible and disposable. Its identity would need to
include at least the OpenGHG query and data provenance, alignment and filtering
settings, relevant software/schema versions, and chunk layout.

An ordinary directory Zarr store is likely more suitable for a lazy working
cache than a zipped Zarr artifact whose store must be closed after reading.
This also requires experimental confirmation in the supported environments.

### Proposed experiment

Compare at least four representative workflows:

1. current separate computations;
2. one combined `dask.compute(...)` for shared downstream products;
3. `.persist()` after retrieval/alignment when memory permits; and
4. one ordinary Zarr working-cache write followed by fresh lazy reads.

Use a realistic inversion and measure:

- wall-clock time and repeated source I/O;
- peak process and worker memory;
- graph task and layer counts before each boundary;
- scheduler overhead and spill behavior;
- Zarr size, write time, read time, and metadata overhead;
- rechunking required for each downstream consumer;
- PyMC preparation time; and
- scientific equality of `H`, basis inputs, `H_bc`, and model-bound values.

Vary the chunk layout rather than assuming that an existing serialization
chunk, such as a fixed time chunk, is appropriate for a working cache. Decide
from evidence whether filtering belongs before or after the checkpoint.

## Serialization and PyMC boundaries

Serialization and PyMC are common, legitimate eager boundaries, but neither
justifies scattered eager conversion upstream.

For serialization:

- collect related variables into one coherent object before writing;
- let the writer execute a shared graph where supported;
- avoid computing the same value once for validation and again for writing;
- record whether the artifact is a working cache, prepared-input artifact, or
  final scientific result; and
- avoid serializing derived numerical caches when they can be reconstructed
  cheaply and checked from semantic inputs.

For PyMC/PyTensor:

- identify the smallest complete set of concrete arrays the model builder
  requires;
- materialize those arrays together where their graphs overlap;
- convert once, close to model construction;
- retain labelled xarray data until the point where labels have served their
  validation and alignment role; and
- document any numerical reason for computing earlier.

## Testing guidance

Tests should verify both values and execution semantics where those semantics
are part of the contract.

Useful tests include:

- constructing or inspecting an object with Dask-backed data does not execute
  tasks unless documented;
- ordinary properties do not copy, compute, persist, or rechunk;
- explicit materialization methods do execute and return the promised backend;
- borrowed-array mutation is documented as unsupported rather than tested as
  impossible everywhere;
- constructor copying protects only the specific eager cache invariant that
  requires it;
- several related collections are computed jointly when graph sharing matters;
- eager block size changes workspace but not numerical results;
- Dask chunk changes do not change scientific results;
- serialization executes the graph only once and round-trips labels, units,
  and provenance; and
- Zarr cache experiments measure task counts and peak memory rather than only
  output equality.

Do not encode extreme defensiveness as a requirement solely by writing tests
that mutate every returned array. First establish that independent mutation is
a real public requirement.

## Review checklist

For any new array-bearing class or numerical workflow, ask:

### Ownership and copying

- Is this array borrowed, owned, lazy, or materialized?
- Does any property allocate or call `copy(deep=True)`?
- What does "copy" mean for the actual wrapped backend?
- Can ordinary caller aliasing invalidate a derived cache?
- Would a narrow behavioral API avoid exposing cache-sensitive state?

### Execution boundary

- Does this code call `.compute()`, `.load()`, `.persist()`, `.values`,
  `.to_numpy()`, `.as_numpy()`, densification, or rechunking?
- Is that transition explicit in the method's name and documentation?
- How much data becomes resident, and what is the peak-memory estimate?
- Could related outputs be submitted together to share upstream work?
- Has the graph already been computed before a later "batching" loop begins?

### Class design

- Is the class a declarative value or a computational object?
- Do generated dataclass initialization, equality, repr, and replacement all
  have useful meanings?
- Are private provenance fields being added only to preserve dataclass
  machinery?
- Can derived caches be rebuilt atomically and explicitly?

### Dask and caching

- Are chunks suitable for the downstream access pattern?
- Is the graph large because chunks are too small or operations too granular?
- Would combined compute, persistence, or a Zarr working cache reduce repeated
  work?
- Is a proposed cache durable output or disposable workflow state?
- Are cache provenance, invalidation, and partial-write behavior defined?

## Open questions

- What is the smallest complete prepared dataset PyMC requires eagerly?
- Can all model-bound values be materialized in one shared computation?
- Which basis algorithms genuinely require eager NumPy data?
- Which current `.values`, `.as_numpy()`, `.compute()`, and deep-copy calls are
  necessary boundaries, and which are incidental?
- Should a working cache contain retrieved/aligned source arrays, filtered
  arrays, derived footprint-times-flux fields, or more than one layer?
- Which chunks best serve basis fitting, sensitivity construction, site/time
  filtering, and `H_bc`?
- Can covariance application remain Dask-native without making the code harder
  to maintain than an explicit eager kernel?
- Which existing copy-returning properties are compatibility commitments?
- Which computational classes genuinely require owned eager buffers because
  they retain derived factors?

## Recommended sequence

1. Use the terminology and review checklist from this note in new work.
2. Inventory hidden eager boundaries, copy-returning properties, and
   array-bearing frozen dataclasses.
3. Add a representative Dask-backed workflow test and record graph and memory
   behavior around `H`, basis fitting inputs, and `H_bc`.
4. Consolidate shared materialization with `dask.compute(...)` before changing
   storage architecture.
5. Compare combined compute, persistence, and ordinary-Zarr caching.
6. Choose the prepared-data/PyMC boundary and cache policy from those results.
7. Review manual batching independently: preserve Dask blocks or expose an
   explicitly eager blocked kernel.
8. Simplify defensive-copy APIs once real compatibility and cache-invariant
   requirements have been identified.

## References

- [xarray: Parallel computing with Dask](https://docs.xarray.dev/en/stable/user-guide/dask.html)
- [xarray: Working with NumPy-like arrays](https://docs.xarray.dev/en/stable/user-guide/duckarrays.html)
- [xarray terminology: dimension, indexed, and non-indexed coordinates](https://docs.xarray.dev/en/stable/user-guide/terminology.html)
- [xarray: `DataArray.copy`](https://docs.xarray.dev/en/stable/generated/xarray.DataArray.copy.html)
- [xarray: Reading and writing files](https://docs.xarray.dev/en/stable/user-guide/io.html)
- [Dask scheduler overview: computing related collections together](https://docs.dask.org/en/latest/scheduler-overview.html#the-compute-function)
- [Dask best practices](https://docs.dask.org/en/stable/best-practices.html)
- [Dask stages of computation](https://docs.dask.org/en/stable/phases-of-computation.html)
- [Python dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [Python typing specification: type qualifiers and `Final`](https://typing.python.org/en/latest/spec/qualifiers.html)
