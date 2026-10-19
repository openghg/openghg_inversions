# Design

> Status: Final. Approved by the specification owner on 2026-09-25.

## Context

See [proposal.md](proposal.md) for motivation and [the delta spec](specs/retained-state-reconstruction/spec.md) for observable behavior. The current `BasisOperator.interpolate` applies a linear basis, while `BasisFunctions.interpolate(flux=True)` weights it with retained flux. The multisource operator gathers source weights onto a ragged `(source, region_in_source)` state MultiIndex, sums source contributions, and calls `.as_numpy()`; its single-source counterpart remains lazy. `InversionOutput.flux` is a time-axis view of `BasisFunctions.flux`, not a different physical prior. The current output functions also materialize completed products.

`AffineFluxMap` already supplies `state_to_native` and `state_to_flux` for the distinct centred equation \(m+U^*(\alpha-\alpha_{ref})\). Its OPE-169 artifacts are final and are not edited here. The multisource `native_prolongation` constructs a logical `(native_source, grid..., state)` array for covariance use; that expansion is not a requirement of ordinary reconstruction. Ragged source-region counts are already supported and remain first-class.

## Goals / Non-Goals

**Goals:**

- Give callers one directional vocabulary and explicit source-summed versus source-preserving results.
- Keep the normal flux source of truth on `BasisFunctions` and make output materialization an owned boundary.
- Retain ragged and shared-state multisource behavior with a profiled memory cost.

**Non-Goals:**

- A generic operator hierarchy, a new basis construction method, or a change to observation sensitivity.
- A change to affine centring, reference-state ownership, or OPE-169 artifact/persistence scope.
- Country aggregation, output routing, or unresolved native posterior uncertainty.

## Decisions

### 1. Name the action by its result, and the map by its mathematical role

Add `BasisOperator.state_to_native(state)` and forward it through `BasisFunctions.state_to_native(state)`. Add `BasisFunctions.state_to_flux(state)` using the paired retained flux. Keep `AffineFluxMap.prolongation` as the public map ingredient required by the approved OPE-169 contract. The existing `BasisOperator.native_prolongation(...)` builds a labelled version of that map; after checking direct users, make it an internal adapter and note the public-name change in the changelog. In documentation and docstrings, define the term on first use as *coarse-to-fine map (prolongation)*. This follows the [restriction/prolongation terminology used by Wu, Bocquet, and colleagues](https://cerea.enpc.fr/HomePages/wu/file/wu11carbon.pdf). In prose, *reconstruction* is the general user-facing term. `interpolate` suggests spatial estimation, *extension* is less precise about direction, and *restoration* can imply recovered fine-scale information that the retained state does not determine. The bucket label describes one-hot region construction and \(U_{bucket}\), not a third kind of application operation.

The basis action stays linear: \(x=U_{bucket}\alpha\), \(f=Fx\). The affine action stays centred: \(\bar x=m+U^*(\alpha-\alpha_{ref})\), \(\bar f=F\bar x\). Matching method names do not justify a generic parent class or force these equations through one implementation.

### 2. Preserve sources until the output requests a total

A source-specific retained vector has one `state` dimension with a `source` MultiIndex level labelling each region; that level is not an independent source dimension. Its reconstructed grid has one source axis, currently named `native_source` by `AffineFluxMap`, whose order follows the operator. Both source roles coexist only while constructing or applying an expanded map \(U(native\_source, grid\ldots, state)\); distinct names avoid an xarray coordinate collision. Contracting `state` leaves only the output source axis. A shared basis produces one scaling field, and multiplying by source-resolved retained flux introduces that flux's source axis. Existing output consumers that need total flux sum the output source axis explicitly. Exact ordered labels are checked before applying the state or flux. Collision handling for independent state sample axes follows the existing affine convention rather than silently aligning same-named axes.

Use `BasisFunctions.flux` in the new flux action. `InversionOutput.flux` currently renames or adds `flux_time`; that convention originally prevented alignment with observation times in a flat container, while modern DataTree serialization keeps trace, inputs, and basis in separate groups. Derived products still expose `flux_time`, and xarray may need time-axis disambiguation before multiplication or merging. OPE-184 preserves existing product values and labels without prescribing when to rename a time axis; [#728](https://github.com/openghg/openghg_inversions/issues/728) owns the later `flux_time`-to-`time` review. The basis loader's explicit binding of validated current-run flux remains the place to replace flux from a saved geometry. There is no routine public flux override on `state_to_flux`.

### 3. Profile application strategies and name eager boundaries

Directional methods should preserve Dask laziness unless a documented application or output boundary requires eager data. The multisource `.as_numpy()` entered with the source-weight broadcasting fix in commit `06a7cfea`, whose message and tests do not explain that conversion. Earlier repository work documents sparse chunks inside Dask, `where`/`dropna` on chunked data, and output writers as separate reasons for eager handling. Test source selection, masking, and actual serialization before moving conversion; do not assume that every Dask `where` call fails. Preserve the existing gathered calculation in the deprecated `interpolate` compatibility path until its consumers are migrated.

For source-preserving application, profile per-source gathered slices, per-source contraction, and an expanded-map contraction; each calculates the same result. The present gathered contraction already sums sources, so use it only as a total-output baseline after explicitly summing each source-preserving candidate. This is one profile-guided implementation choice, not runtime dispatch according to how a retained state was created. Earlier [PR #654](https://github.com/openghg/openghg_inversions/pull/654) retained gathered weighted interpolation after a sourcewise variant was slower for the *summed* product; expanded native application was not profiled there. Record graph construction, execution time, peak RSS, Dask tasks, and largest dense/sparse chunk on representative grid, source, ragged state, and sample shapes. Prefer a source-preserving strategy that avoids a full `(native_source, grid..., state)` intermediate if the profiles show no benefit from expansion.

The postprocessing or writer boundary converts completed output data to a serialization-compatible dense representation where required. A focused sparse NetCDF output check will test the suspected failure path; the present code does not document why multisource `interpolate` originally became eager. An output must not be silently lost after a writer rejects a sparse payload.

### 4. Deprecate the overlapping calls without a second permanent API

Inventory in-tree and known downstream direct use. Move in-tree callers to `state_to_native` or `state_to_flux` and explicit summation. Keep thin deprecated `interpolate` shims with their current source-summed and eager multisource semantics during the migration window, then choose removal timing from the direct-use evidence. No separate interpolation implementation is added for future flexibility.

## Risks / Trade-offs

- **Source-preserving output is larger than a total grid** → Construct it only when requested, benchmark realistic draw counts, and keep aggregate-first postprocessing separate where applicable.
- **A native source name may collide with state or flux coordinates** → Use a distinct native axis and exact label checks; cover collisions in focused tests.
- **Sparse output may fail in a writer** → Convert at the named output boundary and exercise the actual writer path.
- **A compatibility shim can preserve an inefficient eager operation** → Bound it to the deprecation window and remove it after usage review.

## Migration Plan

1. Add and verify the directional methods, including independent linear and affine oracle cases.
2. Profile candidate multisource calculations and record the implementation choice in this design before implementation is accepted.
3. Migrate in-tree postprocessing to the retained flux and explicit source sum; preserve historical output values and time labels, placing any necessary materialization at a named boundary.
4. Deprecate `interpolate`, internalize the native-map adapter after checking direct users, and update usage/API docs and release notes. Rollback can retain the compatibility path while restoring individual output callers if a product regression appears.

## Implementation profile and strategy decision (owner-approved addendum, 2026-09-25)

Tasks 1.2–1.3 were completed against a 48 × 64 labelled grid with 24 × 32 grid
chunks, two chains, 20 draws, and nonlexicographic sources `zeta`, `alpha`,
`mu`. The ragged case has 7, 13, and 5 regions respectively. Every
source-preserving candidate was explicitly summed and checked against the
historical gathered total. `scripts/profile_ope184_reconstruction.py`
reproduces the calculations; the raw runs and caller inventory are retained in
`docs/plans/ope184_reconstruction_implementation.md`.

For one source and a shared state with source-resolved flux, all three strategy
labels use the same direct contraction: neither case needs a source-specific
state map. These are one warm run per label. Wall time includes graph creation,
execution, and parity checking. Peak RSS includes imports; comparisons are
meaningful within this run cohort.

| Case | Strategy | Wall s | Peak RSS MiB | Dask tasks | Largest dense / sparse chunk MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| Single | gathered slices | 0.474 | 340.5 | 37 | 0.469 / 0.012 |
| Single | per-source | 0.545 | 340.7 | 37 | 0.469 / 0.012 |
| Single | expanded | 0.482 | 340.7 | 37 | 0.469 / 0.012 |
| Single | legacy total | 0.442 | 344.3 | 37 | 0.469 / 0.012 |
| Shared state, source flux | gathered slices | 0.508 | 349.2 | 109 | 1.406 / 0.012 |
| Shared state, source flux | per-source | 0.478 | 351.2 | 109 | 1.406 / 0.012 |
| Shared state, source flux | expanded | 0.474 | 350.4 | 109 | 1.406 / 0.012 |
| Shared state, source flux | legacy total | 0.554 | 358.5 | 109 | 1.406 / 0.012 |

Separately timed direct calls took 0.036 s to construct and 0.508 s to execute
for the single case, and 0.038 s and 0.473 s for the shared case. The ragged
case below was measured after replacing an initial per-source candidate that
rebuilt bucket operators during application and computed chunked basis data.
The corrected candidate retains the per-source matrices already made during
operator construction. These ragged runs used separate processes in a second
cohort; their RSS values should not be compared numerically with the table
above. Wall time is graph plus execution time, while the legacy call executes
eagerly inside the method.

| Ragged strategy | Graph s | Execute s | Wall s | Peak RSS MiB | Dask tasks | Largest dense / sparse chunk MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gathered slices | 0.032 | 4.726 | 4.758 | 1748.6 | 197 | 1.406 / 0.035 |
| Retained per-source matrices | 0.029 | 0.524 | 0.553 | 1690.4 | 193 | 1.406 / 0.012 |
| Expanded map | 1.335 | 1.397 | 2.732 | 1741.4 | 213 | 1.406 / 0.079 |
| Legacy total | included in eager call | 0.631 | 0.631 | 1741.5 | 637 | 0.469 / 0.035 |

**Decision:** Use direct contraction for single and shared-state bases, and
retain per-source matrices for every ragged multisource state regardless of
provenance. The ragged per-source action is the fastest source-preserving
candidate in this fixture, avoids a full `(native_source, grid, state)` map,
and returns a lazy Dask result without executing tasks during application.
The total-grid consumer sums the known reconstructed source axis at its output
boundary. In the corrected ragged pass, peak RSS was 51.1 MiB below the
legacy total despite preserving sources until that boundary; the single and
shared cases showed no material RSS increase over their legacy totals. Larger
production grids and draw counts remain a measurement boundary for future
work, not a runtime strategy switch in this change.
