# Design

## Context

See [proposal.md](proposal.md) for motivation and [the delta spec](specs/retained-state-reconstruction/spec.md) for observable behavior. The current `BasisOperator.interpolate` applies a linear basis, while `BasisFunctions.interpolate(flux=True)` weights it with retained flux. The multisource operator gathers source weights onto a ragged `(source, region_in_source)` state MultiIndex, sums source contributions, and calls `.as_numpy()`; its single-source counterpart remains lazy. `InversionOutput.flux` is a time-axis view of `BasisFunctions.flux`, not a different physical prior. The current output functions also materialize completed products.

`AffineFluxMap` already supplies `state_to_native` and `state_to_flux` for the distinct centred equation \(m+U^*(\alpha-\alpha_{ref})\). Its OPE-169 artifacts are final and are not edited here. The multisource `native_prolongation` constructs a logical `(native_source, grid..., state)` array for covariance use; that expansion is not a requirement of ordinary reconstruction. Ragged source-region counts are already supported and remain first-class.

## Goals / Non-Goals

**Goals:**

- Give callers one directional vocabulary and explicit source-summed versus source-preserving results.
- Keep the normal flux source of truth on `BasisFunctions` and make output materialization an owned boundary.
- Retain ragged and shared-state multisource behavior with a measured memory cost.

**Non-Goals:**

- A generic operator hierarchy, a new basis construction method, or a change to observation sensitivity.
- A change to affine centring, reference-state ownership, or OPE-169 artifact/persistence scope.
- Country aggregation, output routing, or unresolved native posterior uncertainty.

## Decisions

### 1. Name the action by its result, and the map by its mathematical role

Add `BasisOperator.state_to_native(state)` and forward it through `BasisFunctions.state_to_native(state)`. Add `BasisFunctions.state_to_flux(state)` using the paired retained flux. Keep `native_prolongation` and `prolongation` for the map \(U\), consistent with the [restriction/prolongation terminology used by Wu, Bocquet, and colleagues](https://cerea.enpc.fr/HomePages/wu/file/wu11carbon.pdf) and with the approved OPE-169 contract. The paper defines prolongation as coarse-to-fine refinement in English; its use here does not establish a French origin for the term. In prose, *reconstruction* is the general user-facing term. `interpolate` suggests spatial estimation, *extension* is less precise about direction, and *restoration* can imply recovered fine-scale information that the retained state does not determine. The bucket label describes one-hot region construction and \(U_{bucket}\), not a third kind of application operation.

The basis action stays linear: \(x=U_{bucket}\alpha\), \(f=Fx\). The affine action stays centred: \(\bar x=m+U^*(\alpha-\alpha_{ref})\), \(\bar f=F\bar x\). Matching method names do not justify a generic parent class or force these equations through one implementation.

### 2. Preserve sources until the output requests a total

A source-specific basis produces a `native_source` dimension distinct from the gathered state MultiIndex level `source`; its order follows the operator. A shared basis produces one scaling field, and multiplying by source-resolved retained flux introduces that flux's source axis. Existing output consumers that need total flux sum this axis explicitly. Exact ordered labels are checked before applying the state or flux. Collision handling for independent state sample axes follows the existing affine convention rather than silently aligning same-named axes.

Use `BasisFunctions.flux` in the new flux action. The `InversionOutput.flux` time rename or singleton period is a view needed by output formatting; perform the equivalent time-axis adaptation after the reconstruction result. The basis loader's explicit binding of validated current-run flux remains the place to replace flux from a saved geometry. There is no routine public flux override on `state_to_flux`.

### 3. Separate application from serialization and measure the kernel

New directional methods do not call `.as_numpy()` or materialize Dask graphs. Preserve the existing gathered kernel only in the deprecated `interpolate` compatibility path until its consumers are migrated. For source-preserving application, start with source selection and per-source contraction, concatenating ordered outputs; this avoids requiring a full `(native_source, grid..., state)` intermediate. Compare it with the gathered and expanded-prolongation paths before finalizing the implementation. An alternative may replace the sourcewise kernel only if it preserves labels and borrowed-array behavior without a material memory regression.

This is a measured choice, not a claim that sourcewise is faster. Earlier [PR #654](https://github.com/openghg/openghg_inversions/pull/654) retained gathered weighted interpolation after a sourcewise variant was slower for the *summed* product; expanded native application was not measured there. The benchmark must compare equivalent outputs, including an explicit source sum where needed, and record graph construction, execution time, peak RSS, Dask tasks, and largest dense/sparse chunk on representative grid, source, ragged state, and sample shapes.

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
2. Benchmark candidate multisource kernels and record the selection in this design before implementation is accepted.
3. Migrate in-tree postprocessing to the retained flux, explicit source sum, and output-boundary materialization; retain historical output values and labels.
4. Deprecate `interpolate` with a replacement message, update usage/API docs, and check direct downstream callers before a later removal. Rollback can retain the compatibility path while restoring individual output callers if a product regression appears.
