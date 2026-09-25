# Tasks

> Status: Final. Approved by the specification owner on 2026-09-25.

## 1. Confirm callers and profile calculation strategies

- [ ] 1.1 Inventory `BasisOperator.interpolate`, `BasisFunctions.interpolate`, `native_prolongation`, postprocessing, and known downstream direct callers; record source, flux, time-axis, eager-output, and map-access needs in the PR or design.
- [ ] 1.2 Profile per-source gathered slices, per-source contraction, and expanded-map contraction on equivalent source-preserving results for representative single, shared-state multisource, and ragged multisource shapes; compare their explicit sums with the current gathered total and record wall time, peak RSS, Dask tasks, and largest chunk in the design.
- [ ] 1.3 Select one source-preserving calculation strategy from the profiles, independent of state provenance, and update the design with the reason; verify no material memory regression for existing total-grid output shapes.

## 2. Add directional basis reconstruction

- [ ] 2.1 Add `state_to_native` application to basis operators and the basis wrapper; verify single-source and ragged multisource results against independent labelled linear oracles, including nonlexicographic source order, one output source axis, and chain/draw dimensions.
- [ ] 2.2 Add `BasisFunctions.state_to_flux` using retained signed flux; verify source-resolved and shared-state cases, exact label checks, and source-summed parity with historical weighted interpolation.
- [ ] 2.3 Verify Dask and sparse inputs remain borrowed and lazy outside any named eager boundary; exercise source selection, masking, sample-axis/source-coordinate collisions, and the output writer without silent alignment or lost output.

## 3. Migrate output consumers safely

- [ ] 3.1 Move in-tree basis postprocessing to directional methods, retain the current product time labels, and sum sources only for total products; verify existing single and multisector output fixtures retain values, labels, and units without prescribing time-axis renaming order.
- [ ] 3.2 Put dense conversion at the named completed-product or writer boundary; verify a sparse-backed NetCDF output is written successfully and that requested output is retained when only postprocessed products are selected.

## 4. Complete compatibility and documentation

- [ ] 4.1 Deprecate overlapping `interpolate` methods with replacement guidance and preserved legacy multisource sum/eager behavior; internalize the native-map adapter after its direct-use check and verify compatibility tests, changelog notice, and removal timing.
- [ ] 4.2 Update basis, affine, and output usage/API documentation to distinguish linear and affine reconstruction, define *coarse-to-fine map (prolongation)* on first use, and explain source summation and eager boundaries; verify documentation builds or renders without broken references.
- [ ] 4.3 Add an OPE-184 Towncrier fragment and run focused basis/postprocessing tests, changed-path Ruff, and `git diff --check`; verify broader relevant pytest coverage before implementation handoff.
