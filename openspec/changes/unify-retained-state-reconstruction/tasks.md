# Tasks

## 1. Confirm callers and kernel costs

- [ ] 1.1 Inventory `BasisOperator.interpolate`, `BasisFunctions.interpolate`, postprocessing, and known downstream direct callers; record each caller's source, flux, time-axis, and eager-output expectations in the PR or design.
- [ ] 1.2 Benchmark gathered, sourcewise, and expanded-native application on representative single, shared-state multisource, and ragged multisource shapes; verify equal-value comparisons after explicit source summation and record wall time, peak RSS, Dask tasks, and largest chunk in the design.
- [ ] 1.3 Select the source-preserving kernel using the benchmark and update the design with the measured decision; verify no material memory regression for existing total-grid output shapes.

## 2. Add directional basis reconstruction

- [ ] 2.1 Add lazy `state_to_native` application to basis operators and the basis wrapper; verify single-source and ragged multisource results against independent labelled linear oracles, including nonlexicographic source order and chain/draw dimensions.
- [ ] 2.2 Add `BasisFunctions.state_to_flux` using retained signed flux; verify source-resolved and shared-state cases, exact label checks, and source-summed parity with historical weighted interpolation.
- [ ] 2.3 Verify Dask and sparse inputs remain borrowed and uncomputed during directional application, and verify sample-axis/source-coordinate collisions do not cause silent alignment.

## 3. Migrate output consumers safely

- [ ] 3.1 Move in-tree basis postprocessing to directional methods, adapt the retained flux time axis at output, and sum sources only for total products; verify existing single and multisector output fixtures retain values, labels, and units.
- [ ] 3.2 Put dense conversion at the named completed-product or writer boundary; verify a sparse-backed NetCDF output is written successfully and that requested output is retained when only postprocessed products are selected.

## 4. Complete compatibility and documentation

- [ ] 4.1 Deprecate overlapping `interpolate` methods with replacement guidance and preserved legacy multisource sum/eager behavior; verify warning and compatibility tests, and record removal timing based on the direct-use inventory.
- [ ] 4.2 Update basis, affine, and output usage/API documentation to distinguish linear reconstruction, affine reconstruction, prolongation, source summation, and materialization; verify documentation builds or renders without broken references.
- [ ] 4.3 Add an OPE-184 Towncrier fragment and run focused basis/postprocessing tests, changed-path Ruff, and `git diff --check`; verify broader relevant pytest coverage before implementation handoff.
