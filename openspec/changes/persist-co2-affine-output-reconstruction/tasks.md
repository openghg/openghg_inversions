# Tasks

> Status: Final. Approved by the specification owner on 2026-09-22. This task list must not be modified without that owner's explicit consent.

## 1. Finalisation Gate

- [x] 1.1 Obtained explicit approval from the specification owner, settled the public object name as `AffineFluxMap`, recorded the 2026-09-22 approval date, and changed every planning-artifact status to `Final` without substantive edits

## 2. PR 1 — Affine Native-Flux Value

- [x] 2.1 Add the small public `AffineFluxMap` value with labelled `native_mean`, `flux`, `prolongation`, state/native dimensions, and `retained_state_conditional` scope; verify construction tests cover arbitrary non-unit native and retained reference values
- [x] 2.2 Add the closed bucket-operator and explicit labelled prolongation representations; verify equivalent representations reconstruct the same native scaling and flux without precomputing \(FU^*\)
- [x] 2.3 Implement explicit-reference `state_to_native` and `state_to_flux` operations; verify every non-state sample dimension is preserved against an independent dense oracle
- [x] 2.4 Validate exact state/native/source alignment and compatible units, with targeted failures for missing, duplicated, reordered-incompatibly, or otherwise incompatible coordinates
- [x] 2.5 Preserve borrowed xarray/Dask ownership without hidden computation, densification, persistence, rechunking, or input mutation; verify the bucket representation is not silently flattened and sparse inputs stay sparse until a named boundary
- [x] 2.6 Document the public equations and prolongation/state-to-flux terminology, run focused pytest/Ruff and `git diff --check`, and merge PR 1 before stacking PR 2

## 3. PR 2 — Coherent-CO2 Persistence and Binding

- [ ] 3.1 Implement a versioned DataTree schema and NetCDF/Zarr save-load paths for native mean, flux, tagged prolongation representation, labels, MultiIndexes, units, scope, identities, intrinsic source provenance, and JSON-safe reconstruction/projection provenance; verify both formats round-trip
- [ ] 3.2 Implement public load-and-bind behavior that obtains authoritative `alpha_prior_mean` from `Co2PreparedInputs` and rejects stale identities, coordinate/unit mismatches, projection mismatches, or conflicting imported reference states
- [ ] 3.3 Reject payloads containing raw `fp_x_flux`, \(\Pi\), native \(B\), native-by-native covariance, precomposed \(FU^*\), named derived-quantity maps, quantity-specific residual blocks, or reporting-sector mappings; verify each prohibited category has a targeted failure
- [ ] 3.4 Produce the bucket-preserving affine flux value from native mean and existing `BasisFunctions` ingredients; verify arbitrary-prior-mean `state_to_native` and `state_to_flux` parity without flattening \(FU\)
- [ ] 3.5 Add a public import path and Verification Games-compatible fixture for exact native mean, signed reference flux, and explicit supplied-restriction \(U^*\), without implementing supplied-\(\Pi\) construction or assuming \(U^*=U_{bucket}\)
- [ ] 3.6 Verify source-aware and gathered-state round trips preserve exact source/native order without introducing a padded public source-state dimension
- [ ] 3.7 Add a representative labelled country-style contraction test that forms reference country values and a country-by-state action before sample dimensions, applies that compact action to draws, and confirms no derived country map is persisted
- [ ] 3.8 Record representative serialized artifact size and peak load, grid-apply, and aggregate-contraction memory for bucket and explicit prolongations; verify materialization occurs only at named boundaries
- [ ] 3.9 Add corruption, incomplete-payload, identity-mismatch, and unsupported-schema tests, and verify all failures occur before reconstructed outputs are produced

## 4. Documentation and PR 2 Merge Gates

- [ ] 4.1 Document native centring, `state_to_native`, `state_to_flux`, bucket/explicit prolongations, prepared-input binding, intrinsic source meaning, aggregate-before-samples ordering, retained-state conditional scope, and limitations deferred to OPE-24, OPE-68, and OPE-164
- [ ] 4.2 Add `newsfragments/169.feature.md` without editing `CHANGELOG.md`
- [ ] 4.3 Run focused pytest and Ruff checks for every changed Python path plus `git diff --check`, and record the exact passing commands in the PR
- [ ] 4.4 Run the applicable registered inversion regression cases and submit compatibility, full-suite, and type-check environments through `scripts/slurm_tox.sh` rather than local tox
- [ ] 4.5 Run `openspec validate persist-co2-affine-output-reconstruction --strict`, compare the implementation with every normative requirement, and obtain owner approval before changing any finalised artifact
