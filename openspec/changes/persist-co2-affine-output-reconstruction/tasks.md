# Tasks

> Status: Draft. Finalisation requires explicit approval from the specification owner. After finalisation, this task list must not be modified without that owner's explicit consent.

## 1. Finalisation Gate

- [ ] 1.1 Obtain explicit approval from the specification owner, record the approval date, change every planning-artifact status to `Final` without substantive edits, and verify all four artifacts have matching status before implementation starts

## 2. PR 1 — Labelled Affine Quantity Operation

- [ ] 2.1 Add the small public affine value with labelled `reference_output`, an exact explicit-or-factorized `sensitivity` action, state/output dimensions, scope metadata, and an explicit-reference `apply`; verify both representation kinds satisfy the same public contract
- [ ] 2.2 Validate exact state alignment and compatible units, with targeted failures for missing, duplicated, reordered-incompatibly, or otherwise incompatible coordinates
- [ ] 2.3 Add labelled left composition by an output functional and verify composition occurs before chain/draw dimensions are introduced
- [ ] 2.4 Preserve borrowed xarray/Dask ownership and every non-state dimension without hidden computation, densification, persistence, rechunking, or input mutation; verify a factorized sensitivity is not silently flattened
- [ ] 2.5 Add focused tests using arbitrary non-unit reference states, explicit/factorized parity, composition-before-samples, and an independent dense affine oracle
- [ ] 2.6 Document the public equation and quantity-of-interest terminology, run focused pytest/Ruff and `git diff --check`, and merge PR 1 before stacking PR 2

## 3. PR 2 — Coherent-CO2 Reconstruction Companion

- [ ] 3.1 Add the frozen CO2 companion containing named affine quantities and their versioned explicit/factorized sensitivity representations, identities, units, scope, projection/reconstruction provenance, and separate source/sector metadata without putting output-storage data in `Co2PreparedInputs.inv_inputs` or `CoherentGaussianReduction`
- [ ] 3.2 Add optional, typed `C_qq`/`C_qy` extension data and `retained_exact` declarations that are validated and preserved but never conditioned or converted to posterior statistics in this change
- [ ] 3.3 Implement a versioned DataTree schema and NetCDF/Zarr save-load paths, preserving representation kinds, arrays or factors, dimensions, MultiIndexes, units, scopes, identities, and JSON-safe provenance
- [ ] 3.4 Implement public load-and-bind behavior that obtains authoritative `alpha_prior_mean` from `Co2PreparedInputs` and rejects stale identities, coordinate/unit mismatches, or conflicting imported reference states
- [ ] 3.5 Reject prohibited payloads containing raw `fp_x_flux`, \(\Pi\), native \(B\), or native-by-native covariance
- [ ] 3.6 Produce the bucket-preserving grid operation from signed reference flux and the bucket `BasisOperator`; verify it against direct `BasisFunctions` evaluation for arbitrary prior means without flattening \(FU\)
- [ ] 3.7 Add a public import path and fixture for an exact explicit supplied-restriction/Verification Games operation, without implementing supplied-\(\Pi\) construction or assuming \(U^*=U_{bucket}\)
- [ ] 3.8 Verify the grid operation always carries `retained_state_conditional` scope, matches independent affine parity expectations, and left-composes to a compact representative country functional before sample application
- [ ] 3.9 Record representative serialized artifact size and peak apply/compose memory for factorized and explicit representations, and verify materialization occurs only at named boundaries without silently densifying factorized or sparse inputs
- [ ] 3.10 Add corruption, incomplete-payload, and round-trip tests, including residual-block observation/likelihood identity validation

## 4. Documentation and PR 2 Merge Gates

- [ ] 4.1 Document the centred affine equation, sensitivity-as-linear-action terminology, explicit/factorized representations, compose-before-samples rule, arbitrary reference states, binding, projection provenance, source/sector distinction, retained-state conditional scope, and the complete-conditioning/output limitations deferred to OPE-68 and OPE-164
- [ ] 4.2 Add `newsfragments/169.feature.md` without editing `CHANGELOG.md`
- [ ] 4.3 Run focused pytest and Ruff checks for every changed Python path plus `git diff --check`, and record the exact passing commands in the PR
- [ ] 4.4 Run the applicable registered inversion regression cases and submit compatibility, full-suite, and type-check environments through `scripts/slurm_tox.sh` rather than local tox
- [ ] 4.5 Run `openspec validate persist-co2-affine-output-reconstruction --strict`, compare the implementation with every normative requirement, and obtain owner approval before changing any finalised artifact
