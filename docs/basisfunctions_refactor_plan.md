# BasisFunctions PR review and refactor sketch (revised)

## What this revision adds

This revision focuses on the practical concerns raised in review:

- temporary support for **legacy basis save format** and **new DataTree format**,
- how the migration stages map to reviewable PRs,
- what to test at each stage,
- and where a cleanup of `fixedbasisMCMC` output wiring should sit before deeper postprocessing migration.

## Updated architectural stance

### 1) Retain BasisFunctions object at runtime

After `basis_functions_wrapper` executes, retain a `BasisFunctions` object in memory (e.g. under a side-channel
entry on the run context) so downstream code can access:

- `operator.sensitivity(...)` for H construction parity,
- `operator.basis_matrix` for flux/country trace calculations,
- `operator.interpolate(...)` for state->grid reconstruction.

### 2) Basis save/read compatibility policy (temporary dual format)

Because `basis_functions_wrapper` can save basis functions today, and the new operator serialization is DataTree,
the migration should explicitly support both for a deprecation window.

Recommended temporary contract:

- **Legacy write (default)**: keep existing flat dataset save behavior unchanged.
- **New write (opt-in)**: add a new option to save DataTree serialization (for operator-aware experiments).
- **Read path**:
  - if DataTree metadata is present, use it to reconstruct `BasisFunctions`/`BasisOperator`;
  - else fall back to legacy flat basis dataset loader.

This gives reproducibility for experiments that reuse fixed basis functions across model variants without forcing
an immediate format switch.

---

## PR slicing plan (review-friendly)

### PR-1: Wrapper plumbing + dual-format IO scaffold (no fixedbasisMCMC behavior change)

Scope:

- Add opt-in return/payload from `basis_functions_wrapper` (non-breaking default).
- Attach basis objects in-memory on `fp_data` side channel.
- Introduce temporary dual write/read support:
  - legacy flat format remains default,
  - DataTree write/read added behind explicit option/flag.

Why this as a standalone PR:

- It is self-contained and low-risk.
- It can be validated with focused unit tests even if broader fixedbasisMCMC reviews are light.

Tests:

- wrapper return contract tests (`legacy` vs `with_basis_object` mode),
- serialization roundtrip tests (legacy and DataTree),
- compatibility tests proving both formats load to equivalent basis partition / basis_matrix.

### PR-2: fixedbasisMCMC output cleanup (structural, minimal logic change)

Scope:

- Refactor/untangle output writing code paths in `fixedbasisMCMC`.
- Separate concerns clearly:
  1. inversion solve outputs,
  2. basis artifact persistence,
  3. postprocessing trigger/output wiring.

Why before postprocessing migration:

- It reduces coupling and makes PR-3/4 easier to review.
- It addresses the “output code is a mess” concern directly.

Tests:

- regression tests on output variables/attrs presence,
- smoke test that current run still writes expected legacy outputs,
- no expected numerical behavior change.

### PR-3: New postprocessing output mode matching legacy `inferpymc_postprocessouts`

Scope:

- Add a postprocessing output mode using the **new postprocessing submodule** but emitting a dataset that matches
  legacy output structure/variables as closely as practical.
- Keep legacy rule differences explicit (e.g. modeled prior obs definition) and configurable if needed.

Notes:

- This PR is about output harmonization, not full operator migration.
- This step enables side-by-side comparison and easier reviewer confidence.

Tests:

- schema/variable parity checks against legacy-like expected outputs,
- numerical parity tests for quantities expected to match,
- explicit tests documenting known/intentional differences.

### PR-4: Postprocessing phase 1 integration with retained BasisFunctions (fallback-first)

Scope:

- Where postprocessing currently rebuilds dummy matrices from flat basis fields, prefer retained
  `BasisFunctions.operator.basis_matrix` when available.
- Keep legacy reconstruction as fallback.

Tests:

- path-selection tests (`basis object present` vs `absent`),
- equality tests for flux/country totals between fallback and operator-backed paths on same inputs.

### PR-5: Postprocessing phase 2 primary operator path + deprecation plan

Scope:

- Make operator-backed basis handling primary.
- Keep legacy path behind feature flag for one release cycle.
- Publish deprecation timeline for legacy basis reconstruction and (optionally) legacy save format.

Tests:

- end-to-end regression suite (legacy vs new mode),
- multisource/ragged basis tests for country and flux traces,
- performance smoke checks on representative runs.

---

## Test strategy summary by layer

1. **Unit tests** (fast)
   - `BasisFunctions` construction from legacy-loaded and DataTree-loaded inputs,
   - basis_matrix equivalence across formats,
   - wrapper API contract / defaults.

2. **Integration tests** (moderate)
   - wrapper -> fixedbasisMCMC (or equivalent pipeline slice) retains basis objects,
   - output writer preserves required legacy fields while adding optional operator metadata.

3. **Parity tests** (heavier)
   - postprocessing outputs: legacy vs new code path comparison,
   - country and flux traces agree within tolerance where definitions match,
   - documented expected differences are asserted explicitly.

4. **Backward-compatibility tests**
   - old saved basis artifacts still load and run,
   - new DataTree artifacts load and run,
   - mixed-run environments do not break rerun workflows.

---

## Practical defaults for transition

To reduce disruption while enabling new workflows:

- Keep all new behavior opt-in at first.
- Keep legacy file format as default writer until parity confidence is high.
- Add clear metadata tags to saved basis artifacts indicating format and loader path.
- Add concise logging at load time: "loaded legacy flat basis" vs "loaded DataTree basis".

---

## Exit criteria before full migration

Before removing ad-hoc postprocessing basis reconstruction:

- PR-1..PR-4 merged,
- parity tests stable across representative domains/species,
- at least one experiment workflow confirmed to reuse basis artifacts in both formats,
- reviewer sign-off that output cleanup in fixedbasisMCMC is sufficient for maintainability.
