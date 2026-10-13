# Tasks

Deliver under #868 before #866 consumes these boundaries. Exact destinations and
selected-resource checks already exist; preserve their tests and scientific behavior.

## 1. Runner-owned resources and exports

- [ ] 1.1 Resolve file resource values and explicit invocation overrides at their
  owning boundary; verify selected paths from a different working directory,
  absolute paths and equivalent full/staged requests.
  Drive conversion from declared path fields, preserving each value's origin;
  keep legacy case/directory translation in entry adapters and remove
  `country_directory` from resolved requests with an early migration error.
- [ ] 1.2 Remove optional saving from `prepare_basis` and migrate every runner,
  staged and copied/example caller to the existing save codec; verify generated,
  reused and partition bases, export metadata, no unrequested writes and exact
  destinations inside/outside checkpoint directories without a repeated prefix.
- [ ] 1.3 Update affected API documentation and add the #868 release note; pass
  focused caller/scientific parity checks, including early missing-resource errors
  without fallback or read-side directory creation.
  Add a 0.8 migration section and a distinct list of planned 0.9 deprecations and
  removals; explain NESW needs no basis file and a legacy directory is unused.
- [ ] 1.4 Move BC basis selection/loading before acquisition and pass resolved
  labelled data to sensitivities. Cover NESW with no file I/O, ignored legacy
  directories with a warning, explicit single/multiple files and time alignment,
  missing files and legacy translation without downstream case/glob discovery.
- [ ] 1.5 Resolve required packaged land/sea and outer-region defaults in runners
  for the selected domain; cover explicit overrides, missing-domain failures,
  algorithms needing no map and independence from explicit reporting `country_file`.

## 2. Completion

- [ ] 2.1 Record merged implementation and acceptance, then sync both deltas and
  archive this change before the grouped-configuration delta is synced. Confirm
  the durable resource spec describes the resulting implemented save boundary.
  As part of sync/archive, update remaining active consumers and the developer
  contract map from this change to the delivered capability; retain historical
  references only via the actual archive location. Check links in promoted specs
  and remaining active changes after archival. Requirement links use durable
  capability URLs so copying them does not depend on source-directory depth.

## Implementation evidence (13 October 2026)

The implementation branch moves exports to the standard, multisector, nested,
staged and copied runners, reuses the existing codec, and resolves resource
paths at their frontend boundary. API guidance and `868.removal.md` describe
both migrations. Configuration construction remains data-free.

Validation on Python 3.13:

- `pytest tests/test_basis_paths.py tests/test_staged_workflow.py tests/integration/test_custom_rhime_runner.py -q`: 144 passed.
- `pytest tests/test_basis_dataset_inputs.py tests/test_nested_rhime.py tests/test_rhime_configuration.py tests/test_rhime_output_resources.py tests/test_co2_staged_workflow.py tests/test_co2_staged_outputs.py tests/test_basis_paths.py::test_full_ini_resources_keep_invocation_directory_convention -q`: 201 passed.
- `pytest tests/test_rhime.py -k 'run_rhime_api_smoke or run_rhime_multisector_api_smoke' -q`: 4 passed.
- Final basis-path rerun after keeping canonical input forwarding unchanged: 79 passed.
- Changed-path Ruff, `git diff --check`, and strict OpenSpec validation passed.

Task 2.1 remains pending implementation merge and acceptance. Do not sync or
archive this prerequisite before that gate; the grouped-configuration delta
must retain its compatibility exception when it subsequently syncs.
