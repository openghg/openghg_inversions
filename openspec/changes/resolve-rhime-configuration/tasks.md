# Tasks

This checklist supersedes the previous completed implementation checklist.
PR #813 originally implemented the former preparation-config design. Prior
locked Python 3.12/3.13 coverage (job 19270831) and docs-full (job 19270662) passed,
but do not verify the revised plan. The planning artifacts are approved and
finalized. The recorded implementation rounds are complete; the newly approved
cleanup and its validation are tracked separately below.

## 1. Revised planning

- [x] 1.1 Reconcile proposal, design and behavioral requirements around one requested configuration, direct preparation fields and explicit roles/contracts; verify strict OpenSpec validation and the key-name table.

## 2. Configuration and compatibility

- [x] 2.1 Remove RhimePreparationConfig and its projection plumbing; put its explicit fields directly on RhimeConfig while reusing existing site/model/output/sampler values; verify direct option access, file/Python equivalence, winning overrides (including date-dependent defaults), early errors and caller ownership; consolidate checks at their owning input boundaries.
- [x] 2.2 Remove RhimeRunnerSetup, make_rhime_runner_setup and resolve_rhime_options and migrate ordinary/nested/staged/shim/example consumers to RhimeConfig; remove encoding-only restoration of raw site forms and sparse priors; verify scientific behavior and entry-point results, absence of removed-type imports and explicit supported staged-contract handling.
- [x] 2.3 Promote the existing acquisition-owned _SiteOptions to SiteOptions and export it from inversion_data; update annotations/imports/consumers while keeping one record; reuse its from_inputs and applicable small alignment helpers independently of file syntax; verify public construction, scalar/list equivalence, resolved-constructor invariants and selection without request mutation or repeated expansion.

## 3. Scientific boundaries and names

- [x] 3.1 Replace load_rhime_config with read_rhime_ini(path, *, overrides=None, multisector=False) returning complete RhimeConfig; retain INI interpretation inside the frontend and apply overrides before shared semantic/site resolution; preserve params_from_config dictionary/normalization behavior through shared internal decoding; migrate configured consumers to use the result without resolving twice and verify reader results, override precedence and existing INI section behavior.
- [x] 3.2 Collapse retrieval/reload forwarding layers into load_rhime_data; verify supplied-data no-I/O behavior, cache loading/fallback, layout/selector checks and retained alignment across ordinary/nested/staged consumers.
- [x] 3.3 Forward needed resolved values explicitly to named stages, retain aligned selectors by label and derive run descriptions after preparation; verify retrieval/reload/filter/drop/supplied-data authority, tracer rejection and requested-versus-retained metadata without request mutation.
- [x] 3.4 Rename fresh acquisition to retrieve_inversion_data, sharing the resolved-selector body; preserve data_processing_surface_notracer as a deprecated same-signature/six-tuple wrapper; verify warning, forwarding, imports and equivalent surface/column results without repeated expansion.

## 4. Documentation and delivery

- [x] 4.1 Reconcile user/developer/API guidance, examples and the existing Issue 804 fragment with the new roles and names; verify examples and affected rendered documentation.
- [x] 4.2 Run focused and relevant broader tests, changed-path Ruff, strict OpenSpec and whitespace checks for the reconciled implementation; inspect ownership and public compatibility before marking complete.
- [x] 4.3 SSH-push the reconciled implementation and update #813 using the repository template; verify remote head, target branch, review readiness and app attachment confirmation.

## Final validation

Implementation commit `436379ca` includes synchronization with `devel` at
`e49632fd`. Relevant broader configuration, acquisition, ordinary/nested/staged
runner, compatibility, serialization and integration coverage passed on Python
3.12 and 3.13 in SLURM job `19278392`. The isolated `docs-full` build passed in
job `19278394`. Both jobs completed with exit status zero. Eight affected
rendered usage, migration, development and API pages were inspected, including
their local links and anchors. Focused tests, changed-path Ruff, strict OpenSpec,
whitespace and executable documentation examples passed; independent subagent
reviews found no actionable correctness, contract or development-guidance issues.
PR #813 targets `devel`, is attached to this task and is ready for review.
This final checklist update changes no implementation or documentation examples.

### Review follow-up: configuration-only finite choices

The shared resolver now rejects invalid `flux_non_finite_check` and
`aggregation_error_mode` values using their existing declared choice types.
Regression tests first reproduced the missing rejection, then verified that
Python and INI requests through both ordinary runners raise before acquisition
and that all seven supported choices are preserved. The 36 configuration tests
and 60 selected broader configuration/consumer tests passed on Python 3.13;
changed-path Ruff, strict OpenSpec and whitespace checks also passed.

The active-preparation follow-up also validates `basis_algorithm` against the
existing live registry only when no saved `fp_basis_case` is supplied, and
rejects unknown named `min_error` methods. Acquisition-uncalled coverage includes
Python and INI requests through both ordinary runners; saved-case precedence,
registered algorithms and both minimum-error methods are preserved. All 56
configuration tests and 67 selected broader configuration, consumer and basis
checks passed on Python 3.13, along with changed-path Ruff, strict OpenSpec and
whitespace checks. This reconciles existing configuration-only late checks;
scientific owners retain their data-dependent checks.

Configuration serialization, resolved-settings logging and an INI writer are
deferred to [#814](https://github.com/openghg/openghg_inversions/issues/814).
No serializer or writer implementation is part of this checklist.

### Review follow-up: configuration ownership and readability

- [x] Place semantic construction on `RhimeConfig.from_params`, retain the
  function compatibility wrapper, and separate the INI frontend.
- [x] Derive supported and required options from their owning configuration
  records and advertised consumer subsets; preserve accepted names and defaults.
- [x] Reuse resolved requests in the standard/multisector runners and HBMCMC shim.
- [x] Move site selectors into the shared selector owner, retain the complete
  record through retrieval, and consolidate copying and layout validation.
- [x] Document changed public contracts and explicit merged-data saving while
  retaining existing opt-in saving behaviour and cache formats.
- [x] Validate the final follow-up on supported Python versions, regenerate and
  inspect affected API documentation, and SSH-push the reviewed changes.

Focused configuration, acquisition, runner, shim and documentation-example
checks passed in the existing environment. That environment contains PyMC
5.26.1; the synthetic staged sampling test fails there on both the untouched
PR head and this follow-up because it returns the former InferenceData type.
The cluster retry passed the relevant configuration, acquisition, ordinary,
nested, staged, shim, serialization and integration suites on Python 3.12 and
3.13 in job `19280326`, including that sampling test. It used locked project
dependencies for both interpreters and ran environments sequentially with one
pytest worker. The initial parallel attempt exhausted its memory allocation;
its Python 3.13 environment also exposed unpinned dependency incompatibilities.
These validation-only environment adjustments do not change repository tox
configuration.

The `docs-full` environment passed in job `19280272`. Both jobs validated
implementation commit `f0f47030`. Nine affected rendered usage, development and
reference pages were checked for expected content and local links/anchors;
the three changed generated reference files are included in the follow-up.
Changed-path Ruff, strict OpenSpec and whitespace checks passed. Earlier
validation records above apply to their named commits. New review comments
about argument forwarding and preparation boundaries are being assessed
separately; this validation does not claim those design concerns are resolved.


### Review follow-up: selected forwarding and canonical preparation

- [x] Add shallow `RhimeConfig.select(*names)` and test borrowed-value ownership,
  missing names and ordinary keyword use without another resolution pass.
- [x] Convert repeated runner, nested, staged and example forwarding to explicit
  selections; keep scientific functions independently callable.
- [x] Remove the named stages' former positional `data_args` adapters and
  update their public docstrings and callers.
- [x] Deprecate `prepare_rhime_inputs`, delegate its science to the named stages,
  and verify matching prepared metadata, footprint provenance and warnings.
- [x] Reconcile user/developer/reference documentation and release notes; review
  any remaining deprecated helper exposure such as `convert_to_list`.
- [x] Run focused and relevant broader tests, changed-path Ruff, strict OpenSpec,
  whitespace and rendered-documentation checks before marking this round complete.

The filtered staged checkpoint, retained-site policy across all execution routes,
and replacement manifest/handoff contracts remain with the planning-only
[PR #802](https://github.com/openghg/openghg_inversions/pull/802), reviewed at
`1ec45fc`. This cleanup shares existing scientific stages and their provenance;
it does not claim to implement that broader workflow replacement.

Focused validation passed: 69 configuration tests, 55 runner/shim/documentation-example
and integration tests, 69 preparation/acquisition/site-resolution and tracer checks,
and 10 nested/runner composition checks (some selections overlap). Independent
review found no actionable forwarding or scientific-contract regressions.

Implementation commit `2b80e490` passed relevant configuration, acquisition,
standard/nested/staged runner, shim, serialization and integration coverage on
Python 3.12 and 3.13, plus `docs-full`, in SLURM job `19280761` (exit zero).
The test environments used locked dependencies and ran sequentially with one
pytest worker. Eight affected rendered API, usage, migration and development
pages were checked for the selection, required-keyword and deprecation contracts;
all 1,316 inspected anchor links had valid local targets where applicable.
Generated reference files match the tracked files. Inspection covered rendered
HTML structure and text, not browser screenshots. Changed-path Ruff, strict
OpenSpec and whitespace checks passed. This final validation record changes no
implementation or examples.

### Review follow-up: decoding and compatibility boundaries

This records the implementation at `7c58af3b`. The approved amendment below
supersedes its canonical-only factory and separate runner translation choices,
and supersedes the resolved-reader contract originally completed in task 3.1.

- [x] Make `read_rhime_ini` a dictionary decoder; apply runner overrides and
  extract recipe-specific choices before calling `RhimeConfig.from_params` once.
- [x] Move historical aliases, fixedbasis translation and the deprecated
  `params_from_config` adapter to `hbmcmc.compatibility`; preserve supported
  entrypoint warnings and keep canonical construction independent of aliases.
- [x] Consume the custom-basis recipe's unused built-in basis options before
  resolution and verify saved project artifacts through the real resolver.
- [x] Keep nested direct fields and composed model/output settings coherent.
- [x] Preserve borrowed flux/BC Dataset attrs at serialization by attaching
  metadata to shallow copies; cover real NetCDF and Zarr saving.
- [x] Reconcile public docstrings, usage/development guidance, release notes and
  the active spec around these responsibilities.
- [x] Validate the combined change on supported Python versions and inspect
  affected rendered documentation. Delivery awaits the approved amendment below.

Scientific default policy, INI template redesign and model selection are deferred.
Existing defaults and automatic-saving policy remain unchanged. Shorthand is
expanded after overrides and recipe-specific option extraction.


Commit `7c58af3b` passed the relevant Python 3.12 and 3.13 suites and `docs-full`
in SLURM job `19281228` (exit zero). Eight rendered pages and 940 local
links/anchors passed inspection. The new compatibility module adds one generated
API reference page. This validation predates the constructor amendment below.

### Approved amendment: one constructor and conditional alias warnings (2026-10-08)

The user approved the decoding/resolution responsibility changes and clarified
that the construction wrapper can be removed because users are working from
0.7.x. The final sequence is: decode file values, apply winning overrides,
consume recipe-owned options, then construct the remaining configuration through
`RhimeConfig.from_params`. This refers to all remaining run options, not only
options unique to a recipe. Existing completed tasks above retain their history.

- [x] Reconcile proposal, design and requirements with that sequence and the
  removal of `resolve_rhime_config`.
- [x] Move the shared alias-translation call into `RhimeConfig.from_params`,
  remove repeated runner translation and the wrapper, and migrate callers/tests.
- [x] Emit `DeprecationWarning` when deprecated names or output values are
  translated or removed; verify canonical inputs remain quiet.
- [ ] Reconcile public documentation and generated API exposure, validate the
  final implementation, SSH-push and update PR #813.

The future dataset-only serialization/provenance contract is recorded in
[#718](https://github.com/openghg/openghg_inversions/issues/718). It does not
change the current artifact format in this PR.

Focused amendment validation passed: 173 consumer tests and 35 compatibility/shim
tests, changed-path Ruff, strict OpenSpec validation and whitespace checks.
Supported-version and documentation validation for this amendment remains pending.
