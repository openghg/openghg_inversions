# Tasks

Current restack acceptance is section 8. Earlier checked sections are dated
revision-specific history; their fixture-retention and named-rejection wording
is superseded by the 26 October direction recorded there.

## 1. Agreed boundary and configuration

- [x] 1.1 Record maintainer-approved post-drop floor semantics, explicit compatibility break and #957 coordination on issue #876; verify the saved issue amendment and bounded delta.
- [x] 1.2 Move BC frequency/anchor and applicable floor choices to model/likelihood owners; verify full, phase-only and model-record roundtrip configuration tests.

## 2. Scientific preparation and model binding

- [x] 2.1 Retain raw spatial BC arrays and remove baked floor/period preparation arguments across shared and direct xarray routes; verify raw assembly, invalid-row selection and extension preservation tests.
- [x] 2.2 Construct model-owned alignments/floors after joint selected-operand/prior materialization, including mechanical nested adoption; verify frequency reuse, retained-row estimators, nonmutation and unused-operand tests.

## 3. Persistence and current consumers

- [x] 3.1 Version affected preparation/sample records, remove redundant durable site codes and reject incompatible baked artifacts; verify actual disk reload, generation binding and rejection tests.
- [x] 3.2 Preserve exact trace operands in graph-free replay and independent modern codecs; verify standard/multisector replay and safe BC legacy-export behavior.

## 4. Review and validation

- [x] 4.1 Update affected usage/API documentation, examples and Towncrier fragment; run documentation-impact/docstring review and applicable checks.
- [x] 4.2 Complete higher-effort independent review, focused and relevant broader pytest, Ruff, diff/OpenSpec checks, and required final CI-equivalent validation; record concrete evidence and remaining limitations.

## 5. PR review follow-up

- [x] 5.1 Migrate Ramsden from unconditional prepared floors/site codes to likelihood-owned floor choices and observation labels; verify actual preparation-to-builder integration.
- [x] 5.2 Define the backend-neutral resolved Dataset handoff, reject conflicting BC/floor settings without recalculation, and keep likelihood-floor policy outside numerical materialization; verify all three eager builders.
- [x] 5.3 Remove redundant raw site-code construction and unused likelihood declarations; update usage/docstrings and complete independent review plus proportionate Python 3.12/3.13 validation.

## Validation evidence

The completed sections above and their evidence describe the 22 October scope.
The accepted amendment below has its own implementation and validation tasks;
the previous full-suite success does not establish delivery of the new policy.

- Configuration worker: 171 focused tests passed; anchor fixes: 86 tests passed.
- Core worker: 183 focused tests passed.
- Caller/handoff integration: 469 passed with two stale failures from concurrent edits; fresh retries of both pass.
- Final broader observation-error/configuration/serialization/documentation checks: 220 passed.
- Final direct-builder boundary suite: 84 passed.
- Actual two-frequency preparation/sample/replay checks: 2 passed, covering standard and multisector with recalculation disabled on replay.
- Full changed-area Ruff, diff checks and strict change validation pass. Higher-effort configuration/core/persistence/documentation reviews completed; reported anchor, direct-builder and tutorial findings were fixed. Counts overlap and are not a combined total.
- Slurm job 19635769 ran the complete Python 3.12/3.13 suites: each had 3,925 passed, 8 skipped, 4 deselected and the same single stale-test failure. Lint and all 4 borrowed-typing tests passed.
- The failing model-option test incorrectly called the now data-only test preparation helper. A test-only correction exercises both public runners with retrieval guarded; both corrected cases and 181 nearby/configuration tests pass locally, with Ruff and independent higher-effort review clear.
- Slurm job 19636938 at test-only correction `a2318985` passed the full affected `tests/test_rhime.py` module through tox on both Python 3.12 and 3.13: 312 passed on each. The original full-suite failures remain part of the evidence; this scoped retry is not a fresh complete-matrix pass.
- The approved configuration amendment is synchronized into the durable contract, including the explicit exception to general compatibility guarantees. CO2 family defaults and unrelated contracts remain unchanged. PR #964 contains the implementation and validation handoff; #957 still owns centered BC arithmetic and its later compatibility changes.

## PR review follow-up evidence

- Raw assembly and observation-error selection: 40 tests passed, including eager/lazy retained label ordering and input nonmutation.
- Ramsden: 26 focused tests passed, including actual raw preparation, per-channel floors, retained-row estimators, lazy scalar/default floors and named-estimator guards. The existing sampling smoke passed before the subsequent narrow guard/doc corrections.
- Resolved-input and broader public-runner coverage: 410 tests passed. Configuration and related alignment coverage after removing the obsolete declaration assertion: 126 passed. Counts overlap.
- Independent higher-effort review found and verified corrections for an overly broad Ramsden eager guard and unit-conversion/doc wording. No remaining implementation findings; full Ruff, diff and strict change checks pass.
- Slurm job 19639181 at review-fix revision `5e1f2b23` passed the complete `py312-openghgCur` and `py313-openghgCur` tox suites: 3,985 passed, 8 skipped and 4 deselected on each interpreter. Lint and all 4 borrowed-typing tests passed.
- The same job passed `docs-full` and regenerated the public resolver API page and parent navigation. The build reported 10 duplicate/ambiguous-reference warnings concerning SiteOptions, DataConfig and RhimePreparedInputs; none concerns the new resolver. The final follow-up commit contains generated documentation and this validation record only.

## 6. Accepted 23 October ownership and error-population amendment

- [x] 6.1 Reconcile the proposal/design with the maintainer-approved #876/PR amendment; preserve dated earlier validation and record the confirmed #954/#945 aggregation-filter retirement decision.
- [x] 6.2 Move likelihood-specific floor preparation/validation beside likelihood inputs and estimator requirements beside MinimumError; verify unchanged floor policy, Ramsden reuse and joint lazy execution tests.
- [x] 6.3 Apply selection before observation-error derivation/imputation in standard, multisector, nested and copied preparation routes; verify rejected-outlier, custom-error, missing-data, retained-label and saved-preparation regressions.
- [x] 6.4 Finalize and record the resampling-filter public-interface disposition with #954/#945; verify affected callers/configuration/import behavior and migration guidance without a silent replacement statistic.
- [x] 6.5 Reconcile approved behavioral contracts and current uncertainty/usage guidance, update the Towncrier fragment, and correct earliest-anchor, resolved-builder and raw/fitted-axis API descriptions; verify source/reference consistency.
- [x] 6.6 Complete higher-effort independent review and relevant broader pytest, Ruff, OpenSpec and final Python 3.12/3.13/documentation validation; record revision-specific evidence and remaining limitations.

## 23 October amendment evidence

- Local floor-owner/configuration/builder coverage: 137 passed. Shared-estimator/assembly/Ramsden coverage: 72 passed, 1 deselected; the final narrower Ramsden lazy-coordinate regression run passed 27, 1 deselected. Counts overlap.
- Retained-population preparation coverage: 124 broader tests passed, including nested contracts, real-data prepared serialization and observation errors; 12 orchestration tests and 5 actual preparation/full-run/save-reopen/selection cases passed.
- Retirement/configuration/CLI coverage: 227 passed. The final combined runner/model/staged/documentation regression run passed 463 tests; these are overlapping focused counts, not a total.
- Independent higher-effort reviews verified scientific ordering, floor ownership, retirement rejection before computation, and corrected public/docstring/active-plan descriptions. Reported lazy-coordinate and stale-ordering findings were fixed; no remaining scoped findings.
- Full Ruff, diff checks and strict validation of both affected active OpenSpec changes pass.
- Slurm job 19650407 at `db5e65b7` ran the complete Python 3.12 and 3.13 suites: each had 4,022 passed, 8 skipped, 4 deselected and the same single stale integration assertion failure. The test incorrectly required mapping identity after observation-error preparation, whose documented contract returns a fresh mapping while borrowing complete supplied-error datasets. A test-only correction checks retained sites and dataset identity instead; all 8 custom-basis integration tests and Ruff passed locally, and independent higher-effort review confirmed the correction. A two-interpreter retry was pending at that point; the final complete-matrix results below supersede that interim status.
- The same job passed lint, all 4 borrowed-typing tests and `docs-full`, generating the likelihood-input API page and navigation. The documentation build reported 5 duplicate/ambiguous-reference warnings concerning SiteOptions, DataConfig and RhimePreparedInputs; none concerns the new API page.
- Later PR review requested a separate #954 aggregation-filter prerequisite, narrower specification placement and removal of the special tracer compatibility rule. The scientific retirement decision remained accepted. Maintainer agreement, implementation and final validation are recorded in section 7 below.

## 7. Later 23 October review and prerequisite split

- [x] 7.1 Extract aggregation-filter retirement to draft PR #971 on current devel, preserving the pre-filter imputation population there; rebase #964 onto that prerequisite and preserve independent quadrature, custom-error and sensitivity assertions when resolving the three conflicts.
- [x] 7.2 Retire special `use_tracer` compatibility handling as a separately identified #944/#719 slice; supplied false and true values receive ordinary unsupported-option validation, with omission valid.
- [x] 7.3 Preserve #960's pinned archive and add dated affected-source comparisons/plain witnesses; adapt #966's live configuration checks to the new model/likelihood owners without changing historical expectations.
- [x] 7.4 Agree the proposed configuration/scientific-input/handoff contract placement before further specification edits; separate behavior from helper instructions, rename the retired-filter scenario and split the nested population/site/configuration claims.
- [x] 7.5 Complete independent review and final Python 3.12/3.13, lint, borrowed typing and documentation validation for both the isolated prerequisite and restacked dependent revision; update the PR and linked issue evidence.
- [x] 7.6 Reject retired global filter names with empty/metadata-only input while preserving inactive absent-site handling for ordinary unknown names; verify the bounded correction on both branches and supported interpreters.

### Split evidence

- PR #971 at `6d93f51c` is based on devel `3b27c542`, including merged #966. Its five affected modules pass 554 tests; changed-path Ruff and diff checks pass. The acquired-population fallback witness is unchanged. Higher-effort independent review found no scoped findings.
- The prerequisite adds inspected-source/independent-arithmetic aggregation witnesses and a dated #960 ledger addendum. All 17 existing archived fixtures and prior ledger rows remain unchanged; no historical execution or replacement resampling algorithm is claimed.
- The separate tracer slice passes 248 focused configuration/compatibility/CLI checks with Ruff and independent higher-effort review clear. These counts precede final restacked-revision validation and overlap other focused runs.
- The maintainer explicitly agreed the bounded configuration/scientific-input/handoff placement after inspecting #971. The configuration contract retains option rules; the new scientific-inputs capability has three numerical requirements; the existing prepared/sample handoff owners retain durable reuse and exact replay. Helper/wrapper instructions remain development guidance. Nested site, population and averaging-policy claims are separate. Final validation is recorded below; both PRs remain draft for maintainer review.
- The restacked migration/configuration-owner, custom-basis, observation-error, alignment, fitted-replay and tracer checks pass 127 tests, with 314 unrelated runner cases deselected. Ruff, diff, independent arithmetic and cited-test checks pass. Independent higher-effort review confirms all 18 prior archive files and the ledger prefix are unchanged; the new dated comparison distinguishes inspection from execution.
- The prerequisite's approved filter contract is published as `ffe0aee4`. Independent higher-effort review confirms the final configuration/scientific-input/handoff split matches the implementation, with no remaining scoped findings. Both active changes and both affected main specs pass strict validation; main/delta numerical requirements and the shared prerequisite filter requirement match exactly. Scenario headings and relative links are checked.

### Final validation, 23 October 2026

- Slurm array `19651924` completed both elements successfully with exit code zero. Element `0` tested isolated prerequisite PR #971 at `ffe0aee4`: the complete `py312-openghgCur` and `py313-openghgCur` suites each passed 3,923 tests, with 8 skipped and 4 deselected. Lint, all 4 borrowed-typing tests and `docs-full` passed.
- Element `1` tested dependent PR #964 at `1487ccf9`: the complete Python 3.12 and 3.13 suites each passed 4,086 tests, with 8 skipped and 4 deselected. Lint, all 4 borrowed-typing tests and `docs-full` passed. This is a fresh complete-matrix pass, including the corrected custom-basis assertion and #966's live owner checks.
- The prerequisite documentation build reported 11 warnings and the dependent build 17. Each contained one introduced short heading underline in the dated migration addendum; the same one-character correction is applied on both branches and verified by a bounded heading/whitespace check. The remaining 10 and 16 warnings concern existing SiteOptions duplicate indexing and ambiguous RhimePreparedInputs/DataConfig references. No missing-download or new-module import warning was reported.
- Independent higher-effort review verified the test counts and warning classification, the rendered scientific-input ownership map and migration/API pages, and the bounded documentation correction. Rendered checks covered 342 local links and 29 downloads on the dependent pages, and 64 links plus one download on prerequisite pages. Two new I6 addendum links used a table caption instead of an existing section anchor; both now target the verified section without changing the historical prefix. Unrelated API-generator formatting noise was discarded.
- The first follow-up changed only documentation and this evidence record; executable source, tests and historical expectations remained those of the tested revisions. The full matrix was not repeated for the underline/link/status corrections, which passed bounded source and existing-rendered-target checks. Broader handoff-programme acceptance, centered BC work and release/merge decisions retain their separate owners.

### Filter request boundary follow-up

- The linked #954/#719 records now reference independently prepared PR #973, reserved for later reconciliation. It does not supersede #971/#964; its source observations prompted a bounded check of our filter request boundary.
- Independent higher-effort reproduction found that global retired names were lost during expansion when there were no observation sites, and that ordinary unknown names in inactive absent-site mapping entries were being rejected rather than ignored as on the base. Both concern the existing retirement/rejection scope; they do not change numerical selection or imputation policy.
- The correction and empty/metadata-only/absent-site regressions are implemented on #971 at `678c8bb4` and propagated to #964. Local filter coverage passes 82 tests, with Ruff, diff checks and independent higher-effort review clear. The complete-matrix successes above remain valid for their exact tested revisions; the changed filter boundary is verified by the scoped retry below.
- Slurm array `19655832` completed both elements successfully with exit code zero. Element `0` tested prerequisite `678c8bb4` through `py312-openghgCur` and `py313-openghgCur`: 267 passed with 2 warnings on each. Element `1` tested dependent `48b05a83`: 274 passed with 4 warnings on each. Compiler preflights used Python 3.12.11 and 3.13.7.
- The retry ran exactly `tests/test_filters.py`, `tests/test_observation_error_preparation.py`, `tests/test_rhime_configuration.py` and `tests/test_standard_phase_cli.py`. It verifies the bounded filter correction and affected callers on both supported interpreters; it is not another complete-suite run.
- Independent higher-effort evidence review confirmed the tested revisions, interpreter versions, module scope and counts. All scoped implementation and validation tasks are complete. The final commit updates this evidence record only; executable source, tests and historical expectations remain those of `48b05a83` and `678c8bb4`. Both PRs remain draft for maintainer review, with #971 preceding #964; no merge or release is performed.

## 26 October 2026 restack validation

The checked historical tasks and results above apply to their named revisions.
They do not establish validation of this restack after PR #974 and updated #971.

- [x] 8.1 Preserve canonical construction/provenance and BC basis setup from #974, generic supported-filter validation from #971, and pinned Git references to earlier #960 evidence while porting #964's retained-population and model-input behavior.
- [x] 8.2 Retain strict ordinary raw prepared schema 2; record the provisional retired-v1/ordinary-v2/optional-centered-v3 map without centered implementation. Future #965 must preserve ordinary BCs; #976 review remains open.
- [x] 8.3 Complete focused restack tests, Ruff, strict OpenSpec validation and independent review.
- [ ] 8.4 Record fresh final Python 3.12/3.13 and documentation validation for the restacked revision.

The latest maintainer direction removes historical aggregation fixtures and their
calculations from the current test tree and maintained documentation. Earlier
23 October statements about retaining plain witness files describe those dated
revisions, not the current acceptance gate. Their pinned Git locations are the
record; current tests cover supported behavior and generic invalid names only.

### Local restack checks, 26 October 2026

- Initial focused model-boundary, floor, observation-error, xarray, configuration,
  serialization and migration-owner coverage: 452 passed and one stale test
  assertion failed. The assertion accidentally checked absence of the current
  canonical schema constants; it now checks absence of the retired names.
- Additional generic filter, direct standard builder, documentation-source, nested
  mask and modern legacy-output coverage: 117 passed.
- Ruff passes for all production Python, tests and examples. Diff whitespace and
  strict validation pass for both active changes and the two affected main specs.
- Broader runner/Ramsden/raw-assembly/custom-basis/handoff/nested coverage passed
  539 tests with one deselected, including the corrected canonical-schema test.
  Its ten paths were `tests/test_prepared_inputs_serialisation.py::test_data_contracts_preserve_canonical_imports_and_prepared_schema`,
  `tests/test_rhime.py`, `tests/experimental/test_ramsden2022.py`,
  `tests/test_inversion_inputs.py`, `tests/integration/test_custom_basis_rhime_runner.py`,
  `tests/test_standard_handoffs.py`, `tests/test_multisector_handoffs.py`,
  `tests/test_standard_handoff_parity.py`, `tests/test_multisector_handoff_parity.py`
  and `tests/test_nested_preparation_contracts.py`. Counts overlap other runs.
- Independent higher-effort review found no scoped findings; its 119 selected
  tests and Ruff checks on 13 relevant modules passed. These counts overlap the
  local checks and must not be added together.
- Final CI-equivalent validation remains pending. Earlier complete-matrix records
  do not establish acceptance of this restacked revision.
