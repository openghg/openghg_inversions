# Tasks

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
- [ ] 6.6 Complete higher-effort independent review and relevant broader pytest, Ruff, OpenSpec and final Python 3.12/3.13/documentation validation; record revision-specific evidence and remaining limitations.

## 23 October amendment evidence

- Local floor-owner/configuration/builder coverage: 137 passed. Shared-estimator/assembly/Ramsden coverage: 72 passed, 1 deselected; the final narrower Ramsden lazy-coordinate regression run passed 27, 1 deselected. Counts overlap.
- Retained-population preparation coverage: 124 broader tests passed, including nested contracts, real-data prepared serialization and observation errors; 12 orchestration tests and 5 actual preparation/full-run/save-reopen/selection cases passed.
- Retirement/configuration/CLI coverage: 227 passed. The final combined runner/model/staged/documentation regression run passed 463 tests; these are overlapping focused counts, not a total.
- Independent higher-effort reviews verified scientific ordering, floor ownership, retirement rejection before computation, and corrected public/docstring/active-plan descriptions. Reported lazy-coordinate and stale-ordering findings were fixed; no remaining scoped findings.
- Full Ruff, diff checks and strict validation of both affected active OpenSpec changes pass. Fresh complete Python 3.12/3.13, lint, borrowed typing and documentation/API regeneration remain pending; the earlier 3,985-pass suites apply to the 22 October executable revision only.
