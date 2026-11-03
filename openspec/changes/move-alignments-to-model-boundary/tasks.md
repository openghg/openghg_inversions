# Tasks

The checklist records delivered 26 October restack acceptance before the review
cleanup below. Scientific decisions and ownership remain in
[proposal.md](https://github.com/openghg/openghg_inversions/blob/a44d724a8d087348300fda87808e8b796db78473/openspec/changes/move-alignments-to-model-boundary/proposal.md) and [design.md](https://github.com/openghg/openghg_inversions/blob/a44d724a8d087348300fda87808e8b796db78473/openspec/changes/move-alignments-to-model-boundary/design.md). Earlier checks apply only
to their named revisions; overlapping counts must not be added together.

## 1. Current restack acceptance

- [x] 1.1 Preserve #974 canonical construction/provenance and BC basis setup and #971 generic supported-filter validation; verify supported behavior without historical aggregation fixtures or calculations in the current test tree.
- [x] 1.2 Keep raw BC preparation reusable and select observations before error imputation across standard, multisector, nested and copied routes; verify retained populations, supplied errors, labels, units, borrowing and shared lazy execution.
- [x] 1.3 Resolve BC temporal policy and likelihood-owned floors after joint selected-operand/prior materialization; verify retained-row estimates, unused-operand laziness, eager builders and rejection of conflicting resolved policies.
- [x] 1.4 Preserve strict ordinary raw prepared schema 2, explicit incompatible-format rejection, exact graph-free fitted replay and independent modern codecs; verify saved/reopened preparations and standard/multisector handoffs.
- [x] 1.5 Keep centered representation provisional: schema 1 is retired, ordinary schema 2 remains supported, and optional future schema 3 is subject to #976 review and #965 preserving ordinary BCs; verify no centered implementation is claimed.
- [x] 1.6 Update configuration, usage/API guidance, examples and release notes; verify source/reference consistency, strict main/delta OpenSpec validation, Ruff, whitespace and independent review.
- [x] 1.7 Record final revision-specific validation; verify failed full-suite results and successful bounded retries remain distinct in section 3.

## 2. Review cleanup, 26 October 2026

- [x] 2.1 Remove the orphan residual-error helper and stale documentation, consolidate this acceptance/evidence record and deduplicate the named-input scenario; verify retained pollution-event and MinimumError behavior, affected tests/lint/documentation, strict main/delta validation and whitespace before marking complete.

## 3. Revision and validation evidence

| Scope / revision | Evidence | Limits |
| --- | --- | --- |
| Local restack | Initial focused coverage: 452 passed, one stale schema-name assertion failed and was corrected. Additional coverage: 117 passed. Broader runner/Ramsden/assembly/handoff/nested coverage: 539 passed, one deselected. Independent review: 119 selected tests and Ruff on 13 modules passed, no scoped findings. Production/test/example Ruff, whitespace, both active changes and both affected main specs passed. | Overlapping runs; detailed paths are in the pinned record below. |
| Full matrix, array `19727663`, element `0`, #971 `57bf326d` | Each of `py312-openghgCur`, `py313-openghgCur` and moving-dependency `py312-openghgDev`: 4,260 passed, one failed, eight skipped, four deselected. | Namespace-unavailable assertion failed in each environment. |
| Full matrix, array `19727663`, element `1`, #964 `9bf0ae66` | Each locked environment: 4,409 passed, one failed, eight skipped, four deselected. Moving environment: 4,407 passed, three failed, eight skipped, four deselected. | Namespace assertion plus two moving-dependency metadata failures; failed full-suite results. |
| Matrix auxiliary checks, both elements | Lint, all four borrowed-typing checks and `docs-full` passed. Each docs build had 16 known warnings: 13 SiteOptions duplicates, one PreparedInputs ambiguity and two DataConfig ambiguities. | Four unrelated generated-RST formatting changes were inspected and restored. |
| Correction, existing environment | `tests/test_hbmcmc_boundary.py` and `tests/test_observation_error_preparation.py`: 30 passed after namespace cleanup and test-only metadata correction; changed-test Ruff and whitespace passed. Independent review accepted the projection and checked seven rendered pages per tree and pinned archive references. | No production scientific behavior changed; no full-suite repeat. |
| Boundary retry, array `19728007`, element `0`, merged #971 `53f6b4b2` | Exactly the same two modules: 23 passed, two warnings in each of the three environments above; exit status `0:0`. | Focused retry, not a fresh full-suite pass. |
| Boundary retry, array `19728007`, element `1`, corrected #964 `d0133911` | Exactly the same two modules: 30 passed, four warnings in each of the three environments above; exit status `0:0`. | Focused retry, not a fresh full-suite pass. |
| Retry preflights | Python 3.12.11 for both Python 3.12 environments and Python 3.13.7 for Python 3.13; selected Ruff passed for both elements. Logs: `slurm-971-964-boundary-retry-19728007_0.out` and `slurm-971-964-boundary-retry-19728007_1.out`. | Complements the original matrix and its auxiliary checks. |
| Review cleanup, `2d805c17`, job `19728899` | Focused input/model-alignment/Ramsden coverage: 73 passed, one deselected; independent AST comparison confirms only the orphan helper was removed. Lint and `docs-full` passed, job `COMPLETED 0:0`. Strict specs and 52 source-heading targets passed; 445 rendered HTML links valid. Removed helper absent from HTML, inventory and search objects; MinimumError and percentile APIs retained. | No full scientific matrix repeat. This incremental documentation invocation reported four existing warnings: one SiteOptions duplicate, one PreparedInputs ambiguity and two DataConfig ambiguities; no warning-backlog fix is claimed. |

The namespace failures came from ignored retired-package directories containing
only generated CPython bytecode files. Inspection
and removal of those files and empty directories resolved discovery without
changing the assertion or production code. The metadata failures required only
projecting required `long_name`/`units` attributes: values `[5, 2]` and labels were
already correct, and retained `source="instrument"` is permitted.

#964 incorporates merged #971 at `53f6b4b2`, including removal of the obsolete
TDMCMC configuration template and its release-note update. #964 remains draft
for maintainer review. Broader handoff acceptance, centered/identifier decisions,
merge, release and archival retain their separate owners.

## 4. Pinned revision history

The full dated process and evidence record remains available at commit
`a44d724a8d087348300fda87808e8b796db78473`:

- [22-23 October implementation, amendments, prerequisite split and validation](https://github.com/openghg/openghg_inversions/blob/a44d724a8d087348300fda87808e8b796db78473/openspec/changes/move-alignments-to-model-boundary/tasks.md#tasks).
- [26 October restack checks, failed full matrix and bounded retry](https://github.com/openghg/openghg_inversions/blob/a44d724a8d087348300fda87808e8b796db78473/openspec/changes/move-alignments-to-model-boundary/tasks.md#26-october-2026-restack-validation).

Historical statements about retained aggregation witnesses describe their dated
revisions. Current acceptance covers supported behavior and generic invalid
names; historical evidence remains in pinned Git locations only.
