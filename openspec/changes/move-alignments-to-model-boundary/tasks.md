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
- [ ] 4.2 Complete higher-effort independent review, focused and relevant broader pytest, Ruff, diff/OpenSpec checks, and required final CI-equivalent validation; record concrete evidence and remaining limitations.

## Validation evidence

- Configuration worker: 171 focused tests passed; anchor fixes: 86 tests passed.
- Core worker: 183 focused tests passed.
- Caller/handoff integration: 469 passed with two stale failures from concurrent edits; fresh retries of both pass.
- Final broader observation-error/configuration/serialization/documentation checks: 220 passed.
- Final direct-builder boundary suite: 84 passed.
- Actual two-frequency preparation/sample/replay checks: 2 passed, covering standard and multisector with recalculation disabled on replay.
- Full changed-area Ruff, diff checks and strict change validation pass. Higher-effort configuration/core/persistence/documentation reviews completed; reported anchor, direct-builder and tutorial findings were fixed. Counts overlap and are not a combined total.
- Final Slurm Python 3.12/3.13 CI-equivalent validation remains pending.
