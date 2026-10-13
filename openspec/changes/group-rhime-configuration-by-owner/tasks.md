# Tasks

Delivery is coordinated with #866. Record implementation evidence before checking
tasks; this planning split does not complete configuration or handoff work.

## 1. Configuration ownership and phase resolution

- [ ] 1.1 Move fields/defaults into `RhimeConfig.data: DataConfig` and migrate
  grouped callers to `data`; verify full/phase-equivalent choices, no duplicate
  defaults, no array access and unchanged caller inputs.
- [ ] 1.2 Reuse alias translation after raw overrides and select applicable groups
  before resolution; verify canonical/deprecated precedence and one early error
  containing all unsupported or wrong-phase keys.
- [ ] 1.3 Bind model choices against requested/retained facts; verify BC defaults,
  omitted/equivalent/conflicting `bc_freq`, retained-site alignment and rejection
  of scientific conflicts without reacquisition or implicit preparation.

## 2. Integration and completion

- [ ] 2.1 Migrate CLI, configured runners and examples with #866; pass relevant
  configuration, CLI and full/saved-route checks, preserving the prerequisite
  resource-I/O compatibility changes and established sampler behavior.
- [ ] 2.2 Record merged delivery and scenario evidence, then sync and archive this
  configuration change; confirm unrelated handoff tasks are not marked complete.
  As part of sync/archive, update remaining active consumers and the developer
  contract map from this change to the delivered capability; retain historical
  references only via the actual archive location. Check links in promoted specs
  and remaining active changes after archival. Requirement links use durable
  capability URLs so copying them does not depend on source-directory depth.
