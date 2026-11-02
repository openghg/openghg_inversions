# Tasks

Delivery is coordinated with #866. Record implementation evidence before checking
tasks; this planning split does not complete configuration or handoff work.

## 1. Configuration ownership and phase resolution

- [x] 1.1 Move fields/defaults into `RhimeConfig.data: DataConfig` and migrate
  grouped callers to `data`; verify full/phase-equivalent choices, no duplicate
  defaults, no array access and unchanged caller inputs.
- [x] 1.2 Reuse runner/frontend alias translation outside HBMCMC after raw overrides and
  select applicable groups before resolution; verify canonical/deprecated
  precedence, independent flux-alias conflict rules, and one early error containing
  all unsupported or wrong-phase keys. Verify phase-only resolution imports no
  HBMCMC code and constructs no unrelated complete request.
- [x] 1.3 Bind model choices against requested/retained facts; verify BC defaults,
  omitted/equivalent/conflicting `bc_freq`, retained-site alignment and rejection
  of scientific conflicts without reacquisition or implicit preparation.

## 2. Integration and completion

- [x] 2.1 Migrate CLI, configured runners and examples with #866; pass relevant
  configuration, CLI and full/saved-route checks, preserving the prerequisite
  resource-I/O compatibility changes and established sampler behavior.
- [x] 2.2 Record merged delivery and scenario evidence, then sync and archive this
  configuration change; confirm unrelated handoff tasks are not marked complete.
  As part of sync/archive, update remaining active consumers and the developer
  contract map from this change to the delivered capability; retain historical
  references only via the actual archive location. Check links in promoted specs
  and remaining active changes after archival. Requirement links use durable
  capability URLs so copying them does not depend on source-directory depth.

## Implementation evidence, 14 October 2026

PR #866 implements this reviewed grouping on devel `52a6b12c`, preserving #869's
canonical constructor, resource/path owners and HBMCMC boundary. Configuration,
shim and full-run coverage passes 480 tests; scientific/resource/custom-consumer
coverage passes 159. Phase CLI tests check independent resolution, canonical-file
versus deprecated-override precedence, all unsuitable keys in one error, flat
model claims, and distinct file/invocation path origins. Model binding tests cover
BC availability, omitted/equivalent/conflicting frequency claims and retained sites.
The handoff change's standard acceptance compares saved results with full
execution and independent density/native-flux calculations. Task 2.2 remains
open until merged delivery and explicit promotion/archive; unrelated family
handoff tasks remain unchecked.

Full standard/multisector grouped aliases now use the same frontend translator;
canonical grouped constructors still reject aliases without warnings. The final
configuration/phase/compatibility/HBMCMC-boundary selection passed 172 tests.

## Closeout — 16 October 2026

PR #866 merged at `040e0c6188905218fd7d46bda2c06f2f2ade12fb` after explicit
maintainer readability acceptance. Runtime head `da832a5a` passed full locked
Python 3.12/3.13 suites (Slurm 19436596), with no introduced strict docs/type
diagnostics. The configuration delta is promoted to the durable capability;
active consumers and the contract map now reference that baseline. This record
is archived at `2026-10-16-group-rhime-configuration-by-owner`. Unrelated
multisector/CO2 handoff tasks remain open in OPE-207/#880.
