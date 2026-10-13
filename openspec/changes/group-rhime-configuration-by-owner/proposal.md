# Proposal

## Why

Full and saved RHIME routes need the same resolved choices without requiring
unrelated model, sampler or output settings. These approved configuration
amendments support #866 but remain unimplemented; they belong in a change rather
than the durable configuration baseline.

## What Changes

- **BREAKING:** Group existing acquisition/preparation fields under
  `RhimeConfig.data: DataConfig`, with `data` as the grouped mapping key.
- Resolve applicable groups independently, select named arguments explicitly,
  preserve alias precedence and report all unsuitable override keys together.
- Bind saved-phase model choices to authoritative prepared facts while preserving
  the agreed BC-use, BC-frequency and sampler contracts.
- Keep path interpretation and optional basis saving with the separate
  [resource-I/O change](../move-rhime-resource-io-to-runners/proposal.md).
- Share runner/frontend alias owners across full and phase-only resolution under the
  separate [HBMCMC isolation/retirement contract](../isolate-and-retire-hbmcmc/proposal.md);
  historical translation remains in HBMCMC without a reverse dependency.

## Capabilities

### Modified Capabilities

- `rhime-configuration`: Grouped requested configuration and phase-only resolution.

## Impact

Delivery is coordinated with #866 and the
[handoff change](../replace-recipe-workflow-with-reusable-handoffs/proposal.md).
Affected owners include `rhime/params.py`, CLI adapters and configured runners.
This record preserves the #867 review decisions; it adds no runtime implementation
or new configuration framework. Sync only after the applicable implementation
and acceptance checks are complete.
