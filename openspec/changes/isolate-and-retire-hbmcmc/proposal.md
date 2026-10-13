# Proposal

## Why

Modern RHIME currently imports HBMCMC compatibility code, preventing the legacy
subpackage from being retired independently. Review of PR #869 also found that
the actual legacy entry point bypasses resource translation even though modern
runners perform it.

## What Changes

- Establish a one-way dependency: HBMCMC may call modern APIs; production code
  outside HBMCMC must not import, re-export or execute through HBMCMC.
- Keep historical fixedbasis policy and resource selection inside the HBMCMC
  input path, after winning overrides and before canonical construction.
- Give retained modern behavior modern owners, including supported runner-owned RHIME aliases,
  INI decoding, path conversion and domain-specific packaged preparation maps.
  Small duplication inside HBMCMC is acceptable.
- **BREAKING (0.8):** Remove modern re-exports of HBMCMC-only adapters and route
  historical selectors through HBMCMC only. Preserve separately supported modern
  aliases and scientific behavior without a reverse dependency.
- **BREAKING (0.9):** Remove the complete `openghg_inversions.hbmcmc` subpackage,
  including its wrapper and legacy data/basis utilities. This is removal, not
  merely deprecation of individual entry points.
- Add a production import-boundary check and migration guidance distinguishing
  this removal from unrelated 0.9 deprecations.

## Capabilities

### New Capabilities

- `hbmcmc-isolation-retirement`: Dependency direction, legacy-only input
  translation, retained modern behavior, and complete HBMCMC removal in 0.9.

### Modified Capabilities

- `rhime-configuration`: Separate modern alias handling and INI decoding from
  HBMCMC translation; remove the requirement to call an HBMCMC-owned translator
  from canonical construction.

## Impact

Coordinate with [PR #869](https://github.com/openghg/openghg_inversions/pull/869)
and the active `move-rhime-resource-io-to-runners` change. The latter owns path
conversion, explicit resource loading and packaged-map selection; this change
owns dependency direction and retirement. The grouped-configuration amendment
must preserve this boundary when its overlapping configuration delta is synced.

Affected consumers include RHIME exports, configuration, full/staged/nested
runners, copied examples, the HBMCMC CLI and dictionary adapter, and their tests.
No model equations, sampling defaults, artifact codecs, grouped configuration or
new handoff formats are introduced. Output formats implemented outside HBMCMC
are not removed merely because their names or products mention HBMCMC.

The initial 0.8 implementation is on PR #869, with approved review corrections
in progress. Acceptance and durable-spec promotion remain pending.
Keep requirements in active changes until their corresponding behavior is
implemented and accepted; 0.9 deletion is a later release task, not a prerequisite
for completing the 0.8 resource-I/O implementation.
