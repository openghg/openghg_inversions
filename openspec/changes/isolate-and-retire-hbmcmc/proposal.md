# Proposal

Positional dates and `--kwargs` remain open for discussion and possible revision.
The finite CLI naming amendment preserves their current meanings for that slice;
it does not establish their final design.

## Development-boundary amendment — 23 October 2026

The maintainer-approved #944 treatment and bounded #961 amendment supersede the
future 0.8/0.9 compatibility schedule in this record. Delivered isolation remains
historical evidence; complete #719 retirement remains pending. Each removal/change
PR must preserve the durable
[evidence guarantees](../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence).
The [delivery checklist](tasks.md#migration-evidence-checklist-and-delivery-order)
records the finite namespace/translation scope, pinned #960 comparison,
per-PR checks, scientific regression migration and current owners. Completed
evidence does not approve unresolved interface choices or a general exception
for public signatures/results.

Finished converters/launchers, maintenance publishing and integrated acceptance
remain M2 release gates, not M1 development-removal prerequisites. Final release
and public-alias decisions stay with #945; independent basis decisions stay with
#951. This amendment removes no production code and establishes no release
readiness.

The later [finite CLI amendment](tasks.md#cli-command-name-amendment--23-october-2026)
permits retiring the second executable and four redundant full-run commands in
favor of `rhime run --model ...`, while preserving their argument handling,
scientific execution and products. It is accepted permission, not delivered
removal. Package/distribution names are unchanged.
The [five-name Python alias map](tasks.md#named-python-alias-amendment--23-october-2026)
also permits retiring those exact aliases without naming-only forwarders,
preserving canonical functions/types/constructors and persisted artifacts.
Broader Python names/imports still require separate exact agreements.

## Why

At the start of PR #869, modern RHIME imported HBMCMC compatibility code, preventing the legacy
subpackage from being retired independently. Review of PR #869 also found that
the actual legacy entry point bypassed resource translation even though modern
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
- **BREAKING (development):** After the affected evidence and regression gate,
  remove the complete `openghg_inversions.hbmcmc` subpackage,
  including its wrapper and legacy data/basis utilities. This is removal, not
  merely deprecation of individual entry points.
- Add a production import-boundary check and migration guidance distinguishing
  this removal from unresolved public-alias and independent basis decisions.

## Capabilities

### New Capabilities

- `hbmcmc-isolation-retirement`: Dependency direction, legacy-only input
  translation, retained modern behavior, and gated HBMCMC development retirement.

### Modified Capabilities

- `rhime-configuration`: Separate modern alias handling and INI decoding from
  HBMCMC translation; remove the requirement to call an HBMCMC-owned translator
  from canonical construction.

## Impact

Coordinate with [PR #869](https://github.com/openghg/openghg_inversions/pull/869)
and the delivered resource-I/O capability. The latter owns path
conversion, explicit resource loading and packaged-map selection; this change
owns dependency direction and retirement. The grouped-configuration amendment
must preserve this boundary when its overlapping configuration delta is synced.

Affected consumers include RHIME exports, configuration, full/staged/nested
runners, copied examples, the HBMCMC CLI and dictionary adapter, and their tests.
No model equations, sampling defaults, artifact codecs, grouped configuration or
new handoff formats are introduced. Output formats implemented outside HBMCMC
are not removed merely because their names or products mention HBMCMC.

The 0.8 implementation is accepted for merge in PR #869 and promoted to
[rhime-configuration](../../specs/rhime-configuration/spec.md),
[rhime-resource-io](../../specs/rhime-resource-io/spec.md) and the
[isolation contract](../../specs/hbmcmc-isolation-retirement/spec.md).
Complete development retirement remains in this change's delta; its deletion
and final archive tasks remain unchecked. Earlier 0.8 descriptions record
delivered isolation rather than a renewed compatibility window.
