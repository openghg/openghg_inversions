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

The accepted [finite CLI target](tasks.md#cli-command-name-amendment--23-october-2026)
has one `rhime` executable and canonical `run --model ...` routes, without the
second executable or four redundant full-run aliases. Preserve effective
arguments, scientific execution and products; package/distribution names stay
unchanged. The [five-name Python target](tasks.md#named-python-alias-amendment--23-october-2026)
uses only the specified canonical functions/types, without naming-only forwarders.
Supported artifact formats/versions remain independent codec decisions; naming
cleanup neither reinterprets scientific data nor implies historical-format support.
Broader Python names/imports still require separate exact agreements. The target
specifies the required outcome; implementation status belongs to these records.

## Prepared-input factory and cleanup addendum — 24 October 2026

The maintainer explicitly approved the additional
[`PreparedInputs.from_legacy_inputs` retirement](https://github.com/openghg/openghg_inversions/pull/974#issuecomment-6380451664),
following the [bounded analysis](https://github.com/openghg/openghg_inversions/pull/974#issuecomment-6380430675).
Its exact replacement is the existing canonical
`PreparedInputs(inv_inputs=..., basis_functions=..., site_metadata=...)` constructor.
This is a separately named compatibility-constructor removal under #945/#663,
not a sixth alias, broad constructor permission or #951 basis disposition.
Canonical xarray coordinates, site/period order, supplied-basis provenance,
borrowed arrays and existing persisted schema/codec behavior remain supported.

[Review 5637015484](https://github.com/openghg/openghg_inversions/pull/974#pullrequestreview-5637015484)
also requests ordinary removed-import absence, rejection of ambiguous sources
without parsing them, one outer CO2 request-name-validation owner and cleanup of
dead documentation branches. Applicable component parsers retain their own value
validation. Independent basis keyword compatibility remains supported with #951;
its resolution logic can stay local to the consumer without a root helper or
new removal schedule.
Mechanisms, evidence and fresh check status belong in design/tasks and the dated
#960 addendum. Prior revision-specific validation remains historical evidence.

## Why

At the start of PR #869, modern RHIME imported HBMCMC compatibility code, preventing the legacy
subpackage from being retired independently. Review of PR #869 also found that
the actual legacy entry point bypassed resource translation even though modern
runners perform it.

## What Changes

- Remove the complete `openghg_inversions.hbmcmc` namespace, executable wrapper,
  configuration adapters and legacy data/basis utilities without forwarding shims.
- Accept only canonical modern runtime options. Reject the enumerated historical
  spellings, `hbmcmc`/`hbmcmc_postprocessing` output aliases and every explicit
  `use_tracer` occurrence through ordinary unsupported-option validation.
- Use the finite canonical CLI and five-name Python targets above, preserving
  scientific operations, arguments, result contents and independently supported
  codecs and basis capabilities.
- Keep INI decoding, path resolution and packaged preparation defaults independent
  of HBMCMC. Normal execution does not invoke migration helpers.
- Preserve historical source evidence and port affected scientific coverage to
  canonical inputs. Keep concrete comparisons, ownership and sequencing in the
  [delivery checklist](tasks.md#migration-evidence-checklist-and-delivery-order).

Historical 0.8 isolation established one-way dependencies and legacy-only request
translation. That is delivery history, not a required transitional runtime branch
in the target interface. The historical mechanism is recorded in [design.md](design.md).

## Capabilities

### New Capabilities

- `hbmcmc-isolation-retirement`: Modern execution without the historical
  namespace, independent retained capabilities and usable migration evidence.

### Modified Capabilities

- `rhime-configuration`: Canonical-only requested options and concrete scientific
  results, with INI decoding independent of execution and no historical translator.

## Impact

The historical [PR #869](https://github.com/openghg/openghg_inversions/pull/869)
and the resource-I/O capability provide the isolation groundwork. The latter owns path
conversion, explicit resource loading and packaged-map selection; this change
owns retirement. Resource selection and independent basis/artifact support
retain their separate scientific owners.

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
