# Proposal

## Why

[Issue #804](https://github.com/openghg/openghg_inversions/issues/804) needs an
inspectable in-memory representation of the resolved RHIME request. Today partly
raw options reach acquisition, and site shorthand is expanded repeatedly;
configuration and retained-run metadata also have overlapping meanings.

## What Changes

- Introduce `RhimeConfig` for the complete resolved standard/multisector request:
  preparation choices, model specification, output specification and the existing
  `RhimeSampler` settings object.
- Keep INI decoding separate from semantic resolution. Apply supported overrides
  first, then resolve aliases, defaults and site shorthand while constructing
  `RhimeConfig`. File and Python inputs share this format-neutral boundary.
- Introduce only the missing `RhimePreparationConfig`; reuse the existing
  site-options, model, sector, likelihood, output and sampler types.
- Keep requested sites in configuration. Acquisition/preparation select retained
  options by label; construct `RhimeRunSpec` with retained sites after preparation.
- Preserve supported public shorthand and return contracts through small adapters,
  caller-owned inputs, scientific behavior and existing direct sampler APIs.
- Document record roles and add focused equivalence, override, early-failure and
  retained-site checks at implementation time.

This is a planning-only change. Implementation tasks remain deferred for review.
INI remains the supported file frontend; format neutrality enables later
frontends without introducing one here. CO2 recipe configuration is unchanged.
Hash/manifest policy remains with
[#808](https://github.com/openghg/openghg_inversions/issues/808) and
[PR #802](https://github.com/openghg/openghg_inversions/pull/802).

## Capabilities

### New Capabilities

- `rhime-configuration`: Resolve a complete requested-run configuration before
  acquisition, with equivalent external shorthand, compatible entry adapters and
  distinct retained-run metadata.

### Modified Capabilities

None. This checkout has no synced durable capability specs.

## Impact

The existing `rhime.params` owner, ordinary runners, acquisition/preparation
consumers and configuration guidance change. Build on landed #773/#774 ownership
in `inversion_data.acquisition`, `inversion_data.prepared_inputs` and
`inference.sampling`, preserving public compatibility exports. Preserve #807's
tracer handling and existing staged transport contracts. Independent builders
and prepared-input runners keep their contracts. No new dependency, scientific
equation, file format, stage ownership, cache policy or identity protocol is
introduced.
