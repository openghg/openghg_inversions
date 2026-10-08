# Proposal

## Why

[Issue #804](https://github.com/openghg/openghg_inversions/issues/804) needs an
inspectable in-memory representation of the complete resolved RHIME request.
The design merged in #809 and implemented in #813 retained a separate preparation
configuration, obscuring that role and preserving the overlap the redesign was
meant to simplify.

## What Changes

- Make `RhimeConfig` the complete resolved standard/multisector request. Put
  acquisition and preparation fields directly on it; remove the proposed
  `RhimePreparationConfig` and its projection methods.
- Reuse the existing site-options record, `RhimeModelSpec`, `RhimeOutputSpec`
  and `RhimeSampler`. Give each a distinct contract; do not duplicate their
  model/output/sampling attributes in another settings class.
- Decode INI into a raw mapping, apply supported overrides, then resolve aliases,
  defaults and all site shorthand before returning configuration. File and
  Python inputs share this format-neutral boundary.
- Keep requested sites in configuration and authoritative retained sites in
  acquired/prepared data. Create `RhimeRunSpec` after preparation.
- Use `read_rhime_ini` for raw decoding and `retrieve_inversion_data` for fresh
  acquisition covering surface and column observations. Retain the old public
  acquisition name as a deprecated wrapper with the same signature and return.
- Preserve public scientific input/return contracts and ownership through small
  adapters. Remove requirements to reconstruct original spellings or sparse
  defaults solely to preserve historical staged hashes.

This revision updates planning on #813; its Python implementation still needs
reconciliation. [tasks.md](tasks.md) tracks that work. INI remains the file
frontend. CO2 recipe configuration is unchanged. Manifest, identity and release
compatibility policy remain with [#808](https://github.com/openghg/openghg_inversions/issues/808)
and [#802](https://github.com/openghg/openghg_inversions/pull/802).

## Capabilities

### New Capabilities

- `rhime-configuration`: One complete resolved requested configuration before
  scientific work, with equivalent shorthand, clear contracts and retained-site
  execution metadata.

### Modified Capabilities

None. This checkout has no synced durable capability specs.

## Impact

Reconcile the existing `rhime.params` owner, ordinary runners, acquisition and
preparation adapters, public exports, focused tests and configuration guidance.
Keep the landed #773/#774 numerical-data and sampler owners and #807 tracer
handling. Existing independent builders, prepared-input APIs and nested/shim
consumers retain their contracts. No dependency, equation, new file format,
shared context hierarchy or hashing framework is introduced.
