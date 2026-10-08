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
- Promote the existing aligned selector record to public `SiteOptions`, exported
  from `inversion_data`; reuse it alongside `RhimeModelSpec`, `RhimeOutputSpec`
  and `RhimeSampler`. Give each a distinct contract; do not duplicate their
  model/output/sampling attributes in another settings class.
- Let `read_rhime_ini` own INI parsing and interpretation, apply supported
  overrides, and return a complete `RhimeConfig`. Share RHIME semantic resolution
  with Python inputs without imposing INI sections or decoding rules on other
  frontends. Reuse site-alignment helpers for common shorthand.
- Keep requested sites in configuration and authoritative retained sites in
  acquired/prepared data. Create `RhimeRunSpec` after preparation.
- Remove `RhimeRunnerSetup`, `make_rhime_runner_setup` and `resolve_rhime_options`;
  migrate ordinary, nested, staged, shim and example consumers to `RhimeConfig`.
- Use `retrieve_inversion_data` for fresh acquisition covering surface and column
  observations. Retain the old public acquisition name as a deprecated wrapper
  with the same signature and return. Preserve `params_from_config` as the
  dictionary-returning INI compatibility adapter.
- Collapse retrieval/reload forwarding layers into `load_rhime_data`, the shared
  supplied-data/cache/fresh-acquisition boundary returning `RhimeMergedData`.
- Preserve public scientific input/return contracts and ownership through small
  adapters. Remove requirements to reconstruct original spellings or sparse
  defaults solely to preserve historical staged hashes.

This change is delivered on #813; [tasks.md](tasks.md) tracks implementation,
validation and delivery. INI remains the file
frontend. CO2 recipe configuration is unchanged. Manifest, identity and release
compatibility policy remain with [#808](https://github.com/openghg/openghg_inversions/issues/808)
and [#802](https://github.com/openghg/openghg_inversions/pull/802).
Configuration serialization, resolved-settings logging and an INI writer are
deferred to [#814](https://github.com/openghg/openghg_inversions/issues/814).
Serializability is the intended direction; no writer/export API is added here.

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
handling. Independent builders and prepared-input APIs retain their scientific
contracts. Nested/shim/staged consumers migrate off the internal setup bundle;
retaining that type or its helper returns is not a compatibility requirement.
No dependency, equation, new file format,
shared context hierarchy or hashing framework is introduced.
