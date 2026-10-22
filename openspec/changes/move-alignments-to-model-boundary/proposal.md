# Proposal

## Why

Prepared standard and multisector data currently bake BC periods and minimum-error choices into numerical values, preventing reuse for compatible model requests. Issue #876 now has maintainer approval for an explicit modern-release break under #944, including minimum-error calculation after invalid observations are dropped.

## What Changes

- Preserve raw spatial BC sensitivities in preparation and resolve temporal alignment at the model boundary.
- Move minimum-error configuration and calculation to the owning likelihood; calculate named methods from retained observations.
- Keep sigma alignment model-owned and derive site positions from observation labels.
- **BREAKING**: remove baked policies from preparation APIs/configuration and reject incompatible older prepared/sample artifacts explicitly, without an adapter.
- Preserve modern output codecs, joint selected-input/prior materialization and graph-free replay.
- Keep BC-related prepared variables intact so ongoing #957 can later apply the same temporal operation to centered operands; centered arithmetic is outside this change.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `rhime-configuration`: model ownership of BC periods and minimum-error choices, retained-observation semantics, and explicit rejection of incompatible baked preparations.

## Impact

Scientific assembly, direct xarray preparation, standard/multisector and shared nested callers, model configuration, prepared and sampled records, regression tests, and maintained usage documentation. Broader persistence and consumer retirement remain with #877/#878; centered BC science remains with #955/#957. No dependency or release-number change.
