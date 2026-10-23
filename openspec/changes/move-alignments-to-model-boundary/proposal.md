# Proposal

## Why

Prepared standard and multisector data previously baked BC periods and minimum-error choices into numerical values, preventing reuse for compatible model requests. Issue #876 has maintainer approval for an explicit modern-release break under #944, including minimum-error calculation after invalid observations are dropped. The accepted 23 October amendment also requires observation-error imputation to use only observations retained by selection filters and localizes floor policy with likelihood-input preparation.

## What Changes

- Preserve raw spatial BC sensitivities in preparation and resolve temporal alignment at the model boundary.
- Move minimum-error configuration and calculation to the owning likelihood; calculate named methods from retained observations.
- Keep estimator requirements with the shared numerical estimator and likelihood-specific floor enablement/preparation/validation with likelihood-input preparation. The backend-neutral model-input stage composes those operations with BC resolution.
- **SCIENTIFIC CHANGE**: apply observation-selection filters before deriving or imputing observation-error standard deviations. Preserve supplied errors and existing component/NaN rules; rejected observations must not contribute to fallback estimates.
- Keep sigma alignment model-owned and derive site positions from observation labels.
- **BREAKING**: remove baked policies from preparation APIs/configuration and reject incompatible older prepared/sample artifacts explicitly, without an adapter.
- Preserve modern output codecs, joint selected-input/prior materialization and graph-free replay.
- Keep BC-related prepared variables intact so ongoing #957 can later apply the same temporal operation to centered operands; centered arithmetic is outside this change.

## Capabilities

### New Capabilities

- `rhime-scientific-inputs`: backend-neutral retained inputs, selected-population observation-error derivation, model-owned BC temporal operands and likelihood-owned floors.

### Modified Capabilities

- `rhime-configuration`: accepted filter/option names, phase/group resolution, model ownership of BC-period and likelihood-floor choices, and explicit tracer-option retirement.
- `prepared-recipe-handoff` and `sampled-recipe-handoff`: reusable raw preparation, explicit older-format rejection and fitted-operand replay.
- `recipe-execution`: full and saved recipes share the same numerical input and handoff contracts.

## Contract placement

Configuration owns accepted values/defaults, group ownership, inherited BC use,
recorded-fact checks and configured anchor resolution. The scientific-input
capability owns numerical populations, labels, units, borrowing, resolved-policy
compatibility and unused-operand execution. Prepared/sample handoff capabilities
own artifact compatibility and replay. Helper placement, direct canonical calls
and wrapper guidance remain development principles rather than behavioral
requirements. The separately approved #954 filter-name requirement is retained
unchanged; no resampling replacement is introduced.

## Impact

Scientific assembly, direct xarray preparation, standard/multisector and shared nested callers, model configuration, prepared and sampled records, regression tests, and maintained usage documentation. Broader persistence and consumer retirement remain with #877/#878; centered BC science remains with #955/#957. No dependency or release-number change.
