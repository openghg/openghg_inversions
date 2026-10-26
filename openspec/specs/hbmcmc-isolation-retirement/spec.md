# Independent modern execution and migration evidence

## Purpose

Define normal execution through supported modern interfaces and preserve usable
historical evidence independently of retired runtime implementations.

## Requirements

### Requirement: Modern execution is independent of migration tooling

Modern full, staged, nested, prepared-input and output-only workflows SHALL
execute through their supported canonical scientific interfaces. Normal
execution SHALL NOT require or invoke historical execution, migration tooling,
option translation or compatibility forwarding. Packaging SHALL provide the
canonical Python interfaces and installed commands under
[model selection](../rhime-model-selection/spec.md).

#### Scenario: Execute a supported modern workflow

- **WHEN** a caller runs a supported full, saved, prepared-input or output-only workflow
- **THEN** execution uses its canonical scientific interfaces directly
- **AND** migration tooling is not a runtime dependency

### Requirement: Retained modern behavior has modern owners

Canonical configuration, INI decoding, phase validation and defaults SHALL follow
[configuration](../rhime-configuration/spec.md#requirement-canonical-requests-and-phase-owned-validation).
Canonical scientific input construction SHALL follow
[scientific inputs](../rhime-scientific-inputs/spec.md).
Exact resource selection, optional exports and independently supported basis
capabilities SHALL follow [resource I/O](../rhime-resource-io/spec.md).
Model likelihood/prior semantics SHALL follow
[model composition](../rhime-model-composition/spec.md#requirement-declared-likelihood-and-prior-semantics);
derived output sample populations SHALL follow
[retained-state reconstruction](../retained-state-reconstruction/spec.md#requirement-derived-products-use-all-retained-chains).

Modern `output_format="legacy"` and `save_inversion_output` SHALL remain supported
by their output owners. A historical-looking name SHALL NOT itself remove a
scientific representation, basis reader/writer or independently supported codec.
Saved format/version admission and replay SHALL follow the prepared/sample
handoff and persistence contracts rather than a runtime retirement inventory.

#### Scenario: Resolve supported resources and outputs

- **WHEN** modern preparation or reporting selects supported resources or a supported output format
- **THEN** its owning capability resolves the selection and preserves the declared scientific behavior
- **AND** ordinary configuration validation and independent codec support remain separate

### Requirement: Development removals require affected migration evidence

Historical inputs and expected interpretations SHALL remain accessible without
importing the retired package. That evidence SHALL identify source versions and
execution routes, distinguish inspected behavior from executed observations, and
keep historical expectations separate from dated modern comparisons. It SHALL
account for the behavior actually retired, including newly discovered cases,
rather than assume that earlier evidence captures every later implementation.

Affected scientific expectations SHALL remain covered independently of retired
compatibility plumbing: unchanged cases SHALL have canonical-input regressions,
and explicit scientific changes SHALL have tests of their agreed new semantics.
An intentional scientific change SHALL NOT overwrite its historical witness or
require restoring a superseded historical expectation.

Supported scientific behavior, numerical ownership/execution, output codecs,
exact resource destinations and independent basis capabilities SHALL retain their
own contracts. Format/version admission SHALL remain a persistence decision;
Python naming changes SHALL NOT imply historical-format compatibility.

#### Scenario: Historical evidence remains usable

- **WHEN** a historical dictionary adapter is unavailable with the retired namespace
- **THEN** its source inputs and expected interpretations remain accessible without that package
- **AND** unchanged scientific cases remain covered through canonical modern inputs
- **AND** independent modern codecs remain supported

#### Scenario: Uncaptured behavior or an intentional scientific change

- **WHEN** retirement exposes an uncaptured historical behavior or an explicit scientific change
- **THEN** historical source behavior remains recorded separately from dated modern comparisons
- **AND** scientific changes are covered against their agreed modern semantics
- **AND** the evidence does not imply permission to remove other interfaces
