# HBMCMC isolation and retirement

## Purpose

Define modern execution without the historical HBMCMC namespace, retaining
independent scientific and artifact capabilities and usable historical evidence.

## Requirements

### Requirement: Modern execution has no HBMCMC dependency

The modern wheel and source distribution SHALL contain no
`openghg_inversions.hbmcmc` subpackage, executable wrapper or HBMCMC-owned
configuration, data or basis utility. No import alias, re-export, forwarding shim
or executable entry point SHALL recreate that namespace. Modern full, staged,
nested, prepared-input and output-only workflows SHALL operate independently of
HBMCMC. Normal modern execution SHALL NOT import or invoke migration tooling.

#### Scenario: Install the modern distribution

- **WHEN** a user installs the modern wheel or source distribution
- **THEN** it contains no HBMCMC subpackage or executable entry point
- **AND** importing the old namespace cannot reach a forwarding shim
- **AND** supported modern workflows remain usable

#### Scenario: Execute without historical tooling

- **WHEN** a modern full, saved, prepared-input or output-only workflow executes
- **THEN** it uses its supported modern interfaces without loading HBMCMC or migration helpers

### Requirement: Historical scientific quirks remain outside modern construction

Modern runner and builder interfaces SHALL NOT accept `preserve_legacy_likelihood`,
`_compatibility_unused_sigma_settings`, `_compatibility_minimum_error_floor`,
`compatibility_output_chain`, `retain_unused_sigma` or `derived_output_chain`.
Modern scientific construction SHALL NOT reproduce historical boundary-only
pollution-event likelihood behavior, disconnected sigma-prior variables or
additive fixed-error minimum-error floors. Modern standard offsets SHALL remain
supported with their declared full-baseline semantics.

Derived outputs SHALL use all retained chains, including legacy-format summaries.
Reproducing historical equations, wrapper defaults or chain-0-only summaries
SHALL require matching historical software rather than a modern compatibility
switch. Modern legacy-format codecs and explicit save paths SHALL remain
independently supported.

#### Scenario: Modern derived products use all retained chains

- **WHEN** a modern run writes derived products, including legacy-format summaries
- **THEN** those products use all retained chains under the ordinary output policy
- **AND** no private compatibility switch selects only chain 0

### Requirement: Retained modern behavior has modern owners

Supported modern INI decoding, canonical configuration construction, path
conversion and vendored preparation defaults SHALL remain available independently
of HBMCMC. Runner/acquisition boundaries SHALL accept canonical names and reject
the historical spellings and output-format aliases enumerated in
[Compatible public adapters and scientific choices](../rhime-configuration/spec.md#requirement-compatible-public-adapters-and-scientific-choices)
through ordinary unsupported-option validation before scientific work. They
SHALL NOT translate those names or emit deprecation warnings in place of rejection.

Modern `output_format="legacy"`, `save_inversion_output` and independently
supported basis readers, writers, discovery and reconstruction operations SHALL
retain their separate contracts. A legacy or compatibility name alone SHALL NOT
remove a scientific representation or artifact codec.

Public Python names SHALL follow only the finite
[five-alias retirement](../rhime-model-selection/spec.md#requirement-retire-named-python-aliases).
CLI names SHALL follow the canonical routes in
[Bounded installed-command retirement](../rhime-model-selection/spec.md#requirement-bounded-installed-command-retirement).
Those named removals SHALL NOT alter other Python paths, canonical signatures,
result contents, scientific behavior or independently supported artifact formats.

#### Scenario: Packaged preparation survives HBMCMC retirement

- **WHEN** a modern run needs the exact domain's packaged preparation map
- **THEN** it obtains that map without importing HBMCMC
- **AND** explicit overrides and missing-default errors retain the resource-I/O contract

#### Scenario: Reject a retired runtime spelling

- **WHEN** a modern runner or acquisition request includes an enumerated historical spelling
- **THEN** ordinary unsupported-option validation rejects it before scientific work
- **AND** canonical `legacy` output remains available independently

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

Scientific requirements, borrowed-array and explicit execution boundaries,
modern `output_format="legacy"`, saved `InversionOutput` codecs, exact artifact
destinations and independently supported labelled basis partitions SHALL remain
supported. Namespace and alias retirement SHALL NOT itself change equations,
canonical defaults, independent artifact support or other public interfaces.
Supported artifact formats and versions SHALL follow their owning codec contracts;
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
