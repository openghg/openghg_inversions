# Spec Delta

> Status: Draft behavioral contract for OPE-207; not an implementation claim.

## Purpose

Provide consistent recipe configuration and independent staged execution while
preserving the installed commands, scientific meanings, and authenticated saved
artifacts used by inversion scientists and downstream workflow callers.

## ADDED Requirements

### Requirement: Consistent resolved recipe configuration

A recipe SHALL have one authoritative resolution of its scientific,
preparation, sampling, and output choices for full and staged execution.
Common sampling and output choices SHALL each have one owner in that
resolution. Family-specific scientific inputs and preparation representations
SHALL retain their distinct meanings. Existing defaults specific to a family
or execution route SHALL remain unchanged.

#### Scenario: Equivalent choices across full and staged execution
- **WHEN** supported full and staged entry points resolve the same recipe,
  scientific configuration, explicit sampling choices, and supported output choices
- **THEN** they supply equivalent resolved scientific and sampling values
- **AND** differences are limited to existing route-specific defaults and
  explicit runtime destinations or stage arguments

#### Scenario: Ordinary and cached CO2 remain distinct
- **WHEN** the existing ordinary or cached CO2 variant is selected
- **THEN** resolution retains that variant's scientific likelihood and matched
  sampling policy without substituting the other variant

### Requirement: Existing configuration contract

Resolution SHALL preserve accepted configuration formats, option meanings,
defaults, aliases, override precedence, and route-specific rejection rules.
File-relative paths SHALL retain their existing resolution semantics.
Invalid or inapplicable configuration SHALL fail at its owning boundary
before scientific artifacts are written.

#### Scenario: Configuration source and relative paths
- **WHEN** an installed stage loads an explicit configuration or parameter file
  containing relative artifact paths and supported overrides
- **THEN** paths resolve relative to the original source and overrides retain
  their existing precedence
- **AND** an unrelated ambient `CONFIG_FILE` does not select staged science

#### Scenario: Unsupported options
- **WHEN** a configuration contains an unknown option or an option unsupported
  by the selected recipe and execution route
- **THEN** resolution reports an error rather than silently ignoring it or
  changing the scientific recipe

### Requirement: Common independent stage contract

Recipe implementations SHALL provide a common explicit contract for
preparation, prior-predictive checks, sampling, and postprocessing. Once the
recipe is resolved, the caller SHALL invoke these operations without repeating
family-specific selection for each operation. Selecting a recipe SHALL NOT
require executing another recipe's preparation or scientific workflow.

#### Scenario: Independent sampling
- **WHEN** sampling receives authenticated prepared inputs and their manifest
- **THEN** it uses the selected recipe's sampling implementation
- **AND** it does not repeat acquisition or preparation

#### Scenario: Independent saved-output production
- **WHEN** postprocessing receives authenticated prepared inputs, a saved
  posterior, and the required manifests
- **THEN** it produces supported requested products without resampling
- **AND** graph construction follows only that family's documented replay policy

#### Scenario: Additional concrete family
- **WHEN** OPE-165 supplies a supported linked CO2/O2 staged implementation
- **THEN** callers use the same stage operation contract for that one joint recipe
- **AND** its unequal channel axes and cross-channel covariance are not replaced
  by two independent tracer workflows

### Requirement: Installed CLI compatibility

Existing installed `prepare`, `prior-predictive`, `sample`, `diagnose`, and
`postprocess` commands SHALL retain their command names, model choices,
arguments, defaults, declared handoff requirements, stdout contract, and exit
policy. This refactor SHALL NOT introduce the separate unified `run` CLI.

#### Scenario: Existing command sequence
- **WHEN** a supported existing standard, multisector, or ordinary/cached CO2
  command sequence is run with the same arguments and controlled inputs
- **THEN** the same handoffs and requested scientific products are produced
- **AND** artifact filenames, manifest schemas, and scientific labels/units remain compatible

### Requirement: Stable scientific configuration identities

For unchanged resolved scientific settings, configuration identity encoding
and hashes SHALL remain compatible with existing saved manifests. The existing
family policy for excluding sampling, output, and transport settings SHALL
remain unchanged. Artifact content identities SHALL continue to authenticate
the numerical handoffs independently.

#### Scenario: Internal configuration representation changes
- **WHEN** an existing scientific configuration is represented by the new
  resolved configuration boundary
- **THEN** its scientific configuration identity matches the pre-refactor identity
- **AND** existing matched saved manifests remain usable

#### Scenario: Replay settings and recorded sampler
- **WHEN** only currently permitted output, transport, or sampling settings change
- **THEN** those changes do not invalidate scientific identity
- **AND** saved-output replay reports the sampler recorded by the sample manifest

#### Scenario: Scientific mismatch
- **WHEN** a scientifically relevant family setting changes or supplied
  numerical artifacts do not match their recorded identities
- **THEN** the handoff is rejected instead of being reused as a matched run

### Requirement: Standard and multisector version-2 replay

Standard and multisector version-2 sample manifests SHALL require an
authenticated saved output binding and SHALL replay without acquisition,
model-input materialization, or model graph construction. Missing, malformed,
altered, escaping, or mismatched bindings SHALL fail before posterior loading
or scientific product writes and SHALL NOT trigger historical graph replay.

#### Scenario: Valid graph-free replay
- **WHEN** either family's version-2 sample manifest binds the exact supplied
  prepared inputs, posterior, and a valid saved output contract
- **THEN** products retain scientific roles, metadata, and separate chain/draw axes
- **AND** replay succeeds when model construction is forbidden

#### Scenario: Invalid output binding
- **WHEN** a binding is missing, its bytes or digest changed, its schema or
  contract is malformed, its artifact path escapes the allowed directory, or
  its prepared-input/posterior identities differ from the sample manifest
- **THEN** validation rejects it before posterior loading or product writes
- **AND** no graph-building fallback is attempted

### Requirement: Genuine historical standard and multisector replay

Genuine version-1 standard and multisector sample manifests without output
bindings SHALL retain their historical graph-building route to recover missing
output roles. Version-1 manifests that advertise output bindings SHALL be
rejected rather than treated as genuine historical artifacts.

#### Scenario: Historical manifest without binding
- **WHEN** a matched historical version-1 standard or multisector manifest has
  no output-binding entries
- **THEN** its established graph-building compatibility route remains available

#### Scenario: Binding-bearing manifest relabelled version 1
- **WHEN** a manifest advertises a binding in its artifacts or identities but
  declares version 1
- **THEN** it is rejected rather than bypassing binding authentication

### Requirement: CO2 version-1 replay and affine authentication

CO2 SHALL retain graph-free replay for version-1 sample manifests and SHALL
reject other sample-manifest versions before posterior loading or output
destination creation. Optional affine reconstruction SHALL retain independent
content authentication and binding to the prepared handoff and sampling record.

#### Scenario: Supported CO2 saved replay
- **WHEN** matched version-1 CO2 artifacts are replayed
- **THEN** supported products can be produced with ordinary and cached model
  construction forbidden

#### Scenario: Unsupported CO2 sample version
- **WHEN** a CO2 sample manifest declares version 2 with an absent binding,
  malformed binding, or incorrect binding digest
- **THEN** it is rejected before posterior loading or output destination creation

#### Scenario: Affine content or binding mismatch
- **WHEN** a supplied affine artifact is altered, bound to other prepared data,
  or disagrees with the preparation/sampling affine identity
- **THEN** it is rejected before scientific product writes
- **AND** currently supported relocation of identical authenticated content remains valid

### Requirement: Diagnostic and scientific output stability

Readiness and convergence checks SHALL retain their schemas, names, threshold
semantics, and existing `pass`, `fail`, and `unknown` decisions. Scientific
outputs SHALL retain the existing values, roles, dimensions, units, and
conditional reconstruction meaning for the same saved inputs and posterior.
This change SHALL NOT alter equations or expand diagnostic policy.

#### Scenario: Diagnosis without recipe resolution
- **WHEN** diagnosis receives a saved posterior, optional sample manifest, and
  convergence options
- **THEN** it requires no scientific configuration, family setup, or model selection
- **AND** optional authentication and existing diagnostic decisions remain unchanged

#### Scenario: Convergence with unavailable metrics
- **WHEN** a finite metric fails its threshold and another metric is unavailable
- **THEN** the convergence check remains `fail`
- **AND** otherwise unassessable evidence retains its existing `unknown` policy

#### Scenario: Strict scientific gate
- **WHEN** an installed check command uses its existing strict mode
- **THEN** a scientific `fail` retains the existing nonzero exit behavior
- **AND** `unknown` is not newly classified as a failed scientific gate

#### Scenario: Scientific gate and serialization failure
- **WHEN** prior-readiness evidence fails scientifically or its artifact writer fails
- **THEN** the scientific failure emits the existing gate-compatible check
- **AND** a serialization failure remains an error rather than a successful check

#### Scenario: Stable saved-posterior products
- **WHEN** the same controlled prepared inputs and saved posterior are processed
  before and after the internal refactor
- **THEN** supported deterministic products and their scientific interpretation agree
- **AND** no identical stochastic trajectory between separate sampling runs is required
