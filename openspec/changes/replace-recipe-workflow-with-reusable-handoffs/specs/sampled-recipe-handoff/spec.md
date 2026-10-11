# Sampled recipe handoff

## Purpose

Preserve completed inference and its scientific output meaning in one saved
handoff, so compatible products can be created without the original configuration
or another model execution.

## ADDED Requirements

### Requirement: Sampling consumes one prepared handoff and current inference choices

Each supported concrete MCMC workflow SHALL accept one prepared-manifest path,
the current model section, a sampler selection and an explicit sample destination,
and return one absolute sample-manifest path. It SHALL authenticate the prepared
handoff and validate the selected model's scientific prerequisites before
materializing model inputs or sampling. It SHALL NOT require the preparation's
original full configuration, a repeated prepared-data path or current output
choices. Compatible new priors or likelihood choices SHALL define new inference
without invalidating the prepared checkpoint. Preparation-dependent scientific
constraints SHALL follow the prepared-recipe-handoff contract.
Facts already owned by preparation SHALL be bound from that handoff; conflicting
claims in the current model section SHALL fail rather than replace saved facts.
The new sampling boundary SHALL accept the existing `RhimeSampler` or
`"default"`, with omission equivalent to `"default"`. Each default selection
SHALL create a fresh sampler for that invocation. A supplied sampler's options
SHALL be snapshotted for the invocation without mutating caller-owned options or
requiring a second sampler-options type. `None` SHALL mean absence of a live
sampler and SHALL be invalid at sampling boundaries. Existing direct entry points
SHALL migrate default selection from explicit `None` to `"default"` consistently
with recipe-execution; no adapter SHALL reinterpret `None` as default selection.

#### Scenario: Reuse preparation for a different compatible model
- **WHEN** two sampling invocations use the same valid prepared handoff with
  different compatible prior or likelihood choices and separate destinations
- **THEN** each samples its selected model and returns its own sample-manifest path
- **AND** neither repeats acquisition/preparation nor changes the prepared files

#### Scenario: Selected component cannot consume the saved preparation
- **WHEN** selected model choices require unavailable arrays, incompatible labels,
  units or a different preparation-dependent scientific calculation
- **THEN** sampling reports the incompatibility before model materialization or
  sample saving, rather than reinterpreting or silently preparing the inputs

#### Scenario: Default and explicit sampler selections
- **WHEN** sampling omits its sampler or selects `"default"` on separate invocations
- **THEN** each invocation uses a fresh default sampler
- **AND** supplying a `RhimeSampler` uses an isolated snapshot of its actual options
- **AND** supplying `None` fails with migration guidance in saved and existing direct sampling entry points

#### Scenario: A model section contradicts prepared facts
- **WHEN** a current model section claims species, domain or source/state facts inconsistent with its prepared handoff
- **THEN** sampling rejects the conflict before construction or sampling rather than replacing the preparation's meaning

### Requirement: The sample record contains bounded scientific replay information

A sample handoff SHALL record its supported envelope version, recipe and MCMC
inference kind; the exact referenced preparation manifest and posterior with
their relative file references and content identities;
the actual model choices, sampler provenance and callable identity/arguments;
and its family's scientific output meaning. Standard/multisector SHALL embed
the existing `OutputContract` representation in the record. CO2 SHALL retain its
authenticated trace roles and affine associations without requiring that generic
contract representation.
The bound preparation manifest and its numerical artifacts SHALL preserve the
original run window, retained sites, averaging periods and other preparation facts
needed to construct the result and its outputs, without requiring duplicate
numerical labels or arrays in the sample record. The sample SHALL add only the
family-specific output context and provenance needed from completed inference.
Recorded inference choices SHALL describe the completed invocation, not later
configuration defaults, using plain serializable metadata rather than a saved
whole configuration or runtime dataclass layout. Saved scientific output meaning
SHALL be validated independently of whether products are requested during sampling.
The record SHALL NOT require a separate output-binding
sidecar or persistence of live models, samplers or steps. Ordinary saves,
manifest-last completion and dependencies SHALL follow the prepared-recipe-handoff
destination and file-reference rules.
Sampling SHALL retain the preparation identity authenticated before inference and
bind its sample record to that identity. Before completing the sample manifest,
saving SHALL verify the current preparation and its required dependencies against
the retained identity rather than substituting a newly computed identity.

#### Scenario: Preparation changes while inference is running
- **WHEN** the preparation manifest or a required dependency changes after its initial verification and before sample saving completes
- **THEN** saving rejects the changed dependency and leaves no completed sample manifest
- **AND** it does not bind the posterior to the replacement preparation

#### Scenario: Record an invocation with retained observations and overrides
- **WHEN** sampling uses a retained subset and invocation-specific sampler choices
- **THEN** the saved record preserves the original run window, actual retained
  metadata through its bound preparation, and the selected model choices and
  sampler settings used for that invocation
- **AND** the returned manifest path identifies the completed record without
  adding a self-referential invocation path to its saved contents

#### Scenario: The referenced artifact tree is relocated
- **WHEN** a completed sample handoff and its dependencies are moved together
  while preserving their manifest-relative relationships
- **THEN** replay resolves those dependencies without the old working directory,
  original configuration file or an ambient run-root setting
- **AND** sampling has not implicitly copied large prepared dependencies to make
  the sample directory independently self-contained

### Requirement: Replay validates saved meaning and numerical associations

Replay SHALL validate the selected family's supported envelope and output-contract
versions, required record structure, artifact-reference associations and referenced
prepared/posterior content identities before loading the posterior or creating a
product destination. Required dependencies SHALL remain available. After loading,
replay SHALL validate scientific roles, labels, units and reconstruction associations
against the numerical data before product writes, without constructing a model.
A matching content digest SHALL NOT substitute for scientific label, unit or
reconstruction validation. Missing, malformed, changed or unsupported records
SHALL fail clearly without graph recovery, implicit preparation or permissive
fallback to another artifact contract.

#### Scenario: A saved numerical dependency is missing or changed
- **WHEN** a required dependency cannot be read or its content identity differs
  from the identity recorded by sampling
- **THEN** replay fails before posterior loading or product writes

#### Scenario: An envelope or output association is invalid
- **WHEN** replay receives an unsupported version, incomplete replay information,
  invalid output contract or inconsistent prepared/posterior association
- **THEN** it reports the invalid handoff without constructing a recovery graph,
  accepting a downgraded contract or creating product artifacts

#### Scenario: Loaded data contradict saved scientific output meaning
- **WHEN** authenticated numerical data have roles, labels, units or reconstruction
  associations incompatible with the saved scientific output contract
- **THEN** replay reports the scientific incompatibility after loading and before
  product writes, without reconstructing a model

### Requirement: Postprocessing consumes one sampled handoff and one output policy

Each supported concrete workflow SHALL accept one sample-manifest path and one
current output policy, including destination, naming and supported product choices,
and return its result with requested products and output metadata. It SHALL use
the saved model/run facts and sampling provenance without requiring the original
full configuration or separate prepared/posterior paths. It SHALL NOT construct
a graph, materialize model inputs, repeat preparation or sample again.
Replay SHALL NOT import or execute recorded construction or likelihood callables.
It SHALL NOT reconstruct a runtime sampler, select fresh sampler defaults or
validate saved provenance as a newly requested current model configuration.
The replay result SHALL have `sampler=None` and carry actual saved sampler
provenance as plain metadata consumed by the shared output owner. It SHALL use
the authenticated preparation's retained flux rather than resolve current native
flux or load historical flux from a separate basis artifact.
Current output choices SHALL NOT replace recorded scientific or inference choices.
Unsupported products or incompatible reporting inputs SHALL fail before product
writes. Scientific result/product construction SHALL share its concrete owner
with the corresponding in-memory route.

#### Scenario: Create different compatible products in a later process
- **WHEN** a later process receives only a valid sample handoff and compatible
  current output choices with a new destination or naming policy
- **THEN** it returns the requested products using recorded inference provenance
  and the same saved scientific output meaning without model execution
- **AND** its result has no runtime sampler and its output metadata describes the completed sampling invocation

#### Scenario: Requested output is unsupported
- **WHEN** current output choices request a format not supported by the saved
  output contract or reporting inputs incompatible with its numerical data
- **THEN** postprocessing fails before product writes rather than changing the
  sampled model or inferring missing output meaning from a new graph

### Requirement: CO2 replay preserves coherent affine reconstruction meaning

CO2 handoffs SHALL retain the family-specific information needed for ordinary or
matched cached inference and its supported concentration/flux products. Any
affine companion used by replay SHALL be authenticated and bound to the same
prepared inputs used by sampling. Replay SHALL preserve its declared conditional
reconstruction and uncertainty scope; it SHALL NOT reinterpret a coherent affine
CO2 result as a standard multiplicative inversion product.

#### Scenario: Request products requiring an affine companion
- **WHEN** compatible CO2 output choices require native or country flux products
- **THEN** replay authenticates the required companion against the recorded
  prepared inputs and preserves its conditional reconstruction meaning
- **AND** a missing or differently bound companion fails before product writes
