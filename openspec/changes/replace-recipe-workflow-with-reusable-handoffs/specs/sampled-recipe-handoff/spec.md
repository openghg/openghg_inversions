# Sampled recipe handoff

## Purpose

Preserve completed inference and its scientific output meaning in one saved
handoff, so compatible products can be created without the original configuration
or another model execution.

## ADDED Requirements

### Requirement: Sampling consumes one prepared handoff and current inference choices

Each supported concrete MCMC workflow SHALL accept one prepared-manifest path,
current model choices, sampler options and an explicit sample destination, and
return one absolute sample-manifest path. It SHALL authenticate the prepared
handoff and validate the selected model's scientific prerequisites before
materializing model inputs or sampling. It SHALL NOT require the preparation's
original full configuration, a repeated prepared-data path or current output
choices. Compatible new priors or likelihood choices SHALL define new inference
without invalidating the prepared checkpoint. Preparation-dependent scientific
constraints SHALL follow the prepared-recipe-handoff contract.

#### Scenario: Reuse preparation for a different compatible model
- **WHEN** two sampling invocations use the same valid prepared handoff with
  different compatible prior or likelihood choices and separate destinations
- **THEN** each samples its selected model and returns its own sample-manifest path
- **AND** neither repeats acquisition/preparation nor changes the prepared files

#### Scenario: Selected component cannot consume the saved preparation
- **WHEN** selected model choices require unavailable arrays, incompatible labels,
  units or a different preparation-dependent scientific calculation
- **THEN** sampling reports the incompatibility before model materialization or
  sample publication, rather than reinterpreting or silently preparing the inputs

### Requirement: The sample record contains bounded scientific replay information

A sample handoff SHALL record its supported envelope version, recipe and MCMC
inference kind; referenced numerical artifacts and their content identities;
the actual model choices, sampler provenance and callable identity/arguments;
and its family's scientific output meaning. Standard/multisector SHALL embed
the existing `OutputContract` representation in the record. CO2 SHALL retain its
authenticated trace roles and affine associations without requiring that generic
contract representation.
It SHALL preserve the original run window and retained sites, averaging periods
and other family-specific facts needed to construct the result and its outputs.
Recorded inference choices SHALL describe the completed invocation, not later
configuration defaults. Saved scientific output meaning SHALL be validated
independently of whether products are requested during sampling.
The record SHALL NOT require a separate output-binding
sidecar or persistence of live models, samplers or steps. Dependencies and
publication SHALL follow the prepared-recipe-handoff path/publication rules.

#### Scenario: Record an invocation with retained observations and overrides
- **WHEN** sampling uses a retained subset and invocation-specific sampler choices
- **THEN** the saved record preserves the original run window, actual retained
  metadata, selected model choices and sampler settings used for that invocation
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
Current output choices SHALL NOT replace recorded scientific or inference choices.
Unsupported products or incompatible reporting inputs SHALL fail before product
writes. Scientific result/product construction SHALL share its concrete owner
with the corresponding in-memory route.

#### Scenario: Create different compatible products in a later process
- **WHEN** a later process receives only a valid sample handoff and compatible
  current output choices with a new destination or naming policy
- **THEN** it returns the requested products using recorded inference provenance
  and the same saved scientific output meaning without model execution

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
