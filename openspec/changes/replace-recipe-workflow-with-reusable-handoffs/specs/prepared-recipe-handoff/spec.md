# Spec Delta

## Purpose

Make saved scientific preparation reusable for compatible inference choices,
with a portable, authenticated handoff that records what its numerical products mean.

## ADDED Requirements

### Requirement: One complete preparation handoff

A successful saved preparation SHALL return one absolute manifest path as its
handoff. The manifest SHALL identify its recipe family and supported contract
version, its numerical artifacts and content digests, and the preparation facts
needed to validate subsequent use without the original full configuration.
Facts SHALL preserve the requested observation window, retained observations and
site metadata, scientific source/state layout, units, basis provenance and any
numerically baked policy. Requested choices SHALL remain separately available
from retained facts. The manifest SHALL NOT imply that later model, sampler or
output choices were already executed or fixed by ordinary preparation.

#### Scenario: Preparation with a retained subset

- **WHEN** a saved preparation requested TAC and MHD but retained only TAC
- **THEN** its returned manifest path identifies the completed numerical handoff
- **AND** the manifest preserves both the original window/request and TAC's retained metadata
- **AND** a consumer needs no original configuration or separately supplied numerical paths

#### Scenario: Baked inverse-model policy

- **WHEN** preparation numerically encodes a boundary temporal parameterization or minimum-error floor
- **THEN** the handoff records that policy as a preparation fact
- **AND** selecting a later model does not silently replace the encoded numerical values

### Requirement: Portable authenticated file references

Prepared and sampled scientific handoffs SHALL share explicit file-reference
rules. Saved artifact and dependency references SHALL be ordinary relative file
paths from the manifest that declares them, with content digests and identifiable
schema contracts; relative parent-directory references SHALL be supported.
Readers SHALL resolve references from that manifest's directory, independently
of the current working directory or environment such as RUN_ROOT. Readers SHALL
validate the selected family's supported contract and required digests before
accepting numerical content. Absolute paths, URI references, environment
placeholders, malformed references and unsupported or falsely labelled contracts
SHALL fail clearly.
Moving an intact dependency tree SHALL preserve its meaning; copying only one
manifest SHALL NOT make unavailable or changed dependencies acceptable.

#### Scenario: Relocated dependency tree

- **WHEN** a prepared handoff and sample handoffs referencing it move together with their relative dependency layout intact
- **THEN** they resolve and authenticate the same content from an unrelated working directory
- **AND** an unrelated RUN_ROOT value cannot select different files

#### Scenario: Invalid reference or changed content

- **WHEN** a required dependency is missing, altered, has an invalid reference or uses an unsupported contract
- **THEN** the reader rejects the handoff before inference or product writes
- **AND** it does not substitute another file found through configuration, working directory or environment defaults

### Requirement: Published handoffs are complete and stable

Saved scientific operations SHALL publish their manifest only after all required
artifacts are successfully written and authenticated. Writer-owned files SHALL
remain under the caller's explicit destination; dependency references SHALL NOT
authorize writes elsewhere. A failure SHALL NOT leave a newly published manifest
claiming completion. Publication SHALL reject a
destination that would overwrite an already published handoff or a referenced
artifact. Several inference runs SHALL be able to reference one published
preparation without mutating it or implicitly copying its large numerical data.
Relocation and additional copies SHALL require explicit caller action.

#### Scenario: Interrupted publication

- **WHEN** writing a required numerical artifact or computing its digest fails
- **THEN** the operation reports the error without publishing a completed handoff

#### Scenario: Shared preparation and protected publication

- **WHEN** two inference runs use one published preparation and distinct new destinations
- **THEN** both reference the same authenticated preparation without an implicit copy
- **AND** attempting to publish over either completed handoff or its artifacts fails without altering them

### Requirement: Preparation owns filtering and retained-site alignment

Standard and multisector preparation SHALL filter pre-filter merged data once,
derive retained sites from kept observations and align per-site metadata/options
by labels before basis and sensitivity construction. Acquisition and compatible
merged-cache reload SHALL accept valid requested-site subsets consistently.
An empty retained set SHALL fail before basis construction or inference;
malformed inputs and incompatible caches SHALL retain their owning validation
and rejection/fallback rules. The optional merged cache SHALL remain acquisition
output before filtering. Saved preparation SHALL omit the retired filtered
merged snapshot and SHALL NOT reinterpret historical filtered files as that cache.
At their owning phase, applicable numeric per-site mappings SHALL cover every retained site and accept
valid extra requested-site entries. All supplied values, including dropped-site
entries, SHALL retain their numeric, finiteness and applicable sign validation;
retained entries SHALL align before entering strict numerical components.
Preparation SHALL align its metadata; inference SHALL align its current model
options against prepared retained labels without making them preparation choices.

#### Scenario: Acquisition, reload or filtering drops a site

- **WHEN** acquisition, compatible cache reload or filtering retains only a nonempty subset of requested sites
- **THEN** full and saved preparation keep equivalent observations and aligned metadata/options
- **AND** no filtered merged snapshot is produced as a preparation handoff

#### Scenario: Numeric mapping for requested and retained sites

- **WHEN** a per-site numeric mapping contains valid entries for retained and dropped requested sites
- **THEN** the retained entries are selected in retained-label order and accepted
- **AND** missing retained entries, boolean/nonnumeric/nonfinite values or violated sign constraints fail even in dropped entries

### Requirement: Reuse requires scientific compatibility rather than full configuration equality

Authenticated preparation SHALL be reusable with different compatible priors,
likelihoods and sampler/output choices without repeating preparation. Acceptance
SHALL check the selected model's required arrays, source/state/observation layout,
units and preparation facts rather than equality to the original full configuration.
Changing acquired data, filtering/averaging, native flux data, basis/sensitivities
or baked numerical policy SHALL require a corresponding new preparation.
Ordinary scaling priors SHALL remain inference choices. A CO2 coherent reduction
SHALL retain its prior moments, effective operator, affine intercept and unresolved
covariance as one linked preparation; changing their defining native prior,
operator or projection SHALL require newly coherent products and matching
reconstruction information. Replacing retained-prior arrays alone SHALL NOT
establish compatibility. Independent boundary/offset priors and supported
observation-mismatch choices SHALL permit reuse when their required inputs exist.

#### Scenario: Two compatible standard or multisector models

- **WHEN** two models select different scaling priors or supported likelihoods from one authenticated preparation with all required arrays
- **THEN** both reuse the same preparation content without retrieval, filtering, basis or sensitivity reconstruction
- **AND** a model requiring absent boundary or minimum-error inputs is rejected before inference

#### Scenario: Prior-dependent CO2 reduction

- **WHEN** a caller changes the native flux prior defining a CO2 reduction
- **THEN** the existing reduced operator/intercept/covariance are not accepted as a newly coherent preparation merely because retained-prior labels match
- **AND** changing only compatible independent boundary/offset or observation-mismatch choices can reuse the unchanged coherent handoff
