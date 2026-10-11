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
The manifest SHALL preserve the original requested observation window,
preparation provenance including basis provenance, and numerically baked policy.
Referenced numerical artifacts SHALL remain authoritative for retained observations and site metadata,
coordinates and labels, scientific source/state layout, units, retained basis
and flux. Consumers SHALL derive these retained facts from the authenticated
artifacts rather than require duplicate arrays or labels in the manifest.
Requested preparation choices SHALL remain distinguishable from retained facts.
The handoff SHALL NOT require a saved whole configuration or runtime dataclass
layout, or imply that later model, sampler or output choices were already
executed or fixed by ordinary preparation.

#### Scenario: Preparation with a retained subset

- **WHEN** a saved preparation requested TAC and MHD but retained only TAC
- **THEN** its returned manifest path identifies the completed numerical handoff
- **AND** the manifest preserves the original window/request and identifies numerical artifacts containing TAC's retained metadata
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
Content digests SHALL bind the referenced file content; they SHALL NOT substitute
for scientific compatibility checks or imply verification of an author's identity.

#### Scenario: Relocated dependency tree

- **WHEN** a prepared handoff and sample handoffs referencing it move together with their relative dependency layout intact
- **THEN** they resolve and authenticate the same content from an unrelated working directory
- **AND** an unrelated RUN_ROOT value cannot select different files

#### Scenario: Invalid reference or changed content

- **WHEN** a required dependency is missing, altered, has an invalid reference or uses an unsupported contract
- **THEN** the reader rejects the handoff before inference or product writes
- **AND** it does not substitute another file found through configuration, working directory or environment defaults

### Requirement: Saved handoffs are complete and stable

Saved scientific operations SHALL use a fresh caller-selected destination and
reject an existing destination before owned writes. Ordinary numerical saves
SHALL complete and required artifact digests SHALL be verified before the final
manifest is made available. The final manifest SHALL be the completion marker;
a failed save SHALL NOT leave one claiming completion. Writer-owned files SHALL
remain under the explicit destination, and dependency references SHALL NOT
authorize writes elsewhere or overwriting referenced artifacts.
This filesystem guarantee SHALL NOT require rollback of incomplete files or a
separate publication framework. Several inference runs SHALL be able to reference
one completed preparation without mutating it or implicitly copying its large
numerical data. Relocation and additional copies SHALL require explicit caller
action.

#### Scenario: Interrupted save

- **WHEN** writing a required numerical artifact or computing its digest fails
- **THEN** the operation reports the error without leaving a final completion manifest
- **AND** any incomplete files do not constitute an accepted handoff

#### Scenario: Shared preparation and protected saves

- **WHEN** two inference runs use one completed preparation and distinct new destinations
- **THEN** both reference the same authenticated preparation without an implicit copy
- **AND** attempting to save into an existing destination fails without altering its completed handoff or referenced artifacts

### Requirement: Preparation owns filtering and retained-site alignment

Standard and multisector preparation SHALL filter pre-filter merged data once,
derive retained sites from kept observations and align per-site metadata/options
by labels before basis and sensitivity construction. Valid nonempty subsets
supplied by acquisition, supported supplied-data entry points or filtering SHALL be accepted without
bypassing the supplied input's owning validation. An empty retained set SHALL
fail before basis construction or inference. Saved preparation SHALL omit the
retired filtered merged snapshot. Acquisition-cache preservation and fallback
policy are outside this capability.
At their owning phase, applicable numeric per-site mappings SHALL cover every retained site and accept
valid extra requested-site entries. All supplied values, including dropped-site
entries, SHALL retain their numeric, finiteness and applicable sign validation;
retained entries SHALL align before entering strict numerical components.
Preparation SHALL align its metadata; inference SHALL align its current model
options against prepared retained labels without making them preparation choices.

#### Scenario: Acquisition, supplied data or filtering drops a site

- **WHEN** acquisition, supported supplied-data input or filtering retains only a nonempty subset of requested sites
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
When new preparation uses an explicitly selected saved basis, it SHALL consume
the current native flux supplied for that preparation rather than silently
substitute historical flux from the basis artifact. The resulting numerical
preparation SHALL retain the flux used by its later inference and replay.
Explicit basis-path selection and native-flux/operator-only basis behavior SHALL
follow their separately owned contracts; the handoff SHALL NOT redefine them.
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

#### Scenario: Saved basis and current native flux

- **WHEN** new preparation explicitly selects a reusable saved basis and current native flux differs from historical flux associated with that basis
- **THEN** the new preparation uses and retains the current flux according to the basis-reuse contract
- **AND** an existing completed preparation and its replay continue to use their own retained flux

#### Scenario: Prior-dependent CO2 reduction

- **WHEN** a caller changes the native flux prior defining a CO2 reduction
- **THEN** the existing reduced operator/intercept/covariance are not accepted as a newly coherent preparation merely because retained-prior labels match
- **AND** changing only compatible independent boundary/offset or observation-mismatch choices can reuse the unchanged coherent handoff
