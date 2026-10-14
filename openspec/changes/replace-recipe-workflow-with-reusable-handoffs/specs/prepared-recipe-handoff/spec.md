# Spec Delta

## Purpose

Make saved scientific preparation reusable for compatible inference choices,
using ordinary files and the matched run context to describe its numerical products.

## ADDED Requirements

### Requirement: One reusable preparation handoff

A successful saved preparation SHALL return one absolute manifest path as its
handoff. The manifest SHALL identify its recipe family and supported contract
format and reference its numerical artifacts. The original requested observation
window, preparation provenance including basis provenance, and numerically baked
policy SHALL remain available in the record, numerical artifacts or matched run
context kept alongside, referenced or explicitly supplied by the caller.
Referenced numerical artifacts SHALL remain authoritative for retained observations and site metadata,
coordinates and labels, scientific source/state layout, units, retained basis
and flux. Consumers SHALL derive these retained facts from the numerical
artifacts rather than require duplicate arrays or labels in the manifest.
Requested preparation choices SHALL remain distinguishable from retained facts.
The manifest SHALL record the data choices that produced the preparation in their
external option form, and loading it SHALL return the same `DataConfig` type the
full runner passes to model resolution, validated by its owner. A saved-route-only
facts mapping, or a configuration reconstructed with defaults, SHALL NOT replace it.
It SHALL NOT depend on a runtime dataclass layout or imply that later
model, sampler or output choices were already executed or fixed by preparation.

#### Scenario: Preparation with a retained subset

- **WHEN** a saved preparation requested TAC and MHD but retained only TAC
- **THEN** its returned manifest path identifies the completed numerical handoff
- **AND** the original window/request remains available in the matched run context and the manifest identifies numerical artifacts containing TAC's retained metadata
- **AND** consumers use TAC's retained metadata from those artifacts

#### Scenario: Baked inverse-model policy

- **WHEN** preparation numerically encodes a boundary temporal parameterization or minimum-error floor
- **THEN** the matched preparation context records that policy as a preparation fact
- **AND** selecting a later model does not silently replace the encoded numerical values

### Requirement: Ordinary file references and current formats

Prepared and sampled handoffs SHALL use ordinary file references. Relative
references, including parent-directory references, SHALL resolve from the record
that declares them. Explicit absolute paths SHALL be accepted for shared caches
or other caller-selected files. Readers SHALL NOT discover replacement files
through environment settings such as RUN_ROOT. Missing files and unsupported
formats SHALL produce clear errors.
Only the current handoff formats SHALL be supported; 0.7 staged compatibility,
schema downgrade recovery and transitive content authentication SHALL NOT be
required. A handoff is used with its matched run context, not as an independently
portable bundle. Content digests SHALL NOT be mandatory handoff requirements.

Records SHALL use JSON and numerical artifacts SHALL use the formats already
supported by their owning codecs. Initial examples MAY use NetCDF; stages SHALL
NOT add a Zarr rejection or conversion requirement solely for content hashing.

#### Scenario: Reuse a supported numerical artifact

- **WHEN** a prepared artifact uses NetCDF or Zarr supported by its existing loader
- **THEN** the stage uses that loader without adding a new codec or conversion step

#### Scenario: Relative references and a shared cache

- **WHEN** a handoff uses a relative reference or an explicit absolute shared-cache path
- **THEN** the relative reference resolves from its declaring record and the absolute path selects that cache directly
- **AND** an unrelated working directory or RUN_ROOT value cannot select different files

#### Scenario: Missing file or unsupported format

- **WHEN** a required file is missing or uses an unsupported format
- **THEN** the reader reports the problem before inference or product writes
- **AND** it does not substitute another file found through configuration, working directory or environment defaults

### Requirement: Fresh destinations and manifest-last completion

Saved scientific operations SHALL use a fresh caller-selected destination and
reject an existing destination before owned writes. Ordinary numerical saves
SHALL complete before the final manifest is written. The final manifest SHALL be
the completion marker; a failed save SHALL NOT leave one claiming completion.
Files owned by the checkpoint writer SHALL remain under the explicit checkpoint
destination, and dependency references SHALL NOT authorize writes elsewhere or
overwriting referenced artifacts.
A separately requested basis export remains outside checkpoint-writer ownership
and follows [rhime-resource-io](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-resource-io/spec.md)
for exact destinations and runner ownership.
This ordinary I/O ordering SHALL NOT require rollback, transactions or repeated
dependency checks before and after execution. Several inference runs SHALL be
able to reference one completed preparation without mutating it or implicitly copying its large
numerical data. Relocation and additional copies SHALL require explicit caller
action.

#### Scenario: Interrupted save

- **WHEN** writing a required numerical artifact fails
- **THEN** the operation reports the error without leaving a final completion manifest
- **AND** any incomplete files do not constitute an accepted handoff

#### Scenario: Shared preparation and protected saves

- **WHEN** two inference runs use one completed preparation and distinct new destinations
- **THEN** both reference the same preparation without an implicit copy
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

Saved preparation SHALL be reusable with different compatible priors,
likelihoods and sampler/output choices without repeating preparation. Acceptance
SHALL check the selected model's required arrays, source/state/observation layout,
units and preparation facts through their existing scientific owners rather than
equality to the original full configuration or repeated downstream validation.
Changing acquired data, filtering/averaging, native flux data, basis/sensitivities
or baked numerical policy SHALL require a corresponding new preparation.
When new preparation uses an explicitly selected saved basis, it SHALL consume
the current native flux supplied for that preparation rather than silently
substitute historical flux from the basis artifact. The resulting numerical
preparation SHALL retain the flux used by its later inference and replay.
Explicit basis-path selection and native-flux/operator-only basis behavior SHALL
follow their separately owned contracts; the handoff SHALL NOT redefine them.
Scaling priors constructed at graph time SHALL remain inference choices. A CO2
coherent reduction SHALL retain its prior moments, effective operator, affine intercept and unresolved
covariance as one linked preparation; changing their defining native prior,
operator or projection SHALL require newly coherent products and matching
reconstruction information. Replacing retained-prior arrays alone SHALL NOT
establish compatibility. Linked CO2/O2 preparation that derives numerical means
or covariances from a prior SHALL likewise be recomputed when that prior changes;
reusable merged inputs need not be reacquired. Independent boundary/offset priors
and supported observation-mismatch choices SHALL permit reuse when their required inputs exist.

#### Scenario: Two compatible standard or multisector models

- **WHEN** two models select different graph-time scaling priors or supported likelihoods from one preparation with all required arrays
- **THEN** both reuse the same preparation content without retrieval, filtering, basis or sensitivity reconstruction
- **AND** a model requiring absent boundary or minimum-error inputs is rejected before inference

#### Scenario: Saved basis and current native flux

- **WHEN** new preparation explicitly selects a reusable saved basis and current native flux differs from historical flux associated with that basis
- **THEN** the new preparation uses and retains the current flux according to the basis-reuse contract
- **AND** an existing completed preparation and its replay continue to use their own retained flux

#### Scenario: Prior-dependent CO2-family preparation

- **WHEN** a caller changes a prior used to derive a CO2 reduction or linked CO2/O2 preparation
- **THEN** the affected numerical preparation is recomputed rather than accepted merely because retained-prior labels match
- **AND** changing only compatible independent boundary/offset or observation-mismatch choices can reuse the unchanged coherent handoff
