# Spec Delta

> Behavioral contract for OPE-207. Implementation and validation remain outstanding.

## Purpose

Keep equivalent supported recipe execution routes scientifically consistent,
with optional checkpoints and authenticated saved-output replay. Preserve
established scientific interfaces and numerical/product contracts while making
staged compatibility and supported schema versions explicit.

## ADDED Requirements

### Requirement: Shared execution across supported routes

A recipe's equivalent supported full, prepared-input, and staged routes SHALL
reuse canonical operations for their applicable preparation, construction,
sampling, and result/product phases. Stages SHALL add checkpoint loading,
authentication, persistence, and reporting without independent scientific policy.
Filtering, retained-site handling, basis construction/application, sensitivities,
and input assembly SHALL belong to one coherent scientific preparation operation
where applicable, without requiring a separate high-level filtering stage.
Full execution SHALL visibly sequence shared operations in memory without
requiring intermediate-file I/O. Builders and direct runners SHALL NOT require
unrelated preparation, configuration, stages, or result formats to be supported.

#### Scenario: Full execution without checkpoint persistence

- **WHEN** an existing full route runs with intermediate saves disabled and
  checkpoint writers unavailable
- **THEN** its scientific phases complete through in-memory handoffs, including
  requested final products without a temporary staged round trip

#### Scenario: Prepared execution joins the same inference operations

- **WHEN** equivalent prepared-input and staged routes receive the same valid
  prepared inputs and scientific/sampling choices
- **THEN** they reuse their recipe's construction and sampling operations without
  acquisition, filtering, basis, or sensitivity reconstruction
- **AND** existing direct Python signatures/returns and likelihood/complete-model
  customization retain their forwarding, conflict, input-ownership, and output contracts

#### Scenario: Products from saved samples

- **WHEN** ordinary product construction and authenticated replay receive the
  same prepared inputs, posterior, and equivalent scientific output information
- **THEN** they reuse the same family product operations without resampling or
  constructing the producing graph for supported saved-output replay

### Requirement: Shared observation-error policy precedes temporal reduction

Equivalent supported routes SHALL derive `mf_error` before temporal filtering
or aggregation, preserving supplied custom errors, the existing zero-error
fallback, borrowed datasets and retained metadata. Nested preparation SHALL
prepare both inner and outer domains before filtering/time alignment. Historical
six-tuple retrieval SHALL call the same operation before returning or saving.
`averaging_error` is preparation configuration; `averaging_period` remains an
acquisition/resampling selector. Actual scientific operations SHALL be shared
without forwarding wrappers added solely to name stages.

#### Scenario: Varying component errors within an averaging interval

- **WHEN** component errors vary among observations in the same interval
- **THEN** routes combine observation-error components before aggregation
- **AND** an explicitly supplied `mf_error` is retained without rederivation

This contract does not claim completed full/staged retained-site parity or add
nested/CO2 checkpoint routes. Those capabilities require their own implementation
and behavioral validation.

### Requirement: Retained-site consistency in preparation

Each recipe SHALL derive retained sites from the observations it keeps and align
applicable per-site metadata and options consistently across equivalent supported
routes, preserving validation of malformed or incompatible inputs.

Standard and multisector SHALL each reuse their recipe's preparation across its
supported routes.
Valid retained subsets after acquisition, supplied in-memory data,
or filtering SHALL be accepted consistently with full preparation, with every
per-site option aligned to retained labels. This replaces staged rejection of
missing requested sites. An empty retained set SHALL fail before basis/inference.
Requested configuration SHALL remain separately available for provenance;
reconciliation SHALL NOT bypass malformed-input or acquisition-compatibility validation.

#### Scenario: Acquisition returns a valid subset

- **WHEN** standard/multisector acquisition returns valid observations for only
  some requested sites
- **THEN** full and staged preparation retain the same observations, labels,
  averaging periods, and other per-site options using requested-site alignment

#### Scenario: Supplied merged data lacks a requested site

- **WHEN** a supported Python route receives compatible in-memory merged data
  containing a subset of requested sites
- **THEN** preparation retains that subset and aligns every per-site option
- **AND** incompatible acquisition facts retain their rejection policy

#### Scenario: Filtering empties a site

- **WHEN** valid standard/multisector filtering removes all observations at one
  site but retains others
- **THEN** full and staged preparation produce equivalent retained observations
  and aligned metadata, which prepared execution uses consistently

#### Scenario: Empty or malformed handoff

- **WHEN** standard/multisector acquisition, supplied in-memory data, or filtering
  leaves no usable sites
- **THEN** preparation fails before basis construction or inference
- **AND** invalid labels/metadata and malformed or mismatched independently
  supplied prepared inputs retain their owning boundary's rejection policy

#### Scenario: CO2 prepared-site alignment

- **WHEN** ordinary/cached CO2 direct and staged routes receive equivalent valid
  prepared inputs whose observed-site labels are a subset of the supplied
  site-metadata labels
- **THEN** they use the same observed-site labels and aligned metadata
- **AND** model-specific per-site options retain their validation rules without
  requiring a new acquisition or filtering route

### Requirement: In-memory acquisition and prepared checkpoints

For standard/multisector, acquisition SHALL remain in memory. Durable reuse
SHALL use the prepared-input artifact; removed merged-data cache options SHALL
be rejected. Optional acquisition replay is deferred to #829.

#### Scenario: Supplied merged data

- **WHEN** a supported Python route receives compatible in-memory merged data
- **THEN** it bypasses acquisition and invokes the same scientific preparation
  operations, including filtering, basis, sensitivities, and assembly

#### Scenario: Fully prepared resume

- **WHEN** valid fully prepared inputs are supplied to a supported route
- **THEN** construction/inference proceeds without repeating any preparation phase

### Requirement: Retire the filtered merged-data checkpoint

Standard/multisector staged preparation SHALL stop producing the filtered merged
snapshot. New preparation manifests SHALL omit its retired `merged_data` artifact
and identity entries. Acquisition stays in memory; optional acquisition replay
is deferred to #829. No resume route for the retired checkpoint SHALL be
provided; removal SHALL NOT reinterpret historical filtered files as pre-filter
caches.
Numerical prepared-input/posterior formats and product schemas SHALL remain
compatible; staged metadata SHALL follow the explicit compatibility boundary
and supported-version policy below.

#### Scenario: Staged preparation

- **WHEN** standard or multisector staged preparation runs
- **THEN** it persists the prepared-input handoff and preparation manifest without
  the former filtered merged artifact or its manifest entries
- **AND** later stages consume the prepared inputs without requiring a merged snapshot

#### Scenario: Supported replay without a merged snapshot

- **WHEN** valid prepared inputs and posterior artifacts use a supported saved-output contract
- **THEN** replay needs no merged snapshot
- **AND** prepared/posterior identity, output-binding, and optional affine
  authentication remain strict

### Requirement: Scientific parity and explicit differences

Equivalent inputs and resolved scientific choices SHALL yield equivalent
scientific behavior across supported routes. Acceptance SHALL establish meaningful
parity using real scientific operations for standard, multisector, and supported
ordinary/cached CO2 variants; routing mocks alone SHALL NOT suffice. Intentional
route differences SHALL have a documented boundary rationale and regression
coverage. Independent stochastic trajectories SHALL NOT be required to match.

#### Scenario: Cross-route scientific parity

- **WHEN** equivalent full, merged/prepared-input, and staged routes are compared
  with controlled inputs and one posterior for product comparisons
- **THEN** preparation arrays/coordinates/retained metadata, model behavior,
  matched sampling policy, and scientific products agree at applicable boundaries
- **AND** roles, units, labelled dimensions, separate chain/draw axes, aggregation,
  conditional reconstruction, and borrowed/lazy input ownership remain compatible

#### Scenario: Supported boundary difference

- **WHEN** routes differ in supported customization, validation, defaults, or destinations
- **THEN** the difference has an explicit purpose and regression coverage without
  introducing conflicting scientific policy for equivalent resolved inputs

### Requirement: Scoped configuration and staged calling contract

Standard, multisector, and ordinary/cached CO2 workflows adopting the shared
staged interface SHALL use a common authoritative resolved configuration, with
one owner each for recipe-specific scientific choices, immutable sampler choices,
and output policy. Resolved choices SHALL be distinct from phase-complete
numerical handoffs, invocation artifact options, and live model/sampler/cache
state. Invocation overrides SHALL NOT mutate resolved choices, including nested
choices, or affect subsequent invocations. Existing direct Python sampler
interfaces SHALL remain supported.
Selection SHALL occur once before named stage operations. Independent builders,
direct runners, nested execution, and future recipes SHALL NOT be required to
adopt this configuration record or a complete stage suite merely to deliver
scientific functionality.
Standard and multisector SHALL have separate concrete scientific stage owners,
sharing identical mechanics without a combined scientific workflow controlled
by a recipe switch.

#### Scenario: Existing staged workflows

- **WHEN** an existing standard, multisector, or ordinary/cached CO2 workflow is selected
- **THEN** preparation, prior prediction, sampling, and postprocessing use explicit
  handoffs without repeated recipe selection or another recipe's scientific execution
- **AND** equivalent configuration-resolving routes supply equivalent scientific
  and sampling choices, retaining supported family/route defaults
- **AND** standard/multisector operations invoke their own canonical recipe
  operations through their separate stage owners

#### Scenario: Invocation choices do not leak

- **WHEN** two invocations derive different sampling overrides from one resolved configuration
- **THEN** each uses its intended effective choices without changing the original,
  including nested keyword choices
- **AND** a subsequent invocation without overrides uses the original choices
  without inheriting either invocation's runtime state

#### Scenario: Prepared handoff and invocation ownership

- **WHEN** a route receives valid prepared inputs and explicit artifact destinations
- **THEN** it uses the handoff's retained labels and phase meaning while retaining
  requested configuration separately for provenance
- **AND** no stage-only setup independently re-resolves scientific or output policy,
  and configuration access does not copy or materialize the numerical handoff

#### Scenario: Independent diagnosis

- **WHEN** diagnosis receives a posterior, optional sample envelope, and convergence options
- **THEN** it requires no scientific configuration, family setup, or model selection

### Requirement: Configuration and installed CLI compatibility

Existing configuration formats, option meanings, defaults, aliases, overrides,
source-relative paths, and rejection rules SHALL remain compatible. Installed
commands SHALL retain names, model choices, arguments, handoffs, stdout, and exit
policy, apart from the specified staged compatibility reset, retained-site
correction, and filtered merged artifact removal. This contract SHALL NOT add a
unified run command or broaden diagnostic policy.

#### Scenario: Existing command sequence and configuration

- **WHEN** a supported existing command sequence uses an explicit source and supported overrides
- **THEN** source-relative paths and override precedence remain unchanged, without
  ambient `CONFIG_FILE` choosing staged science
- **AND** numerical prepared-input/posterior formats and product names/schemas
  remain compatible, with staged metadata governed by its supported-version policy

#### Scenario: Invalid or inapplicable options

- **WHEN** an unknown or unsupported recipe/route option is supplied
- **THEN** the owning boundary reports an error before scientific artifact writes
  rather than ignoring it or changing the recipe

### Requirement: Scientific identities and recorded provenance

Within a supported identity contract, unchanged scientific settings SHALL retain
their family identity encoding and hash independently of incidental runtime
record layout. Requested preparation and retained-run sample/replay projections
SHALL remain distinct; existing exclusions for sampler/output/transport settings
SHALL remain unchanged. Content identities SHALL authenticate numerical handoffs
independently; replay SHALL retain recorded sampler provenance. Pre-refactor
staged hashes need not survive the explicit compatibility reset below.

#### Scenario: Internal representation or permitted runtime change

- **WHEN** representation changes or permitted sampler/output/transport settings
  differ within a supported identity contract
- **THEN** its scientific identity remains unchanged and matched artifacts remain usable
- **AND** replay reports sampling settings recorded in the sample manifest

#### Scenario: Scientific or content mismatch

- **WHEN** a relevant scientific choice or supplied artifact content mismatches its identity
- **THEN** authentication rejects reuse as a matched run

### Requirement: Explicit staged compatibility and schema-version support

This refactor SHALL establish an announced breaking boundary for pre-refactor
staged metadata and internal setup APIs. Compatibility with its existing
manifest/binding contracts and identity hashes SHALL NOT be required;
historical standard/multisector graph-based role recovery SHALL be retired.
Established numerical prepared-input/posterior formats, product contracts,
and the optional pre-filter acquisition cache SHALL retain their separate contracts.

Each family SHALL own and document its supported schema and identity contracts.
Writers SHALL identify their contract, and readers SHALL select and validate an
explicitly supported contract before posterior loading or output destination
creation. Incompatible contract changes SHALL be identifiable by version rather
than silently altering a previously declared version's meaning. Version handling
SHALL permit family-owned readers for selected older schema/identity contracts
alongside the current writer contract. Supporting an older version SHALL require
explicit validation of its identities and authentication obligations; it SHALL
NOT imply accepting every historical artifact or require a generic migration framework.

#### Scenario: Pre-refactor compatibility boundary

- **WHEN** a pre-refactor staged artifact uses a retired schema or identity contract
- **THEN** it is rejected clearly rather than triggering historical graph recovery
  or a compatibility migration
- **AND** this does not change the established standalone numerical artifact formats

#### Scenario: Deliberately supported older version

- **WHEN** a family declares an older schema/identity contract supported and receives
  matched artifacts for that contract
- **THEN** its corresponding reader validates their recorded meaning and identities
  and uses the shared product operations through authenticated graph-free replay
- **AND** current writes identify the current contract without requiring old files
  to be rewritten or weakening authentication for either version

#### Scenario: Unsupported or falsely labelled contract

- **WHEN** a contract is unsupported by the selected family, malformed, or labelled
  as an older version to bypass binding authentication
- **THEN** replay rejects it before posterior loading or output destination creation
  even if a shared envelope loader recognizes the version

### Requirement: Authenticated graph-free saved-output replay

Supported standard/multisector saved samples SHALL require an authenticated
output binding. Supported ordinary/cached CO2 samples SHALL retain authenticated
saved-output replay with independent optional affine authentication. These routes
SHALL replay without acquisition, model-input materialization, graph construction,
or resampling. Missing, malformed, altered, escaping, swapped, or mismatched
bindings SHALL fail before posterior loading or product writes, without graph fallback.

#### Scenario: Valid graph-free replay

- **WHEN** a supported standard, multisector, or ordinary/cached CO2 contract
  authenticates the prepared inputs, posterior, and required output information
- **THEN** supported products preserve scientific roles/metadata and chain/draw axes
  with model-input materialization, construction, and sampling forbidden

#### Scenario: Invalid output binding

- **WHEN** a binding is missing, its bytes/digest or schema/contract is invalid,
  its path escapes the allowed directory, or its prepared/posterior identities mismatch
- **THEN** it is rejected before posterior loading or product writes without graph fallback

#### Scenario: Affine mismatch or supported relocation

- **WHEN** an optional CO2 affine artifact is altered or mismatches prepared/sampling identities
- **THEN** it is rejected before product writes
- **AND** existing supported relocation of identical authenticated content remains valid

### Requirement: Preserve readiness and convergence behavior

Checks SHALL retain names/schemas, thresholds, and existing pass/fail/unknown
semantics. Returned invalid predictive evidence SHALL retain its scientific
readiness classification. Existing family execution-exception boundaries SHALL
remain distinct. External input validation/loading/authentication failures outside
the readiness catch and artifact serialization failures SHALL propagate as errors
rather than successful checks. No diagnostic redesign is required.

#### Scenario: Returned predictive evidence

- **WHEN** prior prediction returns finite, non-finite, or empty evidence
- **THEN** each family applies its existing assessment, including a fail check for
  non-finite/empty evidence and the existing strict-mode exit when that check fails
- **AND** existing predictive and reporting artifacts are still saved, including
  CO2's prior manifest for returned evidence

#### Scenario: Standard or multisector caught readiness error

- **WHEN** a build/predictive operation raises a KeyError or ValueError within
  standard/multisector's existing readiness catch scope
- **THEN** it returns the existing fail check without predictive artifacts,
  preserving strict-mode handling

#### Scenario: CO2 execution error

- **WHEN** ordinary/cached CO2 construction or prior prediction raises an execution error
- **THEN** the error propagates without a readiness check or newly created output destination

#### Scenario: Authentication or serialization error

- **WHEN** external input validation/loading/authentication outside the readiness
  catch or a readiness artifact writer fails
- **THEN** the error propagates without conversion to a readiness check

#### Scenario: Convergence threshold and unavailable metric

- **WHEN** a finite convergence metric fails and another is unavailable
- **THEN** the check remains fail; otherwise unassessable evidence retains unknown
- **AND** strict scientific gates retain existing nonzero behavior for fail
  without newly treating unknown as fail
