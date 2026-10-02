# Spec Delta

> Draft behavioral contract for OPE-207; not an implementation claim.

## Purpose

Keep equivalent supported recipe execution routes scientifically consistent,
with optional checkpoints and authenticated saved-output replay. Preserve the
interfaces and artifact contracts relied on by inversion scientists and callers.

## ADDED Requirements

### Requirement: Shared execution across supported routes

A recipe's equivalent supported full, prepared-input, and staged routes SHALL
reuse canonical operations for their applicable preparation, construction,
sampling, and result/product phases. Stages SHALL add checkpoint loading,
authentication, persistence, and reporting without independent scientific policy.
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
- **THEN** they reuse the same family product operations without resampling,
  following only that family's specified graph-recovery policy

### Requirement: Retained-site consistency in preparation

Standard and multisector preparation SHALL share filtering, retained-site
handling, basis, sensitivities, and labelled assembly across their supported
routes. Valid retained subsets after acquisition, compatible merged-cache reload,
or filtering SHALL be accepted consistently with full preparation, with every
per-site option aligned to retained labels. This replaces staged rejection of
missing requested sites. An empty retained set SHALL fail before basis/inference.
Requested configuration SHALL remain separately available for provenance;
reconciliation SHALL NOT bypass malformed-input or cache-compatibility validation.

#### Scenario: Acquisition returns a valid subset

- **WHEN** acquisition returns valid observations for only some requested sites
- **THEN** full and staged preparation retain the same observations, labels,
  averaging periods, and other per-site options using requested-site alignment

#### Scenario: Compatible merged cache lacks a requested site

- **WHEN** a valid compatible merged cache contains only a subset of requested sites
- **THEN** full and staged preparation retain that subset and align every per-site option
- **AND** incompatible cache inputs retain their existing rejection/fallback policy

#### Scenario: Filtering empties a site

- **WHEN** valid filtering removes all observations at one site but retains others
- **THEN** full and staged preparation produce equivalent retained observations
  and aligned metadata, which prepared execution uses consistently

#### Scenario: Empty or malformed handoff

- **WHEN** acquisition, compatible cache reload, or filtering leaves no usable sites
- **THEN** preparation fails before basis construction or inference
- **AND** invalid labels/metadata and malformed or mismatched independently
  supplied prepared inputs retain their owning boundary's rejection policy

### Requirement: Explicit checkpoint phase boundaries

Supported merged-data and fully prepared checkpoints SHALL resume through the
same remaining recipe operations, preserving formats, names, and existing
validation/authentication. Pre-filter merged caches and staged filtered merged
data SHALL retain distinct meanings. Execution SHALL NOT guess an arbitrary
file's phase, repeat completed transformations, mutate borrowed handoffs, or
require new installed commands or artifact schemas.

#### Scenario: Pre-filter merged resume

- **WHEN** a supported route resumes acquired, external, or reloaded pre-filter merged data
- **THEN** it bypasses acquisition and uses canonical filtering and remaining preparation

#### Scenario: Filtered merged resume

- **WHEN** an explicitly supported Python boundary resumes a known staged filtered
  checkpoint with its required provenance
- **THEN** it starts after filtering and uses the same remaining preparation operations
- **AND** ordinary merged reload is not automatically reinterpreted as filtered resume

#### Scenario: Fully prepared resume

- **WHEN** valid fully prepared inputs are supplied to a supported route
- **THEN** construction/inference proceeds without repeating any preparation phase

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
staged interface SHALL use a common authoritative resolved configuration, with one owner each
for common sampler/output choices and distinct family scientific options.
Selection SHALL occur once before named stage operations. Independent builders,
direct runners, nested execution, and future recipes SHALL NOT be required to
adopt this configuration record or a complete stage suite merely to deliver scientific functionality.
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

#### Scenario: Independent diagnosis

- **WHEN** diagnosis receives a posterior, optional sample envelope, and convergence options
- **THEN** it requires no scientific configuration, family setup, or model selection

### Requirement: Configuration and installed CLI compatibility

Existing configuration formats, option meanings, defaults, aliases, overrides,
source-relative paths, and rejection rules SHALL remain compatible. Installed
commands SHALL retain names, model choices, arguments, handoffs, stdout, and exit
policy. The retained-site correction above is the explicit staged acceptance
change. This contract SHALL NOT add a unified run command or broaden diagnostic policy.

#### Scenario: Existing command sequence and configuration

- **WHEN** a supported existing command sequence uses an explicit source and supported overrides
- **THEN** source-relative paths and override precedence remain unchanged, without
  ambient `CONFIG_FILE` choosing staged science
- **AND** the same handoffs/products and artifact names/schemas remain compatible,
  including the explicitly corrected retained-site cases

#### Scenario: Invalid or inapplicable options

- **WHEN** an unknown or unsupported recipe/route option is supplied
- **THEN** the owning boundary reports an error before scientific artifact writes
  rather than ignoring it or changing the recipe

### Requirement: Stable identities and recorded provenance

Unchanged scientific settings SHALL retain existing family identity encodings
and hashes, including requested preparation and retained-run sample/replay
projections. Existing exclusions for sampler/output/transport settings SHALL
remain unchanged. Content identities SHALL authenticate numerical handoffs
independently; replay SHALL retain recorded sampler provenance.

#### Scenario: Internal representation or permitted runtime change

- **WHEN** configuration representation changes or currently permitted sampler,
  output, or transport settings differ
- **THEN** historical scientific identities remain compatible and matched artifacts remain usable
- **AND** replay reports sampling settings recorded in the sample manifest

#### Scenario: Scientific or content mismatch

- **WHEN** a relevant scientific choice or supplied artifact content mismatches its identity
- **THEN** authentication rejects reuse as a matched run

### Requirement: Standard and multisector version-2 replay

Standard/multisector version-2 samples SHALL require an authenticated output
binding and replay without acquisition, model-input materialization, or graph
construction. Missing, malformed, altered, escaping, swapped, or mismatched
bindings SHALL fail before posterior loading or product writes, without graph fallback.

#### Scenario: Valid graph-free replay

- **WHEN** either family's version-2 manifest binds the supplied prepared inputs,
  posterior, and valid saved output contract
- **THEN** supported products preserve scientific roles/metadata and chain/draw axes
  with materialization and model construction forbidden

#### Scenario: Invalid output binding

- **WHEN** a binding is missing, its bytes/digest or schema/contract is invalid,
  its path escapes the allowed directory, or its prepared/posterior identities mismatch
- **THEN** it is rejected before posterior loading or product writes without graph fallback

### Requirement: Genuine historical standard and multisector replay

Genuine version-1 standard/multisector samples without output bindings SHALL
retain graph-building role recovery through canonical recipe construction,
followed by shared product operations without resampling. Binding-bearing
version-1 manifests SHALL be rejected instead of bypassing authentication.

#### Scenario: Historical manifest

- **WHEN** a matched historical version-1 manifest has no output-binding entries
- **THEN** its established role-recovery route and supported products remain available

#### Scenario: Binding-bearing downgrade

- **WHEN** a version-1 manifest advertises bindings in artifacts or identities
- **THEN** it is rejected rather than treated as genuine historical replay

### Requirement: CO2 version-1 replay and affine authentication

Ordinary/cached CO2 SHALL retain graph-free version-1 sample replay and reject
other sample-manifest versions before posterior loading or output destination
creation. Optional affine reconstruction SHALL retain independent content
authentication and binding to prepared inputs and sampling records.

#### Scenario: Supported CO2 replay

- **WHEN** matched version-1 ordinary/cached CO2 artifacts are replayed
- **THEN** supported products succeed with model-input materialization and graph
  construction forbidden

#### Scenario: Unsupported CO2 sample version

- **WHEN** a CO2 sample manifest declares another version, including version 2
  with an absent/malformed binding or incorrect digest
- **THEN** replay rejects it before posterior loading or output destination creation

#### Scenario: Affine mismatch or supported relocation

- **WHEN** an affine artifact is altered or mismatches prepared/sampling identities
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
