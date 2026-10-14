# Resource-I/O compatibility amendment

## MODIFIED Requirements

### Requirement: Compatible public adapters and scientific choices

Except for the resource-I/O migrations removing `prepare_basis(output_path=...)`
and `country_directory` in 0.8 and replacing scientific BC case/directory inputs
with resolved basis data, the legacy-only routing and namespace migration owned
by the [HBMCMC isolation contract](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/hbmcmc-isolation-retirement/spec.md),
the removed `prepare_rhime_inputs` convenience function, obsolete
modern merged-cache options, former stage adapter, `resolve_rhime_config`,
transitional old-location imports and revised INI reader contract specified below, supported
scientific Python and CLI entry points SHALL retain their signatures, shorthand,
override behavior and return contracts through adapters to the same semantic
resolution or shared applicable site translation. Direct preparation/retrieval
APIs SHALL resolve only applicable choices without requiring likelihood, final
output or sampling settings. Independent builders and prepared-input runners
SHALL remain independent of full configuration. Shared choices such as BC use
and flux-source routing SHALL retain consistent existing meanings across data
preparation and model construction. Current INI section interpretation, path
conventions, file vocabulary, scientific calculations and sampling defaults
SHALL remain unchanged except for the path and legacy-resource migrations specified by
[rhime-resource-io](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-resource-io/spec.md).

Internal orchestration/setup records and their constructor/helper return shapes
SHALL NOT be required compatibility contracts. Their in-repository consumers
SHALL migrate to the unified requested configuration.

Runner/frontend resolution SHALL consume omitted/false `use_tracer` without
retaining a config field. Canonical configuration SHALL accept only canonical
fields and use generic unsupported-option errors for obsolete options. Effective true SHALL retain early rejection at resolution and relevant
direct public boundaries, including supplied merged data. Existing legacy-option
and custom-likelihood conflict precedence SHALL remain unchanged.

#### Scenario: Existing Python shorthand

- **WHEN** standard, multisector or direct public preparation/retrieval callers
  supply supported scalar options instead of expanded sequences
- **THEN** the adapter establishes the same applicable canonical choices before
  scientific work and preserves the established return contract
- **AND** internal canonical calls do not repeat that translation

#### Scenario: Boundary conditions are disabled consistently

- **WHEN** effective configuration sets `use_bc=False`
- **THEN** acquisition/preparation and built-in model construction use the same
  disabled BC choice with existing scientific behavior
- **AND** neither phase infers an independent conflicting default

#### Scenario: Preserve unsupported tracer rejection

- **WHEN** a configured run, direct public preparation call or mapping-based
  retrieval supplies effective `use_tracer=True` after supported overrides,
  including with supplied merged data
- **THEN** it rejects the option before acquisition or scientific execution
- **AND** omitted and false options remain accepted with equivalent resolved
  configuration containing no tracer field

### Requirement: Basis preparation receives explicit scientific inputs

Modern preparation SHALL accept an exact retained `basis_path` or resolved
labelled `basis_partition`, and resolved `landsea_classes` and `outer_regions`.
A labelled partition SHALL remain a valid scientific representation,
independently of historical artifact policy. Scientific preparation and numerical
kernels SHALL NOT depend on legacy discovery or naming policy.

`RhimeConfig.from_params` SHALL remain data-free. Supported deprecated resource
selectors SHALL be translated once at the legacy entry boundary defined by the
separate HBMCMC isolation/retirement contract before
canonical configuration is handed to scientific execution; resolved requests
SHALL NOT retain historical case/directory discovery fields. The removed
`country_directory` option SHALL be rejected rather than translated. Resource
loading, exact paths, packaged defaults, BC inputs and optional exports follow
[rhime-resource-io](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-resource-io/spec.md).

#### Scenario: Resolved basis scientific inputs

- **WHEN** preparation receives a selected operator path or labelled partition and its applicable resolved support maps
- **THEN** it consumes those explicit scientific inputs without historical filename discovery
- **AND** a labelled in-memory partition remains valid independently of artifact format

## ADDED Requirements

### Requirement: Declared filesystem types identify path conversion

Canonical configuration SHALL represent filesystem selections using `Path`
types, including optional paths and explicit collections where supported.
Frontends SHALL accept external path strings and convert the declared path fields
with the owning frontend's base directory. Absolute paths SHALL keep their
selected destination. File-origin values and explicit invocation overrides SHALL
retain their distinct bases during conversion. Ordinary strings such as domain
names and scientific selectors SHALL NOT be interpreted as paths.

Path-field identification SHALL use type annotations through a small
standard-library converter supplied with the owning record's declarations. The
converter SHALL NOT import concrete configuration owners or inspect unrelated
configuration records. No second path-name registry or new configuration framework
dependency SHALL be introduced. Each external value SHALL acquire its base and
canonical path at one owning boundary; downstream factories, runners and identity
encoding SHALL forward resolved paths without repeating normalization. Independent
output configuration SHALL normalize its own external inputs without requiring a
complete run configuration. Historical selection translation, when applicable
under the separate legacy contract, SHALL precede canonical construction;
packaged scientific defaults remain runner-owned. Canonical construction SHALL
NOT glob or open candidate resource artifacts.

#### Scenario: Typed paths retain their frontend origins

- **WHEN** a saved-file request and invocation overrides supply strings for declared filesystem fields
- **THEN** conversion produces absolute `Path` values using each value's file or invocation base
- **AND** non-path strings retain their scientific meaning without path conversion
- **AND** configuration construction performs no scientific resource loading

#### Scenario: Forward a resolved request

- **WHEN** a configured runner or identity encoder receives canonical configuration
- **THEN** it uses the resolved paths without another conversion or legacy adaptation
- **AND** direct canonical construction accepts resolved values, while public
  external-input factories accept supported path strings

#### Scenario: Construct output configuration independently

- **WHEN** a caller constructs only output configuration from external strings
- **THEN** its owner converts only its declared filesystem fields using the invocation base
- **AND** it does not inspect or construct full run configuration or load resources
