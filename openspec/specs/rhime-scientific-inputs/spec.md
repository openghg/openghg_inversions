# RHIME scientific inputs

## Purpose

Define backend-neutral scientific preparation and canonical prepared-input
construction. Requested choices belong to configuration; saved representations,
format/version admission and reuse belong to persistence and handoff contracts.

## Requirements

### Requirement: Canonical prepared scientific inputs

`PreparedInputs` SHALL support construction from `inv_inputs`, `basis_functions`
and labeled `site_metadata`. Construction and xarray preparation SHALL preserve
scientific values and units, site/averaging-period ordering, observation-aligned
release coordinates, and existing basis source/path provenance. The xarray
interface SHALL retain the `"supplied"` fallback for an unannotated supplied basis.
Numerical arrays SHALL remain borrowed and potentially lazy; construction SHALL
NOT mutate caller inputs, copy their payloads or execute their Dask graphs.

Serialization, supported format/version admission and saved-operand reuse SHALL
follow the owning codecs and the
[prepared handoff](../../changes/replace-recipe-workflow-with-reusable-handoffs/specs/prepared-recipe-handoff/spec.md#requirement-ordinary-file-references-and-current-formats)
and [sampled replay](../../changes/replace-recipe-workflow-with-reusable-handoffs/specs/sampled-recipe-handoff/spec.md#requirement-replay-uses-existing-scientific-validation)
contracts. Python constructor naming SHALL NOT freeze stored identifiers or imply
compatibility with unsupported versions.

#### Scenario: Construct canonical prepared inputs

- **WHEN** a caller supplies canonical inversion inputs, a basis and labeled site metadata
- **THEN** the prepared value preserves their scientific values, units, ordering and provenance
- **AND** it borrows numerical payloads without hidden copying or execution

#### Scenario: Prepare supplied xarray inputs

- **WHEN** xarray preparation receives supported stationary or moving release coordinates and a supplied basis
- **THEN** observation-aligned coordinates and existing basis source/path provenance are retained
- **AND** an unannotated supplied basis retains the `"supplied"` fallback without mutation

### Requirement: Scientific preparation owns its operations

Scientific preparation SHALL receive acquired or supplied labelled data and
explicit resolved scientific choices. It SHALL perform basis, sensitivity,
filtering and assembly operations without configuration resolution retrieving
scientific data or implicitly saving artifacts. Scientific stage functions SHALL
remain independently callable with named scientific inputs.

Observation selection, observation-error derivation and missing-data population
rules SHALL belong to scientific preparation rather than requested configuration.
Provider acquisition averaging, selection filters and explicit aggregation SHALL
remain distinct operations; configuration resolution SHALL NOT impose an
additional scientific transformation or error-construction order.

Assembly SHALL preserve minimum-error `None`, integer and per-site mapping
behavior. Prepared-input persistence SHALL remain independent of acquisition
persistence. Public selector construction SHALL use `SiteOptions.from_inputs`,
preserving supported scalar/sequence shorthand and retained-site alignment.
Modern acquisition SHALL NOT load or save merged-data caches.

#### Scenario: Shared nested spatial support

- **WHEN** retained nested sites share one outer domain and one inner domain
- **THEN** preparation normalizes coordinate labels against their domain definitions, falling back to the first retained site with a warning for an unregistered domain
- **AND** incompatible grids are rejected before overlap masking or basis projection
- **AND** compatible sites share one inner rectangular extent, applied on each actual outer footprint and flux grid without mutating or eagerly computing borrowed arrays
- **AND** output metadata retains the applied footprint rectangle; externally pre-masked inputs with no supplied extent retain an unknown mask extent

#### Scenario: Direct scientific stages use named inputs

- **WHEN** a caller supplies acquired data and named required scientific choices
- **THEN** each stage can execute without a `RhimeConfig` or raw parameter mapping
- **AND** omitting a required keyword is rejected at the function-call boundary

