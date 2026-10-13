# rhime-resource-io Specification

## Purpose

Define how RHIME entry points select external resources and preserve export
destinations, independently of configuration representation or checkpoint format.
These guarantees apply to full and saved routes and to direct Python runners.

Option meanings, defaults, aliases and grouping belong to
[rhime-configuration](../rhime-configuration/spec.md). Scientific basis construction
and current-runtime-flux binding remain with their scientific contracts. Manifest
references and checkpoint formats are outside this resource-selection contract.

## Requirements

### Requirement: Exact basis artifact destinations

Modern basis artifact inputs and exports SHALL accept arbitrary exact `.nc` or
`.zarr` paths without filename parsing or glob discovery. Full and staged routes
SHALL preserve the caller-selected export destination, including a separately
requested export outside the checkpoint directory. Checkpoint containment SHALL
NOT rebase or restrict that separate export.

Historical case/glob discovery, filename construction, legacy schema adaptation
and explicit legacy writing SHALL remain in `basis.compatibility`. For devel/0.8,
saved bases SHALL default to modern retained artifacts; legacy output SHALL remain
explicit opt-in through that adapter. Its 0.9 deprecation SHALL NOT imply automatic
removal in 0.9. Canonical scientific operations SHALL receive exact paths or
labelled arrays without depending on historical discovery or naming policy.

#### Scenario: Arbitrary retained artifact destination

- **WHEN** a runner selects an arbitrary exact `.nc` or `.zarr` basis input or output
- **THEN** no filename parsing or glob discovery is required
- **AND** staged preparation preserves the caller's exact destination, including a separately selected export outside its checkpoint directory

#### Scenario: Historical basis entry boundary

- **WHEN** supported deprecated basis arguments are supplied
- **THEN** a compatibility adapter warns and translates them once
- **AND** canonical preparation receives exact paths or labelled arrays
- **AND** explicit legacy writing preserves partitions and scientific results

### Requirement: Selected resources resolve before acquisition

The application boundary SHALL check selected basis artifact paths and resolve
support maps before acquisition, then pass canonical inputs to scientific
preparation. Configured `basis_partition_path` SHALL load the explicit `basis`
variable there. Packaged basis support maps SHALL remain explicitly selectable.
A missing selected file or unavailable requested domain resource SHALL fail
without another-domain substitution or read-side directory creation.

Country reporting maps SHALL remain separate caller-selected inputs. Complete
runs requesting `basic` or `paris` reporting SHALL require an explicit
`country_file` and check selected files before acquisition. Resource loading
SHALL remain separate from data-free configuration construction; the existing
`resolve_basis_inputs` application boundary supplies the basis inputs.

#### Scenario: Selected support map is unavailable

- **WHEN** a selected path is missing or the requested domain lacks a packaged map
- **THEN** the application boundary fails before acquisition or weight computation
- **AND** it neither substitutes another domain nor creates a read directory

### Requirement: Optional basis saving during preparation

Canonical `prepare_basis` SHALL accept an optional exact `output_path`. When
supplied, preparation SHALL save the resulting basis through its existing codec
and retain the selected artifact-path metadata. Saving SHALL apply to generated
bases, reused operators and supplied partitions. `None` SHALL disable that save.

#### Scenario: Prepare with an optional basis export

- **WHEN** a caller prepares a basis with an explicit `output_path`
- **THEN** preparation saves the resulting basis at that exact destination
- **AND** omitting the destination performs no separate basis export
