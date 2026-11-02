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

For independently deprecated standalone basis adapters, historical case/glob
discovery, filename construction, legacy schema adaptation and explicit legacy
writing SHALL remain independently supported through `basis.compatibility`.
Modern run-resource selection SHALL use explicit canonical inputs without
HBMCMC or historical discovery. Saved bases SHALL default to modern retained
artifacts; legacy output SHALL remain explicit opt-in through that adapter. Its independent deprecation SHALL NOT imply
automatic removal. Canonical scientific operations SHALL receive exact paths or
labelled arrays without depending on historical discovery or naming policy.
HBMCMC retirement under its isolation contract SHALL NOT authorize deletion of
these independent basis capabilities or imply their automatic removal date.
Exact destinations and useful labelled
partition inputs SHALL remain independently supported.

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

### Requirement: Runners own resource and export paths

Runner/frontends SHALL resolve external resource and export paths before calling
scientific operations. Direct Python/CLI paths, including explicit overrides,
SHALL resolve relative to the invocation working directory, including resource
and output values read from INI or JSON configuration in full and saved phases.
Absolute paths SHALL retain their selected destination.

Scientific preparation SHALL receive selected exact input paths or resolved
labelled arrays and SHALL return its basis result without saving an export.
Canonical `prepare_basis` SHALL NOT accept `output_path`. The runner SHALL own
the optional `basis_output_path` choice and call the existing basis save codec
on the returned result when requested. It SHALL coordinate that export separately
from checkpoint-owned saves, following the existing exact-destination contract.
Relative artifact references inside saved manifests remain governed by the
prepared handoff contract, not these invocation rules.

#### Scenario: File resource and invocation override

- **WHEN** a saved-phase file at `/configs/run.json` supplies `basis_path="basis.nc"`
- **THEN** it selects `basis.nc` beneath the invocation working directory
- **AND** an explicit `basis_path="basis.nc"` invocation override selects the same file
- **AND** scientific preparation receives the selected absolute path without another resolution pass

#### Scenario: Runner saves the scientific basis result

- **WHEN** scientific preparation constructs or reuses a basis
- **THEN** it returns the basis without an export destination or export write
- **AND** a runner with a requested `basis_output_path` saves that result through its existing codec after scientific preparation
- **AND** omitting `basis_output_path` performs no separate basis export

### Requirement: Boundary-condition basis selection precedes scientific preparation

Runners SHALL resolve and load selected file-backed BC bases before acquisition,
then pass labelled basis data to sensitivity construction. Scientific sensitivity
functions SHALL NOT accept case/directory discovery arguments, glob for files,
infer resource filenames from the domain or create resource directories.
Modern requests SHALL select exact BC basis paths; explicitly selected multiple
files SHALL retain the existing time combination and alignment semantics.

The built-in NESW calculation SHALL use acquired directional BC contributions
without reading any BC basis file. Disabled BCs SHALL NOT trigger basis-resource
loading. File-backed selections SHALL fail early for missing or unreadable
selected files rather than silently dropping them.

#### Scenario: NESW does not select a file

- **WHEN** a canonical modern request selects the built-in NESW calculation
- **THEN** no BC basis-file lookup is performed
- **AND** sensitivities use acquired directional BC contributions

#### Scenario: Explicit time-dependent BC basis

- **WHEN** a request explicitly selects a file-backed BC basis spanning multiple times or files
- **THEN** the runner loads the selected resources before acquisition and passes labelled data onward
- **AND** sensitivity construction preserves existing time alignment without case/directory discovery
- **AND** a missing or unreadable selected file fails before acquisition

### Requirement: Packaged preparation defaults remain distinct from reporting maps

For algorithms requiring land/sea or outer-region maps, runners SHALL resolve
omitted `landsea_path` or `outer_regions_path` to the vendored default for the
exact requested domain and load it through the existing support-map loader.
Explicit paths SHALL take precedence and SHALL NOT fall back when missing.
An unavailable requested-domain default SHALL fail before acquisition without
selecting another domain or creating directories. Algorithms not needing a map
SHALL NOT require its default. Scientific preparation SHALL receive labelled maps.

`country_file` SHALL remain the explicit reporting country definition, independent
of these preparation defaults; the domain name SHALL NOT infer a reporting map.
Reporting SHALL use `country_file`; preparation SHALL use `landsea_path` or
`outer_regions_path` according to the selected operation.

#### Scenario: Packaged preparation with explicit reporting

- **WHEN** a run selects EUROPE, an algorithm needing support maps and an explicit reporting `country_file`
- **THEN** the runner selects available EUROPE packaged preparation defaults for omitted support paths
- **AND** the reporting country definition remains the explicitly selected file
- **AND** an explicit preparation path overrides only its corresponding packaged default

### Requirement: Explicit partition collections have deterministic ordering

Configured runners SHALL accept `basis_partition_paths` and nested
`inner_basis_partition_paths` as nonempty explicit path collections, mutually
exclusive with the corresponding singular partition and retained-basis choices.
Runners SHALL sort resolved paths lexically before calling the existing partition
loader. The first sorted file SHALL provide its latitude/longitude reference;
existing combination and alignment semantics SHALL be preserved. Selection SHALL
perform no globbing. All selected files SHALL be checked before acquisition.

#### Scenario: Reverse a partition collection

- **WHEN** a caller reverses the same selected partition paths
- **THEN** loading uses the same sorted order and coordinate reference
- **AND** the prepared partition is unchanged
