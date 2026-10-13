# Resource-I/O amendments

## REMOVED Requirements

### Requirement: Optional basis saving during preparation

**Reason**: Optional export ownership moves from scientific preparation to runners.
**Migration**: Remove `prepare_basis(output_path=...)`; explicitly save the returned
basis in the runner with the existing codec and selected `basis_output_path`.

## ADDED Requirements

### Requirement: Runners own resource and export paths

Runner/frontends SHALL resolve external resource and export paths before calling
scientific operations. Direct Python/CLI paths, including explicit overrides,
SHALL resolve relative to the invocation working directory. Saved-phase file
resource paths SHALL resolve relative to the configuration file's directory.
Absolute paths SHALL retain their selected destination. These rules clarify the
saved-phase override convention; other established file-frontend conventions
remain unchanged and SHALL be documented by their frontend.

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
- **THEN** it selects `/configs/basis.nc`
- **AND** an explicit `basis_path="basis.nc"` invocation override instead selects the file beneath the invocation working directory
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

The built-in NESW calculation SHALL use the acquired directional BC contributions
without reading any BC basis file. Supported historical case/directory requests
SHALL be translated once at the configuration compatibility boundary, after
winning overrides and before scientific work. For NESW, a supplied legacy
`bc_basis_directory` SHALL produce an unused/deprecated-option warning and SHALL
NOT cause file discovery or a missing-directory error. Disabled BCs SHALL NOT
trigger basis-resource loading. File-backed selections SHALL fail early for
missing or unreadable selected files rather than silently dropping them.

#### Scenario: NESW does not select a file

- **WHEN** a request uses NESW and includes a historical `bc_basis_directory`
- **THEN** the input adapter explains that the directory is unused and deprecated
- **AND** the runner performs no BC basis-file lookup, even if that directory is absent
- **AND** sensitivities use the acquired directional BC contributions

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
`country_directory` SHALL be removed from run configuration in 0.8. Entry adapters
SHALL reject it before scientific work with guidance to use `country_file`,
`landsea_path` or `outer_regions_path` according to the intended operation.

#### Scenario: Packaged preparation with explicit reporting

- **WHEN** a run selects EUROPE, an algorithm needing support maps and an explicit reporting `country_file`
- **THEN** the runner selects available EUROPE packaged preparation defaults for omitted support paths
- **AND** the reporting country definition remains the explicitly selected file
- **AND** an explicit preparation path overrides only its corresponding packaged default

#### Scenario: Removed country directory

- **WHEN** a 0.8 run request supplies `country_directory`
- **THEN** its entry adapter rejects the option with the operation-specific replacements before acquisition
- **AND** no downstream compatibility branch interprets that directory
