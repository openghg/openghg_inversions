# rhime-resource-io Specification

## Purpose

Define how RHIME entry points select external resources and preserve export
destinations, independently of configuration representation or checkpoint format.
These guarantees apply to full and saved routes and to direct Python runners.

Option meanings, defaults, aliases and grouping belong to
[rhime-configuration](../rhime-configuration/spec.md). Scientific basis construction
and current-runtime-flux binding remain with their scientific contracts. References
inside manifests, checkpoint-owned files and save-completion rules belong to the
[prepared handoff contract](../../changes/replace-recipe-workflow-with-reusable-handoffs/specs/prepared-recipe-handoff/spec.md).

## Review and delivery status

This capability separates resource I/O from configuration in
[#867](https://github.com/openghg/openghg_inversions/pull/867).

- Exact basis artifact destinations and selected-resource checks below were
  delivered by #840 for #838. Moving their requirements here does not change
  their behavior or make them pending. #866's export rebasing is a regression
  against that existing contract.
- The final requirement, "Runners own resource and export paths", includes
  pending #868 work: consistent invocation-override path interpretation and
  moving optional saving out of `prepare_basis`. Its `output_path` argument
  remains in the implementation until that migration lands.

[#868](https://github.com/openghg/openghg_inversions/issues/868) delivers the
bounded prerequisite before #866 adopts it. No new resource manager or numerical
codec is required. BC case/directory migration and acquisition replay (#829)
remain outside this scope.

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
