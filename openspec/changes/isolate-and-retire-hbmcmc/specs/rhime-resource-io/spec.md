# Explicit boundary-condition resource selection

## MODIFIED Requirements

### Requirement: Boundary-condition basis selection precedes scientific preparation

Runners SHALL resolve and load selected file-backed BC bases before acquisition,
then pass labelled basis data to sensitivity construction. Scientific sensitivity
functions SHALL NOT accept case/directory discovery arguments, glob for files,
infer resource filenames from the domain or create resource directories.
Canonical `bc_basis_algorithm="NESW"` SHALL select the resource-free built-in
calculation when no exact `bc_basis_paths` are supplied. With BCs enabled, exact
paths SHALL take precedence and the algorithm value SHALL remain inactive.
Selected multiple files SHALL retain existing time combination and alignment
semantics. `use_bc=False` SHALL bypass BC basis resolution and file loading.

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

#### Scenario: Resolve a BC operand once

- **WHEN** a runner executes a complete or staged preparation request
- **THEN** the application boundary resolves its enabled algorithm or exact-file selection before acquisition
- **AND** scientific sensitivity functions consume the resulting labeled dataset or resource-free built-in operand without re-resolving configuration

