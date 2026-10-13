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
