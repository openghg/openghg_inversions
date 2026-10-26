# Explicit boundary-condition basis selection

## ADDED Requirements

### Requirement: Explicit boundary-condition basis choices

Canonical data configuration SHALL accept `bc_basis_algorithm`, defaulting to
`"NESW"`, separately from exact `bc_basis_paths`. `use_bc=False` SHALL disable BC
basis resolution and loading. With BCs enabled, explicit `bc_basis_paths` SHALL
select the file-backed basis and make `bc_basis_algorithm` inactive. Otherwise,
`bc_basis_algorithm="NESW"` SHALL select the built-in directional calculation.

Supported-name validation SHALL apply ordinarily, including inactive groups.
An active algorithm SHALL be validated against supported algorithms by its
configuration owner before acquisition or scientific preparation. An inactive
algorithm value SHALL remain deferred, following the shared phase policy.
No historical-name or historical-token validation branch SHALL be introduced.
Resource resolution SHALL follow
[Boundary-condition basis selection precedes scientific preparation](../../../../specs/rhime-resource-io/spec.md#requirement-boundary-condition-basis-selection-precedes-scientific-preparation).

#### Scenario: Select built-in NESW explicitly or by default

- **WHEN** BCs are enabled without exact file paths and the algorithm is omitted or `"NESW"`
- **THEN** configuration selects built-in NESW before acquisition
- **AND** numerical sensitivity construction receives the same resource-free BC operand

#### Scenario: Exact files take precedence

- **WHEN** BCs are enabled and exact BC basis paths are supplied with an algorithm value
- **THEN** the paths select the file-backed basis and the algorithm is inactive
- **AND** an invalid selected file fails rather than falling back to NESW

#### Scenario: Disabled BCs do not resolve resources

- **WHEN** `use_bc=False` accompanies BC basis paths or an algorithm value
- **THEN** no BC basis file is read and no BC algorithm value is consumed
- **AND** unsupported option names still receive ordinary validation
