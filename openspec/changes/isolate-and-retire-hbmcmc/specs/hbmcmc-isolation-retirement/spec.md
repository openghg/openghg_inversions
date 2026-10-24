# HBMCMC retirement

## ADDED Requirements

### Requirement: HBMCMC is unavailable in the modern distribution

The modern wheel and source distribution SHALL contain no
`openghg_inversions.hbmcmc` subpackage, executable wrapper, configuration adapter
or legacy data/basis utility. No import alias, re-export, forwarding shim or
executable entry point SHALL recreate that namespace. Modern execution SHALL NOT
import or invoke migration tooling.

Modern runner/acquisition boundaries SHALL validate supported option names and
applicable values generically under
[Compatible public adapters and scientific choices](../../../../specs/rhime-configuration/spec.md#requirement-compatible-public-adapters-and-scientific-choices).
Unknown names SHALL be rejected before scientific work. Known inactive values
SHALL remain deferred until their consuming phase. Validation SHALL NOT recognize
historical names or output tokens specially.
Every explicit `use_tracer` occurrence SHALL be unsupported, regardless of value;
omission SHALL remain valid without enabling tracer execution.

Public Python naming removals SHALL follow the finite
[five-alias retirement](../../../../specs/rhime-model-selection/spec.md#requirement-retire-named-python-aliases).
The separately approved
[prepared-input factory retirement](../../../../specs/rhime-configuration/spec.md#requirement-retire-the-prepared-input-compatibility-factory)
SHALL remove only its named compatibility constructor, preserving canonical
construction and existing schema/codec behavior.
Installed commands SHALL use the canonical `rhime` routes in
[Bounded installed-command retirement](../../../../specs/rhime-model-selection/spec.md#requirement-bounded-installed-command-retirement).
Other Python names, canonical signatures, result contents and scientific meanings
SHALL retain their separately specified contracts.

Historical observations and canonical scientific coverage SHALL retain the
[migration evidence guarantees](../../../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence).
Modern `output_format="legacy"`, saved `InversionOutput` exports, exact artifact
destinations and independently supported basis capabilities SHALL remain
available. Supported artifact formats and versions SHALL follow their owning
codec contracts, without implied historical-format compatibility from Python
naming changes.

#### Scenario: Install the modern distribution

- **WHEN** a user installs the modern wheel or source distribution
- **THEN** it contains no HBMCMC subpackage or executable entry point
- **AND** the HBMCMC module path cannot resolve through a compatibility shim
- **AND** modern workflows and independently supported output formats remain usable

#### Scenario: Execute with canonical choices

- **WHEN** a caller supplies supported canonical modern choices
- **THEN** normal execution performs no HBMCMC or migration-tool translation
- **AND** old request spellings and every explicit `use_tracer` occurrence are rejected before scientific work
- **AND** independent scientific and artifact capabilities remain available
