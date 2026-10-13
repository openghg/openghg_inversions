# HBMCMC isolation and retirement

## Purpose

Keep modern inversion workflows independent of historical HBMCMC compatibility,
preserve a bounded legacy migration route in 0.8, and remove that route in 0.9.

## ADDED Requirements

### Requirement: Production dependencies point from legacy to modern

Production code outside `openghg_inversions.hbmcmc` SHALL NOT import or re-export
anything from that subpackage, or route modern execution back through it. This
includes lazy and dynamic imports and indirect forwarding through another
module. HBMCMC entry points MAY depend on modern APIs. Small duplication inside
HBMCMC MAY preserve this boundary; sharing code SHALL NOT justify a reverse
dependency. Tests MAY import HBMCMC to exercise its remaining legacy entry points.

#### Scenario: Modern execution without HBMCMC

- **WHEN** a modern full, staged, nested, prepared-input or output-only workflow
  is imported or executed with HBMCMC unavailable
- **THEN** its supported operation does not require any HBMCMC module
- **AND** modern public exports do not expose HBMCMC-owned adapters

#### Scenario: Enforce the production boundary

- **WHEN** a production module outside HBMCMC acquires a forbidden import or re-export
- **THEN** the repository's import-boundary check fails
- **AND** tests of HBMCMC itself remain permitted

### Requirement: Historical requests are translated only at legacy entry points

During 0.8, supported historical fixedbasis policy and case/directory selection
SHALL be interpreted inside HBMCMC after winning overrides and before canonical
configuration construction. The actual HBMCMC CLI and normalized dictionary
adapter SHALL perform the applicable translation; calling an adapter from modern
runners SHALL NOT substitute for wiring these legacy entry points.

Translation SHALL preserve established override precedence, applicable warnings,
scientific choices, borrowed inputs and output policy, except for explicitly
specified removals. Canonical requests SHALL contain explicit selections rather
than historical discovery fields. Modern boundaries SHALL reject HBMCMC-only
selectors before scientific work and identify the explicit modern replacements
or the temporary legacy route. Removed options such as `country_directory`
SHALL remain errors on both routes, not regain support through HBMCMC.

The resource-I/O contract owns BC file loading, NESW's file-free calculation,
path origins and preparation defaults. HBMCMC translation SHALL supply the
explicit inputs needed by that contract rather than add downstream discovery.

#### Scenario: Legacy NESW request reaches canonical configuration

- **WHEN** the 0.8 HBMCMC CLI receives `bc_basis_case="NESW"` with an optional
  historical directory after overrides
- **THEN** translation removes the historical selectors before construction
- **AND** a supplied directory receives the unused-option warning without lookup
- **AND** the run reaches modern execution with built-in NESW selected

#### Scenario: Winning legacy selection is translated

- **WHEN** an HBMCMC override replaces a historical basis selection from a file
- **THEN** only the winning selection is discovered and translated
- **AND** a missing superseded selection does not cause failure
- **AND** the canonical constructor performs no globbing or artifact inspection

#### Scenario: Modern caller supplies historical discovery fields

- **WHEN** a modern request supplies `fp_basis_case`, `basis_directory`,
  `bc_basis_case` or `bc_basis_directory`
- **THEN** it fails before acquisition with migration guidance
- **AND** no modern adapter calls HBMCMC to interpret those fields

### Requirement: Retained modern behavior has modern owners

Supported modern INI decoding, RHIME aliases, configuration construction, path
conversion and vendored preparation defaults SHALL remain available outside
HBMCMC. Moving their ownership SHALL preserve their separately specified
semantics and deprecation schedules. In particular, modern alias support SHALL
NOT disappear merely because its implementation previously lived in HBMCMC.

HBMCMC-only adapters, including its deprecated dictionary-returning
`params_from_config`, SHALL be available only through HBMCMC during 0.8; modern
re-exports SHALL be removed with documented replacements. Modern callers SHALL
use independent INI decoding, apply overrides, and construct modern configuration.
This requirement does not remove modern legacy-format output codecs or
`save_inversion_output`; their ownership and schedules remain independent.

#### Scenario: Packaged preparation survives legacy isolation

- **WHEN** a modern run needs the exact domain's packaged preparation map
- **THEN** the modern resource owner selects it without importing HBMCMC
- **AND** explicit overrides and missing-default errors retain the resource-I/O contract

#### Scenario: Supported modern alias survives relocation

- **WHEN** a modern caller supplies a separately supported deprecated RHIME alias
- **THEN** a modern owner preserves its warning, precedence and canonical value
- **AND** no historical fixedbasis policy or resource discovery is invoked

### Requirement: HBMCMC is completely removed in 0.9

Version 0.9 SHALL remove the entire `openghg_inversions.hbmcmc` subpackage,
including the executable wrapper, compatibility adapters and legacy data/basis
utilities. No import aliases, forwarding shims or entry points SHALL preserve
that namespace. Supported modern workflows SHALL continue independently.

Migration guidance SHALL distinguish 0.8 isolation from complete 0.9 removal,
list modern replacements for retained workflows, and direct users needing
legacy-only utilities or unsupported historical artifacts to a compatible older
release. It SHALL distinguish removals from unrelated 0.9 deprecations without
inventing a removal date for those other features.

#### Scenario: Install the 0.9 distribution

- **WHEN** a user installs the 0.9 wheel or source distribution
- **THEN** it contains no HBMCMC subpackage or executable entry point
- **AND** the HBMCMC module path cannot be imported through a compatibility shim
- **AND** modern workflows and separately supported output formats remain usable

#### Scenario: Follow the 0.8 migration guidance

- **WHEN** a user reviews the legacy migration documentation before upgrading
- **THEN** complete HBMCMC removal is explicitly listed for 0.9
- **AND** RHIME execution and prepared-input/basis artifacts are identified where applicable
- **AND** unsupported legacy-only operations are identified without promising new codecs
