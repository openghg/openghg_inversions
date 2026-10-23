# HBMCMC isolation and retirement

## Purpose

Keep modern inversion workflows independent of historical HBMCMC compatibility
and define the evidence gate for removing runtime compatibility on devel.
Permission to remove behavior is distinct from its implemented removal; the
active retirement change records outstanding delivery.
Requirements describing transitional HBMCMC calls apply while those entry
points remain; they SHALL NOT require restoring an entry point after its
evidenced removal.

## Requirements

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

While the transitional HBMCMC entry points remain, supported historical
fixedbasis policy and case/directory selection
SHALL be interpreted inside HBMCMC after winning overrides and before canonical
configuration construction. The actual HBMCMC CLI and normalized dictionary
adapter SHALL perform the applicable translation; calling an adapter from modern
runners SHALL NOT substitute for wiring these legacy entry points.

Translation SHALL preserve established override precedence, applicable warnings,
scientific choices, borrowed inputs and output policy, except for explicitly
specified removals. Canonical requests SHALL contain explicit selections rather
than historical discovery fields. Modern runner boundaries SHALL reject HBMCMC-only
selectors before scientific work and identify the explicit modern replacements
or the temporary legacy route. Removed options such as `country_directory`
SHALL remain errors on both routes, not regain support through HBMCMC.

Historical discovery used by HBMCMC SHALL be self-contained in that package,
without importing private discovery helpers from `basis.compatibility`. Small
duplication with independently deprecated basis adapters is permitted. Missing
domain information required for discovery SHALL produce a clear validation error.

The resource-I/O contract owns BC file loading, NESW's file-free calculation,
path origins and preparation defaults. HBMCMC translation SHALL supply the
explicit inputs needed by that contract rather than add downstream discovery.

#### Scenario: Legacy NESW request reaches canonical configuration

- **WHEN** the still-present transitional HBMCMC CLI receives `bc_basis_case="NESW"` with an optional
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

- **WHEN** a modern runner request supplies `fp_basis_case`, `basis_directory`,
  `bc_basis_case` or `bc_basis_directory`
- **THEN** it fails before acquisition with migration guidance
- **AND** no modern adapter calls HBMCMC to interpret those fields

### Requirement: Historical scientific quirks require version 0.7

Version 0.8 SHALL remove `preserve_legacy_likelihood`,
`_compatibility_unused_sigma_settings`, `_compatibility_minimum_error_floor` and
`compatibility_output_chain` from modern runner and builder operations. The
lower-level `retain_unused_sigma` and `derived_output_chain` options SHALL also be
removed. The HBMCMC CLI SHALL remove `--all-chains`, because all retained chains
are always used. Historical boundary-only pollution-event likelihood behaviour, disconnected sigma-prior
variables and additive fixed-error minimum-error floors SHALL NOT be reproduced
through modern scientific construction. The HBMCMC boundary SHALL reject affected
historical configurations before scientific execution with guidance to use
OpenGHG Inversions 0.7. The rejected historical inputs SHALL include
`no_model_error=True` for both pollution-event and additive likelihoods, and the
default pollution-event selection (omitted `likelihood`) with both
`add_offset=True` and `pollution_events_from_obs=True`. Guidance SHALL also identify
direct modern `run_standard` configuration for selecting the current model.
Other accepted wrapper requests SHALL use modern full-baseline and output-role
semantics; modern standard offsets SHALL remain supported.

Derived outputs SHALL use all retained chains, including outputs requested through
the transitional HBMCMC wrapper and legacy-format summaries. Reproduction of the
historical chain-0-only summary policy SHALL require OpenGHG Inversions 0.7. Modern
legacy-format codecs and explicit save paths SHALL remain supported.

#### Scenario: Historical likelihood configuration

- **WHEN** an HBMCMC request requires a removed likelihood or unused-prior quirk
- **THEN** the legacy boundary rejects it with guidance to use OpenGHG Inversions 0.7
- **AND** no modern runner reconstructs that compatibility graph

#### Scenario: Wrapper output uses retained chains

- **WHEN** the transitional HBMCMC wrapper writes derived products
- **THEN** it follows ordinary RHIME output policy over all retained chains
- **AND** it does not select only chain 0 through a private compatibility switch

### Requirement: Retained modern behavior has modern owners

Supported modern INI decoding, RHIME aliases, configuration construction, path
conversion and vendored preparation defaults SHALL remain available outside
HBMCMC. Moving their ownership SHALL preserve their separately specified
semantics, subject to the bounded development removals below. Temporary
runner/frontend spelling, output-format and `emissions_*` flux-selector
translation MAY be removed on devel after that gate is satisfied, without
waiting for finished converters or launchers. This supersedes their former
devel/0.8 warning and 0.9 removal schedule. Until a removal lands, supported
translation SHALL preserve its warnings, canonical-spelling precedence and
separate flux old/new conflict errors. Canonical
configuration SHALL contain no alias definitions, translation, alias warnings or
alias-specific errors; ordinary unknown-option validation handles old names.

HBMCMC-only adapters, including its deprecated dictionary-returning
`params_from_config`, SHALL be available only through HBMCMC while retained; modern
re-exports SHALL be removed with documented replacements. Modern callers SHALL
use independent INI decoding, apply overrides, and construct modern configuration.
This requirement does not remove modern legacy-format output codecs or
`save_inversion_output`; their support remains independent. Established public
Python and CLI aliases SHALL retain the model-selection contract pending
explicit decisions in #945. Independent basis readers, writers, discovery and
deprecated reconstruction adapters SHALL retain their own contracts pending
#951; a compatibility or legacy name alone SHALL NOT authorize deletion.

#### Scenario: Packaged preparation survives legacy isolation

- **WHEN** a modern run needs the exact domain's packaged preparation map
- **THEN** the modern resource owner selects it without importing HBMCMC
- **AND** explicit overrides and missing-default errors retain the resource-I/O contract

#### Scenario: Supported modern alias survives relocation

- **WHEN** a runner caller supplies a separately supported deprecated RHIME alias
- **THEN** the runner frontend preserves its warning, precedence and canonical value
- **AND** no historical fixedbasis policy or resource discovery is invoked

### Requirement: Development removals require affected migration evidence

Under the accepted #944 treatment and #961 boundary, removal of the complete
HBMCMC executor namespace, its wrapper and legacy data/basis utilities, and
temporary runtime parameter/output-format/flux-selector translation MAY proceed
on devel in bounded #719 PRs. This supersedes the promise to retain the HBMCMC
route through 0.8 and remove it in 0.9. Historical execution SHALL belong to
the maintained 0.7 line; normal modern execution SHALL NOT invoke migration
tooling. The boundary SHALL NOT itself change equations, defaults, independent
artifact support or unresolved public interfaces.

Before each migration-related removal or deliberate behavior change merges,
its PR SHALL:

- Identify affected #960 ledger entries and plain cases, adding bounded evidence
  for newly discovered behavior with explicit source version and route.
- Compare its base and affected source with pinned development revision
  `0a32d73a5c4a72be0505a4ed682472e1613e49e9`, accounting for intervening changes
  and relevant merged PRs.
- Record a disposition and follow-up owner: retained, mechanically migrated,
  deliberate scientific change, unsupported or unresolved. Historical inputs
  and expectations SHALL remain available; modern comparisons SHALL be dated
  additions rather than replacements for source observations.
- Preserve or port affected scientific coverage before retiring adapter/test
  plumbing. Unchanged cases SHALL have canonical regression coverage; deliberate
  scientific changes SHALL test their separately agreed new semantics. Evidence
  SHALL distinguish inspection from execution and identify the validated revision.

Completion of #960 SHALL NOT constitute advance approval of future deletions.
Only affected evidence and live checks need refreshing; a whole-audit rerun
SHALL NOT be required. Finished conversion and launcher interfaces (#949/#950),
maintenance publication (#947), complete guidance (#432), final API/version
decisions (#945) and integrated acceptance (#952) remain M2 release gates,
not prerequisites for these bounded development removals. Unresolved interface
decisions SHALL remain with #945 and independent basis decisions with #951.

Scientific requirements, borrowed-array and explicit execution boundaries,
modern `output_format="legacy"`, saved `InversionOutput` codecs, exact artifact
destinations and independently supported labelled basis partitions SHALL remain
supported. A separately agreed scientific change SHALL meet the same evidence
gate without restoring a superseded historical expectation.

#### Scenario: Remove an evidenced runtime adapter during development

- **WHEN** a #719 PR meets the affected-entry, source-diff, disposition and scientific-test gate
- **THEN** its agreed runtime compatibility can be removed before the converter or launcher is finished
- **AND** plain historical evidence remains usable without the deleted package
- **AND** this does not establish release readiness or remove independent codecs

#### Scenario: Discover an uncaptured behavior or an intentional scientific change

- **WHEN** a migration-related PR discovers behavior absent from #960 or changes an agreed scientific expectation
- **THEN** it adds bounded source evidence and an owned disposition before merging
- **AND** it retains historical expectations separately from dated tests of the agreed modern semantics
- **AND** unresolved interfaces are not approved for deletion by this gate
