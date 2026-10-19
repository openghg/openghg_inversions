# Pending HBMCMC retirement

## ADDED Requirements

### Requirement: HBMCMC is completely removed in 0.9

Version 0.9 SHALL remove the entire `openghg_inversions.hbmcmc` subpackage,
including the executable wrapper, compatibility adapters and legacy data/basis
utilities within that subpackage. Separately supported `basis.compatibility`
adapters retain their own schedules and SHALL NOT depend on HBMCMC. No import aliases, forwarding shims or entry points SHALL preserve
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
