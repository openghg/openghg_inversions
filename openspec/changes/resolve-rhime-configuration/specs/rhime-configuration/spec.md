# RHIME configuration

## Purpose

Resolve standard and multisector RHIME configuration into inspectable typed
values before acquisition, preserving supported external shorthand and correct
alignment when scientific preparation retains only some requested sites.

## ADDED Requirements

### Requirement: Format-neutral resolution before acquisition

Standard and multisector configured runs SHALL resolve raw configuration into
named preparation, model, sampling and output values before acquisition or
scientific preparation. INI decoding SHALL remain separate from semantic
resolution. Equivalent file-derived and Python mappings, after supported
overrides, SHALL produce equivalent canonical values. Existing aliases,
precedence, option meanings and rejection rules SHALL remain supported.

#### Scenario: Equivalent file and Python inputs

- **WHEN** an INI file and a Python mapping express the same standard or
  multisector options, with equivalent winning overrides
- **THEN** resolution yields equivalent configuration values and defaults
- **AND** no acquisition, cache loading or scientific preparation is performed

#### Scenario: Invalid options fail early

- **WHEN** a configured run supplies unknown, unsupported or malformed options,
  including site-option lengths that do not match the request
- **THEN** it fails at resolution before retrieval, reload or scientific execution
- **AND** the error identifies the offending option

### Requirement: Complete requested-site options

Resolution SHALL normalize requested site labels to uppercase, reject empty or
duplicate requests, and establish one complete ordered site-options record.
Scalar selectors SHALL expand to the requested site count; explicit sequences
SHALL match that count. Optional selectors SHALL retain their existing meaning,
including unspecified values. Internal acquisition/preparation SHALL consume
these canonical selectors without another scalar-expansion pass.

#### Scenario: Scalar and expanded averaging periods

- **WHEN** two requested sites use `averaging_period="1h"` or
  `averaging_period=["1h", "1h"]`
- **THEN** both resolve to the same two site-aligned period values
- **AND** `averaging_period=["1h"]` fails before acquisition

#### Scenario: Labels and optional selectors

- **WHEN** sites are `["tac", "MHD"]`, `time_resolved=None`, inlet selectors
  include a supported slice, and maximum levels include integers or `None`
- **THEN** labels resolve to `("TAC", "MHD")`, unspecified time resolution
  remains unspecified for both sites, and supported selector values are preserved
- **AND** boolean maximum levels and case-insensitive duplicate sites are rejected

### Requirement: Resolution owns configuration values without scientific work

Resolution SHALL leave caller mappings and nested configuration containers
unchanged on success or failure. The concrete configuration SHALL contain
configuration values rather than a runtime sampler or acquired/prepared numerical
handoff. It SHALL NOT acquire, copy, compute, persist, densify or rechunk scientific
data. Configuration-only defaults SHALL be resolved at this boundary; conversions
requiring retained labels or numerical data SHALL remain with their owning phase.

#### Scenario: Caller inputs remain unchanged

- **WHEN** resolution normalizes aliases, prior mappings, sampling keywords and
  site-selector lists, or rejects an invalid option
- **THEN** the original mapping and its nested containers retain their values
- **AND** subsequent resolution of those inputs is unaffected

#### Scenario: Inspect configuration before data access

- **WHEN** a caller resolves configuration with acquisition and scientific
  materialization unavailable
- **THEN** the named choices can be inspected without creating a runtime sampler,
  loading data or materializing numerical arrays

### Requirement: Retained-site selection preserves complete alignment

Acquisition and preparation SHALL retain responsibility for data-dependent site
selection. Dropping or reordering sites SHALL select every applicable resolved
site option together by label, without reparsing the external request or changing
its configuration. Existing empty-set, malformed-handoff and cache-compatibility
rules SHALL remain at their owning boundaries. A supplied valid merged handoff
SHALL retain its authoritative options and existing no-acquisition behavior.

#### Scenario: Acquisition returns reordered retained sites

- **WHEN** TAC, MHD and BSD have unequal per-site options and acquisition returns
  BSD followed by TAC
- **THEN** all applicable selectors follow BSD and TAC in that order
- **AND** the original requested configuration remains unchanged

#### Scenario: Reload or filtering drops a middle site

- **WHEN** a compatible reload or filtering retains TAC and BSD from the
  established TAC, MHD, BSD order
- **THEN** every applicable selector retains the TAC, BSD pairing and order
- **AND** selection preserves the existing ordering policy without reparsing
  or changing the requested configuration

#### Scenario: Supplied merged data

- **WHEN** a configured run receives a valid compatible merged handoff with its
  own retained-site options
- **THEN** it preserves those options, performs no acquisition and continues
  preparation without replacing them with requested defaults

### Requirement: Compatible public adapters

Supported Python and CLI entry points SHALL retain their signatures, shorthand,
override behavior and return contracts through adapters to the same semantic
resolver or its shared site-option translation. Existing direct preparation and
retrieval APIs SHALL continue to accept scalar and site-aligned options;
independent builders and prepared-input runners SHALL NOT require the complete
configured-run record. INI section interpretation, existing path conventions and
the supported file vocabulary SHALL remain unchanged.

#### Scenario: Existing Python shorthand

- **WHEN** standard, multisector or direct public preparation/retrieval callers
  supply supported scalar options instead of expanded sequences
- **THEN** the adapter establishes the same applicable canonical choices before
  its scientific work, preserving the established return contract
- **AND** internal canonical calls do not repeat that translation
