# RHIME configuration

## Purpose

Represent the complete resolved standard/multisector RHIME request before data
access, with equivalent external shorthand and a clear separation between
requested configuration and retained execution metadata.

## ADDED Requirements

### Requirement: Complete format-neutral requested configuration

Standard and multisector configured runs SHALL construct one inspectable,
complete resolved requested-run configuration before acquisition or scientific
preparation. It SHALL expose named preparation, model, output and sampling
choices with established configuration-only defaults and aliases resolved. It
SHALL contain no raw site shorthand, acquired/prepared scientific handoff or
retained-run description. INI decoding SHALL remain separate from semantic
resolution. Equivalent file-derived and Python mappings with equivalent winning
overrides SHALL yield equivalent resolved choices. INI SHALL remain the current
file frontend without requiring another format.

#### Scenario: Equivalent file and Python inputs

- **WHEN** an INI file and a Python mapping express the same supported standard
  or multisector request with equivalent winning overrides
- **THEN** they produce equivalent complete configuration choices and defaults
- **AND** the result can be inspected before data access or scientific execution

#### Scenario: Configuration defaults and aliases are explicit

- **WHEN** a request uses supported aliases or omits options with established
  configuration-only defaults
- **THEN** the returned configuration exposes their resolved canonical values
- **AND** downstream canonical consumers need no alias or default-resolution pass

### Requirement: Apply overrides before resolving shorthand

Supported overrides SHALL be applied before semantic resolution and site-option
expansion. Resolution SHALL use the effective requested site list and winning
values, preserve established precedence and rejection rules, and reject malformed
effective options before acquisition, reload or scientific preparation. A returned
configuration SHALL already contain the complete equivalent of external
shorthand; no scientific phase SHALL be responsible for completing the request.

#### Scenario: Override changes the requested site count

- **WHEN** file options request TAC and MHD with scalar averaging period `"1h"`
  and a supported override changes the sites to TAC, MHD and BSD
- **THEN** the resolved request contains all three sites and three `"1h"` periods
- **AND** expansion happens before acquisition

#### Scenario: Override replaces a site sequence

- **WHEN** a supported override replaces an averaging-period sequence with a
  scalar or a correctly sized sequence
- **THEN** resolution uses the winning value against the effective site list
- **AND** the replaced sequence does not undergo site-length validation

#### Scenario: Invalid effective options fail early

- **WHEN** effective options include an unknown, unsupported or malformed choice,
  including a sequence whose length does not match the effective site count
- **THEN** resolution fails before retrieval, reload or scientific execution
- **AND** the error identifies the offending option

### Requirement: Complete requested-site options

Resolution SHALL normalize requested site labels to uppercase, reject empty or
case-insensitive duplicate requests, and establish one complete ordered
site-options record. Scalar selectors SHALL expand to the requested site count;
explicit sequences SHALL match that count. The same convention SHALL apply to
file-derived and supported direct Python inputs. Optional selectors SHALL retain
their existing meaning. Canonical acquisition/preparation SHALL consume the
resolved entries without another scalar-expansion pass.

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

### Requirement: Configuration ownership without scientific execution

Resolution SHALL leave caller mappings and nested configuration containers
unchanged on success or failure. Configured execution SHALL preserve the resolved
request when preparation selects retained inputs. Resolving configuration SHALL
NOT acquire, copy, compute, persist, densify or rechunk scientific data, bind a
model, generate posterior/predictive draws or write output products. Constructing
and holding the existing sampler settings object SHALL be permitted; inference
SHALL execute only after a model is supplied at the inference boundary.
Conversions requiring retained labels or numerical data SHALL remain with their
owning scientific phase.

#### Scenario: Caller inputs remain unchanged

- **WHEN** resolution normalizes aliases, priors, sampling keywords and
  site-selector lists, or rejects an invalid option
- **THEN** the original mapping and nested containers retain their values
- **AND** subsequent resolution of those inputs is unaffected

#### Scenario: Inspect configuration without executing sampling

- **WHEN** a caller resolves configuration while acquisition, sampling and
  scientific materialization are unavailable
- **THEN** the resolved preparation, model, output and sampler settings can be
  inspected without data access, a bound model or numerical execution
- **AND** creating sampler settings does not invoke posterior or predictive draws

### Requirement: Retained execution metadata is distinct from the request

Ordinary configured runs SHALL derive their execution run description after
preparation, using retained sites and averaging periods from the prepared
handoff, requested date bounds, the selected prepared layout, and the resolved
model/output choices. Requested configuration SHALL remain unchanged and SHALL
NOT contain a pre-preparation execution run description. Model and output
specifications SHALL be composed into the retained run description without
reinterpreting raw options. Established public prepared-input and compatibility
setup contracts SHALL remain supported through adapters.

#### Scenario: One requested site is removed

- **WHEN** configuration requests TAC and MHD, and preparation retains only TAC
- **THEN** the configuration continues to describe TAC and MHD
- **AND** the ordinary run description contains only TAC and its retained period,
  with the requested dates and resolved model/output choices

#### Scenario: Run an independently prepared artifact

- **WHEN** a caller supplies valid prepared inputs and the existing public run,
  model, output and sampler arguments
- **THEN** execution retains the established prepared-input API and alignment
  behavior without requiring the complete requested-run configuration

### Requirement: Retained-site selection preserves complete alignment

Acquisition and preparation SHALL retain responsibility for data-dependent site
selection. Dropping or reordering sites SHALL select every applicable resolved
site option together by label, without reparsing external shorthand or changing
the request. Existing empty-set, retained-label and cache-compatibility rules
SHALL remain at their owning boundaries. A supplied valid compatible merged
handoff SHALL retain its authoritative options and no-acquisition behavior.

#### Scenario: Acquisition returns reordered retained sites

- **WHEN** TAC, MHD and BSD have unequal per-site options and acquisition returns
  BSD followed by TAC
- **THEN** all applicable selectors follow BSD and TAC in that order
- **AND** the original requested configuration remains unchanged

#### Scenario: Reload or filtering drops a middle site

- **WHEN** a compatible reload or filtering retains TAC and BSD from the
  established TAC, MHD, BSD order
- **THEN** every applicable selector retains the TAC, BSD pairing and order
- **AND** selection preserves the existing ordering policy without resolving the
  original shorthand against the reduced site count

#### Scenario: Redundant legacy retrieval metadata

- **WHEN** the legacy retrieval adapter returns valid retained labels and unused
  metadata lists that disagree with the resolved request
- **THEN** requested site options are selected using those labels
- **AND** unused legacy metadata does not replace them or add validation failures

#### Scenario: Supplied merged data

- **WHEN** a configured run receives a valid compatible merged handoff with its
  own retained-site options
- **THEN** it preserves those options, performs no acquisition and continues
  preparation without replacing them with requested defaults

### Requirement: Compatible public adapters and scientific choices

Supported Python and CLI entry points SHALL retain their signatures, shorthand,
override behavior and return contracts through adapters to the same semantic
resolution or shared applicable site translation. Direct preparation/retrieval
APIs SHALL resolve only applicable choices without requiring likelihood, final
output or sampling settings. Independent builders and prepared-input runners
SHALL remain independent of full configuration. Shared choices such as BC use
and flux-source routing SHALL retain consistent existing meanings across data
preparation and model construction. Current INI section interpretation, path
conventions, file vocabulary, scientific calculations and sampling defaults
SHALL remain unchanged.

Resolution SHALL consume omitted/false `use_tracer` without retaining a config
field. Effective true SHALL retain early rejection at resolution and relevant
direct public boundaries, including supplied merged data. Existing legacy-option
and custom-likelihood conflict precedence SHALL remain unchanged.

#### Scenario: Existing Python shorthand

- **WHEN** standard, multisector or direct public preparation/retrieval callers
  supply supported scalar options instead of expanded sequences
- **THEN** the adapter establishes the same applicable canonical choices before
  scientific work and preserves the established return contract
- **AND** internal canonical calls do not repeat that translation

#### Scenario: Boundary conditions are disabled consistently

- **WHEN** effective configuration sets `use_bc=False`
- **THEN** acquisition/preparation and built-in model construction use the same
  disabled BC choice with existing scientific behavior
- **AND** neither phase infers an independent conflicting default

#### Scenario: Preserve unsupported tracer rejection

- **WHEN** a configured run, direct public preparation call or mapping-based
  retrieval supplies effective `use_tracer=True` after supported overrides,
  including with supplied merged data
- **THEN** it rejects the option before acquisition, reload or scientific execution
- **AND** omitted and false options remain accepted with equivalent resolved
  configuration containing no tracer field
