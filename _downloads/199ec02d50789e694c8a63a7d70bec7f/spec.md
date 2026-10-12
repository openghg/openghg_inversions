# RHIME configuration

## Purpose

Represent the complete resolved standard/multisector RHIME request before data
access, with equivalent external shorthand and a clear separation between
requested configuration and retained execution metadata.

Serialization and INI writing are deferred to
[#814](https://github.com/openghg/openghg_inversions/issues/814); they are not
acceptance requirements for this implementation.

## ADDED Requirements

### Requirement: Complete format-neutral requested configuration

Standard and multisector configured runs SHALL construct one inspectable,
complete resolved requested-run configuration before acquisition or scientific
preparation. It SHALL expose named preparation, model, output and sampling
choices with established configuration-only defaults and aliases resolved. It
SHALL contain no raw site shorthand, acquired/prepared scientific handoff or
retained-run description. File-specific parsing and interpretation SHALL belong
to the corresponding frontend. Runners SHALL combine decoded file options with
winning overrides, consume options handled by their own recipe, and pass the
remaining configuration options to `RhimeConfig.from_params` before scientific work.
Equivalent file-derived and Python requests with equivalent winning overrides
SHALL yield equivalent resolved choices. INI SHALL remain the current file
frontend without requiring another format or imposing its document structure
on other frontends.

The public requested configuration SHALL expose acquisition and preparation
choices directly, without requiring a second preparation-configuration object.
Existing cohesive site, model, output and sampler values SHALL retain their
distinct roles; the request SHALL NOT introduce duplicate settings objects for
those choices.

#### Scenario: Inspect one complete requested configuration

- **WHEN** a caller resolves a complete request and inspects requested store,
  basis, filter and date options
- **THEN** those choices are available directly on the returned configuration
- **AND** accessing them does not require a nested preparation configuration,
  data acquisition or an execution run description

#### Scenario: Construct and execute one resolved request

- **WHEN** a caller uses `RhimeConfig.from_params` after decoding and editing any file options
- **THEN** a standard or multisector runner can consume the returned `config`
  without resolving its options again
- **AND** supplying that configuration together with a file or raw overrides
  is rejected before acquisition
- **AND** `RhimeConfig.from_params` is the single construction entry point;
  `resolve_rhime_config` is removed

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

### Requirement: Explicit shallow selection for scientific keyword calls

`RhimeConfig.select(*names)` SHALL return a new dictionary containing the
explicitly named attributes. Selected values SHALL remain borrowed, including
mutable containers and opaque objects. Selection SHALL NOT copy nested values,
resolve defaults, inspect consumer signatures, execute scientific work or define
a serialization schema. An unknown attribute SHALL raise `AttributeError`.
Scientific stage functions SHALL retain named inputs and remain callable without
configuration. A private orchestration helper MAY receive resolved configuration
when composing several stages.

#### Scenario: Forward selected resolved values

- **WHEN** a caller unpacks `config.select("filters")` into the observation filter
- **THEN** the function receives the same resolved value as `filters=config.filters`
- **AND** changing the selection dictionary does not change the configuration
- **AND** mutable values inside the dictionary retain their existing identity

#### Scenario: Selection does not resolve overrides

- **WHEN** a caller needs to change raw options that affect shorthand or dependent defaults
- **THEN** it applies those overrides before semantic resolution
- **AND** selection and direct dataclass replacement do not recompute those defaults

### Requirement: Apply overrides before resolving shorthand

Supported overrides SHALL be applied before semantic resolution and site-option
expansion. Resolution SHALL use the effective requested site list and winning
values, preserve established precedence and rejection rules, and reject malformed
effective options before acquisition or scientific preparation. A returned
configuration SHALL already contain the complete equivalent of external
shorthand; no scientific phase SHALL be responsible for completing the request.
Date-dependent configuration defaults SHALL use effective overridden dates.

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
- **THEN** resolution fails before retrieval or scientific execution
- **AND** the error identifies the offending option

#### Scenario: Override changes a date-dependent default

- **WHEN** a supported override changes the start date and the selected
  likelihood derives a configuration default from that date
- **THEN** resolution derives the default from the winning start date
- **AND** the returned configuration needs no downstream default-resolution pass

### Requirement: Complete requested-site options

Resolution SHALL normalize requested site labels to uppercase, reject empty or
case-insensitive duplicate requests, and establish one complete ordered
site-options record. Scalar selectors SHALL expand to the requested site count;
explicit sequences SHALL match that count. The same convention SHALL apply to
file-derived and supported direct Python inputs. Optional selectors SHALL retain
their existing meaning. Canonical acquisition/preparation SHALL consume the
resolved entries without another scalar-expansion pass.

The existing aligned selector record SHALL be public `SiteOptions`, exported
from `inversion_data` and used by requested configuration and merged-data
handoffs. Its `from_inputs` factory and applicable alignment helpers SHALL
provide reusable shorthand normalization independent of file syntax. Direct
construction SHALL accept complete resolved values and enforce structural
alignment without another shorthand-expansion pass. Selection SHALL return a
new complete record without modifying the requested record.

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

#### Scenario: Public site-options construction

- **WHEN** a caller imports `SiteOptions` from `inversion_data` and constructs
  it through `from_inputs` with supported external selectors
- **THEN** the result has the same complete aligned values used by requested
  configuration and acquisition handoffs
- **AND** construction does not require INI sections or a complete model,
  output or sampler configuration

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
reinterpreting raw options. Independent prepared-input scientific contracts SHALL
remain supported. In-repository configured orchestration SHALL use the same
requested configuration rather than a second setup bundle holding a
pre-preparation run description and a preparation dictionary.

#### Scenario: Unified configured orchestration

- **WHEN** ordinary, nested, staged, shim or example consumers resolve supported
  standard/multisector requested options
- **THEN** they use the same complete requested configuration
- **AND** no distinct setup bundle or pre-preparation execution description is
  required to access configuration choices

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
the request. Existing empty-set, retained-label and acquisition-compatibility rules
SHALL remain at their owning boundaries. A supplied valid compatible merged
handoff SHALL retain its authoritative options and no-acquisition behavior.

#### Scenario: Acquisition returns reordered retained sites

- **WHEN** TAC, MHD and BSD have unequal per-site options and acquisition returns
  BSD followed by TAC
- **THEN** all applicable selectors follow BSD and TAC in that order
- **AND** the original requested configuration remains unchanged

#### Scenario: Supplied data or filtering drops a middle site

- **WHEN** compatible supplied data or filtering retains TAC and BSD from the
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

Except for the former stage adapter, `resolve_rhime_config` and transitional
old-location import removals, and the revised INI reader contract specified below, supported
scientific Python and CLI entry points SHALL retain their signatures, shorthand,
override behavior and return contracts through adapters to the same semantic
resolution or shared applicable site translation. Direct preparation/retrieval
APIs SHALL resolve only applicable choices without requiring likelihood, final
output or sampling settings. Independent builders and prepared-input runners
SHALL remain independent of full configuration. Shared choices such as BC use
and flux-source routing SHALL retain consistent existing meanings across data
preparation and model construction. Current INI section interpretation, path
conventions, file vocabulary, scientific calculations and sampling defaults
SHALL remain unchanged.

Internal orchestration/setup records and their constructor/helper return shapes
SHALL NOT be required compatibility contracts. Their in-repository consumers
SHALL migrate to the unified requested configuration.

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
- **THEN** it rejects the option before acquisition or scientific execution
- **AND** omitted and false options remain accepted with equivalent resolved
  configuration containing no tracer field

### Requirement: Scientific preparation owns its operations

Recipes SHALL derive observation errors before temporal filtering/aggregation,
call `make_basis_functions` directly with resolved arguments, and use the actual
sensitivity, filtering and assembly implementations. Forwarding wrappers SHALL
NOT be retained solely for uniform stage names, timing or keyword translation.
Shared indexing utilities SHALL remain independently reusable. Assembly SHALL
preserve minimum-error `None`, integer and per-site mapping behavior.

`prepare_rhime_inputs` and its exclusive acquisition dispatcher are removed.
Modern acquisition SHALL NOT load or save merged-data caches. Historical
six-tuple retrieval SHALL explicitly call the shared observation-error operation
before returning or saving its established result.

#### Scenario: Error derivation precedes temporal reduction

- **WHEN** component errors vary across observations to be aggregated
- **THEN** preparation combines the components before temporal reduction
- **AND** custom `mf_error`, zero-error fallback behavior, labels and metadata
  are preserved without mutating borrowed datasets

#### Scenario: Nested observation-error preparation

- **WHEN** a nested recipe receives acquired or supplied inner and outer data
- **THEN** both domains have observation errors prepared before filtering and
  time alignment; only sites with both domains can enter nested preparation
- **AND** `averaging_error` is preparation policy, not a binding acquisition fact

#### Scenario: Direct scientific stages use named inputs

- **WHEN** a caller supplies acquired data and named required scientific choices
- **THEN** each stage can execute without a `RhimeConfig` or raw parameter mapping
- **AND** omitting a required keyword is rejected at the function-call boundary

### Requirement: Neutral acquisition naming with a compatible deprecated entry point

Fresh acquisition SHALL expose a name covering both surface and column
observations. The established acquisition entry point SHALL remain a deprecated
forwarding wrapper with its existing signature, shorthand, six-tuple return and
error behavior. Internal canonical calls SHALL share the acquisition body and
SHALL NOT route through the deprecated wrapper.

#### Scenario: Call the deprecated acquisition name

- **WHEN** a caller invokes the established acquisition name with valid surface
  or column inputs
- **THEN** a deprecation warning identifies the replacement
- **AND** the call returns the same six-tuple and scientific data as the neutral
  entry point for equivalent inputs

### Requirement: INI decoding is independent of semantic construction

`read_rhime_ini(path)` SHALL return a dictionary of decoded file options. It
SHALL own INI syntax, section flattening and value decoding, and preserve the
existing first-occurrence rule for repeated bare keys. It SHALL NOT require
complete run settings, select a model type, accept overrides, translate legacy
names or resolve defaults/site shorthand. No new file format or section schema
is introduced by this change.

Standard, multisector, nested and custom runners SHALL combine decoded options
with winning overrides, extract their own recipe choices, and construct canonical
configuration once. Shorthand SHALL remain editable until that construction.
Shared semantic/site-alignment helpers SHALL remain independent of file syntax.

Historical fixedbasis translation and deprecated alias definitions SHALL be
owned by a lightweight HBMCMC compatibility module. `RhimeConfig.from_params`
SHALL invoke alias translation before modern coercion, validation and construction.
Runners SHALL call that classmethod without repeating alias translation.
The translator SHALL emit `DeprecationWarning` when it replaces or removes a
deprecated option name or output-format value; canonical options SHALL pass
without a deprecation warning. Canonical spellings SHALL win when both spellings
are supplied. Fixedbasis-specific scientific policy SHALL remain in its separate
compatibility adapter. The `resolve_rhime_config` wrapper SHALL be removed.
The deprecated `params_from_config` adapter SHALL preserve its dictionary,
normalization and override behavior using the same decoder. It SHALL NOT be
needed by modern scientific runners.

#### Scenario: Read recipe-specific or incomplete options

- **WHEN** a caller reads an INI file with incomplete ordinary settings or
  project-specific choices
- **THEN** the reader returns decoded values without trying to construct a run
- **AND** the recipe can remove its own options and apply overrides before
  validating the remaining canonical request

#### Scenario: Override shorthand after reading

- **WHEN** a caller reads site shorthand and changes the requested sites or periods
- **THEN** semantic construction expands only the winning values
- **AND** the reader performs no premature expansion or date-dependent resolution

#### Scenario: Construction translates deprecated spellings once

- **WHEN** a runner or direct factory caller supplies a supported deprecated spelling
- **THEN** `RhimeConfig.from_params` invokes the compatibility translator and resolves
  the canonical value, emitting a deprecation warning for the changed option
- **AND** canonical inputs do not emit deprecation warnings
- **AND** HBMCMC-specific historical behavior remains in the HBMCMC translator

#### Scenario: Dictionary compatibility adapter

- **WHEN** a caller uses `params_from_config` with supported overrides and normalization controls
- **THEN** the deprecated adapter retains its established dictionary contract
- **AND** modern runners can instead read, combine and resolve without using it

### Requirement: Configuration is not a historical identity schema

Requested configuration SHALL represent resolved choices without reconstructing
original scalar/list spellings or sparse defaults solely to preserve historical
staged hashes. Public scientific behavior and return contracts SHALL remain
supported; equality of historical encoding hashes SHALL NOT be an acceptance
requirement. Changes affecting a currently supported persisted contract SHALL
be identified explicitly rather than silently assigned its old contract.

#### Scenario: Equivalent shorthand and historical encodings

- **WHEN** scalar and expanded site inputs resolve to equivalent choices
- **THEN** canonical consumers use those equivalent resolved values
- **AND** configuration adapters are not required to reconstruct different raw
  forms to preserve their historical staged identities

### Requirement: In-memory supplied and freshly acquired data paths

Configured recipes SHALL reuse compatible supplied merged data without
object-store I/O or call `RhimeMergedData.from_options` with resolved selectors.
They SHALL preserve authoritative retained selections and validate known
acquisition facts. Modern merged-data cache options SHALL be rejected.
Acquisition replay remains deferred to #829; durable reuse uses prepared inputs.
Selected merged-data provenance remains on the in-memory handoff and SHALL NOT
be added to prepared inputs or the preparation manifest by this change.

#### Scenario: Supplied data bypasses acquisition

- **WHEN** valid compatible merged data is supplied to the recipe
- **THEN** its data and authoritative site options are reused without acquisition
- **AND** the same observation-error preparation policy applies

#### Scenario: Removed modern cache options

- **WHEN** a modern caller requests a removed acquisition-cache option
- **THEN** configuration rejects it and directs file reuse to prepared inputs
- **AND** historical legacy codecs retain their separately documented behavior

### Requirement: Modern flux selectors have their own deprecation boundary

Modern configuration and acquisition SHALL use `flux_store`, `flux_domain`,
`inner_flux_store` and `inner_flux_domain`. Their `emissions_*` predecessors
SHALL remain accepted with `DeprecationWarning` until removal in 0.9. Supplying
both spellings SHALL raise even if the values agree or are explicitly `None`.
Canonical configuration exports SHALL contain only the new names. This modern
compatibility boundary SHALL remain separate from historical HBMCMC translation.

`flux_domain` SHALL select the source flux dataset domain, defaulting to the
corresponding footprint domain; it does not inherently mean the nested inner
domain. Existing `make_basis_functions(emissions_name=...)` calls SHALL receive
the same deprecation treatment for canonical `flux_sources`. New dataset helper
interfaces SHALL use canonical names directly without forwarding wrappers.

#### Scenario: Deprecated flux selector supplied alone

- **WHEN** a modern caller supplies an old selector without its new spelling
- **THEN** it warns, selects the same dataset, and exports the canonical name

#### Scenario: Ambiguous modern flux selector

- **WHEN** a caller supplies both old and new selector spellings
- **THEN** it raises before acquisition or basis execution, even for equal values

#### Scenario: Source flux domain differs from footprint domain

- **WHEN** a caller selects a different `flux_domain`
- **THEN** the requested flux dataset uses that domain and footprint retrieval
  retains its own domain and existing interpolation behavior
