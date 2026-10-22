# rhime-configuration Specification

## Purpose

Represent the complete resolved standard/multisector RHIME request before data
access, with equivalent external shorthand and a clear separation between
requested configuration and retained execution metadata.

Serialization and INI writing are deferred to
[#814](https://github.com/openghg/openghg_inversions/issues/814); they are not
acceptance requirements for this implementation.

## Requirements

### Requirement: Complete format-neutral requested configuration

Standard and multisector configured runs SHALL construct one inspectable,
complete resolved requested-run configuration before acquisition or scientific
preparation. It SHALL expose named data, model, output and sampling
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

The public complete requested configuration SHALL group data, model,
sampler and output choices. `RhimeConfig.data` SHALL hold one
`DataConfig` containing the moved acquisition/preparation fields and
their defaults; the complete configuration SHALL NOT retain a second authoritative
flat copy. The model, sampler and output groups SHALL reuse `RhimeModelSpec`,
`RhimeSampler` and `RhimeOutputSpec`, respectively, rather than introduce duplicate
settings types. Preparation SHALL resolve independently without constructing a
model, sampler or output policy. Complete configured runs SHALL resolve their
applicable groups once before scientific execution and SHALL keep requested
choices distinct from retained scientific facts.

`DataConfig` names the combined acquisition/preparation choices. Grouped input
mappings SHALL use the key `data`, matching `RhimeConfig.data`; the proposed
`preparation` field/key SHALL NOT remain as a compatibility alias. Runners SHALL
explicitly select each operation's named arguments from this record. Scientific
preparation SHALL receive the acquired handoff and its selected preparation
arguments. Sharing the record SHALL NOT combine acquisition and preparation
operations, whose separation is specified by the recipe-execution contract.

The legacy INI frontend SHALL preserve its documented decoding and precedence.
An adapter SHALL group its flat options into ordinary section mappings without
requiring new INI syntax or a new file format. Grouping SHALL NOT perform
scientific execution or force resolution of unrelated groups. Option names and
defaults SHALL have concrete owners rather than independent duplicate schemas.

#### Scenario: Inspect one complete requested configuration

- **WHEN** a caller resolves a complete request and inspects requested store,
  basis, filter and date options
- **THEN** those choices are available in the returned configuration's `data` group
- **AND** accessing them does not require data acquisition or an execution run description

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

#### Scenario: Inspect a grouped complete request
- **WHEN** a caller resolves a complete supported request
- **THEN** acquisition/preparation choices are available through `config.data`, with
  model, sampler and output choices available through their respective groups
- **AND** each choice and default has one authoritative owner

#### Scenario: Resolve preparation alone
- **WHEN** a caller supplies a valid `data` mapping without model, sampler or
  output choices
- **THEN** preparation resolution succeeds without constructing those other groups
  or requiring their inputs

#### Scenario: Group a familiar INI file
- **WHEN** a caller groups a supported existing INI request
- **THEN** its decoded options and winning values retain their established meaning
- **AND** the grouped representation is an ordinary mapping suitable for the same
  group resolvers as a Python request
- **AND** acquisition/preparation options use the `data` key without changing the
  existing INI section syntax

### Requirement: Explicit shallow selection for scientific keyword calls

`DataConfig.select(*names)` SHALL return a new dictionary containing the
explicitly named attributes. Selected values SHALL remain borrowed, including
mutable containers and opaque objects. Selection SHALL NOT copy nested values,
resolve defaults, inspect consumer signatures, execute scientific work or define
a serialization schema. An unknown attribute SHALL raise `AttributeError`. Grouping SHALL NOT introduce
implicit flattened attribute lookup; selection addresses the owning record's
direct attributes. `RhimeConfig` SHALL expose `data`, `model`, `output` and
`sampler` explicitly without a global `select` API.
Scientific stage functions SHALL retain named inputs and remain callable without
configuration. A private orchestration helper MAY receive resolved configuration
when composing several stages.

#### Scenario: Forward selected resolved values

- **WHEN** a caller unpacks `config.data.select("filters")` into observation filtering
- **THEN** the function receives the same resolved value as `filters=config.data.filters`
- **AND** changing the selection dictionary does not change the configuration
- **AND** mutable values inside the dictionary retain their existing identity

#### Scenario: Select acquisition and preparation arguments separately

- **WHEN** a runner selects store/site/date arguments and filter/basis/error arguments from `config.data`
- **THEN** acquisition receives its named selectors and returns the acquired handoff
- **AND** preparation receives that handoff and its own named arguments, without retrieving data or receiving export destinations

#### Scenario: Selection does not resolve overrides

- **WHEN** a caller needs to change raw options that affect shorthand or dependent defaults
- **THEN** it applies those overrides before semantic resolution
- **AND** selection and direct dataclass replacement do not recompute those defaults

### Requirement: Apply overrides before resolving shorthand

Supported overrides SHALL be applied before semantic resolution and site-option
expansion. Resolution SHALL use the effective requested site list and winning
values, preserve established precedence and rejection rules, and reject malformed
effective options before acquisition, reload or scientific preparation. A returned
configuration SHALL already contain the complete equivalent of external
shorthand for the selected phase; no scientific phase SHALL be responsible for
completing those selected choices. Known unrelated groups in a shared file SHALL
remain inactive; unsupported names and explicit wrong-phase overrides SHALL fail.
The external boundary SHALL report all unsupported and explicit wrong-phase
keys together before acquisition, preparation or inference. Supported aliases
SHALL use their runner/frontend alias owner and the same precedence in full and saved
routes; translating file and override mappings separately SHALL NOT change which
effective value wins.
Date-dependent configuration defaults SHALL use effective overridden dates for a
full request and recorded preparation dates for saved-input sampling.

Configuration owners SHALL validate their choices once during parsing or
construction, including the sampler's exclusion of inference-data `coords` and
`dims`. Later operations SHALL trust resolved choices rather than repeat those
checks because Python objects are mutable or attempt to reproduce all
model-dependent PyMC validation. Existing sampler settings SHALL remain the
owner; this change SHALL NOT require a separate sampler options class.

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

- **WHEN** effective options for the selected phase include an unknown, unsupported or malformed choice,
  including a sequence whose length does not match the effective site count
- **THEN** resolution fails before retrieval, reload or scientific execution
- **AND** the error identifies the offending option

#### Scenario: Override changes a date-dependent default

- **WHEN** a supported override changes the start date and the selected
  likelihood derives a configuration default from that date
- **THEN** resolution derives the default from the winning start date
- **AND** the returned configuration needs no downstream default-resolution pass

#### Scenario: Canonical file option and deprecated override

- **WHEN** a file supplies canonical `x_prior` and an override supplies deprecated `xprior`
- **THEN** the canonical spelling wins in both full and saved routes under the existing alias rule

#### Scenario: Several unsuitable overrides

- **WHEN** an invocation supplies several unsupported or explicit wrong-phase override keys
- **THEN** one error lists all those keys before scientific work

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
remain supported. Full configured orchestration SHALL use the same requested configuration rather
than a second setup bundle holding a pre-preparation run description and a
preparation dictionary. Saved phases SHALL resolve only their applicable groups
and use recorded preparation facts. Required run context MAY be retained,
referenced or supplied by the caller; the applicable inputs together SHALL
provide the facts the phase needs without requiring each record to stand alone
or the original full configuration to be unavailable. These phase choices SHALL
share their owners with complete requests.

#### Scenario: Unified configured orchestration

- **WHEN** ordinary, nested, shim or example consumers resolve a complete supported
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

### Requirement: Explicit 0.8 standard API migration

The Python `run_rhime` API SHALL be removed in 0.8 with a helpful import error
pointing to `run_standard`. This explicitly supersedes earlier promises to
preserve that Python entry point and its `RhimeResult` return. The `run-rhime`
CLI command SHALL call `run_standard` directly and preserve its command name,
written files, export defaults and logging.

Standard prepared execution SHALL use `run_standard(prepared=..., config=...)`,
skipping acquisition/preparation and then using the ordinary scientific pipeline.
`prepared` and `merged_data` SHALL be mutually exclusive. The former standard
`run_rhime_from_prepared_inputs`/one-sector `RhimeRunSpec` call SHALL be removed;
that entry point SHALL retain only its multisector compatibility contract.

Standard full, prepared-input and saved postprocessing SHALL return
`InversionOutput`, including with `output_format="none"`. The in-memory view SHALL
borrow its numerical inputs; `none` SHALL suppress writes, not result construction.
Its output metadata SHALL contain the actual inference description, without output
format, destinations, naming, save flags, written paths or live PyMC references.
`RhimeResult`, `RhimeRunSpec` and their retained-site construction helpers SHALL
be removed after the final cached CO2 output consumer migrates to `Co2Output`.
Model, output and retained preparation facts SHALL have separate owners.
The three standard builder entry points SHALL be replaced by one eager
`build_standard_model`; materialization SHALL be explicit in its caller.
New configuration helpers SHALL use `load_params`, `group_params` and
`option_groups`; `read_rhime_ini` SHALL retain its established name.
`RHIME_PREPARATION_OPTION_NAMES` SHALL be removed in favour of explicit owner
selection. These migrations supersede prior signature/return preservation
requirements for those named standard interfaces.

#### Scenario: Migrate the Python entry point without changing the CLI

- **WHEN** a caller imports the removed Python `run_rhime` API in 0.8
- **THEN** the import error directs the caller to `run_standard`, which returns `InversionOutput`
- **AND** the retained `run-rhime` CLI command calls `run_standard` directly with unchanged command naming, written files, export defaults and logging

### Requirement: Compatible public adapters and scientific choices

Except for the explicit standard API/result migration above, the sampler-default
migration, the model-owned BC/minimum-error migration specified below, and the
resource-I/O migrations removing `prepare_basis(output_path=...)` and `country_directory` in 0.8
and replacing scientific BC case/directory inputs with resolved basis data,
the legacy-only routing and namespace migration owned by the
[HBMCMC isolation contract](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/hbmcmc-isolation-retirement/spec.md),
removed `prepare_rhime_inputs` convenience function, obsolete
modern merged-cache options, former stage adapter, `resolve_rhime_config`,
transitional old-location imports and revised INI reader contract specified below, supported
scientific Python and CLI entry points SHALL retain their signatures, shorthand,
override behavior and return contracts through adapters to the same semantic
resolution or shared applicable site translation. Direct preparation/retrieval
APIs SHALL resolve only applicable choices without requiring likelihood, final
output or sampling settings. Independent builders and prepared-input runners
SHALL remain independent of full configuration. Shared choices such as BC use
and flux-source routing SHALL retain consistent existing meanings across data
preparation and model construction. Current INI section interpretation, path
conventions, file vocabulary, scientific calculations and sampling defaults
SHALL remain unchanged except for the path and legacy-resource migrations specified by
[rhime-resource-io](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-resource-io/spec.md),
the model-owned BC/minimum-error migration and retained-observation estimates
specified below, and the 0.8 historical scientific-quirk and chain-policy removals
specified by
[hbmcmc-isolation-retirement](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/hbmcmc-isolation-retirement/spec.md).

Established selector and shared-handoff aliases in the current
[model-selection capability](https://github.com/openghg/openghg_inversions/blob/7269f50f09dc4b618766cb9d69abe260a564eebc/openspec/specs/rhime-model-selection/spec.md)
SHALL remain supported for their documented compatibility period. The removed
transitional imports above do not include those public aliases.

Internal orchestration/setup records and their constructor/helper return shapes
SHALL NOT be required compatibility contracts. Their in-repository consumers
SHALL migrate to the unified requested configuration.

Default-selecting live APIs SHALL accept `"default"` or a supplied sampler, with
omission equivalent to `"default"`. Existing explicit `None` calls SHALL fail with
guidance to omit the optional argument or pass `"default"`; no compatibility
adapter SHALL retain `None`-means-default selection. Default settings and scientific
behavior SHALL otherwise remain unchanged. Result fields SHALL use `None` only
for absence of a live sampler, as specified by the recipe-execution contract.

Runner/frontend resolution SHALL consume omitted/false `use_tracer` without
retaining a config field. Canonical configuration SHALL accept only canonical
fields and use generic unsupported-option errors for obsolete options. Effective true SHALL retain early rejection at resolution and relevant
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

#### Scenario: Migrate an explicit null sampler

- **WHEN** a caller passes `sampler=None` to an existing live entry point
- **THEN** it fails before inference with guidance to omit the optional argument
  or pass `"default"`
- **AND** omitting that argument continues to select the established defaults

### Requirement: Scientific preparation owns its operations

Recipes SHALL derive observation errors before temporal filtering/aggregation,
call canonical `prepare_basis` directly with resolved arguments, and use the actual
sensitivity, filtering and assembly implementations. Forwarding wrappers SHALL
NOT be retained solely for uniform stage names, timing or keyword translation.
Shared indexing utilities SHALL remain independently reusable. Assembly SHALL
retain backend-neutral observations and spatial sensitivities;
model-owned numerical policies SHALL be resolved at the model boundary.

`prepare_rhime_inputs` and its exclusive acquisition dispatcher are removed.
Modern acquisition SHALL NOT load or save merged-data caches. Historical
six-tuple retrieval SHALL explicitly call the shared observation-error operation
before returning or saving its established result.
Prepared-input persistence SHALL remain available independently of acquisition
persistence. Public selector construction SHALL use `SiteOptions.from_inputs`.
The internal `inversion_data._site_options.convert_to_list` helper SHALL retain
its calling and list-return contract through shared selector expansion without a
warning; its old `get_data` import alias SHALL remain removed.

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

#### Scenario: Shared nested spatial support

- **WHEN** retained nested sites share one outer domain and one inner domain
- **THEN** preparation normalizes coordinate labels against their domain
  definitions, falling back to the first retained site with a warning for an
  unregistered domain
- **AND** incompatible grids are rejected before overlap masking or basis projection
- **AND** compatible sites share one inner rectangular extent, applied on each
  actual outer footprint and flux grid without mutating or eagerly computing
  borrowed arrays
- **AND** output metadata retains the applied footprint rectangle; externally
  pre-masked inputs with no supplied extent retain an unknown mask extent

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
configuration once. A legacy grouping adapter SHALL arrange the decoded flat
options by their existing owners; complete and phase-only resolution SHALL reuse
those owners. Phase-only resolution SHALL select its groups before validating
values or defaults. Shorthand SHALL remain editable until that construction.
Shared semantic/site-alignment helpers SHALL remain independent of file syntax.

Historical fixedbasis translation SHALL be owned by HBMCMC entry points.
Supported RHIME spelling and output-format aliases, including deprecated
`emissions_*` flux selectors, SHALL be translated at runner/frontend boundaries
outside HBMCMC after winning overrides. Supported runner option translation SHALL
remain available through `run_standard` on devel and in 0.8, with `DeprecationWarning` for deprecated
options, until their removal in 0.9. Canonical spelling precedence and the
separate flux old/new conflict errors SHALL remain unchanged.
`RhimeConfig.from_params` and canonical normalization SHALL know only canonical
names: they SHALL NOT import alias definitions, translate aliases, issue alias
warnings, or provide alias-specific replacement messages. Unsupported names
SHALL use ordinary unsupported-option validation. Canonical inputs SHALL pass
without alias warnings. Complete and phase-only frontends SHALL share the runner
compatibility owner; phase-only routes SHALL select applicable groups before
value validation/defaults without constructing an unrelated complete request.
HBMCMC-only resource selectors SHALL be translated by the legacy entry point,
or rejected with guidance by modern runner boundaries before construction.
Fixedbasis scientific policy SHALL remain in its separate compatibility adapter.
The `resolve_rhime_config` wrapper SHALL be removed.

During 0.8, HBMCMC's deprecated `params_from_config` SHALL return a dictionary
using the modern decoder and apply overrides before normalization. With
`normalise=False`, it SHALL preserve decoded values and historical names.
With `normalise=True`, it SHALL translate historical names and resource selections
into canonical keys and resolve declared filesystem values to absolute `Path`
objects, preserving file/override origins. This is an explicit return-value
migration from original path strings and historical selector keys; documentation
SHALL describe it. The adapter SHALL NOT be re-exported by RHIME or needed by
modern runners. Its removal belongs to the HBMCMC retirement contract.

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

- **WHEN** a caller supplies a supported deprecated spelling to `run_standard`
- **THEN** its frontend preserves the canonical value and established precedence,
  emitting a deprecation warning before canonical construction
- **AND** canonical construction receives only canonical names and emits no alias warning

#### Scenario: Direct canonical factory receives an old name

- **WHEN** a caller supplies a deprecated spelling directly to `RhimeConfig.from_params`
- **THEN** it rejects the unsupported name through ordinary option validation
- **AND** it neither translates the name nor emits an alias warning

#### Scenario: Phase-only aliases remain independent of HBMCMC

- **WHEN** a saved-phase caller supplies a supported deprecated modern alias
  for an applicable group alongside unrelated groups in a shared file
- **THEN** its external boundary uses the runner/frontend alias owner once with the same
  precedence as a complete request, without importing or calling HBMCMC
- **AND** it selects applicable groups before validating values or defaults,
  without constructing an unrelated complete request

#### Scenario: Dictionary compatibility adapter

- **WHEN** a 0.8 caller uses HBMCMC's `params_from_config` with supported overrides and normalization controls
- **THEN** the deprecated adapter returns raw overridden values when normalization is disabled,
  or canonical keys and absolute path values when normalization is enabled
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

### Requirement: Basis preparation receives explicit scientific inputs

Modern preparation SHALL accept an exact retained `basis_path` or resolved
labelled `basis_partition`, and resolved `landsea_classes` and `outer_regions`.
A labelled partition SHALL remain a valid scientific representation,
independently of historical artifact policy. Scientific preparation and numerical
kernels SHALL NOT depend on legacy discovery or naming policy.

`RhimeConfig.from_params` SHALL remain data-free. Supported deprecated resource
selectors SHALL be translated once at the legacy entry boundary defined by the
separate HBMCMC isolation/retirement contract before
canonical configuration is handed to scientific execution; resolved requests
SHALL NOT retain historical case/directory discovery fields. The removed
`country_directory` option SHALL be rejected rather than translated. Resource
loading, exact paths, packaged defaults, BC inputs and optional exports follow
[rhime-resource-io](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-resource-io/spec.md).

#### Scenario: Resolved basis scientific inputs

- **WHEN** preparation receives a selected operator path or labelled partition and its applicable resolved support maps
- **THEN** it consumes those explicit scientific inputs without historical filename discovery
- **AND** a labelled in-memory partition remains valid independently of artifact format

### Requirement: Modern flux selectors have their own deprecation boundary

Modern configuration and acquisition SHALL use `flux_store`, `flux_domain`,
`inner_flux_store` and `inner_flux_domain`. Their `emissions_*` predecessors
SHALL remain accepted at supported runner/acquisition frontends with
`DeprecationWarning` until removal in 0.9. Canonical configuration SHALL accept
only the new names and SHALL NOT translate aliases or emit alias warnings. Supplying
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

### Requirement: Native runtime flux has an explicit preparation owner

Each compatible domain/grid preparation context SHALL resolve native
`runtime_flux` once from its retained source datasets, after scientific masking
where applicable. Runners SHALL pass this labelled array and a singular
`weighting_flux` to canonical basis preparation. Weighting SHALL use the first
requested source or the first mapping entry; retained flux SHALL include all
runtime sources. Configuration SHALL NOT retain these numerical arrays. Native
prior flux SHALL remain distinct from observation-aligned `fp_x_flux` and
`fp_x_flux_sectoral`. Existing source alignment and signed-flux semantics SHALL
be preserved. Historical dataset and `fp_all` translation SHALL end at named
compatibility boundaries.

#### Scenario: Current prior binds a reused operator

- **GIVEN** a retained basis saved with prior A and acquired current prior B
- **WHEN** new preparation selects that artifact
- **THEN** `BasisFunctions.reuse_operator(path, flux=runtime_flux)` SHALL load and
  validate the operator without eagerly reading the saved flux payload
- **AND** source-specific operators SHALL follow the current ordered sources
- **AND** sensitivities and reconstructed flux SHALL use current prior B

#### Scenario: Reconstruction restores the retained prior

- **GIVEN** a retained basis saved with prior A
- **WHEN** postprocessing calls `BasisFunctions.load(path)`
- **THEN** the complete operator and saved flux SHALL be restored
- **AND** reconstruction SHALL use prior A without acquisition

### Requirement: Declared filesystem types identify path conversion

Canonical configuration SHALL represent filesystem selections using `Path`
types, including optional paths and explicit collections where supported.
Frontends SHALL accept external path strings and convert the declared path fields
with the owning frontend's base directory. Absolute paths SHALL keep their
selected destination. File-origin values and explicit invocation overrides SHALL
retain their distinct bases during conversion. Ordinary strings such as domain
names and scientific selectors SHALL NOT be interpreted as paths.

Path-field identification SHALL use type annotations through a small
standard-library converter supplied with the owning record's declarations. The
converter SHALL NOT import concrete configuration owners or inspect unrelated
configuration records. No second path-name registry or new configuration framework
dependency SHALL be introduced. Each external value SHALL acquire its base and
canonical path at one owning boundary; downstream factories, runners and identity
encoding SHALL forward resolved paths without repeating normalization. Independent
output configuration SHALL normalize its own external inputs without requiring a
complete run configuration. Historical selection translation, when applicable
under the separate legacy contract, SHALL precede canonical construction;
packaged scientific defaults remain runner-owned. Canonical construction SHALL
NOT glob or open candidate resource artifacts.

#### Scenario: Typed paths retain their frontend origins

- **WHEN** a saved-file request and invocation overrides supply strings for declared filesystem fields
- **THEN** conversion produces absolute `Path` values using each value's file or invocation base
- **AND** non-path strings retain their scientific meaning without path conversion
- **AND** configuration construction performs no scientific resource loading

#### Scenario: Forward a resolved request

- **WHEN** a configured runner or identity encoder receives canonical configuration
- **THEN** it uses the resolved paths without another conversion or legacy adaptation
- **AND** direct canonical construction accepts resolved values, while public
  external-input factories accept supported path strings

#### Scenario: Construct output configuration independently

- **WHEN** a caller constructs only output configuration from external strings
- **THEN** its owner converts only its declared filesystem fields using the invocation base
- **AND** it does not inspect or construct full run configuration or load resources

### Requirement: Phase-only resolution and authoritative model facts

Each phase SHALL select its applicable option groups before applying value
validation or defaults. Known unrelated options in a shared configuration file
SHALL NOT be forwarded to that phase or make its resolution depend on their
values. Unsupported option names and explicitly supplied phase-inapplicable
arguments SHALL fail at the applicable outer boundary before scientific artifact
writes. Repeated claims about authoritative prepared facts SHALL be checked
against the prepared inputs and applicable run context rather than silently
overwriting their facts.

Saved-input sampling and prior readiness SHALL bind model choices to recorded
preparation facts, including species, domain, source/layout information and the
recorded requested date anchor. Callers SHALL NOT have to repeat those facts to
resolve a model-only request. Complete runs SHALL use the same model resolution
rules with resolved requested preparation choices, then validate compatibility
with the retained prepared inputs. Changed model or likelihood choices SHALL be
accepted only when their required prepared inputs and structure are available;
they SHALL NOT reinterpret baked preparation choices or reacquire inputs.

For standard and multisector, preparation SHALL own the established `use_bc`
default (`True`) and the presence of spatial BC inputs. A flat `use_bc` request
SHALL route consistently to preparation and model selection. Omitted model
`use_bc` SHALL inherit the requested or recorded preparation choice. Explicit
model `use_bc=False` SHALL permit omitting available BC terms; explicit true
SHALL require those inputs and SHALL NOT cause acquisition or silent fallback.
Full requests SHALL reject model BC use with preparation BC disabled before
acquisition.

For these families, BC temporal grouping and minimum-error choices SHALL belong
to model resolution, not acquisition/preparation. Compatible raw prepared inputs
SHALL permit changing these choices without new acquisition, filtering or spatial
sensitivity work.
Omitted BC frequency SHALL select one period. Configured fixed-duration grouping
SHALL use the recorded requested start date unless an explicit model anchor is
supplied. Named minimum-error methods SHALL use retained observations after
unusable fitted rows are removed. Scalar and retained-site mapping choices SHALL
retain their validated numerical meaning. Unused minimum-error calculations SHALL
NOT execute for a likelihood that does not consume a floor.
CO2 SHALL retain its independent family-specific defaults.

Incompatible baked preparations and sample records SHALL be rejected explicitly
with guidance to use their matching software or prepare again. Readers SHALL NOT
silently reinterpret baked values as raw operands. Independent modern output
codecs SHALL retain their support. Scientific replay SHALL consume recorded fitted
operands without silently recomputing alignment or minimum error.

#### Scenario: Resolve disabled BC consistently across phases
- **WHEN** standard or multisector preparation resolves `use_bc=False` and the
  model group omits `use_bc`
- **THEN** both a complete request and independent sampling of that preparation
  resolve model BC use to false
- **AND** an empty model group does not apply a conflicting true default

#### Scenario: Resolve default BC consistently across phases
- **WHEN** standard or multisector preparation omits `use_bc` and the model group
  also omits it
- **THEN** preparation resolves the existing true default and both full and saved
  model resolution inherit true
- **AND** missing required BC arrays cause an error rather than disabling BC use

#### Scenario: Change BC use without changing preparation
- **WHEN** a preparation contains BC inputs and a later model explicitly selects
  `use_bc=False`
- **THEN** model construction omits BC terms without changing the saved preparation
- **AND** explicit `use_bc=True` on a preparation with BC disabled fails before
  inference or artifact writes

#### Scenario: Bind the saved boundary temporal policy
- **WHEN** a preparation contains raw spatial BC inputs and two models select
  different valid BC frequencies
- **THEN** both reuse those inputs without changing the preparation
- **AND** each model and its saved samples retain their own fitted alignment

#### Scenario: Minimum error uses retained fitted observations
- **WHEN** invalid sensitivities or observation errors remove otherwise finite
  observation values before a named minimum-error calculation
- **THEN** the calculation uses the retained observation population
- **AND** its selected values are retained with the fitted model for replay

#### Scenario: Reject an incompatible baked artifact
- **WHEN** a caller loads an older preparation or sample format whose numerical
  policies are incompatible with the raw-input contract
- **THEN** loading fails with migration guidance before inference or product writes
- **AND** no historical adapter silently substitutes model-time values

#### Scenario: Sample with model and sampler choices only
- **WHEN** a caller supplies a prepared handoff, applicable run context and valid
  model and sampler groups without acquisition or product configuration
- **THEN** sampling resolves those groups using authoritative saved facts and does
  not require unrelated configuration

#### Scenario: Unrelated shared-file value
- **WHEN** a preparation command reads a shared file containing an invalid sampler
  value alongside valid preparation choices
- **THEN** the preparation entry point selects and validates the `data` group without resolving or
  validating that inactive sampler value

#### Scenario: Conflicting saved-input claim
- **WHEN** a sampling request repeats a species, domain, source/layout or date
  claim that conflicts with the preparation facts
- **THEN** the request fails before model execution or scientific artifact writes
- **AND** the handoff's facts remain unchanged
