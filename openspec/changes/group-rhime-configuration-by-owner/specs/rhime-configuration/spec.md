# Configuration amendments

## MODIFIED Requirements

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

### Requirement: Explicit 0.8 standard API migration

The Python `run_rhime` API SHALL be removed in 0.8 with a helpful import error
pointing to `run_standard`. This explicitly supersedes earlier promises to
preserve that Python entry point and its `RhimeResult` return. The `run-rhime`
CLI command SHALL call `run_standard` directly and preserve its command name,
written files, export defaults and logging.

Standard full, prepared-input and saved postprocessing SHALL return
`InversionOutput`, including with `output_format="none"`. The in-memory view SHALL
borrow its numerical inputs; `none` SHALL suppress writes, not result construction.
Its output metadata SHALL contain the actual inference description, without output
format, destinations, naming, save flags, written paths or live PyMC references.
`RhimeResult` and `RhimeRunSpec` SHALL remain only for families awaiting handoff
adoption. The three standard builder entry points SHALL be replaced by one eager
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

Except for the explicit sampler-default migration, the prerequisite #868 resource-I/O
migrations removing `prepare_basis(output_path=...)` and `country_directory` in 0.8
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
[rhime-resource-io](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-resource-io/spec.md)
and the 0.8 historical scientific-quirk and chain-policy removals specified by
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

## ADDED Requirements

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
default (`True`) and `bc_freq` default (`None`). A legacy flat `use_bc` choice SHALL
route to preparation and the full run's model choice consistently. When resolving
a model group, omitted `use_bc` SHALL inherit the resolved requested preparation
choice for a full run or its recorded fact for saved sampling. Explicit model
`use_bc=False` MAY omit BC terms even when preparation contains BC inputs.
Explicit model `use_bc=True` SHALL require preparation to provide those inputs;
it SHALL NOT trigger acquisition or silent fallback. Full requests SHALL reject
model BC use with preparation BC disabled before acquisition.

`bc_freq` SHALL remain a preparation-owned temporal policy. Sampling SHALL use
the recorded policy when the key is omitted and SHALL accept a repeated equivalent
claim; any conflicting claim SHALL fail before model construction, even when
model BC terms are disabled. An explicit `None` claim SHALL be compared as the
existing no-frequency policy, not treated as omission. Model binding SHALL NOT
apply another default to replace saved temporal expansion. The same rules SHALL
hold for full and independently resolved standard/multisector phases without
changing CO2's family-specific defaults.

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
- **WHEN** preparation baked `bc_freq="3h"` into its numerical inputs
- **THEN** a later model request omitting `bc_freq` uses that saved temporal policy
  and a repeated `"3h"` claim is accepted
- **AND** an explicit different frequency or `None` is rejected before model
  construction, including when model `use_bc=False`

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
