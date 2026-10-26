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
choices with established configuration-only defaults resolved. It
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

The existing INI frontend SHALL preserve its documented decoding and precedence.
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

#### Scenario: Configuration defaults are explicit

- **WHEN** a canonical request omits options with established configuration-only defaults
- **THEN** the returned configuration exposes their resolved values
- **AND** downstream canonical consumers need no default-resolution pass

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
keys together before acquisition, preparation or inference. Full and saved
routes SHALL apply the same override precedence to canonical option names.
Unsupported historical names SHALL NOT acquire support through an override
or translation step.
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

#### Scenario: Deprecated override remains unsupported

- **WHEN** a file supplies canonical `x_prior` and an override supplies unsupported `xprior`
- **THEN** ordinary unsupported-option validation rejects `xprior` in full and saved routes
- **AND** the presence of a canonical value does not cause the old name to be ignored

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

- **WHEN** resolution normalizes priors, sampling keywords and
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

- **WHEN** ordinary, nested or example consumers resolve a complete supported
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

#### Scenario: Supplied merged data

- **WHEN** a configured run receives a valid compatible merged handoff with its
  own retained-site options
- **THEN** it preserves those options, performs no acquisition and continues
  preparation without replacing them with requested defaults

### Requirement: Canonical standard and multisector execution

Standard execution SHALL use `run_standard`; multisector execution SHALL use
`run_multisector`. Prepared execution SHALL accept `prepared` with the resolved
`config`, skip acquisition/preparation and use the same scientific operations as
full and saved execution. `prepared` and `merged_data` SHALL be mutually exclusive.
Standard CLI execution SHALL use `rhime run --model standard` or `rhime run`,
preserving effective configuration, written files, export defaults and logging.

Standard and multisector full, prepared-input and saved postprocessing SHALL
return `InversionOutput`, including with `output_format="none"`. The in-memory
view SHALL borrow its numerical inputs; `none` SHALL suppress writes, not result
construction. Output metadata SHALL describe actual inference without output
format, destinations, naming, save flags, written paths or live PyMC references.
Model choices, output choices and retained preparation facts SHALL remain distinct.

The supported eager builders SHALL be `build_standard_model` and
`build_multisector_model`; related numerical inputs SHALL be materialized
explicitly before construction. Configuration helpers SHALL expose `load_params`,
`group_params`, `option_groups` and `read_rhime_ini`. Named selection from the
owning configuration record SHALL supply applicable component inputs.

#### Scenario: Run canonical full or prepared execution

- **WHEN** a caller executes `run_standard` or `run_multisector` with valid full or prepared inputs
- **THEN** it receives the corresponding scientific `InversionOutput`
- **AND** prepared execution performs no acquisition or preparation
- **AND** disabling writes does not suppress the returned scientific result

### Requirement: Canonical requests and phase-owned validation

Modern runner/acquisition requests SHALL use the supported canonical option
names and scientific interfaces of their owning capabilities. Normal execution
SHALL operate directly on those interfaces without migration translation.

Unknown option names SHALL fail ordinary unsupported-option validation before
acquisition, reload or scientific execution, including names in known inactive
groups. Effective values applicable to the selected phase SHALL be validated by
their owning configuration against supported values before that phase performs
scientific work. Known inactive groups in shared requests SHALL defer value
validation and resource resolution until consumed. Runner/acquisition validation
SHALL describe supported names and applicable values without special recognition
of historical spellings or tokens, alias translation, warning-and-ignore behavior
or fallback. Supplying a canonical spelling alongside an unknown name SHALL NOT
make that unknown name supported. Canonical names and other supported modern
signatures and result types SHALL retain their separate contracts.

Public commands and Python recipe/data interfaces SHALL follow
[Canonical installed commands](../rhime-model-selection/spec.md#requirement-canonical-installed-commands)
and [Canonical public Python interfaces](../rhime-model-selection/spec.md#requirement-canonical-public-python-interfaces).
Supported artifact formats and versions SHALL follow their separate persistence
contracts, without implied historical-format support from Python names.

Scientific calculations, canonical site shorthand and override behavior, labelled
alignment, borrowed inputs, named materialization boundaries, modern legacy-format
output and saved `InversionOutput` codecs SHALL retain their independent contracts.
Direct preparation/retrieval APIs SHALL resolve only applicable choices without
requiring likelihood, final output or sampling settings. Independent builders and
prepared-input runners SHALL remain independent of full configuration. Shared
choices such as BC use and flux-source routing SHALL have consistent meanings
across preparation and model construction. Existing INI section interpretation,
canonical vocabulary, scientific calculations and sampling defaults SHALL remain
supported, subject to their separately specified resource and scientific contracts.

Historical observations and scientific regression coverage SHALL retain the
[migration evidence guarantees](../hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence).
Internal orchestration/setup records SHALL NOT impose public constructor or
helper-return compatibility requirements.

Default-selecting live APIs SHALL accept `"default"` or a supplied sampler, with
omission equivalent to `"default"`. Explicit `None` SHALL fail before inference
with guidance to omit the argument or pass `"default"`. Result fields MAY use
`None` for absence of a live sampler. Established defaults and scientific behavior
SHALL otherwise remain unchanged.

#### Scenario: Unsupported applicable output value

- **WHEN** a full or output-only request supplies an `output_format` value outside the output owner's supported values
- **THEN** ordinary value validation rejects it before the selected phase performs scientific work
- **AND** it receives the ordinary invalid-output diagnostic without alias translation

#### Scenario: Known inactive output value

- **WHEN** a preparation-only command reads a shared file with supported output option names and an invalid `output_format` value
- **THEN** preparation defers validation of that inactive output value
- **AND** a later output consumer validates its effective value against its supported formats
- **AND** unknown names in that group still fail ordinary unsupported-option validation

#### Scenario: Existing Python shorthand

- **WHEN** standard, multisector or direct public preparation/retrieval callers supply supported scalar options instead of expanded sequences
- **THEN** the external boundary establishes the same applicable canonical choices before scientific work
- **AND** the supported scientific return contract is preserved

#### Scenario: Boundary conditions are disabled consistently

- **WHEN** effective configuration sets `use_bc=False`
- **THEN** acquisition/preparation and built-in model construction use the same disabled BC choice
- **AND** neither phase infers an independent conflicting default

#### Scenario: Migrate an explicit null sampler

- **WHEN** a caller passes `sampler=None` to a live entry point
- **THEN** it fails before inference with guidance to omit the argument or pass `"default"`
- **AND** omission continues to select the established defaults

### Requirement: Accepted observation-selection filter names

Observation-selection requests SHALL use supported registered filter names.
Global scalar and sequence requests SHALL be validated even when no observation
sites are present. Site-specific requests SHALL validate only entries for sites
present in the supplied observations; absent-site entries SHALL remain deferred.
Unsupported active names SHALL raise ordinary `KeyError` before applying any
preceding filter or executing lazy scientific data, without historical-name
recognition or migration translation.

Observation selection SHALL NOT provide aggregation filters or direct Python
aggregation-filter functions. Resampling and propagation of the resulting
observation uncertainty SHALL be explicit scientific choices separate from
selection. No aggregation statistic SHALL be substituted automatically.

#### Scenario: Reject an unsupported active filter

- **WHEN** a global scalar or sequence request, or an active site-specific request, names an unregistered filter
- **THEN** it raises ordinary `KeyError` before any earlier requested filter or lazy scientific payload executes
- **AND** global requests receive the same validation with empty or metadata-only observations
- **AND** no replacement aggregation statistic is selected automatically

#### Scenario: Defer absent-site filter choices

- **WHEN** a site-specific request includes an unsupported filter for a site absent from the observations
- **THEN** that site's choice is deferred without executing or validating it
- **AND** requests for present sites are validated before any filter executes

### Requirement: Configuration supplies explicit scientific choices

Configuration SHALL resolve requested choices without acquiring data, preparing
numerical arrays or saving scientific artifacts. Scientific preparation SHALL
remain independently callable under
[Scientific preparation owns its operations](../rhime-scientific-inputs/spec.md#requirement-scientific-preparation-owns-its-operations).
Prepared-input construction SHALL follow
[Canonical prepared scientific inputs](../rhime-scientific-inputs/spec.md#requirement-canonical-prepared-scientific-inputs).
Saved format/version admission and replay SHALL remain with the prepared/sample
handoff and persistence owners rather than configuration resolution.

#### Scenario: Resolve choices without scientific execution

- **WHEN** a caller constructs complete or phase-specific configuration
- **THEN** it resolves applicable requested choices without acquisition, numerical preparation or artifact writes
- **AND** direct scientific callers can supply those choices without a full run configuration

### Requirement: INI decoding is independent of semantic construction

`read_rhime_ini(path)` SHALL return a dictionary of decoded file options. It SHALL
own INI syntax, section flattening and value decoding, preserving the existing
first-occurrence rule for repeated bare keys. It SHALL NOT require complete run
settings, select a model type, accept overrides, translate historical names or
resolve defaults/site shorthand. No new file format or section schema is required.

Standard, multisector, nested and custom runners SHALL combine decoded canonical
options with winning overrides, consume their own recipe choices, and construct
canonical configuration once. Complete and phase-only resolution SHALL use the
same option meanings and precedence. Phase-only resolution SHALL select applicable
groups before validating values or applying defaults, without constructing an
unrelated complete request. Known unrelated groups SHALL remain inactive;
unsupported names SHALL fail before scientific work.

`RhimeConfig.from_params` and modern full/saved frontends SHALL accept supported
canonical request names and use ordinary unsupported-option validation. Resource
selectors SHALL identify exact selected paths or resolved scientific inputs;
semantic construction SHALL NOT perform historical discovery or invoke migration
translation.

#### Scenario: Read recipe-specific or incomplete options

- **WHEN** a caller reads an INI file with incomplete ordinary settings or project-specific choices
- **THEN** the reader returns decoded values without trying to construct a run
- **AND** the recipe can consume its own options and apply overrides before validating the remaining canonical request

#### Scenario: Override shorthand after reading

- **WHEN** a caller reads site shorthand and changes the requested sites or periods
- **THEN** semantic construction expands only the winning values
- **AND** the reader performs no premature expansion or date-dependent resolution

#### Scenario: Reject an unsupported name at every modern request boundary

- **WHEN** a full runner, saved phase or `RhimeConfig.from_params` receives an unsupported request name
- **THEN** ordinary unsupported-option validation rejects it before scientific work
- **AND** no boundary translates or warns-and-ignores the unsupported name

### Requirement: Configuration is not a historical identity schema

Requested configuration SHALL represent resolved choices without reconstructing
original scalar/list spellings or sparse defaults solely to preserve historical
staged hashes. Public scientific behavior and independently supported codecs
SHALL remain supported. Historical hashes SHALL NOT justify changing supported signatures or result
wrappers. Equality of historical encoding hashes SHALL NOT be an acceptance
requirement. Changes affecting a currently supported persisted contract SHALL
be identified explicitly rather than silently assigned its old contract.

#### Scenario: Equivalent shorthand and historical encodings

- **WHEN** scalar and expanded site inputs resolve to equivalent choices
- **THEN** canonical consumers use those equivalent resolved values
- **AND** configuration adapters are not required to reconstruct different raw
  forms to preserve their historical staged identities

### Requirement: In-memory supplied and freshly acquired data paths

Configured recipes SHALL reuse compatible supplied merged data without
object-store I/O or call `MergedData.from_options` with resolved selectors.
They SHALL preserve authoritative retained selections and validate known
acquisition facts. Modern merged-data cache options SHALL be rejected.
Durable scientific reuse SHALL use prepared inputs rather than acquisition caches.
Selected merged-data provenance remains on the in-memory handoff and SHALL NOT
be added to prepared inputs or the preparation manifest by this change.

#### Scenario: Supplied data bypasses acquisition

- **WHEN** valid compatible merged data is supplied to the recipe
- **THEN** its data and authoritative site options are reused without acquisition
- **AND** the same observation-error preparation policy applies

#### Scenario: Removed modern cache options

- **WHEN** a modern caller requests a removed acquisition-cache option
- **THEN** configuration rejects it and directs file reuse to prepared inputs
- **AND** independently supported scientific codecs retain their separate contracts

### Requirement: Basis preparation receives explicit scientific inputs

Modern preparation SHALL accept an exact retained `basis_path` or resolved
labelled `basis_partition`, and resolved `landsea_classes` and `outer_regions`.
A labelled partition SHALL remain a valid scientific representation,
independently of historical artifact policy. Scientific preparation and numerical
kernels SHALL NOT depend on legacy discovery or naming policy.

`RhimeConfig.from_params` SHALL remain data-free. Resolved requests SHALL contain
explicit canonical scientific selections. Resource loading, exact paths, packaged
defaults, BC inputs and optional exports SHALL follow
[rhime-resource-io](../rhime-resource-io/spec.md).

#### Scenario: Resolved basis scientific inputs

- **WHEN** preparation receives a selected operator path or labelled partition and its applicable resolved support maps
- **THEN** it consumes those explicit scientific inputs without historical filename discovery
- **AND** a labelled in-memory partition remains valid independently of artifact format

### Requirement: Canonical modern flux selectors

Modern configuration and acquisition SHALL use `flux_store`, `flux_domain`,
`inner_flux_store` and `inner_flux_domain`. Configuration exports SHALL contain
canonical names; unknown names SHALL follow ordinary boundary validation.

`flux_domain` SHALL select the source flux dataset domain, defaulting to the
corresponding footprint domain; it SHALL NOT inherently mean the nested inner
domain. Modern scientific source selection SHALL use canonical `flux_sources`.
Independently supported basis representations, artifact codecs and standalone
adapter capabilities SHALL retain their separate contracts.

#### Scenario: Source flux domain differs from footprint domain

- **WHEN** a caller selects a different `flux_domain`
- **THEN** the requested flux dataset uses that domain and footprint retrieval retains its own domain and existing interpolation behavior

### Requirement: Native runtime flux has an explicit preparation owner

Each compatible domain/grid preparation context SHALL resolve native
`runtime_flux` once from its retained source datasets, after scientific masking
where applicable. Runners SHALL pass this labelled array and a singular
`weighting_flux` to canonical basis preparation. Weighting SHALL use the first
requested source or the first mapping entry; retained flux SHALL include all
runtime sources. Configuration SHALL NOT retain these numerical arrays. Native
prior flux SHALL remain distinct from observation-aligned `fp_x_flux` and
`fp_x_flux_sectoral`. Existing source alignment and signed-flux semantics SHALL
be preserved. Historical `fp_all` utilities SHALL NOT be required by modern
preparation; independently supported labelled inputs SHALL retain their contracts.

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
values, including optional paths and explicit collections where supported.
Modern Python/CLI paths, INI/JSON resource and output paths, and explicit
invocation overrides SHALL resolve relative to the invocation working directory,
as specified by
[rhime-resource-io](../rhime-resource-io/spec.md#requirement-runners-own-resource-and-export-paths).
A configuration file's location SHALL NOT rebase its resource or output values.
Absolute paths SHALL retain their selected destination. Ordinary strings such
as domain names and scientific selectors SHALL NOT be interpreted as paths.

Resolved selections SHALL preserve the caller's resource and export choices.
Canonical construction SHALL NOT glob, inspect candidate artifacts or load
scientific resources. Constructing output configuration SHALL NOT require a
complete run configuration. Packaged scientific defaults SHALL remain
runner-owned. Relative dependency references inside saved manifests SHALL
retain their separate handoff contract rather than these invocation rules.

#### Scenario: File and invocation paths share the invocation base

- **WHEN** an INI/JSON file supplies `basis_path="basis.nc"` and an invocation override selects that same relative path
- **THEN** both select `basis.nc` beneath the invocation working directory
- **AND** moving the configuration file does not change that resource selection
- **AND** configuration construction performs no scientific resource loading

#### Scenario: Preserve an absolute selected destination

- **WHEN** a configured request selects an absolute resource or export path
- **THEN** the resolved configuration preserves that destination
- **AND** scientific consumers receive the resolved selection without historical discovery

#### Scenario: Construct output configuration independently

- **WHEN** a caller constructs only output configuration from external path strings
- **THEN** the paths resolve using the invocation base
- **AND** unrelated full-run settings and scientific resources are not required

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
default (`True`) and `bc_freq` default (`None`). A flat `use_bc` choice SHALL
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
