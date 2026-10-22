# Spec Delta

## MODIFIED Requirements

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
