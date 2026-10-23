# Spec Delta

## MODIFIED Requirements

### Requirement: Compatible public adapters and scientific choices

Except for the explicit standard API/result migration above, the sampler-default
migration, the model-owned BC/minimum-error and retained-population uncertainty
migrations specified by `rhime-scientific-inputs`, aggregation-filter retirements
specified below, tracer-option retirement, and the
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
the model-owned BC/minimum-error migration and retained-population observation-error
and minimum-error estimates specified by `rhime-scientific-inputs`, the
aggregation-filter retirements specified below, tracer-option retirement,
and the 0.8 historical scientific-quirk and chain-policy removals
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

Explicit `use_tracer` options SHALL fail ordinary unsupported-option validation
for either boolean value before acquisition or scientific execution, including
requests using supplied merged data. Omission SHALL remain valid without a tracer
configuration field. Alias translation SHALL NOT silently discard the option or
provide a special true-only guard. Existing legacy-option and custom-likelihood
conflict precedence SHALL remain unchanged.

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

#### Scenario: Reject an explicit tracer option

- **WHEN** a configured request explicitly supplies `use_tracer=False` or
  `use_tracer=True`, including with supplied merged data
- **THEN** ordinary unsupported-option validation rejects the name before
  acquisition or scientific execution
- **AND** omitting the option remains valid without a tracer configuration field

#### Scenario: Migrate an explicit null sampler

- **WHEN** a caller passes `sampler=None` to an existing live entry point
- **THEN** it fails before inference with guidance to omit the optional argument
  or pass `"default"`
- **AND** omitting that argument continues to select the established defaults

### Requirement: Scientific preparation owns its operations

`prepare_rhime_inputs` and its exclusive acquisition dispatcher are removed.
Modern acquisition SHALL NOT load or save merged-data caches. Historical
six-tuple retrieval SHALL retain its signature and return contract, including
prepared observation errors before its result is returned or saved; this
interface SHALL NOT acquire observation-selection options it does not accept.
Prepared-input persistence SHALL remain available independently of acquisition
persistence, as specified by `prepared-recipe-handoff`.
Public selector construction through `SiteOptions.from_inputs` SHALL remain
supported.
The internal `inversion_data._site_options.convert_to_list` helper SHALL retain
its calling and list-return contract without a warning; its old `get_data` import alias SHALL remain removed.

`averaging_error` SHALL remain a preparation option rather than an acquisition
compatibility fact. Its default (`True`) and established interpretation SHALL remain
available when preparing freshly acquired or compatible supplied merged data.
The numerical population and borrowing rules belong to `rhime-scientific-inputs`.

#### Scenario: Change averaging error during preparation

- **GIVEN** compatible supplied merged data with uncertainty components and no
  supplied final `mf_error`
- **WHEN** a caller changes the valid `averaging_error` preparation choice
- **THEN** preparation uses the new choice without rejecting the handoff or
  requiring another acquisition
- **AND** configuration resolution does not calculate observation errors

#### Scenario: Preserve the selector helper contract

- **WHEN** an existing caller uses
  `inversion_data._site_options.convert_to_list` with supported selector shorthand
- **THEN** its calling and list-return contract is retained without a warning
- **AND** the removed old `get_data` import alias is not restored

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

For standard and multisector, `bc_freq` and `bc_freq_anchor` SHALL belong to the
model group; `min_error` and `min_error_options` SHALL belong to the selected
likelihood's model options, not acquisition/preparation. Model-only resolution
SHALL accept valid choices using authoritative preparation facts without
requiring acquisition choices to be resolved again. Omitted `bc_freq` SHALL
select the established one-period policy. Configured requests SHALL resolve an
omitted `bc_freq_anchor` from the requested start date, using the recorded date
for saved-input sampling. Explicit `None` SHALL remain an explicit no-anchor
choice. `bc_freq` SHALL accept `None`, calendar `"monthly"` and positive fixed
durations; malformed, zero or negative fixed durations SHALL fail configuration
resolution. `bc_freq_anchor` SHALL accept a valid timestamp or `None`.

Likelihood floor options SHALL accept the established nonnegative finite scalar
or site-mapping values, `None` as a zero scalar, and named `"residual"` or
`"percentile"` methods. Omitted `min_error` SHALL resolve to `0.0`;
`min_error_options` SHALL accept only boolean `by_site`, defaulting to `False`.
Additive-sigma floor enablement SHALL remain an explicit boolean defaulting to
`False`. Invalid option values SHALL fail configuration resolution without
executing estimators. CO2 SHALL retain its independent family-specific defaults.

Numerical resolution, retained estimator populations and supplied-operand
compatibility follow `rhime-scientific-inputs`; durable preparations and fitted
operand replay follow `prepared-recipe-handoff` and `sampled-recipe-handoff`.

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

#### Scenario: Resolve model-owned options from saved facts

- **WHEN** a sampling request supplies valid BC-period or likelihood-floor choices
  with recorded preparation facts
- **THEN** model resolution accepts those choices without asking the caller to
  repeat acquisition configuration
- **AND** omitted `bc_freq` resolves to the one-period policy and an omitted
  `bc_freq_anchor` resolves from the recorded requested start date
- **AND** explicit `bc_freq_anchor=None` remains `None`

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
