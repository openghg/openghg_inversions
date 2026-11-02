# RHIME configuration

## MODIFIED Requirements

### Requirement: Configuration supplies explicit scientific choices

Configuration SHALL resolve requested choices without acquiring data, preparing
numerical arrays or saving scientific artifacts. Scientific preparation SHALL
remain independently callable under
[Scientific preparation owns its operations](../rhime-scientific-inputs/spec.md#requirement-scientific-preparation-owns-its-operations).
Prepared-input construction SHALL follow
[Canonical prepared scientific inputs](../rhime-scientific-inputs/spec.md#requirement-canonical-prepared-scientific-inputs).
Saved format/version admission and replay SHALL remain with the prepared/sample
handoff and persistence owners rather than configuration resolution.

`averaging_error` SHALL remain a preparation option rather than an acquisition
compatibility fact, preserving its default (`True`) and established component
interpretation. Compatible supplied data SHALL permit changing this choice
without reacquisition. Numerical population and borrowing follow
`rhime-scientific-inputs`.

#### Scenario: Change averaging error during preparation

- **GIVEN** compatible supplied merged data with uncertainty components and no supplied final `mf_error`
- **WHEN** a caller changes the valid `averaging_error` preparation choice
- **THEN** preparation uses it without rejecting the handoff or repeating acquisition
- **AND** configuration resolution does not calculate observation errors

#### Scenario: Resolve choices without scientific execution

- **WHEN** a caller constructs complete or phase-specific configuration
- **THEN** it resolves applicable requested choices without acquisition, numerical preparation or artifact writes
- **AND** direct scientific callers can supply those choices without a full run configuration

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
