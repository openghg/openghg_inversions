# Spec Delta

## Purpose

Preserve scientific consistency across equivalent recipe routes with phase-owned
choices, explicit numerical ownership, customization and established check behavior.

Saved preparation and inference follow [prepared-recipe-handoff](../prepared-recipe-handoff/spec.md)
and [sampled-recipe-handoff](../sampled-recipe-handoff/spec.md), respectively.

Configuration ownership and phase resolution follow the
[rhime-configuration capability](../../../../specs/rhime-configuration/spec.md).
External resource paths and optional basis exports follow
[rhime-resource-io](../../../../specs/rhime-resource-io/spec.md).

Existing public selection and aliases follow the current
[model-selection capability](https://github.com/openghg/openghg_inversions/blob/7269f50f09dc4b618766cb9d69abe260a564eebc/openspec/specs/rhime-model-selection/spec.md).
Registered coordinates and the standard/multisector custom-likelihood seam
follow the current [model-composition capability](https://github.com/openghg/openghg_inversions/blob/7269f50f09dc4b618766cb9d69abe260a564eebc/openspec/specs/rhime-model-composition/spec.md).
This change adds route and handoff behavior without redefining those contracts.

These requirements describe pending implementation. Landed configuration and
preparation foundations do not establish completion of the phase APIs or handoffs.

## ADDED Requirements

### Requirement: Canonical scientific execution across supported routes

Equivalent supported full, prepared-input and staged routes SHALL reuse their
family's scientific preparation, construction, inference and product operations.
Full execution SHALL sequence those operations in memory without mandatory
intermediate persistence. Preparation SHALL consistently derive retained labels
from retained observations and align applicable per-site choices, accepting valid
nonempty subsets without bypassing their owning input validation.
Runners SHALL keep acquisition or supplied-data selection visible before
scientific preparation and explicitly select each operation's named arguments
from `config.data`. Scientific preparation SHALL consume an explicit acquired
handoff and SHALL NOT retrieve data internally. Acquired data SHALL remain before
preparation-derived errors, filtering, basis/sensitivity construction and stacking;
existing `ModelScenario` products such as `fp_x_flux` and BC sensitivities remain
acquisition products. A combined `DataConfig` SHALL NOT obscure this boundary.
Runners SHALL consume the resource-I/O contract's save and resource-resolution
boundary supplied by #868.
Direct scientific customization, signatures and numerical returns SHALL retain
their contracts except for the explicit sampler-default migration below and
#868's removal of the basis-preparation export argument;
staged/setup APIs and owned CO2 configuration conventions SHALL
follow the declared reset instead of requiring compatibility aliases.
The existing selector forwarding, full-run CLI model choices and established
aliases SHALL remain supported under the model-selection capability. Saved-phase
commands SHALL NOT implicitly extend prepared-only CO2 recipes with acquisition
or configuration-file arguments. Existing registered-coordinate ownership,
custom-likelihood arguments/results, conflict rules and descriptive callable
metadata SHALL follow the model-composition capability. Shared output-role
handling SHALL NOT relax the canonical custom-likelihood seam.
Families supporting partial workflows SHALL remain usable without implementing
unrelated stages or adopting a common inference algorithm or result format.

Preparation SHALL derive missing observation errors before filtering or
aggregation and preserve caller-supplied custom errors and borrowed inputs.
Supported nested preparation SHALL establish the corresponding errors on both
domains before filtering/time alignment. These rules SHALL reuse their scientific
owners without implying additional nested or linked checkpoint support.

#### Scenario: Observation errors precede filtering
- **WHEN** supported preparation filters or aggregates observations
- **THEN** required errors have been established before that transformation
- **AND** supplied custom errors and caller-owned datasets remain unchanged

#### Scenario: Preparation consumes acquired data without retrieval

- **WHEN** a runner supplies acquired merged data to scientific preparation with retrieval disabled
- **THEN** preparation performs its scientific operations without attempting acquisition
- **AND** full and saved entry points expose acquisition or supplied-data selection before that same preparation operation
- **AND** optional basis export is a separate runner call after the scientific result is returned

#### Scenario: Full execution without intermediate saves
- **WHEN** a full route runs with intermediate saves disabled
- **THEN** it completes through in-memory scientific handoffs and produces
  requested final products without writing or reloading intermediate checkpoints

#### Scenario: Scientific parity and retained subsets
- **WHEN** equivalent routes receive equivalent inputs and choices, including a
  valid retained-site subset after acquisition, supported supplied-data input or filtering
- **THEN** their retained arrays, labels, aligned metadata, scientific terms and
  products agree within justified numerical tolerances
- **AND** validation uses real science and an independent calculation or existing
  regression reference rather than requiring identical stochastic trajectories

#### Scenario: Direct customization and partial recipes
- **WHEN** a caller uses supported likelihood or complete-model customization,
  or a family supports only prepared construction and its own inference
- **THEN** its explicit scientific arguments, conflict rules and return contract,
  subject to the declared sampler-default and prerequisite basis-export migrations,
  remain available without requiring acquisition, MCMC staging or saved replay

### Requirement: Existing sampler API and invocation isolation

Public live sampling and runner APIs offering default sampler selection,
including existing prepared-input entry points, SHALL accept `RhimeSampler` or the literal
`"default"` as their sampler argument. When no complete configured request supplies sampling choices, an omitted
argument SHALL mean `"default"` and SHALL select a fresh family-appropriate runtime
sampler for each inference invocation. A configured full run SHALL consume its
resolved `config.sampler`; omission of a separate sampler argument SHALL NOT replace
those configured choices. Supplying a resolved complete configuration together
with a separate sampler override SHALL follow the existing conflicting-input
rejection rule. A supplied sampler SHALL retain its selected behavior while mutable
configuration containers used by an invocation remain isolated from the caller
and later invocations, including nested keyword containers. These APIs SHALL
reject `None`; existing `None`-means-default behavior SHALL be removed rather than
retained by compatibility adapters. Omitted optional arguments SHALL continue to
select defaults, while explicit `None` SHALL produce a migration error directing
callers to omit the argument or use `"default"`. Lower-level operations requiring
an already resolved sampler MAY retain their object-only argument.
The contract SHALL NOT require a new sampler options class.

Resolving or overriding choices SHALL NOT mutate the caller's choices or affect
later invocations. Resolving supported options SHALL preserve their values and
container types, including lists and tuples. Model coordinate and dimension
registration SHALL remain authoritative: inference-data keyword
configuration SHALL reject `coords` and `dims` once at configuration parsing or
sampler construction rather than overriding the registered model. Downstream
resolution and sampling SHALL trust valid resolved choices rather than repeat
those checks because the Python object is mutable. The application SHALL NOT
duplicate PyMC's model-dependent validation. Ownership copies SHALL protect
caller settings from actual backend side effects while borrowing model, step,
xarray and Dask objects.

#### Scenario: Fresh default sampler
- **WHEN** separate live inference invocations omit the sampler or pass `"default"`
- **THEN** each invocation selects its family's default behavior with a fresh
  runtime sampler, without inheriting mutable state from another invocation

#### Scenario: Preserve configured sampling choices
- **WHEN** a complete resolved request supplies nondefault sampler settings
- **THEN** full execution uses those settings without requiring a repeated sampler
  argument or replacing them with defaults

#### Scenario: Explicit null sampler at a live boundary
- **WHEN** a caller supplies `None` to a public live sampling or runner API,
  including an existing prepared-input entry point
- **THEN** the boundary rejects it before inference rather than treating it as a
  request for a default sampler
- **AND** the error directs the caller to omit the optional argument or pass `"default"`

#### Scenario: Migrate an existing default-selecting call
- **WHEN** an existing caller omits the optional sampler or replaces explicit
  `None` with `"default"`
- **THEN** execution selects the same supported default settings with a fresh
  sampler, without requiring the caller to import `RhimeSampler`

#### Scenario: Repeated sampling with an override
- **WHEN** one invocation overrides sampler options containing nested lists,
  tuples or mappings and a later invocation uses the original options
- **THEN** the original options and later behavior remain unchanged
- **AND** resolving the supported options preserves their values and
  list-versus-tuple distinctions

#### Scenario: Attempted inference-data coordinate override
- **WHEN** inference-data keyword options contain `coords` or `dims`
- **THEN** configuration parsing or sampler construction rejects the keyword
  before sampling or scientific artifact writes, including when its supplied
  value is empty
- **AND** the resulting valid sampler needs no repeated downstream check

### Requirement: Optional live samplers and descriptive replay metadata

`RhimeResult.sampler` SHALL be optional and default to `None` without a sampler
factory. Saved sampler provenance SHALL be plain metadata describing the actual
inference invocation. Product replay and diagnosis SHALL NOT rebuild the
producing model or execute inference. They MAY use existing configuration
construction to inspect supplied or recorded choices; constructing a settings
object does not imply execution. Missing provenance SHALL NOT be filled from
current defaults, and descriptive sampling metadata SHALL NOT impose an
additional scientific compatibility gate. No strict-metadata mode is required.
Live execution SHALL retain access to a sampler only when its result contract
explicitly supplies one.
Here `None` SHALL mean absence of a live sampler, not absence of previously
completed inference. `"default"` SHALL be an execution request only and SHALL NOT
be retained as a result sampler or substituted for actual inference provenance.

#### Scenario: Result without a live sampler
- **WHEN** a caller constructs a result without a sampler, or loads saved samples
  for product replay or diagnosis
- **THEN** its live sampler remains `None` without selecting execution defaults
- **AND** recorded sampler provenance remains inspectable as metadata without
  rebuilding a model or running inference

### Requirement: Explicit numerical ownership and execution boundaries

Routes SHALL treat caller-supplied numerical inputs as borrowed and potentially
lazy. Choice resolution and access SHALL NOT mutate, copy, compute, persist,
densify or rechunk those inputs implicitly. Related arrays SHALL preserve shared
execution graphs until an explicit assembly, model-input, serialization or
eager-kernel boundary materializes the related arrays required there together.
Preparation SHALL NOT be required to keep every array lazy across an explicit
scientific execution boundary. Borrowed inputs SHALL remain unchanged.
Built-in construction SHALL own its remaining model-specific materialization;
a supplied complete-model builder SHALL receive inputs after the documented
preparation boundaries, including any arrays that remain lazy, and own any further
materialization required by its implementation. Custom builders SHALL NOT be
forced through a built-in model's additional execution boundary.

#### Scenario: Borrowed lazy preparation and construction
- **WHEN** a route receives borrowed inputs with shared lazy arrays
- **THEN** resolution and preparation preserve borrowed input values and shared
  execution until an applicable explicit materialization boundary
- **AND** related arrays required by assembly, construction or an eager kernel
  materialize together at the declared boundary rather than through incidental
  configuration access

#### Scenario: Custom complete-model builder
- **WHEN** a caller supplies a supported complete-model builder
- **THEN** the route forwards its resolved scientific arguments and prepared
  inputs, including remaining lazy arrays, without imposing the built-in model's
  additional construction boundary

### Requirement: One consumed output policy per invocation

Each configuration-driven execution SHALL consume one resolved output policy
through its family's product operations. Derived result metadata SHALL NOT become
an independently resolved output policy. The configured CO2 route SHALL consume
its supported model, sampler and output choices and expose its scientific results
and requested products consistently with that family; raw scientific runners
SHALL retain their separate numerical return contracts. Equivalent live and saved
product routes SHALL retain labelled dimensions, units, chain/draw axes and
family-specific reconstruction meaning without broadening unsupported products.
Construction SHALL validate its declared scientific variable roles and registered
coordinates independently of product selection, including when output is disabled.
Saved product execution SHALL validate its retained scientific output contract
without rebuilding the producing graph.

#### Scenario: Configured CO2 products
- **WHEN** configured CO2 execution requests supported products and writing choices
- **THEN** those choices govern the resulting products and destinations rather
  than being ignored or requiring a separately resolved stage-only output setup

#### Scenario: Live and saved products
- **WHEN** live execution and saved replay use the same samples,
  scientific output context and equivalent product policy
- **THEN** their supported scientific products agree, including conditional or
  affine reconstruction where applicable
- **AND** changing reporting or writing choices does not redefine inference

#### Scenario: Invalid scientific roles with output disabled
- **WHEN** construction declares an inconsistent scientific role or coordinate
  contract while output products are disabled
- **THEN** validation rejects the inconsistent contract before inference rather
  than skipping validation because no products were requested

### Requirement: Explicit readiness and independent diagnosis behavior

Readiness and diagnosis SHALL remain independently selectable checks with their
existing schemas, thresholds and pass/fail/unknown meanings. Returned empty or
non-finite predictive evidence SHALL yield a fail check and preserve applicable
predictive/report writes. Standard/multisector SHALL retain their scoped
construction/prediction `KeyError` and `ValueError` catch; CO2 execution errors
SHALL propagate before readiness destination creation. Loading outside that
catch and serialization errors SHALL propagate as errors. Strict
gates SHALL retain nonzero behavior for fail without newly treating unknown as
fail. Diagnosis SHALL consume the posterior and its check choices without
rebuilding the producing graph. This change SHALL NOT add historical staged
record compatibility.

Prior readiness SHALL consume prepared inputs, compatible model choices and its
check choices without requiring MCMC or final-product configuration or running
posterior inference. The existing family-specific predictive execution and write
behavior SHALL remain unchanged.

#### Scenario: Prior readiness without inference configuration
- **WHEN** a caller requests prior readiness using a prepared handoff and valid
  model/check choices without sampler or final-product choices
- **THEN** readiness runs its supported prior checks without posterior inference
  or requiring unrelated configuration

#### Scenario: Readiness execution and artifact failures
- **WHEN** prediction returns invalid evidence or raises an execution error
- **THEN** the family preserves the stated returned-evidence and exception behavior
- **AND** loading outside the catch or serialization failure remains an
  error rather than being reported as a completed scientific check

#### Scenario: Independent diagnosis and unavailable metrics
- **WHEN** diagnosis receives a posterior, optional supported run context and
  convergence choices
- **THEN** it assesses convergence without construction or sampling, retaining
  fail when any available metric fails and otherwise unknown when unassessable
