# Spec Delta

## Purpose

Preserve scientific consistency across equivalent recipe routes with phase-owned
choices, explicit numerical ownership, customization and established check behavior.

Saved preparation and inference follow [prepared-recipe-handoff](../prepared-recipe-handoff/spec.md)
and [sampled-recipe-handoff](../sampled-recipe-handoff/spec.md), respectively.

Configuration ownership and phase resolution follow the
[rhime-configuration capability](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-configuration/spec.md).
Numerical preparation, model-input alignment and error operands follow
[rhime-scientific-inputs](../../../../specs/rhime-scientific-inputs/spec.md).
External resource paths and optional basis exports follow
[rhime-resource-io](https://github.com/openghg/openghg_inversions/blob/devel/openspec/specs/rhime-resource-io/spec.md).

Existing public selection and aliases follow the current
[model-selection capability](../../../../specs/rhime-model-selection/spec.md),
including its canonical installed commands and five retired Python names.
Registered coordinates and the standard/multisector custom-likelihood seam
follow the current [model-composition capability](https://github.com/openghg/openghg_inversions/blob/7269f50f09dc4b618766cb9d69abe260a564eebc/openspec/specs/rhime-model-composition/spec.md).
Supported standard entry points, scientific results and eager-builder boundaries
follow the configuration and model-composition contracts.

## ADDED Requirements

### Requirement: Matched fixed-OU amplitude priors

The matched cached CO2 fixed-OU model SHALL accept the existing supported
independent positive-prior families through `site_amplitude_prior`. The
conditional amplitude update SHALL use the scientific model's prior density,
transform and Jacobian, including bounded positive distributions, without a
second independently specified prior. It SHALL retain sigma-then-state ordering,
the accepted-cache lifecycle and native correlated prior/posterior prediction.
Hierarchical amplitude priors and sampled OU timescales are outside this scope.

#### Scenario: A non-HalfNormal amplitude prior uses the matched update

- **WHEN** a supported independent positive amplitude prior is supplied
- **THEN** the scientific graph, conditional sampler and prior prediction use
  that same distribution
- **AND** transformed conditional values and gradients agree with an independent
  density calculation, with the Jacobian included exactly once
- **AND** physical amplitudes used to refresh the cache follow the model transform

#### Scenario: Bounded amplitude initialization

- **WHEN** a bounded positive prior supplies default or explicit initial amplitudes
- **THEN** initialization has finite positive physical amplitudes, finite
  transformed values and finite prior density
- **AND** invalid endpoints or out-of-support starts fail before inference

#### Scenario: Explicit LogNormal reparameterisation on the matched route

- **WHEN** a matched amplitude prior explicitly requests `reparameterise=true`
- **THEN** construction warns and records the effective direct LogNormal representation
- **AND** the physical prior and caller-owned configuration remain unchanged

### Requirement: CO2 inference follows scientific choices

Configured CO2 full execution and shared saved inference SHALL choose their
implementation from the scientific likelihood and amplitude policy. Fixed OU
timescales with inferred independent positive site amplitudes SHALL use the
matched cached update, without a public algorithm switch or silent fallback.
Sampling lengths, chains and seeds SHALL remain configurable. Unsupported
sampler substitutions SHALL fail before loading or materializing scientific
inputs. Prior readiness SHALL resolve only its own scientific and seed choices.
Configured CO2 SHALL reject `sampling.nuts_sampler`. The supported automatic
policy SHALL be:

| Likelihood | Backend / update | Reason |
| --- | --- | --- |
| `additive_sigma` | NumPyro NUTS | All steps are NUTS and the model runs under JAX via NumPyro |
| `site_sigma`, fixed or inferred amplitudes | PyMC NUTS | Conservative default, not an inherent backend restriction |
| `scalar_sigma` | PyMC NUTS | Conservative default, not an inherent backend restriction |
| `fixed_ou`, fixed amplitudes | PyMC NUTS | Custom likelihood operation has no JAX implementation |
| `fixed_ou`, inferred amplitudes | Matched PyMC amplitude update followed by state NUTS when states exist | Requires the custom cached amplitude update and, when states exist, its ordered compound step in PyMC |

Matched target-accept controls SHALL belong to `sampling`, not the scientific
likelihood. Equivalent full and saved requests SHALL select the same inference
policy from their resolved scientific choices. The actual algorithm and explicit
target-accept values, or a `target_accept_source="pm.sample_default"` marker
when ordinary acceptance is omitted, SHALL be recorded in the trace and sample
manifest and retained by product replay without constructing a sampler. Presence of any
active flux, boundary or offset state SHALL require the state update.

#### Scenario: Full and saved sampling retain actual inference details

- **WHEN** equivalent CO2 model and sampling choices run through full or saved execution
- **THEN** both choose the same backend, update sequence and tuning controls
- **AND** trace serialization and replay retain the actual algorithm and explicit tuning or the delegated-default marker
- **AND** absent historical descriptions are not replaced by current execution defaults

#### Scenario: All affine coefficients are fixed

- **WHEN** fixed-OU amplitudes are inferred while all flux and baseline
  coefficients are fixed
- **THEN** the matched amplitude update remains available
- **AND** no empty state NUTS step is constructed
- **AND** the complete fixed concentration mean remains in the likelihood

#### Scenario: Dropped-site options are validated before selection

- **WHEN** OU timescales or initial amplitudes include entries for dropped sites
- **THEN** every supplied entry is validated before selecting retained labels
- **AND** valid extras are allowed while missing retained-site coverage fails

### Requirement: Canonical scientific execution across supported routes

Equivalent supported full, prepared-input and staged routes SHALL preserve their
family's scientific preparation, construction, inference and product semantics.
Full execution SHALL remain usable in memory without mandatory intermediate
persistence. Preparation SHALL derive retained labels from retained observations
and align applicable per-site choices, accepting valid nonempty subsets without
bypassing input validation.

Scientific preparation of supplied acquired data SHALL perform no retrieval.
The acquired handoff SHALL retain acquisition facts and values independently of
preparation-derived errors, selected observations and assembled sensitivities.
Preparation SHALL NOT mutate the supplied acquired data to attach those derived
values. Existing acquisition products such as `fp_x_flux` and BC sensitivities
SHALL retain their meanings.

Supported customization, concrete scientific result types and resource/export
behavior SHALL follow the configuration, model-composition and resource-I/O
contracts. Normal modern execution SHALL contain no HBMCMC namespace dependency
or historical option translation. The enumerated obsolete options, special
tracer handling, installed command names and five Python aliases SHALL follow
the canonical-only target in
[Canonical requests and phase-owned validation](../../../../specs/rhime-configuration/spec.md#requirement-canonical-requests-and-phase-owned-validation).
The [migration evidence guarantees](../../../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence)
SHALL preserve historical observations and relevant scientific coverage.

Scientific calculations, borrowed-array/execution guarantees, modern
legacy-format output, `InversionOutput` exports and independent basis capabilities
SHALL retain their separate support. Supported artifact formats and versions
SHALL follow the prepared/sample persistence contracts independently of Python
naming. No general signature/result-wrapper change or incidental scientific
change follows from the enumerated naming removals.

Linked prepared-input scientific runners SHALL NOT gain acquisition or
`config_file` handling from saved-phase support. Registered-coordinate ownership,
custom-likelihood arguments/results, conflict rules and descriptive callable
metadata SHALL follow the model-composition contract. Families supporting partial
workflows SHALL remain usable without unrelated phases or a common inference
algorithm or result type.

Observation population, numerical preparation, error derivation and alignment
SHALL follow the focused scientific-input contract. Equivalent full, prepared
and saved routes SHALL preserve those semantics rather than reconstruct a
historical preparation policy.

#### Scenario: Preparation consumes acquired data without retrieval

- **WHEN** a supported recipe receives supplied acquired data
- **THEN** preparation performs its scientific operations without acquisition
- **AND** preparation leaves the supplied acquired values unchanged
- **AND** optional exports preserve the requested artifact destinations

#### Scenario: Equivalent scientific routes

- **WHEN** equivalent inputs and effective choices enter full, prepared-input or saved execution
- **THEN** their retained arrays, labels, aligned metadata, scientific terms and products agree within justified numerical tolerances
- **AND** equivalence does not require identical stochastic trajectories or private helper calls

#### Scenario: Full execution without intermediate saves

- **WHEN** a full route runs with intermediate saves disabled
- **THEN** it produces requested final products without writing or reloading intermediate checkpoints

#### Scenario: Direct customization and partial recipes

- **WHEN** a caller uses supported likelihood customization or independent scientific composition
- **THEN** its explicit scientific arguments, conflict rules and concrete result contract remain available
- **AND** prepared-only execution requires neither acquisition nor unrelated checkpoint phases

### Requirement: Existing sampler API and invocation isolation

Configured CO2 full and saved execution are an explicit exception to the
generic sampler-argument contract below: they SHALL consume a `sampling`
mapping, and the resolved model SHALL select the backend and update. They SHALL
NOT retain a second saved-route sampler override. Direct scientific composition
and linked CO2/O2 retain their separate numerical sampling APIs.

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

### Requirement: Scientific outputs and descriptive replay metadata

Standard and multisector results SHALL be `InversionOutput`; nested results SHALL
be `NestedOutput`, and ordinary and cached CO2 results SHALL be `Co2Output`.
These scientific views SHALL NOT hold live models or sampler fields.
Saved sampler provenance SHALL be plain metadata describing the actual inference
invocation. Product replay and diagnosis SHALL NOT rebuild the producing model or
execute inference. Replay SHALL normalize available `draws`, `burn`, `tune`,
`chains` and `nuts_sampler` values as descriptive metadata without
constructing a sampler. Missing or invalid descriptions SHALL become `{}`; valid
partial descriptions SHALL retain only their supplied fields.
Missing provenance SHALL NOT be filled from current defaults, and descriptive
sampling metadata SHALL NOT impose an additional scientific compatibility gate.
No strict-metadata mode is required. `"default"` SHALL remain an execution
request only and SHALL NOT replace actual inference provenance.
CO2 SHALL additionally retain valid recorded algorithm and effective target-accept
fields through its plain metadata reader; it SHALL NOT drop them when replaying
the sample manifest.

#### Scenario: Result without a live sampler
- **WHEN** a caller constructs a scientific output, or loads saved samples
  for product replay or diagnosis
- **THEN** the output retains no live sampler field or execution defaults
- **AND** recorded sampler provenance remains inspectable as metadata without
  rebuilding a model or running inference

### Requirement: Scientific construction across supported routes

Numerical borrowing and execution boundaries SHALL follow the scientific-inputs
contract. Standard prepared-input construction SHALL accept model choices without
requiring output configuration and preserve the full/saved route's scientific meaning.
Its complete-model-builder hook SHALL be rejected; its custom-likelihood seam
SHALL remain available. Multisector construction SHALL use the same separation.
Caller-owned complete models MAY directly compose scientific functions with
explicit inputs and SHALL own any further materialization they require.

#### Scenario: Direct multisector complete-model composition
- **WHEN** a caller needs a multisector complete graph beyond the built-in recipe
- **THEN** the caller directly composes preparation, materialization, its scientific
  construction function, sampling and compatible output operations with explicit
  arguments, without a runner builder hook or run-specification context

### Requirement: Standard scientific results and explicit exports

Standard and multisector full, prepared-input and postprocessing execution SHALL
return `InversionOutput`. Prepared execution SHALL accept
`run_standard(prepared=..., config=...)` and
`run_multisector(prepared=..., config=...)`, respectively. Nested results SHALL
be `NestedOutput`; ordinary CO2 results SHALL be `Co2Output`. The obsolete
`run_rhime_from_prepared_inputs` entry point, multisector build/result/output
adapters, `RhimeRunSpec` and `RhimeResult` SHALL be unavailable. Complete-model
customization SHALL remain available through explicit scientific composition.

Dates SHALL follow the effective data request or supplied prepared inputs;
retained selectors SHALL come from the prepared inputs. Requested products and
basis, trace and self-contained inversion-output exports SHALL preserve selected
destinations, existing export defaults, formats and overwrite behavior.
An optional export SHALL NOT replace the scientific value to record an export
receipt or overwrite its original source provenance. Requested destinations
SHALL remain distinct from scientific provenance. The `basic` output choice SHALL
NOT compute an inaccessible summary; `basic_output` SHALL remain available from
the returned scientific view independently of file output.

#### Scenario: Scientific result with optional exports

- **WHEN** a standard full, prepared-input or postprocessing invocation completes
- **THEN** it returns an `InversionOutput` borrowing the prepared scientific arrays and trace
- **AND** `output_format="none"` suppresses file writes while retaining that scientific return value
- **AND** requested basis, trace and inversion-output exports preserve selected paths, scientific inputs and existing export defaults

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
