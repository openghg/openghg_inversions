# Sampled recipe handoff

## Purpose

Preserve completed inference and its scientific output meaning in a saved
handoff used with its matched run context, so compatible products can be created
without another model execution.

## ADDED Requirements

### Requirement: Sampling consumes one prepared handoff and current inference choices

Each supported concrete MCMC workflow SHALL accept one prepared-manifest path,
the current model section, a sampler selection and an explicit sample destination,
and return one absolute sample-manifest path. Required run context MAY be kept
alongside, referenced or explicitly supplied; the preparation record need not be
independently complete. Existing scientific loaders and model boundaries SHALL
check the selected model's prerequisites before materializing model inputs or
sampling, without repeating their checks throughout the workflow. Sampling SHALL
NOT require equality to the preparation's original full configuration or current
output choices. Compatible new priors or likelihood choices SHALL define new inference
without invalidating the prepared checkpoint. Preparation-dependent scientific
constraints SHALL follow the prepared-recipe-handoff contract.
Facts already owned by preparation SHALL be taken from its numerical artifacts
and matched preparation context; conflicting claims in the current model section
SHALL fail rather than replace saved facts.
The new sampling boundary SHALL accept the existing `RhimeSampler` or
`"default"`, with omission equivalent to `"default"`. Each default selection
SHALL create a fresh sampler for that invocation. A supplied sampler's options
SHALL be snapshotted for the invocation without mutating caller-owned options or
requiring a second sampler-options type. `None` SHALL mean absence of a live
sampler and SHALL be invalid at sampling boundaries. Existing direct entry points
SHALL migrate default selection from explicit `None` to `"default"` consistently
with recipe-execution; no adapter SHALL reinterpret `None` as default selection.

#### Scenario: Reuse preparation for a different compatible model
- **WHEN** two sampling invocations use the same valid prepared handoff with
  different compatible prior or likelihood choices and separate destinations
- **THEN** each samples its selected model and returns its own sample-manifest path
- **AND** neither repeats acquisition/preparation nor changes the prepared files

#### Scenario: Selected component cannot consume the saved preparation
- **WHEN** selected model choices require unavailable arrays, incompatible labels,
  units or a different preparation-dependent scientific calculation
- **THEN** sampling reports the incompatibility before model materialization or
  sample saving, rather than reinterpreting or silently preparing the inputs

#### Scenario: Default and explicit sampler selections
- **WHEN** sampling omits its sampler or selects `"default"` on separate invocations
- **THEN** each invocation uses a fresh default sampler
- **AND** supplying a `RhimeSampler` uses an isolated snapshot of its actual options
- **AND** supplying `None` fails with migration guidance in saved and existing direct sampling entry points

#### Scenario: A model section contradicts prepared facts
- **WHEN** a current model section claims species, domain or source/state facts inconsistent with its prepared handoff
- **THEN** sampling rejects the conflict before construction or sampling rather than replacing the preparation's meaning

### Requirement: The sample record contains bounded scientific replay information

A sample handoff SHALL identify its current supported format, recipe and MCMC
inference kind, and reference its preparation and posterior through ordinary
file paths. Actual model choices, sampler provenance, callable identity/arguments
and the family's scientific output meaning SHALL remain available from the
record or matched run context kept alongside, referenced or explicitly supplied.
Standard/multisector SHALL embed the existing `OutputContract` representation in
the record. CO2 SHALL retain its
trace roles and affine associations without requiring that generic
contract representation.
The preparation artifacts and matched run context SHALL preserve the
original run window, retained sites, averaging periods and other preparation facts
needed to construct the result and its outputs, without requiring duplicate
numerical labels or arrays in the sample record. The sample SHALL add only the
family-specific output context and provenance needed from completed inference.
Fixed prior values, their scientific coordinates and model-record references
SHALL follow the [pending model-composition amendment for #872](../../../register-and-persist-prior-parameters/specs/rhime-model-composition/spec.md).
That amendment owns prior-data registration and persistence; this handoff
contract does not duplicate numerical payloads or define another serializer.
Recorded inference choices SHALL describe the completed invocation, not later
configuration defaults. Keeping the run configuration alongside or referencing
it SHALL be permitted; records SHALL NOT depend on a runtime dataclass layout
or duplicate the whole configuration at each stage. Saved scientific output meaning
SHALL be validated independently of whether products are requested during sampling.
Default trace retention SHALL include the scientific quantities needed by the
declared output contract, including standard flux scales `x`. Existing named
deterministic expressions SHALL remain available as those scientific quantities;
internal `_latent` variables alone SHALL NOT substitute for their output roles.
The sampler SHALL reject posterior `sample_kwargs["var_names"]` overrides at its
owning boundary. Intentional posterior-predictive variable selection SHALL remain
supported. Result and inversion-output construction SHALL trust PyMC-produced
traces after supported coordinate restoration and burn operations, rather than
revalidate the same scientific trace. Necessary external-input validation SHALL
remain at its owning loading or construction boundary. General
deterministic reconstruction and compact-storage policy remain with the issues
indexed by [#850](https://github.com/openghg/openghg_inversions/issues/850).
One `load_samples`/`save_samples` owner SHALL decode/encode the referenced
scientific result core, including the preparation generation. Replay orchestration
SHALL consume that decoder rather than reconstruct domain records inline.
The record SHALL NOT require a separate output-binding
sidecar or persistence of live models, samplers or steps. Ordinary saves,
manifest-last completion and dependencies SHALL follow the prepared-recipe-handoff
destination and file-reference rules. The workflow SHALL assume ordinary trusted
use of a matched run and SHALL NOT require mutation-during-run detection,
transitive authentication or rehashing dependencies before and after inference.

#### Scenario: Record an invocation with retained observations and overrides
- **WHEN** sampling uses a retained subset and invocation-specific sampler choices
- **THEN** the saved record preserves the original run window, actual retained
  metadata through its preparation, and the selected model choices and sampler
  settings used for that invocation in the record or matched run context
- **AND** the returned manifest path identifies the completed record without
  adding a self-referential invocation path to its saved contents

#### Scenario: Sampling references shared prepared data
- **WHEN** sampling uses a completed preparation in a shared cache
- **THEN** the sample record references it through an ordinary relative or explicit absolute path
- **AND** sampling does not implicitly copy large prepared dependencies or make
  the sample directory independently self-contained

#### Scenario: Retain scientific state for output replay

- **WHEN** the standard model constructs `x` as a deterministic transform of `x_latent`
- **THEN** default sampling retains `x` for postprocessing
- **AND** an explicit posterior trace selection keeping only `x_latent` and `mu` is rejected at the sampler boundary
- **AND** replay cannot write an apparently valid inversion output with an absent flux-scale role

#### Scenario: Retain the shared multisector physical state

- **WHEN** multisector sampling completes with the shared flux-state formulation
- **THEN** its saved contract declares the complete `x` as `flux_scale` and default
  trace retention includes its gathered source/local-state labels
- **AND** per-source products select from that state using the recorded source
  associations without requiring per-sector full-state aliases

### Requirement: Replay uses existing scientific validation

Replay SHALL read the current supported record and output-contract formats and
report missing required files or unsupported formats clearly. Existing scientific
loaders SHALL perform their necessary external-input checks on roles, labels,
units and reconstruction associations without constructing a model. Result and
product construction SHALL NOT repeat validation of the same scientific trace.
The workflow SHALL trust values validated by those owners
rather than repeat their checks at every stage. It SHALL NOT require artifact
digests, tamper detection, schema-downgrade handling or 0.7 compatibility paths.
Missing required scientific context SHALL fail clearly without graph recovery or
implicit preparation.

#### Scenario: A saved numerical dependency is missing
- **WHEN** a required dependency cannot be read
- **THEN** replay reports the missing file before product writes

#### Scenario: An envelope or output association is invalid
- **WHEN** replay receives an unsupported format, incomplete scientific context,
  invalid output contract or inconsistent prepared/posterior association
- **THEN** its owning reader or scientific output component reports the problem
  without constructing a recovery graph or creating product artifacts

#### Scenario: Loaded data contradict saved scientific output meaning
- **WHEN** loaded numerical data have roles, labels, units or reconstruction
  associations incompatible with the saved scientific output contract
- **THEN** replay reports the scientific incompatibility after loading and before
  product writes, without reconstructing a model

#### Scenario: Replay an existing current-format per-sector trace

- **WHEN** a supported current-format built-in sample record declares
  `x_<suffix>` per-sector scaling variables and lacks an explicit canonical
  `flux_scale` role
- **THEN** the existing saved variable names and source associations remain
  readable without reconstructing a model or changing the preparation generation
- **AND** a record declaring a canonical role cannot recover a missing or
  incompatible canonical state by silently selecting per-sector variables
- **AND** this reader compatibility does not restore historical staged envelopes
  or create alias variables in new sampled graphs

### Requirement: Postprocessing consumes one sampled handoff and one output policy

Each supported concrete workflow SHALL accept a sample-manifest path and a
current output policy, including destination, naming and supported product choices,
and return its result with requested products and output metadata. Required run
context MAY be kept alongside, referenced or explicitly supplied. Postprocessing
SHALL use the saved model/run facts and sampling provenance from that matched
context rather than require every fact to be embedded in the sample record. It
SHALL NOT construct a graph, materialize model inputs, repeat preparation or sample again.
Replay SHALL NOT import or execute recorded construction or likelihood callables.
Sampler provenance SHALL be a small description of completed sampling consumed
by the shared output owner. It SHALL NOT be a scientific acceptance gate or be
replaced by fresh sampler defaults. Missing descriptive fields SHALL remain
unavailable rather than be invented. A shared pure metadata operation SHALL
normalize available `draws`, `burn`, `tune`, `chains` and `nuts_sampler` values
without constructing a `RhimeSampler`. Missing or invalid descriptions SHALL
become `{}`, while valid partial descriptions SHALL remain partial. No separate
sampler settings type or metadata strict flag SHALL be required. Replay results
SHALL retain saved sampler provenance as metadata without a live sampler field. Replay SHALL
use the preparation's retained flux rather than resolve current native
flux or load historical flux from a separate basis artifact.
Current output choices SHALL NOT replace recorded scientific or inference choices.
Unsupported products or incompatible reporting inputs SHALL fail before product
writes. Scientific result/product construction SHALL share its concrete owner
with the corresponding in-memory route.

#### Scenario: Create different compatible products in a later process
- **WHEN** a later process receives a valid sample handoff with its matched run
  context and compatible current output choices with a new destination or naming policy
- **THEN** it returns the requested products using recorded inference provenance
  and the same saved scientific output meaning without model execution
- **AND** its result has no live sampler field and its output metadata describes the completed sampling invocation without resampling or selecting new defaults

#### Scenario: Requested output is unsupported
- **WHEN** current output choices request a format not supported by the saved
  output contract or reporting inputs incompatible with its numerical data
- **THEN** postprocessing fails before product writes rather than changing the
  sampled model or inferring missing output meaning from a new graph

### Requirement: CO2 replay preserves coherent affine reconstruction meaning

CO2 handoffs SHALL retain the family-specific information needed for ordinary or
matched cached inference and its supported concentration/flux products. Any
affine companion used by replay SHALL be associated with the same
prepared inputs used by sampling. Replay SHALL preserve its declared conditional
reconstruction and uncertainty scope; it SHALL NOT reinterpret a coherent affine
CO2 result as a standard multiplicative inversion product. The existing CO2
affine loader SHALL own the numerical association and compatibility checks;
stages SHALL NOT add a second authentication or validation layer around it.

#### Scenario: Request products requiring an affine companion
- **WHEN** compatible CO2 output choices require native or country flux products
- **THEN** the existing affine loader checks the required companion against the
  prepared inputs and preserves its conditional reconstruction meaning
- **AND** a missing or differently bound companion fails before product writes
