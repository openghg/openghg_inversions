# Spec Delta

> Status: Draft behavioral contract for OPE-207; not an implementation claim.

## Purpose

Provide one scientific execution implementation per recipe across full,
prepared-input, and staged routes, with meaningful parity and explicit checkpoint
boundaries. Preserve the installed interfaces and authenticated saved artifacts
used by inversion scientists and downstream workflow callers.

## ADDED Requirements

### Requirement: Shared recipe execution

Each recipe SHALL own canonical operations for its applicable preparation,
construction, sampling, and result/product phases. Full, prepared-input, and
staged entry points SHALL reuse the same applicable operations for that recipe.
Staged execution SHALL add
checkpoint loading, authentication, persistence, and execution reporting without
independently implementing scientific preparation, inference, or product policy.
Full execution SHALL visibly sequence the shared operations using in-memory
handoffs without requiring intermediate-file I/O.

#### Scenario: Full execution without checkpoint persistence
- **WHEN** a supported acquisition-to-product full route runs with checkpoint/cache
  saves disabled and checkpoint writers unavailable
- **THEN** it completes through the recipe's preparation, construction, sampling,
  and requested result/product operations using in-memory handoffs
- **AND** requested final products do not require a temporary staged round trip

#### Scenario: Existing direct prepared CO2 execution
- **WHEN** a direct CO2 runner starts from its coherent prepared handoff and
  returns a trace
- **THEN** it reuses the same applicable construction/sampling operations as staging
- **AND** its established return meaning remains unchanged without adding an
  acquisition phase or requiring staged result products

#### Scenario: Prepared execution shares the inference tail
- **WHEN** supported prepared-input and staged sampling routes receive the same
  valid prepared inputs and equivalent resolved scientific/sampling choices
- **THEN** they use the same recipe construction and sampling operations as full execution
- **AND** they perform no acquisition, filtering, basis, or sensitivity reconstruction

#### Scenario: Shared product construction from saved samples
- **WHEN** ordinary result construction and authenticated replay receive the same
  prepared inputs, posterior, and equivalent scientific output information
- **THEN** they use the same recipe result/product operations
- **AND** replay requires no sampling and follows only the documented graph-recovery policy

### Requirement: Canonical preparation and retained-site policy

Each recipe SHALL have one implementation of its applicable preparation phases.
For standard and multisector this SHALL include filtering, retained-site policy,
basis construction, sensitivities, and labelled assembly.
Standard and multisector routes SHALL permit valid filtering to remove empty
sites and SHALL consistently use the aligned retained observations and metadata.
This explicitly replaces staged rejection of every missing requested site.
An empty retained set SHALL fail before basis construction or inference.
Independent prepared inputs SHALL retain their owning boundary's validation;
retained-site reconciliation SHALL NOT bypass malformed-input rejection.

#### Scenario: Filtering drops one requested site
- **WHEN** the same standard or multisector inputs and filtering choices remove
  an empty requested site while leaving valid observations at other sites
- **THEN** full and staged preparation retain the same observations, site labels,
  averaging periods, and aligned per-site metadata
- **AND** prepared execution uses those retained settings consistently
- **AND** requested configuration remains available separately for provenance

#### Scenario: No retained observations
- **WHEN** preparation removes all usable sites for standard or multisector
- **THEN** full and staged preparation reject the empty retained set before
  basis construction or inference

#### Scenario: Invalid independently supplied prepared data
- **WHEN** a prepared handoff has a malformed layout, invalid required scientific
  inputs, or mismatched authenticated content
- **THEN** the owning boundary rejects it rather than treating it as a valid
  filtering-induced site reduction

### Requirement: Distinct checkpoint boundaries

Saving or resuming supported merged-data and fully prepared checkpoints SHALL
reuse the recipe's scientific operations. Each checkpoint SHALL enter at its
defined scientific phase, preserving the existing data formats, filenames, and
validation/authentication contracts. Pre-filter merged caches and staged
filtered merged data SHALL retain distinct meanings; execution SHALL NOT guess
the phase of an arbitrary merged file or repeat a completed transformation.
This requirement SHALL NOT introduce new installed commands or artifact schemas.

#### Scenario: Resume pre-filter merged data
- **WHEN** a supported route resumes acquired, external, or reloaded pre-filter
  merged data with equivalent preparation choices
- **THEN** it bypasses acquisition and uses the same remaining filtering, basis,
  sensitivity, and assembly operations as full execution
- **AND** it produces equivalent labelled prepared inputs without mutating the
  supplied merged handoff

#### Scenario: Resume a known filtered merged checkpoint
- **WHEN** an explicit supported Python checkpoint boundary resumes the staged filtered
  merged handoff with its required provenance
- **THEN** it uses the same remaining basis, sensitivity, and assembly operations
- **AND** it does not apply the already-completed filters again
- **AND** ordinary merged reloads are not automatically reinterpreted as
  filtered-stage resume

#### Scenario: Resume fully prepared inputs
- **WHEN** valid fully prepared inputs are supplied through a supported execution route
- **THEN** the route starts at the recipe's construction/sampling boundary
- **AND** no earlier preparation phase is executed

### Requirement: Scientific consistency and explicit route differences

Equivalent resolved scientific choices and inputs SHALL produce equivalent
scientific behavior across a recipe's execution routes. Every intentional route
difference SHALL have a named boundary rationale and regression coverage;
historical duplication alone SHALL NOT justify divergence. Scientific input and
output capability decisions SHALL belong to the consuming recipe. Existing
Python-only customization and independently authored input validation SHALL
retain their supported contracts without extending the installed CLI.

#### Scenario: Route policy is explicit
- **WHEN** routes retain different supported customization, input validation,
  defaults, or output destinations
- **THEN** the difference has a documented purpose and a regression check
- **AND** equivalent resolved scientific settings still execute the same recipe policy

#### Scenario: Existing Python scientific customization
- **WHEN** a supported direct Python route receives an existing custom likelihood
  or complete-model callable and its explicit options
- **THEN** it preserves argument forwarding, conflict rejection, numerical input
  ownership, and declared output capabilities
- **AND** it requires no registration, executable-callable serialization, or staged files
- **AND** established runner signatures and return meanings remain compatible

### Requirement: Meaningful cross-route scientific parity

Acceptance SHALL compare scientific preparation, construction, sampling policy,
and products across equivalent full, merged/prepared-input, and staged routes.
Routing or call-order checks alone SHALL NOT establish parity. Comparisons SHALL
exercise representative real scientific operations for standard, multisector,
and the supported ordinary/cached CO2 variants. Independently sampled trajectories
SHALL NOT be required to match exactly.

#### Scenario: Preparation parity on controlled data
- **WHEN** full and staged preparation use the same controlled merged inputs and
  applicable scientific choices
- **THEN** real filtering, basis/sensitivity, and assembly operations produce
  equivalent prepared arrays, coordinates, basis representation, and retained metadata
- **AND** checks cover unequal per-site metadata, a retained-site change, and a
  source-resolved multisector layout
- **AND** borrowed inputs and existing lazy/eager execution boundaries remain intact

#### Scenario: Construction and sampling parity
- **WHEN** equivalent full, prepared-input, and staged routes consume the same
  prepared handoff for one supported recipe/variant
- **THEN** model inputs, roles, deterministic forward calculations, and log
  probability at controlled parameter values agree
- **AND** equivalent sampler choices use the same recipe sampling policy
- **AND** small real-sampling checks establish execution of supported routes

#### Scenario: Product parity through authenticated replay
- **WHEN** one controlled posterior and prepared handoff are processed by ordinary
  construction and authenticated replay for the same supported product request
- **THEN** scientific values, roles, units, labelled dimensions, aggregation, and
  conditional reconstruction meaning agree
- **AND** separate chain/draw axes are preserved
- **AND** differences are limited to documented destinations and execution reporting

### Requirement: Consistent resolved recipe configuration

A recipe SHALL have one authoritative resolution of its scientific,
preparation, sampling, and output choices for its configuration-resolving full,
prepared-input, and staged entry points.
Common sampling and output choices SHALL each have one owner in that
resolution. Family-specific scientific inputs and preparation representations
SHALL retain their distinct meanings. Existing defaults specific to a family
or explicitly supported execution boundary SHALL remain unchanged.

#### Scenario: Equivalent choices across full and staged execution
- **WHEN** supported full and staged entry points resolve the same recipe,
  scientific configuration, explicit sampling choices, and supported output choices
- **THEN** they supply equivalent resolved scientific and sampling values
- **AND** differences are limited to explicitly documented/tested defaults and
  explicit runtime destinations or stage arguments

#### Scenario: Ordinary and cached CO2 remain distinct
- **WHEN** the existing ordinary or cached CO2 variant is selected
- **THEN** resolution retains that variant's scientific likelihood and matched
  sampling policy without substituting the other variant

### Requirement: Existing configuration contract

Resolution SHALL preserve accepted configuration formats, option meanings,
defaults, aliases, override precedence, and route-specific rejection rules.
File-relative paths SHALL retain their existing resolution semantics.
Invalid or inapplicable configuration SHALL fail at its owning boundary
before scientific artifacts are written.

#### Scenario: Configuration source and relative paths
- **WHEN** an installed stage loads an explicit configuration or parameter file
  containing relative artifact paths and supported overrides
- **THEN** paths resolve relative to the original source and overrides retain
  their existing precedence
- **AND** an unrelated ambient `CONFIG_FILE` does not select staged science

#### Scenario: Unsupported options
- **WHEN** a configuration contains an unknown option or an option unsupported
  by the selected recipe and execution route
- **THEN** resolution reports an error rather than silently ignoring it or
  changing the scientific recipe

### Requirement: Common independent stage contract

Recipe implementations SHALL provide a common explicit contract for
preparation, prior-predictive checks, sampling, and postprocessing. Once the
recipe is resolved, the caller SHALL invoke these operations without repeating
family-specific selection for each operation. Selecting a recipe SHALL NOT
require executing another recipe's preparation or scientific workflow.
Standard and multisector SHALL have separately selectable concrete stage owners
that invoke their own canonical recipe operations. Identical mechanics SHALL
retain common ownership where appropriate without replacing distinct scientific
recipes with one workflow controlled by a recipe switch.

#### Scenario: Standard and multisector retain concrete scientific ownership
- **WHEN** standard or multisector staged execution is selected once
- **THEN** subsequent operations invoke that recipe's canonical preparation,
  construction, sampling, and product policy
- **AND** shared mechanics preserve the selected single-source or source-resolved
  scientific layout without dispatching a second complete workflow

#### Scenario: Independent sampling
- **WHEN** sampling receives authenticated prepared inputs and their manifest
- **THEN** it uses the selected recipe's sampling implementation
- **AND** it does not repeat acquisition or preparation

#### Scenario: Independent saved-output production
- **WHEN** postprocessing receives authenticated prepared inputs, a saved
  posterior, and the required manifests
- **THEN** it produces supported requested products without resampling
- **AND** graph construction follows only that family's documented replay policy

#### Scenario: Additional concrete family
- **WHEN** OPE-165 supplies a supported linked CO2/O2 staged implementation
- **THEN** callers use the same stage operation contract for that one joint recipe
- **AND** its unequal channel axes and cross-channel covariance are not replaced
  by two independent tracer workflows

### Requirement: Installed CLI compatibility

Existing installed `prepare`, `prior-predictive`, `sample`, `diagnose`, and
`postprocess` commands SHALL retain their command names, model choices,
arguments, defaults, declared handoff requirements, stdout contract, and exit
policy. The explicitly specified retained-site correction changes staged
acceptance of valid filtering reductions; all other command behavior SHALL remain
compatible. This refactor SHALL NOT introduce the separate unified `run` CLI.

#### Scenario: Existing command sequence
- **WHEN** a supported existing standard, multisector, or ordinary/cached CO2
  command sequence is run with the same arguments and controlled inputs
- **THEN** the same handoffs and requested scientific products are produced
- **AND** artifact filenames, manifest schemas, and scientific labels/units remain compatible
- **AND** newly accepted filtering-induced site reductions follow the common
  retained-site policy rather than introducing a different scientific workflow

### Requirement: Stable scientific configuration identities

For unchanged resolved scientific settings, configuration identity encoding
and hashes SHALL remain compatible with existing saved manifests. The existing
family policy for excluding sampling, output, and transport settings SHALL
remain unchanged. Artifact content identities SHALL continue to authenticate
the numerical handoffs independently.

#### Scenario: Internal configuration representation changes
- **WHEN** an existing scientific configuration is represented by the new
  resolved configuration boundary
- **THEN** its scientific configuration identity matches the pre-refactor identity
- **AND** existing matched saved manifests remain usable

#### Scenario: Replay settings and recorded sampler
- **WHEN** only currently permitted output, transport, or sampling settings change
- **THEN** those changes do not invalidate scientific identity
- **AND** saved-output replay reports the sampler recorded by the sample manifest

#### Scenario: Scientific mismatch
- **WHEN** a scientifically relevant family setting changes or supplied
  numerical artifacts do not match their recorded identities
- **THEN** the handoff is rejected instead of being reused as a matched run

### Requirement: Standard and multisector version-2 replay

Standard and multisector version-2 sample manifests SHALL require an
authenticated saved output binding and SHALL replay without acquisition,
model-input materialization, or model graph construction. Missing, malformed,
altered, escaping, or mismatched bindings SHALL fail before posterior loading
or scientific product writes and SHALL NOT trigger historical graph replay.

#### Scenario: Valid graph-free replay
- **WHEN** either family's version-2 sample manifest binds the exact supplied
  prepared inputs, posterior, and a valid saved output contract
- **THEN** products retain scientific roles, metadata, and separate chain/draw axes
- **AND** replay succeeds when model construction is forbidden

#### Scenario: Invalid output binding
- **WHEN** a binding is missing, its bytes or digest changed, its schema or
  contract is malformed, its artifact path escapes the allowed directory, or
  its prepared-input/posterior identities differ from the sample manifest
- **THEN** validation rejects it before posterior loading or product writes
- **AND** no graph-building fallback is attempted

### Requirement: Genuine historical standard and multisector replay

Genuine version-1 standard and multisector sample manifests without output
bindings SHALL retain their historical graph-building route to recover missing
output roles. Version-1 manifests that advertise output bindings SHALL be
rejected rather than treated as genuine historical artifacts.

#### Scenario: Historical manifest without binding
- **WHEN** a matched historical version-1 standard or multisector manifest has
  no output-binding entries
- **THEN** its established graph-building compatibility route remains available
- **AND** it uses the same recipe construction operation to recover roles,
  followed by the common result/product operations without resampling

#### Scenario: Binding-bearing manifest relabelled version 1
- **WHEN** a manifest advertises a binding in its artifacts or identities but
  declares version 1
- **THEN** it is rejected rather than bypassing binding authentication

### Requirement: CO2 version-1 replay and affine authentication

CO2 SHALL retain graph-free replay for version-1 sample manifests and SHALL
reject other sample-manifest versions before posterior loading or output
destination creation. Optional affine reconstruction SHALL retain independent
content authentication and binding to the prepared handoff and sampling record.

#### Scenario: Supported CO2 saved replay
- **WHEN** matched version-1 CO2 artifacts are replayed
- **THEN** supported products can be produced with ordinary and cached model
  construction forbidden

#### Scenario: Unsupported CO2 sample version
- **WHEN** a CO2 sample manifest declares version 2 with an absent binding,
  malformed binding, or incorrect binding digest
- **THEN** it is rejected before posterior loading or output destination creation

#### Scenario: Affine content or binding mismatch
- **WHEN** a supplied affine artifact is altered, bound to other prepared data,
  or disagrees with the preparation/sampling affine identity
- **THEN** it is rejected before scientific product writes
- **AND** currently supported relocation of identical authenticated content remains valid

### Requirement: Diagnostic and scientific output stability

Readiness and convergence checks SHALL retain their schemas, names, threshold
semantics, and existing `pass`, `fail`, and `unknown` decisions. Scientific
outputs SHALL retain the existing values, roles, dimensions, units, and
conditional reconstruction meaning for the same saved inputs and posterior.
This change SHALL NOT alter equations or expand diagnostic policy.

#### Scenario: Diagnosis without recipe resolution
- **WHEN** diagnosis receives a saved posterior, optional sample manifest, and
  convergence options
- **THEN** it requires no scientific configuration, family setup, or model selection
- **AND** optional authentication and existing diagnostic decisions remain unchanged

#### Scenario: Convergence with unavailable metrics
- **WHEN** a finite metric fails its threshold and another metric is unavailable
- **THEN** the convergence check remains `fail`
- **AND** otherwise unassessable evidence retains its existing `unknown` policy

#### Scenario: Strict scientific gate
- **WHEN** an installed check command uses its existing strict mode
- **THEN** a scientific `fail` retains the existing nonzero exit behavior
- **AND** `unknown` is not newly classified as a failed scientific gate

#### Scenario: Scientific gate and serialization failure
- **WHEN** prior-readiness evidence fails scientifically or its artifact writer fails
- **THEN** the scientific failure emits the existing gate-compatible check
- **AND** a serialization failure remains an error rather than a successful check

#### Scenario: Stable saved-posterior products
- **WHEN** the same controlled prepared inputs and saved posterior are processed
  before and after the internal refactor
- **THEN** supported deterministic products and their scientific interpretation agree
- **AND** no identical stochastic trajectory between separate sampling runs is required
