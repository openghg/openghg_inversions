# Clean Up Inversions Refactor

This note tracks the near-term refactor sequence for moving from the legacy
`fixedbasisMCMC` path to the modern `run_rhime` pipeline.

## Goal

Switch over from `fixedbasisMCMC` to `run_rhime` as quickly as possible.
Maintaining both paths is slowing development of the modern inversion pipeline,
and `fixedbasisMCMC` is now radically different from the currently released
version, which partially negates the benefit of keeping it as the main route.

## Issue sequence

- [x] #431:  move `RhimePreparedInputs` toward consuming RHIME specs, keeping `fp_data`,
  `fp_all`, flat `.basis`, materialised basis arrays, and optional basis-object
   side channels out of the modern contract.
- [x] #400: keep for specs/terminology. `RhimePreparedInputs` should consume these
  specs later, but changing that prepared-input contract is not part of #400.
- [x] #401: use to separate `LegacyInversionOutput` from modern `InversionOutput`.
- [x] #435: migrate postprocessing consumers from `LegacyInversionOutput` to the
  modern `InversionOutput`; opened as the follow-up from #401.
- [ ] #383: revise toward "postprocessing consumes `BasisFunctions`, no `fp_data`
  in modern path." Standard postprocessing should consume modern
  `InversionOutput` directly; old-format HBMCMC output may remain only as a
  dataset formatter over modern postprocessing results.
- [ ] #429: keep as operator-backed output/postprocessing, but make it depend on
  the preparation split.
- [ ] #416: should become active sooner; classify/deprecate `fixedbasisMCMC`,
  `inferpymc`, `inferpymc_postprocessouts`, and hbmcmc helpers.
- [ ] #415: serializable RHIME bundle should serialize `RhimePreparedInputs`
  artifacts, not `fp_data`.
- [ ] #405: complete sector-aware RHIME outputs and PARIS-compatible total outputs.
  PR #436 makes multisector runs carry modern `InversionOutput`, but sector
  diagnostics and PARIS totals are still transitional.

# Issue #400 model specs and terminology

## Completed in PR #434 / Issue #400

- 2026-05-31: #400 / PR #434 added RHIME specs and terminology, normalized
  `flux_sources` as the modern public key, kept legacy `emissions_name` as a
  compatibility alias, and added `sector_sources` so model sector labels can
  differ from OpenGHG flux `source` values. Follow-up: #431 should make
  `RhimePreparedInputs` consume specs directly.
- 2026-05-31: #400 / PR #434 moved RHIME parameter/config parsing into the
  `openghg_inversions.rhime.params` package module, added simple integer
  coercion before spec construction, and marked runner-local basis
  materialisation and multi-sector diagnostics as transitional output shims.
- 2026-05-31: #400 / PR #434 moved RHIME orchestration into
  `openghg_inversions.rhime.runner`, kept the package `__init__` as the public
  re-export surface, kept lightweight model specs with the RHIME builders in
  `openghg_inversions.models.rhime`, kept `RhimeRunSpec` as run metadata, and
  carried the executable `RhimeSampler` on runner setup/results.
- 2026-05-31: #400 / PR #434 moved RHIME setup construction into
  `openghg_inversions.rhime.params` and output adapters into
  `openghg_inversions.rhime.outputs`, leaving `runner.py` focused on
  preparation, model build, sampling, and output dispatch.

## Recorded decisions for PR #434 / Issue #400

- Keep one public concrete `RhimeModelSpec` with
  `sectors: tuple[SectorSpec, ...]` for both single-sector and multisector
  RHIME. Do not split the public API into separate `SingleSectorRhimeModelSpec`
  and `MultisectorRhimeModelSpec` dataclasses in PR #434.
- The current invariant difference is sector count and builder choice. The
  shared metadata is otherwise the same: `species`, `domain`, sector/source
  mappings, priors, boundary-condition settings, model-error settings, offset
  settings, and likelihood options.
- Revisit separate dataclasses only when model families gain genuinely
  different invariants, such as non-shared bases, inner/outer region
  components, multi-species or tracer ports, sector-specific likelihood/error
  terms, or distinct persisted output schemas.
- Represent future richer model composition with a `SemanticModel` IR rather
  than growing branching logic around concrete RHIME runner flags.

## Current PR #434 / Issue #400 cleanup notes

- Keep the `openghg_inversions.rhime` package public-facing: specs, result
  objects, and `run_rhime` / `run_rhime_multisector` orchestration should
  remain visible.
- Sequester RHIME config parsing, alias handling, simple type coercion, and
  validation into a RHIME-specific parser module rather than growing the
  top-level runner module.
- Mark transitional code explicitly where #400 intentionally stops short:
  multi-sector outputs remain a direct diagnostic special case until
  postprocessing supports sectors, and output reconstruction still materialises
  a flat basis until postprocessing consumes `BasisFunctions` directly.
- Do not implement the future YAML/schema/semantic-IR design in #400. Use it to
  guide boundaries: INI compatibility should parse into the same Python spec
  model that future YAML and Python DSL frontends will target.
- Do not add Pydantic, attrs/cattrs, or YAML/schema dependencies in #400.
  Keep the public spec dataclasses as stdlib dataclasses, and reserve schema
  validation for the later YAML/semantic-IR work.

# Issue #401 Introduce `LegacyInversionOutput` and keep `InversionOutput` modern

## Completed in PR #436 / Issue #401

- 2026-05-31: #401 / PR #436 renamed the current postprocessing carrier to
  `LegacyInversionOutput` and reserved `InversionOutput` for the modern RHIME
  result contract.
- 2026-05-31: #401 / PR #436 added modern `InversionOutput` save/load support
  for `InferenceData`, canonical `inv_inputs`, retained `BasisFunctions`, run
  metadata, model metadata, output metadata, and provenance.
- 2026-05-31: #401 / PR #436 routed fixedbasis output explicitly:
  `output_format="hbmcmc"` remains the only path that calls
  `inferpymc_postprocessouts`, while `hbmcmc_postprocessing`, `basic`, `paris`,
  and `inv_out` use `LegacyInversionOutput`.
- 2026-05-31: #401 / PR #436 routed standard `run_rhime` output explicitly:
  `inv_out` saves and returns modern `InversionOutput`, while `basic` and
  `paris` build a temporary `LegacyInversionOutput` adapter for the existing
  postprocessing functions.
- 2026-05-31: #401 / PR #436 made non-`none` multisector RHIME output bundles
  carry the same modern `InversionOutput` contract while leaving their
  sector-flux diagnostics as a transitional output product.
- 2026-05-31: #401 / PR #436 opened #435 to migrate postprocessing consumers to
  modern `InversionOutput`.
- 2026-05-31: #401 / PR #436 hardened modern `InversionOutput` load handling
  for serialized MultiIndex metadata by accepting bytes attrs and ignoring
  malformed index metadata instead of failing load.

## Recorded decisions for PR #436 / Issue #401

- The true legacy output remains `inferpymc_postprocessouts` in
  `hbmcmc.inversion_pymc`. It should stay legacy/fixedbasis-only and should not
  appear in modern RHIME output construction.
- `LegacyInversionOutput` is the legacy-shaped postprocessing carrier, not the
  true legacy `inferpymc_postprocessouts` output object.
- Modern `InversionOutput` keeps retained `BasisFunctions` and canonical
  `inv_inputs`; it does not materialise flat basis/flux arrays for RHIME
  `inv_out` output.
- `LegacyInversionOutput.from_modern_output(...)` is the only temporary bridge
  for standard RHIME `basic` and `paris` output modes in #401.
- Multisector RHIME should also expose modern `InversionOutput` for `inv_out`.
  The remaining multisector gap is not the durable run artifact but the
  sector-aware postprocessing/PARIS output contract, which belongs to #405.
- #401 does not migrate `postprocessing` consumers to modern `InversionOutput`.
  That is deferred to #435.

## Current PR #436 / Issue #401 cleanup notes

- Keep `inferpymc_postprocessouts` isolated to
  `fixedbasisMCMC(output_format="hbmcmc")`.
- Keep current `postprocessing` functions typed against
  `LegacyInversionOutput` until #435 changes their input contract.
- Keep modern RHIME save/load serialization focused on durable artifacts:
  `InferenceData`, `inv_inputs`, `BasisFunctions`, metadata, and provenance.
  Do not add fixedbasis `fp_data` or materialised `.basis` side channels to the
  modern object.
- The #401 adapter still materialised basis/flux arrays for `basic` and
  `paris`. #435 removes the RHIME use of that adapter; #383/#429 should remove
  the remaining flat-basis assumptions from postprocessing itself.

## Review findings for PR #436 / Issue #401

- Architecture review found the standard #401 split sound: modern
  `InversionOutput` owns `InferenceData`, `inv_inputs`, `BasisFunctions`, and
  metadata; `LegacyInversionOutput` remains the temporary postprocessing
  carrier; and `inferpymc_postprocessouts` stays fixedbasis-only.
- Architecture review flagged multisector `inv_out` as the only boundary risk:
  `run_rhime_multisector` originally returned sector diagnostics but did not
  set `result.inv_out`. PR #436 now builds the modern `InversionOutput` for
  non-`none` multisector bundles as well.
- The multisector output product itself remains transitional. #405 should turn
  the current `sector_flux_diagnostics` product into documented sector-aware
  outputs and PARIS-compatible total outputs.
- Copilot review found that serialized MultiIndex metadata could be bytes or
  malformed. PR #436 now decodes bytes and skips malformed records safely
  before applying `set_index`.

# Issue #435 postprocessing consumer migration target

#435 should move `make_outputs`, `make_paris_outputs`, `countries`,
`diagnostics`, and `legacy_outputs` off `LegacyInversionOutput` where possible.
`LegacyInversionOutput` should become an adapter over modern output, while
`legacy_outputs` remains isolated to fixedbasis/HBMCMC compatibility output.

This work links #401, #383, #429, #416, and #381. It should remove the RHIME
`basic`/`paris` dependency on `LegacyInversionOutput.from_modern_output(...)`
rather than expanding that adapter.

## Completed in Current PR / Issue #435

- 2026-05-31: #435 / current PR added a transitional
  `StandardPostprocessingOutput` contract and `ModernPostprocessingOutput`
  view so standard single-sector postprocessing consumers can derive trace,
  model-data, basis, and flux conveniences from modern `InversionOutput`.
- 2026-05-31: #435 / current PR changed the modern view to carry observation
  inputs as one dataset rather than split `obs`/`obs_err`/optional satellite
  fields. The split fields remain only on `LegacyInversionOutput` for
  fixedbasis and explicit compatibility callers.
- 2026-05-31: #435 / current PR restores model-managed coordinates onto RHIME
  `InferenceData` in `RhimeSampler.sample(...)` using the `models.coords`
  `CoordRegistry`, after predictive groups are attached.
- 2026-05-31: #435 / current PR routed standard `run_rhime`
  `output_format="basic"` and `output_format="paris"` directly through modern
  `InversionOutput`, removing the RHIME dependency on
  `LegacyInversionOutput.from_modern_output(...)`.
- 2026-05-31: #435 / current PR updated `make_outputs`,
  `make_paris_outputs`, `countries`, and `diagnostics` to consume the narrow
  postprocessing contract while continuing to accept fixedbasis
  `LegacyInversionOutput`.
- 2026-05-31: #435 / current PR kept `legacy_outputs` explicitly
  `LegacyInversionOutput`-only because it formats fixedbasis/HBMCMC
  compatibility arrays and still requires legacy `mcmc_results`, sigma
  indexes, and sensitivity matrices.

## Recorded decisions for Current PR / Issue #435

- Use a narrow standard single-sector helper/view layer rather than making
  modern `InversionOutput` mimic every legacy attribute or splitting each
  postprocessing function into many smaller argument bundles.
- Treat `StandardPostprocessingOutput` as transitional. It must not become the
  durable architecture for multisector, multi-species, or inner/outer-domain
  outputs; those should use smaller product-specific contracts.
- Keep `LegacyInversionOutput.from_modern_output(...)` as an explicit
  compatibility adapter for callers that request a legacy-shaped carrier, but
  do not use it on standard RHIME `basic` or `paris` paths.
- Do not move `inferpymc_postprocessouts` into any RHIME path.
- Do not remove `inferpymc_postprocessouts` in #435 because it is still the
  default fixedbasis `output_format="hbmcmc"` product. Remove it only after
  fixedbasis itself is retired.
- Defer operator-backed flux/country reconstruction to #383/#429. This PR
  still derives the postprocessing basis matrix through retained
  `BasisFunctions` only to satisfy the current single-sector output contract.
- Defer sector-aware PARIS totals and sector diagnostics to #405 rather than
  broadening the standard single-sector postprocessing contract in this PR.
- Remove `fixedbasisMCMC` as soon as `run_rhime` can replace it for production
  scripts. The next compatibility step should add a shim in
  `hbmcmc/run_hbmcmc.py` so old script invocations and `.ini` config files
  translate to `run_rhime`, then deprecate and remove the fixedbasis path.

## Recommended next PR after Current PR / Issue #435

#435 defines the postprocessing consumer contract around modern
`InversionOutput` and removes the standard RHIME `basic`/`paris` adapter
dependency. That creates the right place for #383 to replace legacy flat-basis
assumptions with retained `BasisFunctions` in flux/country computations.

After #435, use #383 for the first operator-backed postprocessing slice:
prefer retained `BasisFunctions.operator.basis_matrix` where available and keep
legacy basis reconstruction as fallback. #429 should then make the
operator-backed path primary and define the deprecation policy for legacy
basis reconstruction.

# Issue #383 Use retained basis operators in postprocessing

## Completed in Current PR / Issue #383

- 2026-05-31: #383 / current PR added private operator-backed postprocessing
  helpers for standard single-sector flux and country products.
- 2026-05-31: #383 / current PR routed modern `make_flux_outputs`, country
  totals, and PARIS flux outputs through retained `BasisFunctions` /
  `BasisOperator` data when a modern `InversionOutput` is supplied.
- 2026-06-01: #383 / current PR moved fixedbasis `basic`, `paris`,
  `hbmcmc_postprocessing`, `inv_out`, and saved inversion-output handling onto
  modern `InversionOutput` backed by retained `BasisFunctions`, while keeping
  the true legacy `hbmcmc` formatter isolated.
- 2026-06-01: #383 / current PR removed the transitional
  `StandardPostprocessingOutput` / `PostprocessingInput` /
  `ModernPostprocessingOutput` layer. Standard postprocessing helpers now take
  modern `InversionOutput` directly and use its existing `trace`,
  `inv_inputs`, `BasisFunctions`, and metadata fields.
- 2026-06-01: #383 / current PR removed `LegacyInversionOutput`,
  `make_inv_out_for_fixed_basis_mcmc(...)`, and the old RHIME-output
  reprocessing helpers. The remaining old-format `hbmcmc_postprocessing`
  product is a formatter over modern `InversionOutput`, not a separate
  inversion-output carrier.

## Recorded decisions for Current PR / Issue #383

- Keep operator-backed flux/country reconstruction as internal postprocessing
  implementation detail for now. The public input is the modern
  `InversionOutput`; do not add legacy convenience properties or formatter
  methods to `InversionOutput` itself.
- Treat saved legacy inversion-output artifacts and old RHIME-output
  reprocessing as disposable debugging compatibility. New postprocessing starts
  from modern `InversionOutput` or reruns the model to create one.
- Modern RHIME `basic`, fixedbasis `basic`, fixedbasis `paris`, countries, and
  PARIS flux/country paths must use retained `BasisFunctions` rather than
  modern-to-legacy adapters or `fp_data`.
- Leave sector-aware PARIS totals and multisector diagnostics to #405, and
  leave default/deprecation policy for operator-backed reconstruction to #429.

Track multisector output work under #405. It can proceed after the modern
postprocessing input contract is clearer, or in parallel if it stays limited to
sector diagnostics and PARIS-compatible total outputs.

Track fixedbasis removal under #416. The first slice should be a
`run_hbmcmc.py` compatibility shim that translates legacy config names such as
`outputpath`, `outputname`, `nit`, and `nchain` to the `run_rhime` API, while
raising targeted errors for fixedbasis-only output formats. Once old scripts
can run through `run_rhime`, remove `fixedbasisMCMC` and then remove
`inferpymc_postprocessouts`.

# Issue #416 Fixedbasis transition and legacy adapter removal

Target this transition for the immediate cleanup window starting 2026-06-01.
The current priority is to complete the move to the modern path before
multisector postprocessing, because keeping the changing fixedbasis pathway and
the modern pathway side by side is obscuring the architecture and slowing
review.

## Required short-term behavior

- `fixedbasisMCMC` may remain as a compatibility entrypoint while #416 is
  active, but it should keep only the operational contract that current users
  need: same inputs in, successful run, and matching `output_format="paris"`
  products.
- Saved legacy inversion-output compatibility and direct legacy carrier return
  behavior are not strategic compatibility goals. #383 removes that carrier
  once fixedbasis postprocessing can construct modern `InversionOutput`.
- `inferpymc_postprocessouts` should not remain the compatibility target.
  `postprocessing/legacy_outputs.py` exists so the old-format dataset can be
  produced from the modern postprocessing path by renaming variables and
  arranging dimensions, then `inferpymc_postprocessouts` can be deleted.

## Preferred #416 implementation direction

- Done in #383: pipe retained `BasisFunctions` through `FixedBasisPreparedData`
  for fixedbasis output modes after data preparation, construct modern
  `InversionOutput` from canonical `inv_inputs`, retained
  `basis_objects["emissions"]`, the `InferenceData` trace, and metadata, and
  route fixedbasis `basic`, `paris`, `hbmcmc_postprocessing`, `inv_out`, and
  saved inversion output through that modern carrier.
- Keep any old-format `hbmcmc_postprocessing` output as a formatter over modern
  postprocessing results, not as a reason to keep the old inversion output
  carrier.
- Done in #383: remove the transitional postprocessing protocol/view layer and
  require standard postprocessing helpers to consume `InversionOutput` directly.

## Aggressive cleanup after #416 lands and passes parity tests

- Done in #383: remove `LegacyInversionOutput.from_modern_output(...)`,
  `make_inv_out_for_fixed_basis_mcmc(...)`, `make_inv_out_from_rhime_outputs(...)`,
  and `make_paris_flux_outputs_from_rhime(...)`.
- Done in #383: remove flat-basis fallback branches from modern flux/country
  postprocessing helpers.
- Remove `inferpymc_postprocessouts` after old-format output parity is covered
  by `postprocessing/legacy_outputs.py` using modern postprocessing inputs.
- Remove `fixedbasisMCMC` once the #416 compatibility shim can run current
  fixedbasis-style scripts through `run_rhime`.

## Tests needed before removal

- A focused fixedbasis compatibility test that proves the same fixedbasis input
  still runs and produces PARIS flux/concentration outputs matching the current
  behavior.
- Done in #383: guard tests prove fixedbasis `basic`, `paris`, `inv_out`, and
  saved inversion-output paths construct/pass modern `InversionOutput`, not
  `LegacyInversionOutput`.
- Done in #383: old-format `hbmcmc_postprocessing` smoke tests cover the
  dataset shape and variable names using modern `InversionOutput` inputs.
- Done in #383: standard RHIME `basic` and `paris` dispatch tests assert the
  output helpers receive modern `InversionOutput` directly.

## Follow-up operator work for #429

- Consider adding a `BasisOperator.project(...)` or similar reduction API for
  grid-to-state products, based on the prototype
  `~/Documents/inversions/src/inversions/basis_functions.py`. The prototype's
  `project()` method covers weighted and normalised reductions that would make
  postprocessing country totals, region means, and uncertainty reductions less
  dependent on ad hoc `sparse_xr_dot` calls.
- Keep this as #429 operator cleanup unless it is needed to remove fixedbasis
  legacy carriers in #416.

# Deferred 

## Deferred SemanticModel plan

PR #434 / Issue #400 keeps the production fast path on `RhimeModelSpec`,
`SectorSpec`, and the current RHIME preparation/builders. That path should stay
small and concrete for standard and multisector RHIME. The near-term dependency
direction is `openghg_inversions.rhime` -> `openghg_inversions.models`, so the
RHIME-specific model specs live beside the concrete RHIME model builders.

The deferred architecture is a separate `SemanticModel` IR:

```text
RhimeModelSpec/config
-> SemanticModel
-> required-data plan
-> PreparedComponentData
-> backend compiler
```

Future TOML/YAML configs and Python DSL frontends should normalize to the same
IR. Compile-time choices such as multisector `loop_sum` versus `stacked_dot`
belong in a future `CompilationPlan`, not in PR #434.

Once that abstraction exists, the runner/pipeline can pass a concrete
`SemanticModel` or backend-specific concrete spec to model builders through an
abstract model-spec contract. That later dependency inversion is the right
place for model builders to depend on abstract specs rather than RHIME-specific
dataclasses.

## Deferred sampler extensions

PR #434 / Issue #400 should keep `RhimeSampler` focused on the current PyMC
MCMC path: `pm.sample`, post-sampling burn slicing, and optional prior/posterior
predictive groups. Follow-up sampler variants can add quick MAP checks,
prior-predictive-only validation runs, variational inference, and custom PyMC
step methods without reintroducing runner-level sampling branches.

## Deferred Issue #431 data-preparation spec

PR #434 / Issue #400 should not introduce `RhimeDataSpec`. The current
`prepare_rhime_inputs` signature still mixes data gathering, filtering, basis
function construction, and `make_inv_inputs` concerns, so a one-piece data spec
would mostly mirror the INI template rather than clarify the architecture.

#431 was closed by #433. Any remaining preparation split should break
preparation into smaller contracts for data gathering, filtering, basis
functions, and inversion input construction. A future `RhimeDataSpec` or
`RequiredDataPlan` can then be introduced at the right boundary, with
`sigma_freq`, `bc_freq`, and related model-input concerns moving closer to the
model components that consume them.

## Deferred Issue #383 / Issue #429 output boundary

- #383 landed the first modern postprocessing slice that consumes retained
  `BasisFunctions` for standard flux/country products while preserving flat
  legacy fallback.
- #429 remains the boundary for operator-backed output and postprocessing.
  It should make the operator-backed path primary, add durable reconstruction
  metadata, and define the deprecation policy for legacy basis reconstruction.
