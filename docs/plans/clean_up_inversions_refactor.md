# Clean Up Inversions Refactor for PR #434 / Issue #400

This note tracks the near-term refactor sequence for moving from the legacy
`fixedbasisMCMC` path to the modern `run_rhime` pipeline while keeping each PR
scoped enough to review.

## Goal for PR #434 / Issue #400

Switch over from `fixedbasisMCMC` to `run_rhime` as quickly as possible.
Maintaining both paths is slowing development of the modern inversion pipeline,
and `fixedbasisMCMC` is now radically different from the currently released
version, which partially negates the benefit of keeping it as the main route.

## Deferred issue sequence

- #400: keep for specs/terminology. `RhimePreparedInputs` should consume these
  specs later, but changing that prepared-input contract is not part of #400.
- #383: revise toward "postprocessing consumes `BasisFunctions`, no `fp_data`
  in modern path." Legacy reconstruction from `fp_data[".basis"]` should remain
  only in compatibility adapters.
- #401: use to separate `LegacyInversionOutput` from modern `InversionOutput`.
- #429: keep as operator-backed output/postprocessing, but make it depend on
  the preparation split.
- #416: should become active sooner; classify/deprecate `fixedbasisMCMC`,
  `inferpymc`, `inferpymc_postprocessouts`, and hbmcmc helpers.
- #415: serializable RHIME bundle should serialize `RhimePreparedInputs`
  artifacts, not `fp_data`.

Prep split note: #431 is the dependency that should make `RhimePreparedInputs`
consume RHIME specs and keep `fp_data`, `fp_all`, flat `.basis`, materialised
basis arrays, and optional basis-object side channels out of the modern
contract.

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

Track the future split under #431. That work should break preparation into
smaller contracts for data gathering, filtering, basis functions, and inversion
input construction. A future `RhimeDataSpec` or `RequiredDataPlan` can then be
introduced at the right boundary, with `sigma_freq`, `bc_freq`, and related
model-input concerns moving closer to the model components that consume them.

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

## Deferred Issue #383 / Issue #429 output boundary

- #383 remains the boundary for modern postprocessing consuming retained
  `BasisFunctions` directly. Until that lands, runner-local basis/flux
  materialisation is a temporary adapter for the current `InversionOutput`
  boundary.
- #429 remains the boundary for operator-backed output and postprocessing.
  PR #434 should only mark runner-local basis materialisation and multisector
  output diagnostics as transitional, not move the output contract.
