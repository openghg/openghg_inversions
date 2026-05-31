# Clean Up Inversions Refactor

This note tracks the near-term refactor sequence for moving from the legacy
`fixedbasisMCMC` path to the modern `run_rhime` pipeline while keeping each PR
scoped enough to review.

## Goal

Switch over from `fixedbasisMCMC` to `run_rhime` as quickly as possible.
Maintaining both paths is slowing development of the modern inversion pipeline,
and `fixedbasisMCMC` is now radically different from the currently released
version, which partially negates the benefit of keeping it as the main route.

## PR sequence

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

## Recently completed

- 2026-05-31: #400 / PR #434 added RHIME specs and terminology, normalized
  `flux_sources` as the modern public key, kept legacy `emissions_name` as a
  compatibility alias, and added `sector_sources` so model sector labels can
  differ from OpenGHG flux `source` values. Follow-up: #431 should make
  `RhimePreparedInputs` consume specs directly.
- 2026-05-31: #400 / PR #434 sequestered RHIME parameter/config parsing in a
  private `_rhime_params` module, added simple integer coercion before spec
  construction, and marked runner-local basis materialisation and multi-sector
  diagnostics as transitional output shims.
- 2026-05-31: #383 / PR #412 merged. Modern postprocessing now accepts
  `BasisFunctions`; remaining `fp_data` usage is limited to the legacy
  compatibility layer. Follow-up: #418 to remove redundant adapter.

## Current PR cleanup notes

- Keep the top-level `rhime.py` public-facing: specs, result objects, and
  `run_rhime` / `run_rhime_multisector` orchestration should remain visible.
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
