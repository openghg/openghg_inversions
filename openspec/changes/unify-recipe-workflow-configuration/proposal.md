# Proposal

> Status: Draft for review. This spec PR captures the proposed OPE-207 contract;
> it does not implement or complete the issue. Revise these artifacts together
> as the PR discussion settles the design.

## Why

The newly released staged workflow repeatedly selects families and exposes
incompatible configuration boundaries: standard/multisector use
`RhimeRunnerSetup`, while CO2 stages add output resolution around
`Co2RunSetup` in `Co2StageSetup`. Resolve that ownership disparity before
OPE-165 adds linked CO2/O2 staging, without delaying its prepared-input
persistence slice.

## What Changes

- Establish one resolved recipe configuration shape for common sampling and
  output policy plus typed, family-owned scientific/preparation options. Full
  and staged execution reuse the same family resolution implementation.
- Select a family once and expose a narrow, explicit staged execution contract.
  Keep concrete scientific workflows procedural and diagnosis independently
  owned; remove duplicate stage-only configuration wrappers and repeated
  per-operation family selection.
- **BREAKING**: New staged Python setup classes and helper calling conventions
  can change. Preserve installed CLI commands/options, existing configuration
  syntax and meanings, established scientific runner contracts, artifact names,
  schemas, configuration identities, and saved replay behavior.
- Preserve standard/multisector version-2 graph-free replay and historical
  version-1 graph replay. Preserve CO2 version-1 graph-free replay, rejection of
  other sample-manifest versions, and independent affine authentication.
- Reconcile these owners with PRs #773–#776; a namespace move must carry the
  resolved boundaries rather than restore the older monolithic stage module.

## Capabilities

### New Capabilities

- `recipe-workflow-contract`: The first OpenSpec contract for consistent recipe
  resolution and independent installed stages, including existing CLI,
  diagnostic, saved-artifact, and replay guarantees. It records those
  compatibility obligations without introducing another CLI or model family.

### Modified Capabilities

None. The project currently has no durable capability specs for these workflows.

## Impact

The eventual implementation affects recipe configuration owners, staged family
entry points, CLI routing, and their tests/API documentation. It introduces no
scientific equation, configuration format, dependency, schema migration, or
unified `run` CLI change. This PR changes OpenSpec planning artifacts only.

[OPE-207](https://linear.app/openghg-inversions/issue/OPE-207/unify-resolved-recipe-configuration-and-staged-workflow-interfaces)
blocks OPE-165's completion and remaining linked staged integration until the
refactor is implemented and validated. PR #778 has merged; PR #779's linked
prepared-input persistence can proceed independently. Merging this spec PR
does not close OPE-207 or unblock OPE-165.
