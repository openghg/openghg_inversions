# Proposal

> Status: Draft for review. This spec PR captures the proposed OPE-207 contract;
> it does not implement or complete the issue. Revise these artifacts together
> as the PR discussion settles the design.

## Why

Full, prepared-input, and staged standard/multisector execution independently
compose the same scientific functions, and have already diverged on retained-site
policy. Establish one execution implementation per recipe before OPE-165 adds
linked CO2/O2 staging; consistent configuration and dispatch support that goal.

## What Changes

- Make each recipe own one implementation of its applicable preparation phases.
  For standard/multisector, this includes filtering, retained-site policy, basis
  construction, sensitivities, and labelled assembly. Full, prepared-input, and
  staged routes reuse the same applicable construction, sampling, and
  result/product operations. Full runners visibly sequence these
  ordinary operations without requiring intermediate-file I/O.
- Give standard and multisector separate concrete stage implementations. Share
  scientific functions and genuinely identical mechanics without a combined
  workflow that switches recipes internally. Stages add checkpoint loading,
  authentication, persistence, and execution reporting around recipe operations.
- Preserve distinct merged-data and fully prepared checkpoints, including the
  existing filtered merged artifact produced by staging. Saving or resuming a
  checkpoint must not introduce a second scientific workflow or repeat completed
  transformations.
- Align standard/multisector staging with the full runners' retained-site policy:
  preparation may drop empty sites and continue with aligned retained metadata;
  an empty retained set still fails. This explicitly corrects staged rejection
  of every missing requested site and requires regression coverage and a release
  note. Other intentional route differences must be named and tested.
- Support shared execution with one resolved configuration shape for common
  sampling/output choices and typed recipe options, one authoritative family
  resolver, and a narrow stage interface selected once. Remove stage-only setup
  wrappers and repeated dispatch; diagnosis remains independently owned.
- Require meaningful parity checks across full, merged/prepared-input, and staged
  routes: preparation products, model behavior, sampling policy, and products
  from the same posterior, together with checkpoint independence.
- **BREAKING**: New staged Python setup classes and helper calling conventions
  can change. Preserve installed CLI commands/options, existing configuration
  syntax and meanings, established scientific runner contracts, artifact names,
  schemas, configuration identities, and saved replay behavior.
- Preserve standard/multisector version-2 graph-free replay and historical
  version-1 graph replay. Preserve CO2 version-1 graph-free replay, rejection of
  other sample-manifest versions, and independent affine authentication.
- Reconcile these owners with PRs #773–#776; their namespace migration must carry
  the canonical recipe operations and checkpoint wrappers without retaining
  competing execution implementations.

## Capabilities

### New Capabilities

- `recipe-workflow-contract`: The first OpenSpec contract for consistent recipe
  execution across full, checkpoint, and staged routes, supported by common
  resolution and independent stages. It includes parity, retained-site policy,
  and existing CLI, diagnostic, saved-artifact, and replay guarantees.

### Modified Capabilities

None. The project currently has no durable capability specs for these workflows.

## Impact

The eventual implementation affects full and prepared-input runners, scientific
preparation/construction/output owners, separate standard/multisector stage
implementations, configuration, CLI routing, and their tests/API documentation.
The retained-site correction is the identified staged behavior change. No
equations, configuration formats, dependencies, artifact schema migrations, or
unified `run` CLI are introduced. This PR changes planning artifacts only.

[OPE-207](https://linear.app/openghg-inversions/issue/OPE-207/unify-resolved-recipe-configuration-and-staged-workflow-interfaces)
blocks OPE-165's completion and remaining linked staged integration until the
refactor is implemented and validated. PR #778 has merged; PR #779's linked
prepared-input persistence can proceed independently. Merging this spec PR
does not close OPE-207 or unblock OPE-165.
