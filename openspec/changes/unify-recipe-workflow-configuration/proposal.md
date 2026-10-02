# Proposal

> Draft for review. This PR records OPE-207's design; it does not implement or
> complete the issue. The proposal, design, and spec remain open to revision.

## Why

Full, prepared-input, and staged standard/multisector runners independently
compose the same scientific functions and have diverged on retained-site policy.
Establish shared recipe execution before OPE-165 completes linked CO2/O2 staging.

## What Changes

- Give each recipe canonical operations for its supported scientific phases.
  Equivalent full, prepared-input, and staged routes reuse preparation, model
  construction, sampling, and result/product construction. Full runners visibly
  sequence these operations in memory; stages add checkpoint I/O, authentication,
  and reporting around them.
- Give standard and multisector separate concrete stage owners, sharing identical
  scientific operations and artifact mechanics where appropriate. Preserve
  distinct pre-filter merged, staged filtered-merged, and fully prepared
  checkpoints without repeating completed scientific transformations.
- Correct standard/multisector staging to accept the full runners' valid retained
  subsets after acquisition, compatible merged-cache reload, or filtering. Align
  all per-site metadata; reject an empty retained set or malformed input. This is
  an explicit staged behavior change, requiring regression coverage and a release
  note in the implementation.
- Use one authoritative configuration resolution and a common sampling/output
  record for workflows adopting the shared staged interface: standard,
  multisector, and ordinary/cached CO2 initially. These types and the complete
  stage suite are not prerequisites for independent builders, direct runners,
  nested execution, or every future recipe.
- Require meaningful scientific parity across equivalent supported routes,
  including model behavior and products from the same posterior. Preserve
  authenticated standard/multisector version-2 graph-free replay, genuine
  historical version-1 replay, and CO2 version-1 graph-free/affine replay.
- **BREAKING**: Newly introduced staged setup classes and helper signatures can
  change. Preserve installed CLI behavior, configuration meanings, established
  scientific Python APIs, artifact names/schemas/identities, and family-specific
  readiness error reporting, apart from the retained-site correction above.

## Capabilities

### New Capabilities

- `recipe-workflow-contract`: Consistent execution across supported recipe routes,
  checkpoint boundaries, scientific parity, and existing CLI/artifact/replay
  guarantees.

### Modified Capabilities

None; the project has no durable workflow capability specs yet.

## Impact

Implementation affects recipe runners and their scientific operations,
configuration, staged wrappers, and associated tests/documentation. Reconcile
these owners with the recipes migration in PRs #773–#776. No equations,
configuration formats, dependencies, artifact schema migrations, diagnostic
redesign, unified `run` CLI, MAP route, or new recipe are added here.

[OPE-207](https://linear.app/openghg-inversions/issue/OPE-207/unify-resolved-recipe-configuration-and-staged-workflow-interfaces)
blocks OPE-165's completion until implementation and validation; merging this
planning PR does not unblock it. PR #779's linked prepared-input persistence can
proceed independently. This PR contains no implementation or tasks file.
