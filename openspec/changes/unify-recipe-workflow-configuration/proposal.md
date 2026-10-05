# Proposal

> Draft for review. This PR records OPE-207's design; it does not implement or
> complete the issue. The proposal, design, and spec remain open to revision.

## Why

Full, prepared-input, and staged standard/multisector runners independently
compose the same scientific functions and have diverged on retained-site policy.
Establish shared recipe execution before OPE-165 completes linked CO2/O2 staging.

## What Changes

- Give each recipe canonical operations for its supported scientific phases.
  Filtering, retained-site alignment, basis construction/application, sensitivities,
  and input assembly belong inside one scientific preparation operation.
  Equivalent full, prepared-input, and staged routes reuse preparation, model
  construction, sampling, and result/product construction. Full runners visibly
  sequence these operations in memory; stages add checkpoint I/O, authentication,
  and reporting around them.
- Give standard and multisector separate concrete stage owners, sharing identical
  scientific operations and artifact mechanics where appropriate. Keep the
  existing optional merged-data cache at the acquisition/preparation boundary
  and the fully prepared handoff at the preparation/construction boundary.
  Full and staged routes reuse the same merged-cache save/reload mechanism.
- **BREAKING (next minor release)**: Remove the newer filtered merged-data
  checkpoint (`merged-data/merged-data.nc`) and its preparation-manifest entries.
  Filtered merged data remains an in-memory preparation intermediate, with no
  separate filtering stage or filtered-checkpoint resume interface.
- Correct standard/multisector staging to accept the full runners' valid retained
  subsets after acquisition, compatible merged-cache reload, or filtering. Align
  all per-site metadata; reject an empty retained set or malformed input. This is
  an explicit staged behavior change, requiring regression coverage and a release
  note in the implementation.
- Use one authoritative configuration resolution and a common sampling/output
  record for workflows adopting the shared staged interface: standard,
  multisector, and ordinary/cached CO2 initially. Give resolved recipe choices,
  immutable sampler choices, numerical handoffs, invocation options, and runtime
  execution state clear roles and consistent names; remove stage-only setup
  wrappers. These types and the complete stage suite are not prerequisites for
  independent builders, direct runners, nested execution, or every future recipe.
- Require meaningful scientific parity across equivalent supported routes,
  including model behavior and products from the same posterior. Preserve
  authenticated graph-free replay and independent CO2 affine authentication for
  artifacts supported by the revised workflows.
- **BREAKING (next minor release)**: Existing staged setup APIs, manifests,
  bindings, and identity encodings need not remain compatible. Retire historical
  standard/multisector graph-recovery replay; no migration or compatibility aliases
  are required for pre-refactor staged artifacts. Keep explicit version handling
  so families can support selected older schema/identity contracts going forward,
  with strict authentication rather than permissive fallback.
- Preserve installed CLI behavior, configuration meanings, established
  scientific Python APIs, numerical prepared-input/posterior formats, product
  names/schemas, and family-specific readiness error reporting, apart from the
  explicit staged compatibility reset, retained-site correction, and filtered
  checkpoint removal above.

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
these owners with the recipes migration in PRs #773–#776. The implementation
must announce the checkpoint removal and staged compatibility reset as breaking
changes in the next minor release, with user documentation and a Towncrier
removal fragment. Numerical artifact codec/schema migrations, equations,
configuration formats, dependencies, diagnostic redesign, unified `run` CLI,
MAP execution, and new recipes remain outside this change.

[OPE-207](https://linear.app/openghg-inversions/issue/OPE-207/unify-resolved-recipe-configuration-and-staged-workflow-interfaces)
blocks OPE-165's completion until implementation and validation; merging this
planning PR does not unblock it. PR #779 has merged its linked prepared-input
persistence; linked staged integration remains separate. This PR contains no
implementation or tasks file.
