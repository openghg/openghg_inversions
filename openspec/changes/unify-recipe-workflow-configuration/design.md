# Design

> **Superseded:** Read the [replacement design](../replace-recipe-workflow-with-reusable-handoffs/design.md).
> The historical design below is not the current implementation instruction.

> Implementation design for OPE-207. Proposed names are not implemented APIs.
> The [proposal](proposal.md) defines scope; the
> [behavioral spec](specs/recipe-workflow-contract/spec.md) defines acceptance requirements.

## Context

Standard/multisector full and prepared-input runners and `_standard_stages.py`
independently sequence the same scientific work. Their retained-site policies
have diverged. CO2 staged sampling already calls its ordinary/cached prepared
runners, but configuration is split between `Co2RunSetup` and `Co2StageSetup`;
standard/multisector use `RhimeRunnerSetup`.

An **execution route** is an entry point into all or part of a recipe: a full
runner, a runner supplied with merged or prepared inputs, staged commands, or
saved-output replay. Equivalent routes must call one implementation of each
scientific operation they need. Staged functions also load, authenticate, and
save checkpoints.

Follow [RHIME development guidance](../../../docs/development/rhime_model_development.rst)
and the [architecture principles proposed in PR #793](https://github.com/openghg/openghg_inversions/blob/92be84ec2e697f2a7c2dbf8a6f0266b9965c0cb7/docs/development/architecture_principles.rst):
keep scientific composition visible and choose responsibilities by the decisions
that must change together.

## Goals / Non-Goals

**Goals:** Share scientific operations across equivalent routes, keep full
runners procedural, and give configuration and artifact handling clear owners.

**Non-Goals:** New scientific recipes, equations, optimizers, configuration
formats, numerical artifact codecs, CLI commands, or diagnostic policies.
Import isolation and a possible `_model_building.py` split remain separate work:
graph-free replay forbids graph construction, not PyMC imports.

## Decisions

### 1. Full, prepared-input, and staged execution call shared recipe functions

Each recipe provides ordinary functions for preparation, construction, sampling,
and products where those operations are supported.

- **Preparation** orders filtering, retained-site alignment, basis construction,
  sensitivities, and labelled input assembly. Filtering stays inside preparation.
- **Construction** selects required inputs, materializes related arrays together,
  builds the model, and establishes its scientific output meaning.
- **Sampling** applies the recipe's inference policy. Cached CO2 keeps its graph,
  sigma-then-state `CompoundStep`, cache updates, conditional prediction, and trace
  annotation together.
- **Products** use one family implementation, accepting either live construction
  information or authenticated saved output information.

A full runner visibly sequences these functions in memory. Supplied merged
data starts at preparation; fully prepared inputs start at construction. Saved
samples and output information start at products.
Requested final products and explicitly enabled merged/basis saves remain supported.

```text
acquire merged --> prepare --> build model --> sample --> products
       |              |                                    ^
       v              v                                    |
 optional merged   prepared inputs               authenticated saved
    checkpoint       checkpoint                  samples/output information
```

The diagram shows scientific order and optional persistence. It does not
prescribe one class or module per phase. A recipe can own several functions;
mathematically coupled decisions stay together. Sharing low-level helpers is
insufficient if the entry points still independently choose scientific policy.

Preserve direct runner/builder signatures and returns, including CO2's trace
return, likelihood customization, and complete-model builders. Forward resolved
values explicitly. Built-in construction owns coordinated materialization;
a custom complete-model builder receives potentially lazy inputs and owns its
materialization. Validate external inputs at their owning boundary without
mutating borrowed arrays.

### 2. Stage functions belong to their recipes

Standard and multisector get separate stage implementations, replacing the
combined scientific workflow in `_standard_stages.py`. They can share functions
whose equations and option meanings agree, along with artifact and reporting
mechanics. Small repeated entry-point sequences are acceptable.

Select the workflow once at the CLI boundary. Standard, multisector, and
ordinary/cached CO2 expose `prepare`, `prior_predictive`, `sample`, and
`postprocess` through a small structural calling interface. Each function
receives explicit artifact paths and invocation options, then calls its recipe's
scientific operations. Implementations import shared helpers from their owners,
not the public dispatcher.

This interface applies to standard, multisector, and ordinary/cached CO2.
Nested and future recipes can expose only the operations they need. Diagnosis
remains independently callable with a posterior, optional sample manifest, and
convergence options.

Independent builders and direct runners retain explicit scientific arguments.
For example, a future maximum a posteriori (MAP) experiment could call
`build_rhime_co2`, then its own optimizer and result construction; a prepared-only
recipe could expose a builder
and runner without acquisition or staging. These are design checks, not features
to implement here. MAP suitability and its objective would require separate
scientific work; a cached graph is not automatically a valid MAP target.

### 3. Configuration records choices; runtime objects execute them

One family resolver supplies one immutable resolved configuration for its equivalent
configuration-driven routes. File loading, source-relative paths, aliases, and
override precedence retain their existing rules.

Use consistent names for these roles. Reuse suitable existing types; the table
does not require a class for every row.

| Role | Proposed name or existing representation | Contents or responsibility |
| --- | --- | --- |
| Resolved recipe configuration | `RecipeConfig` | Recipe selection, recipe-specific options, sampler choices, and one output policy |
| Recipe-specific choices | `StandardOptions`, `MultisectorOptions`, `Co2Options` | Resolved acquisition and scientific choices for that recipe |
| Sampling choices | `SamplerOptions` | Immutable settings, including nested keyword choices |
| Product policy | Existing `RhimeOutputSpec` where suitable | Products and reporting/writing choices; derived result metadata is not another policy |
| Numerical inputs | Existing merged/prepared input types | Values, labels, units, provenance, and the preparation already performed |
| Invocation arguments | Explicit function arguments | Per-call artifact paths, destination, and overrides |
| Runtime objects | Existing build results, models, `RhimeSampler`, steps, caches | Invocation-local execution state; build results associate a graph with its output meaning |
| Saved output contract | Versioned output information and identities | Scientific roles and authenticated associations needed after the graph is gone |

Remove CO2's stage-only setup wrapper. Keep one authoritative output policy
rather than copying it into a separately resolved result configuration.
Configuration contains choices, not numerical payloads or live execution
objects. Prepared inputs supply actual retained labels; requested choices remain
available for provenance.

Construct or adapt runtime samplers, steps, and caches for the invocation.
Overrides must leave the original choices unchanged and must not affect later
calls, including through nested dictionaries. Existing direct Python sampler
interfaces remain supported. Configuration access must not copy or materialize
borrowed arrays.

The proposed naming distinguishes `Config` (resolved execution choices),
`Options` (a related set of choices), and existing `Inputs` types (numerical
values). Do not rename public scientific types for cosmetic uniformity or create
generic input/result containers.

### 4. Persist merged data before preparation and prepared inputs after it

| Starting values | Remaining work |
| --- | --- |
| Acquired, external, or reloaded merged data | Preparation, construction, sampling, products |
| Fully prepared inputs | Construction, sampling, products |
| Prepared inputs and authenticated saved samples/output information | Products without construction or resampling |

For standard/multisector, acquisition must honour `save_merged_data` and
`reload_merged_data`. A requested save writes acquisition output before configured
filtering, basis construction, and sensitivities; acquisition may already include
averaging. A valid cache reload bypasses acquisition and enters the same scientific
preparation operation. Full and staged execution must share the existing save/load
functions, naming and format rules, and cache-validation/fallback policy. Stage
artifact paths retain their directory-containment checks. Cache saves remain opt-in.

**Breaking change for the next minor release:** Stop writing the staged filtered
`merged-data/merged-data.nc` snapshot. Omit its `merged_data` path/digest entries
from new preparation manifests, even when acquisition caching is enabled.
Do not repurpose those entries or reinterpret old filtered files as pre-filter
caches. Filtering creates an in-memory intermediate, not another checkpoint or
public stage.

Prepared inputs remain the staged inference checkpoint. CO2 already uses prepared
inputs and optional affine companions without a merged snapshot; this change
does not require it to add acquisition or merged caching.

### 5. Correct retained-site handling and preserve readiness behavior

Each recipe must derive retained sites from the observations it keeps and align
applicable per-site metadata and options. Equivalent routes must use the same
policy, preserving validation of malformed or incompatible inputs.

For standard/multisector, full preparation already accepts valid subsets after
acquisition, compatible cache reload, and filtering; staging rejects missing
requested sites. OPE-207 must remove this policy divergence by sharing preparation.
Align every per-site option and reject an empty set before basis construction or
inference. Cover all three causes with unequal per-site options and document the
behavior change.

CO2 staging currently starts with prepared inputs, without acquisition or configured
observation filtering. Shared canonical-input validation already aligns site metadata
to observed sites. Check this alignment across ordinary/cached execution routes;
retain validation of model-specific per-site options. This does not require a new
CO2 acquisition or filtering route.

Preserve other supported route differences with a stated reason and regression
coverage. For example,
[`run_rhime_from_prepared_inputs`](../../../openghg_inversions/rhime/prepared.py)
with the built-in model rejects `basic`, `paris`, and `legacy` products when
aggregation-error mode is not `none`; this refactor does not expand those outputs.

The table describes current prior-predictive readiness behavior. The refactor must
preserve these exception boundaries and artifact-writing rules:

| Event | Standard / multisector | Ordinary / cached CO2 |
| --- | --- | --- |
| Returned finite evidence | Existing assessment | Existing assessment |
| Returned empty/non-finite evidence | Existing `fail` check | Existing `fail` check |
| Build/prior-predictive `KeyError` or `ValueError` | Within the existing catch: `fail` check; no predictive artifacts | Error propagates; no check or new destination |
| External loading/authentication outside that catch, or artifact serialization error | Error propagates | Error propagates |

When prediction returns invalid evidence, preserve the existing predictive/report
writes, including CO2's prior manifest. Keep standard/multisector authentication
and serialization outside the readiness catch, and CO2 construction/prediction
before destination creation. Strict-mode exits, convergence thresholds, and
`unknown` handling follow the [behavioral spec](specs/recipe-workflow-contract/spec.md#requirement-preserve-readiness-and-convergence-behavior).

### 6. Version readers own artifact compatibility

**Breaking change for the next minor release:** Pre-refactor staged setup APIs,
manifests, bindings, and scientific identity hashes need not remain compatible.
Retire standard/multisector's historical version-1 graph-recovery replay.
Direct scientific APIs, numerical prepared-input/posterior formats, product
contracts, and the pre-filter acquisition cache remain supported.

A scientific identity hashes the settings relevant to scientific reuse.
Each family defines those settings explicitly rather than serializing runtime
dataclasses. Keep requested preparation and retained-run identities distinct,
retain existing sampler/output/transport exclusions, and report sampling
provenance recorded in the sample manifest. Within a declared identity version,
incidental record changes must not silently redefine the hash.

Each family declares supported schema and identity versions. Writers identify
their version; readers validate the corresponding contract before loading the
posterior or creating a product destination. A binding associates saved output
information with the prepared-input and posterior content identities.
Incompatible metadata changes require an identifiable new version.

Going forward, a family can retain readers for selected older versions.
Those readers decode and authenticate the saved information, then call shared
product functions. They do not require rewritten artifacts or a second
scientific workflow. A version recognized by a shared loader is not necessarily
supported by the selected family.

All supported saved-output replay is strictly authenticated and avoids acquisition,
model-input materialization, graph construction, and resampling. Standard/multisector
requires a bound output contract; CO2 independently authenticates optional affine
companions. Unsupported versions, malformed bindings, and authentication
downgrades fail before posterior loading or product writes, without graph fallback.
No pre-refactor migration or speculative older-version reader is required now.

Shared artifact/path/digest and manifest/binding helpers own storage and
authentication mechanics. Recipes own scientific compatibility and supported
versions. Provenance collection remains shared, with CO2's revision requirement
local to CO2.

## Risks / Trade-offs

Shared configuration can conceal duplicated science; verify actual operation
owners and scientific results. Consolidation can also alter retained-site policy,
error reporting, or artifact acceptance. Characterize those behaviors before
moving implementations, and identify the agreed changes separately from
structural edits.

## Implementation and validation

1. Record supported entry points, options, checkpoints, identities, and readiness
   behavior in a short `entry point -> scientific operation -> owner` table.
   Coordinate with PRs #773–#776, which reorganize acquisition, inference, model
   components, and recipe composition. Move shared operations and their callers
   together so the package changes do not leave competing implementations.
2. Consolidate one recipe at a time. During review, trace a retained-site policy
   change through full and staged execution, then trace a checkpoint-format change
   through its artifact owner. Updating callers and tests is normal; repeating the
   policy in those callers is not.
3. Compare real preparation arrays, coordinates, retained metadata, basis and
   sensitivities, multisector source data, model inputs, deterministic terms or log
   probability at controlled parameters, and matched sampling policy. Include
   small real-sampling checks for ordinary/cached variants and product comparisons
   from one posterior: units, labelled dimensions, chain/draw axes, and conditional
   reconstruction. Independent stochastic trajectories need not match.
4. Supplement route agreement with existing independent reference calculations
   or pre-refactor expected values. Expected values establish regression stability;
   independent calculations provide correctness evidence. Check array ownership
   and lazy execution separately from numerical results.
5. With intermediate saves disabled, forbid intermediate checkpoint writes during
   full execution. Forbid preparation during prepared execution and model-input
   materialization/construction/sampling during saved-output replay. Check sampler
   override isolation, version dispatch, malformed/mismatched artifact rejection,
   optional pre-filter caching, and omission of the retired filtered snapshot.
   Run relevant workflow/configuration/replay tests, changed-path Ruff, and
   whitespace checks.
6. Review public-operation docstrings for the decision owned, required input phase,
   validation, borrowing/materialization, returns, and failures. Update user/API
   documentation and add a Towncrier removal fragment announcing the checkpoint
   removal and staged compatibility change in the next minor release, alongside
   the retained-site correction.

OPE-207 blocks OPE-165's completion until implementation and validation; this
planning PR does not complete it. PR #779 has merged linked CO2/O2 prepared-input
persistence; linked staged integration and the unified `run` CLI remain separate work.
After implementation, sync durable requirements and archive the change with its
tasks and validation evidence.
