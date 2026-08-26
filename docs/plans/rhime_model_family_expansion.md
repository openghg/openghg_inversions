# RHIME Model-Family Expansion and Landing Plan

Status: active companion plan to the P0 `run_rhime` roadmap
Date: 2026-08-18
Owners: OpenGHG Inversions maintainers and model authors

## Purpose

OpenGHG Inversions needs to absorb two near-term, substantially tested model
prototypes:

1. a nested-domain inversion with a high-resolution 6 km inner region; and
2. a CO2 model family containing a CO2-only model and a CO2 model with an O2
   tracer, sharing preparation and components where that is scientifically
   honest.

Verification Games also needs reusable prior-covariance, aggregation-error,
and coherent native-to-basis reduction capabilities to land in the released,
DOI-backed package.

These requirements must not wait for a generic model framework. This plan
permits a temporary menagerie of explicit production model recipes and records
the minimum structure needed to keep that menagerie understandable.

The normative programming principles are in
[`docs/development/rhime_model_development.rst`](../development/rhime_model_development.rst).
The main workflow roadmap remains
[`run_rhime_readability_and_modifiability.md`](run_rhime_readability_and_modifiability.md).

## Decisions made now

### 1. Named recipes are the production extension unit

A model whose preparation topology, state sharing, observation channels, or
outputs differ materially from standard RHIME receives a named procedural
runner and a nearby concrete model builder. It does not become a
`builder_strategy` or a case in a semantic compiler.

The intended near-term landing is:

```text
openghg_inversions/rhime/
  nested_domain.py       readable nested-domain runner and concrete graph
  co2/
    __init__.py          run_rhime_co2 and run_rhime_co2_o2
    co2_model.py         concrete CO2-only graph
    co2_runner.py        readable CO2-only pipeline
    co2_o2_preparation.py  CO2/O2 shared-state preparation
    co2_o2_model.py      concrete CO2/O2 graph
    co2_o2_runner.py     CO2/O2 prepared-input replay seam
    outputs.py           tracer-aware output construction
  config/
    nested_domain.ini
    co2.ini
    co2_o2.ini
```

The current CO2 model is already complex enough to justify a dedicated family
subpackage. It owns two explicit recipes: CO2 only and CO2 with an O2 tracer.
This is code locality, not a framework layer: both public runners must still
show their whole sequence, and both model functions must remain readable
concrete graphs. Keep `nested_domain.py` as one module initially and split it
only if real size prevents top-to-bottom reading.

Do not reorganize the existing standard and multisector runners before these
models can land. When the existing files are next changed substantially, they
may converge toward the same one-recipe-per-module layout with compatibility
re-exports.

### 2. Proliferation control is deferred

It is acceptable to add readable pipelines for urgently needed, tested models.
The project will review their repeated structure after the menagerie exists.
No generic pipeline class, model registry, component graph, or dependency-
injection mechanism is required now.

This does not permit an unstructured branch matrix inside `run_rhime`. A
structurally distinct model receives a distinct named recipe.

### 3. Composite scientific components are allowed

Components may be composed at scientifically meaningful scales. In particular:

- a reported baseline may compose separately stored boundary conditions, an
  offset, and an output-side outer-domain flux projection;
- a likelihood may combine observation error, aggregation error, model-data
  mismatch, pollution-event treatment, and a distribution; and
- a pollution or flux component may combine several source-resolved forward
  terms.

Initially, a composite is an ordinary function beside the model recipe that
uses it. Extract it only when two production models use the same equations and
option meanings.

### 4. Configuration options are owned by components

INI options should be documented and grouped by the scientific component that
owns them. This is separate from runtime function introspection: runtime
signatures include derived scientific values that are not user options.

Examples:

| Component | User options |
| --- | --- |
| Baseline | `bc_prior`, `bc_freq`, offset choices |
| Model-data mismatch | `sigma_prior`, `sigma_freq`, `sigma_per_site` |
| Aggregation error | representation/mode and preparation choices |
| Flux or pollution term | source, basis, state grouping, and prior choices |

A first implementation may use explicit option-name tuples or a small table.
It must not require a component class or infer the configuration schema from
all function parameters.

The current config reader flattens INI sections. Linked-channel models need a
small section-preserving resolver so CO2 and O2 can each own values such as
`sigma_prior`. Do not build a general component-config engine as a prerequisite.

Store new RHIME templates under `openghg_inversions/rhime/config` with names
matching their recipes. This provides an obvious parallel layout without
putting new model configuration under the legacy `hbmcmc/config` directory.
The CO2-family subpackage may own its section-preserving resolver, while the INI
template remains visible in the common RHIME config directory. Moving the
existing standard template from `openghg_inversions/config/templates` can be a
later compatibility task.

### 5. Callable seams remain useful but are not the only home for citable models

`likelihood_builder` and copied runners remain useful for incubation and
project-specific work. When a variation is repeatedly used in papers, its
stable form should become a named, tested in-tree component or recipe. A
released OpenGHG Inversions version can then identify the implementation under
the project's DOI.

## Nested-domain model

The source prototype is draft
[#359](https://github.com/openghg/openghg_inversions/pull/359), branch
`inner_domain_adaptation`. The branch is too divergent to update or cherry-pick
as a whole. It remains scientific and regression evidence.

The model is:

```text
mu_pollution = H_outer @ x_outer + H_inner @ x_inner
```

with the ordinary baseline, observation-error, model-data mismatch, and
likelihood components added afterward.

The production recipe should visibly:

1. retrieve outer-domain and inner-domain data;
2. align and filter their common observations;
3. build separate outer and inner basis functions;
4. build separate labelled sensitivities without flattening the grids;
5. assemble one validated nested-domain prepared handoff;
6. add the outer and inner pollution components and sum them;
7. add the baseline and likelihood;
8. sample and construct separate native-grid output products.

One `NestedDomainPreparedInputs`-style object is justified because it represents
two related domains, two basis artifacts, and their alignment invariants. It
must not grow into a generic model graph.

Use the scientific name `nested_domain` in code. The 6 km domain is the first
supported inner domain, not the definition of the abstraction.

Port the scientific behavior and fixtures, not the old acquisition mutations,
NumPy assembly, sampler, or output splicing. Make outer and inner basis counts
and priors explicit. Validate the provenance and licence of the prototype's
EUHROB region asset before inclusion.

Existing issues [#407](https://github.com/openghg/openghg_inversions/issues/407),
[#408](https://github.com/openghg/openghg_inversions/issues/408), and
[#409](https://github.com/openghg/openghg_inversions/issues/409) remain a useful
input/model/output split. Revise #407 so a generic `InnerDomainSpec` is not a
required outcome. Use #410 for the readable recipe and extension documentation.

## CO2 model family

The production subpackage must support a CO2-only recipe and a CO2 recipe with
an O2 tracer. The current linked Verification Games prototype has a small model
graph but substantial scientific preparation. Its corrected case uses shared
GPP, TER, and fossil-fuel states, tracer-specific ocean states, correlated
arithmetic-lognormal priors, and dense coherent aggregation-error covariance
with cross-tracer blocks.

The CO2-only recipe should be a first-class model, not the linked model with a
dummy or empty tracer. It can share source selection, covariance preparation,
coherent reduction, likelihood terms, output roles, and configuration parsing
with the linked recipe where their meanings agree.

The production recipe should visibly:

1. prepare the CO2 and O2 channels, allowing different observation axes;
2. declare shared and tracer-specific source/state groups directly;
3. construct native covariance products;
4. coherently reduce the joint native model to retained basis states;
5. validate the shared-state labels and tracer loadings;
6. build one joint, readable PyMC graph;
7. sample once and produce tracer-aware outputs.

The concrete graph is approximately:

```text
flux_scaling = correlated_lognormal_state(...)
co2_o2_flux_contribution = joint_sensitivity @ flux_scaling
modelled_concentration = (
    fixed_prior_contribution + co2_o2_flux_contribution
)
y_obs = multivariate_normal(
    modelled_concentration,
    aggregation_covariance + mismatch_covariance,
)
```

### Fixed and uncertain cross-channel couplings

A fixed oxidative ratio or other cross-channel coupling is a labelled
scientific input. Its contract must state its units, direction, sign,
coordinate scope, alignment, and provenance. A prepared sensitivity may embed a
fixed coupling, but its metadata and the consuming recipe must say so
explicitly to prevent the coupling from being applied twice or omitted.

An uncertain coupling is an explicit sampled parameter in the concrete model
recipe, not metadata attached to a fixed sensitivity. For example, a latent
O2-per-CO2 ratio requires a ratio-free O2 sensitivity and a visible graph of the
form:

```text
oxidative_ratio = positive_parameter(...)
o2_contribution = H_o2_ratio_free @ (-oxidative_ratio * x_shared)
```

This is bilinear in the ratio and shared flux state. Preparation must therefore
not silently reuse a sensitivity or aggregation covariance derived for a fixed
ratio. Any covariance held fixed under the uncertain coupling must have a
scientific justification and be recorded as an approximation. Configuration
may offer a recipe-owned choice between a fixed value and a prior
specification; this does not require a generic parameter graph or nested-prior
language.

Keep the first model literal and local: shared sources and tracer-specific
sources should be plainly visible rather than normalized through a generic
multigas schema. A `Co2O2PreparedInputs`-style handoff is justified because it
represents distinct observation channels with shared state coordinates.

The supplied prototype is currently uncommitted and partly untracked in its
Verification Games worktree. Preserve it as a golden scientific reference, but
establish a reproducible commit or archived evidence bundle before claiming
exact parity.

This prototype intentionally uses the exact inner region. Same-grid outer
entries in the CO2-only recipe are now ordinary labelled members of its one
retained state rather than a separate model subsystem.

The CO2-family subpackage is a landing and integration point, not a permanent silo.
After the model is reproduced, promote its generally useful features—prior
covariance, coherent reduction, aggregation covariance, and any genuinely
shared likelihood pieces—into ordinary functions that standard RHIME and
multisector RHIME can call directly.

OPE-119 supersedes OPE-76's experimental outer-state model surface. Same-grid
inner and outer entries share one labelled sensitivity, one retained prior,
and one ``flux_scaling`` state. ``basis_group`` identifies geography;
``StateActivity(fixed_groups=("outer",), fixed_value=1.0)`` preserves the
fixed-at-one case, while leaving that group active infers it. Group-specific
moments are resolved into the complete ordered prior before graph construction,
including any intended inner/outer cross-covariance.

Gaussian outer-state marginalization and outer-only sector collapse are not
production behavior. General sector combinations belong to GitHub #628.

## Verification Games scientific components

These features should land by scientific role, not as compiler infrastructure:

### Native prior covariance

Continue developing the labelled numerical objects in `native_covariance.py`,
`source_covariance.py`, and `basis/covariance_products.py`. They are reusable
inputs to more than one model recipe.

### Coherent reduction

Add a plainly named preparation operation, provisionally
`openghg_inversions/coherent_reduction.py`. It should return one mathematically
meaningful result containing the retained prior mean/covariance, transformed
forward sensitivity, and unresolved aggregation covariance. These quantities
must be reduced together, including cross-tracer covariance blocks.

This result is a justified scientific data object. It is not a semantic model,
compiler plan, or execution manifest.

### Aggregation error

Keep aggregation error as an explicit observation/likelihood input using the
existing representations in `observation_error.py`. Each model recipe decides
which representation it supports and makes that choice visible at preparation
and likelihood construction.

### Grouped inner/outer reporting

Verification Games provided prototype evidence that fixed outer-region flux
contributions could be grouped with the atmospheric baseline for reporting. Its
implemented scientific operation was:

```text
composite_baseline = atmospheric_baseline + sum(H_outer_fixed)
```

The prototype marks outer states inactive with a fixed scale of one and retains
their sensitivity columns. The historical behavior is distributed across
`scripts/prepare_met_office_followup_input.py`,
`src/verification_games/rhime_calibration/model.py`, and
`src/verification_games/rhime_calibration/analytic.py`. Relevant provenance is
recorded in Verification Games commits `c840a2d`, `25931e1`, and `df1c704`.

The CO2 recipe keeps enough state and sensitivity metadata for outputs to
project the single full flux contribution into inner and outer reporting
views. It does not construct separate group terms in the model. Atmospheric
``mu_bc``, ``offset``, the coherent affine prior term, and reconstructed outer
flux remain scientifically distinct. Reporting may compose boundary, offset,
and outer flux as a baseline without changing model construction or stored
provenance.

### Source and state selection

The production compiler path and `models/_rhime_flux.py` have now been removed.
The reusable labelled source selection and safe-naming operations retained from
that work live in `models/_flux.py`; RHIME-specific sector binding, state
activity, and prior resolution now remain visible in the multisector recipe.
Future model-family work should reuse only the demonstrated ordinary operations
from `_flux.py` and keep new recipe-specific decisions beside the recipe rather
than recreating compiler plans or backend lowering.

### Backend-neutral scientific variable roles

Retain a small CF-like vocabulary for quantities consumed by outputs and
postprocessing, for example `modelled_concentration`,
`pollution_concentration`, `baseline_concentration`, `flux_scaling`, and
`model_data_mismatch`. Each concrete backend maps those roles to its own
variable or result names.

This mapping allows PyMC and analytic-Gaussian models to share labelled
preparation and postprocessing without requiring the same internal variable
names. It is not a semantic model, execution graph, or compiler manifest.
Each role should have one short scientific definition and the dimensional or
unit expectations needed by its consumers, analogous to a small project-level
CF `standard_name` vocabulary.

## Deferred questions

The following are deliberately not prerequisites for the urgent models:

- a universal component protocol;
- a general nested/composite component representation;
- a generic multigas model beyond the concrete CO2 and CO2/O2 recipes;
- a single configuration engine for all model families;
- automatic routing based on function-signature introspection;
- a registry controlling all model recipes; and
- a final policy for deduplicating common runner code.

Revisit these only after the nested-domain, CO2-only, and CO2/O2 recipes provide
concrete evidence.

## Delivery sequence

The two recipes and the scientific components can proceed in parallel with the
P0 W4-W6 cleanup. They do not need to wait for a compiler replacement or replay
bundle.

1. Preserve reproducible prototype evidence and focused mathematical fixtures.
2. Land the small reusable numerical/scientific functions needed by each model.
3. Add the dedicated concrete builder and model-only equation tests.
4. Add the procedural runner, model-specific config resolver/template, and one
   end-to-end smoke test.
5. Add tracer/domain-aware outputs and a literal, tested model walkthrough.
6. Release under the normal OpenGHG Inversions version and DOI process.
7. Review the resulting recipes together and extract only demonstrated common
   components.

Structural refactoring and scientific changes should remain separate PRs where
practical. Each intermediate PR must leave existing standard and multisector
RHIME behavior intact.

## Tracking

- Linear project: [RHIME model families — nested domain and CO2](https://linear.app/openghg-inversions/project/rhime-model-families-nested-domain-and-co-80bc5cea9108)
- Linear delivery plan: [RHIME model-family expansion delivery plan](https://linear.app/openghg-inversions/document/rhime-model-family-expansion-delivery-plan-dc4a7b002964)
- Shared foundations: [OPE-17](https://linear.app/openghg-inversions/issue/OPE-17),
  [OPE-18](https://linear.app/openghg-inversions/issue/OPE-18),
  [OPE-20](https://linear.app/openghg-inversions/issue/OPE-20), and
  [OPE-80](https://linear.app/openghg-inversions/issue/OPE-80).
- Nested-domain / 6 km family:
  [OPE-25](https://linear.app/openghg-inversions/issue/OPE-25), with delivery
  slices [OPE-71](https://linear.app/openghg-inversions/issue/OPE-71),
  [OPE-72](https://linear.app/openghg-inversions/issue/OPE-72), and
  [OPE-73](https://linear.app/openghg-inversions/issue/OPE-73).
- CO2 model family: [OPE-26](https://linear.app/openghg-inversions/issue/OPE-26),
  with prototype preservation
  [OPE-74](https://linear.app/openghg-inversions/issue/OPE-74), CO2-only
  [OPE-75](https://linear.app/openghg-inversions/issue/OPE-75), CO2 with O2
  tracer [OPE-77](https://linear.app/openghg-inversions/issue/OPE-77), one-state
  grouped inner/outer construction superseding separate treatments
  [OPE-119](https://linear.app/openghg-inversions/issue/OPE-119), grouped outer
  reporting and composite baseline
  [OPE-76](https://linear.app/openghg-inversions/issue/OPE-76), and
  outputs/docs [OPE-79](https://linear.app/openghg-inversions/issue/OPE-79).
- Later promotion of demonstrated shared features:
  [OPE-78](https://linear.app/openghg-inversions/issue/OPE-78).

## Acceptance criteria for every production recipe

- The runner reads as a complete scientific sequence without IDE navigation.
- The concrete PyMC graph is readable in mathematical order.
- Component options are documented with their owning scientific concept.
- Unknown or unused configuration options fail before retrieval or sampling.
- Model-only fixtures verify the expected mean and covariance equations.
- Borrowed xarray and named materialization boundaries are preserved.
- Outputs retain domain, tracer, source, state, and unit labels needed for
  interpretation.
- A runnable config/example and focused end-to-end smoke test are present.
- Provenance records the recipe name, package version, and relevant scientific
  references.
- A representative scientist can locate and modify one component without
  learning a compiler or manifest system.
