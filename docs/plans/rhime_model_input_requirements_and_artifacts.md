# RHIME model input requirements and prepared artifacts

> **Status (August 2026):** The bill-of-materials object sketched below was
> superseded by the locality guidance in
> `docs/development/rhime_model_development.rst` and the OPE-82 implementation.
> Current recipes keep a small tuple or pure input-name function beside each
> consumer and pass named scientific arrays directly. The durable prepared and
> replay-artifact distinctions in this note remain current.

This is a design note for the work that follows the visible `run_rhime`
orchestration spine. It separates three ideas that are currently too easy to
conflate:

1. reusable prepared scientific data, including cached expensive calculations;
2. the labelled arrays that a selected model actually needs at the PyMC
   boundary; and
3. a replayable run artifact that binds data to a serialized model and run
   specification. Here, replayable means re-executable from the prepared-data
   boundary; it does not promise bitwise-identical posterior draws.

The names and sketches below are illustrative. The intended outcome is a small
set of ordinary typed contracts, not a workflow framework or semantic compiler.

## Motivation

The visible orchestration introduced by OPE-45 makes two forms of extension
possible without changing all of the standard plumbing:

- a custom runner can replace one scientific stage, for example supplying a
  custom basis while reusing retrieval, filtering, sensitivity construction,
  model building, sampling, and outputs; and
- a runner can start from externally supplied or cached data rather than
  retrieving it through OpenGHG, which is especially useful when footprints,
  sensitivities, or other calculations are expensive.

Those extensions need an honest downstream contract. A selected model should
state which inputs it requires and accepts; callers should not have to infer
that contract from implementation details or construct a semantic-role
manifest themselves.

The same distinction matters for persistence. A cache of prepared scientific
data can be useful with more than one compatible model. It should not claim to
be an exactly replayable model run. Conversely, storing “the inputs for a
model” is incomplete unless the artifact also identifies a serialized model
specification that explains how those inputs were interpreted.

## Proposed concepts

### Prepared scientific data

A prepared-data artifact stores reusable, labelled scientific products and
their preparation provenance. Depending on the supported cache boundary this
may include aligned observations, footprints, flux products, basis functions,
sensitivities, site metadata, or fully assembled inversion arrays.

Its purpose is to bypass OpenGHG acquisition or avoid repeating expensive
scientific calculations. It therefore:

- preserves dimensions, coordinates, attrs, units, provenance, and the
  borrowed/Dask ownership contract;
- records the preparation stages and options needed to interpret the cached
  products;
- may be bound later to more than one compatible model specification; and
- makes no claim that loading it alone reproduces a previous inference run.

Existing `RhimePreparedInputs` and merged-data serialization should evolve
incrementally. This plan does not require an immediate rename or a single
monolithic cache format.

### Model input requirements: component-adjacent declarations

Each concrete recipe derives a small immutable sequence of the labelled inputs
needed to build that model. Each model component owns the requirements it
introduces, expressed as a nearby constant or pure function. Explicit recipe
composition produces the selected input names without a generic requirements
object.

A first implementation should record only what the built-in model needs to
select, validate, and jointly materialize its inputs:

- the exact required input names and accepted diagnostic inputs;
- the component and model option that own each requirement;
- whether the model consumes the input or retains it only for diagnostics,
  reconstruction, output, or provenance;
- whether PyMC construction requires an eager dense value, permits a lazy or
  sparse representation, or does not consume the value at all.

Dimensions, coordinates, units, and scientific validity should remain checked
by plainly named component functions rather than being encoded in a generic
declarative schema. Alternatives such as aggregation-error representations
should resolve to one concrete inventory before the PyMC boundary.

For example, aggregation-error requirements should follow the selected mode:

| Mode | Required PyMC inputs | Optional retained inputs |
| --- | --- | --- |
| `none` | none | diagnostic arrays |
| `diagonal` | aggregation-error standard deviation | other diagnostic representations |
| `dense` | dense covariance | marginal standard deviation |
| `low_rank` | factor and residual diagonal | marginal standard deviation |

The model spec and its components, not an external runner, select these names.
Current usage is:

```python
input_names = standard_model_input_names(prepared, run_spec.model)
model_inputs = materialize_pymc_inputs(
    prepared,
    variable_names=input_names,
)
built = build_standard_rhime_model_result(
    prepared=prepared,
    model_inputs=model_inputs,
    run_spec=run_spec,
)
```

This makes parameter ownership and the eager boundary explicit. It also avoids
the current transitional pattern in which input selection and validation are
partly repeated by replay, materialization, and model construction.

The input-name declaration is not a stage registry, dependency-injection
mechanism, caller-authored manifest, or validation schema. Ordinary custom
runners call the declaration owned by the concrete recipe. The consuming
component remains responsible for scientific validation after coordinated
materialization.

### Configuration option ownership is a separate contract

Model input requirements describe labelled scientific arrays. They must not be
used as the schema for all user configuration, because component functions also
receive derived arrays and PyMC terms that are not INI options.

User options should instead be grouped explicitly by the scientific component
that owns them. For example:

- `bc_prior` and `bc_freq` belong to the baseline component;
- `sigma_prior`, `sigma_freq`, and `sigma_per_site` belong to the model-data
  mismatch component; and
- aggregation-error representation and preparation options belong to the
  aggregation-error component.

A first implementation may record this ownership in small option-name tuples
or a documentation table. It may use the information to validate INI files and
keep templates current. Do not infer the configuration schema from every
runtime function parameter, and do not introduce a component class hierarchy
or generic routing engine to hold it.

### Replayable run artifact

A replay artifact binds prepared scientific data to the specification that
interpreted it. At minimum it should persist:

- a versioned, serialized model specification;
- the run and sampling options required for replay;
- prepared-data identity or embedded prepared arrays;
- the derived requirements/schema identity used to validate those arrays;
- output/reconstruction capabilities and safe provenance; and
- package/schema versions needed to diagnose compatibility.

The artifact must not serialize executable Python callables. A supported
callable seam may persist a safe import identity and JSON-compatible metadata,
but standard replay is available only when that identity can be resolved and
its contract is compatible. Sampler seeds/state and backend versions must be
recorded when they matter, without promising bitwise-identical stochastic
results across environments.

A complete custom model builder has two honest choices:

1. provide a serializable model specification and component-owned input
   requirements, gaining standard validation, materialization, and replay; or
2. own its validation and materialization explicitly and be recorded as an
   advanced, non-standard-replayable build.

This resolves the aggregation-error ambiguity exposed during OPE-45 review. A
custom model is not required to use built-in aggregation-error inputs merely
because they are present in a prepared dataset. If its serialized spec selects
that component, the derived bill of materials requires and validates the
corresponding representation. If it does not, the data remains optional cache
content and should not be computed at the PyMC boundary.

## Custom-stage substitution

The copied public spine is useful for more than recording the standard order.
It lets a project provide its own version of a step while reusing the rest of
the supported workflow. The custom-basis example tracked in OPE-54 should be
the first executable preparation-stage proof:

```python
merged = retrieve_or_reload_rhime_data(setup.data_args, multisector=False)
filtered = filter_rhime_observations(merged, setup.data_args)
basis = build_project_basis(filtered, setup.data_args)
sensitivities = build_rhime_sensitivities(
    filtered,
    basis,
    setup.data_args,
    multisector=False,
)
prepared = assemble_rhime_inputs(
    filtered,
    basis,
    sensitivities,
    setup.data_args,
)
```

The example should use tested source and public handoffs. It should show both
the power and the limit of the escape hatch: the custom stage owns the
scientific validity of its output, while the following stage contracts and the
model bill of materials validate the structural inputs they consume. If a
custom basis becomes common and stable, a later task can expose it through a
lower-ceremony argument; the copied runner remains the path for genuinely new
or project-specific steps.

## Delivery sequence

1. **OPE-44 / OPE-54 — executable extension proofs.** Publish the copied
   public spine and exercise both a model/likelihood substitution and a custom
   basis stage without private runner glue.
2. **OPE-46 — prepared-data and cache boundaries.** Make supported reload and
   externally supplied data paths explicit, including ownership, provenance,
   compatibility checks, and the scientific stage at which each artifact can
   re-enter the workflow.
3. **OPE-47 / OPE-82 — component-owned requirements.** Colocate the concrete
   model graph, component contracts, output roles, and recipe-owned input-name
   declarations. Use those declarations for PyMC materialization and keep
   validation in the consuming component.
4. **OPE-48 — outputs and reconstruction.** Consume the same backend-neutral
   scientific roles and concrete model contract for output compatibility and
   reconstruction. This work does not wait for serialized replay.
5. **OPE-55 / W5b — serialized model-bound replay, in parallel.** Define the
   versioned replay bundle that binds prepared data to a serialized model/run
   specification without turning reusable caches into model-specific
   artifacts. This branch may proceed after OPE-47 but does not block OPE-48.
6. **OPE-49 — user-facing consolidation.** Document choosing between standard
   options, a low-ceremony seam, a custom stage, a prepared-data cache, and an
   re-executable model-bound run.

OPE-55 is the concrete P0 delivery slice for GitHub issue #415 and the broader
component and reproducible-run-bundle roadmap tracked by OPE-21. It should
remain incremental, must not wait for or introduce a general semantic kernel,
and is not a prerequisite for W6.

## Acceptance evidence

The combined work should eventually demonstrate:

- a custom runner supplies cached prepared data without OpenGHG retrieval;
- a custom basis stage composes with the unchanged downstream public stages;
- two compatible model specs can validate and use the same prepared-data
  cache without mutating it;
- missing, malformed, or incompatible inputs fail with the owning component
  and option named;
- only inputs selected by the recipe-owned declaration cross the PyMC
  materialization boundary, and related Dask graphs compute together;
- a built-in model spec and prepared data serialize and round-trip as a
  replayable run artifact;
- a model-bound replay rejects a missing or incompatible spec before sampling
  or output writes; and
- complete custom builders clearly own validation and materialization and opt
  out of the standard recipe's replayable input contract.

## Non-goals

- No workflow or pipeline class, stage registry, dependency-injection
  framework, or semantic compiler.
- No caller-authored input, role, or output manifest for ordinary runners.
- No serialization of executable Python code.
- No requirement that a reusable prepared-data cache be tied to one model.
- No promise that arbitrary custom Python models are standard-replayable or
  produce bitwise-identical stochastic results.
- No broad module move outside the existing W4-W6 sequence.
