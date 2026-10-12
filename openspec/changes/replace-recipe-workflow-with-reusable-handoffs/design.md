# Design

## Context

See [proposal.md](proposal.md) for the replacement's motivation. The inspected
landed base is `devel` at `80b0ebf0` (11 October 2026). PRs #773/#774 and #822–#825
provide acquisition, prepared-value, inference and concrete configuration owners.
This plan consumes that groundwork; #798 remains unmerged prototype evidence.

Merged #826/#830 remove the obsolete checkpoint/cache workflow and provide dataset
basis inputs. The remaining stack is not completion of this change. #832 isolates
legacy utilities; #834 changes
observation-error ordering; #835/#836 own terminology/public entry points; #837
owns joint input materialization. PR #840 implements #838's explicit
basis paths and support maps. PR #841 implements #839's native runtime-flux
resolution and operator-only loading and remains under review. Recheck their
merged revisions before implementation.
Acquisition replay is deferred to #829 and does not block prepared/sample handoffs.

The [current configuration capability](../../specs/rhime-configuration/spec.md)
describes the landed flat preparation fields and complete requested configuration.
This proposal deliberately replaces those two constraints with grouped ownership
and independently resolved phases through an explicit
[configuration delta](specs/rhime-configuration/spec.md). Completed baseline
tasks do not establish implementation of this amendment. Only reconciled delivered
capabilities are synced; the four deltas in this change remain pending.

Read the [execution](specs/recipe-execution/spec.md),
[preparation](specs/prepared-recipe-handoff/spec.md) and
[sample](specs/sampled-recipe-handoff/spec.md) contracts for normative acceptance.

## Goals / Non-Goals

**Goals:** Readable scientific order, reusable merged/prepared data, compatible
inference reuse, and repeatable products from a completed run. Follow the
[RHIME development rules](../../../docs/development/rhime_model_development.rst)
and [array ownership rules](../../../docs/plans/numerical_data_ownership_and_execution_boundaries.md).

**Non-goals:** A workflow engine, phase inheritance, publisher/storage framework,
new numerical codecs, new inference algorithms or mandatory acquisition persistence.
TOML support, nested/linked checkpoint implementation and MAP remain separate.
Graph-free replay forbids graph construction, not imports of modules using PyMC.

The supported workflow uses trusted package-produced files and the configuration
for the run. Keep that context together or supply its existing parts explicitly;
an individual stage record need not explain the whole run by itself. This change
does not build portable authenticated bundles, protect against edited internal
objects, or support adversarial combinations of files. Validate external inputs
at their owning boundary and trust the resulting objects. Missing files and
genuine scientific incompatibilities still fail through the applicable loader or
scientific operation.

## Decisions

### 1. Group choices once, then resolve the selected phase

Use one complete requested `RhimeConfig` for full configured execution. Move its
existing preparation fields/defaults into `PreparationConfig`; do not keep
another authoritative flat copy or parallel default table. Keep `SiteOptions`,
`RhimeModelSpec`, `RhimeSampler` and `RhimeOutputSpec` as their existing owners.
One preparation record has a concrete purpose: it can be resolved and consumed
without constructing a model, sampler or output request. No replacement sampler
settings class, `ModelChoices` class or generic phase registry is required.

Illustrative pseudocode below describes the intended interfaces, not installed
API names. Decoded sections are ordinary mappings, not resolved records:

```python
sections = group_legacy_options(read_rhime_ini("run.ini"))
# {"preparation": {...}, "model": {...}, "sampler": {...}, "output": {...}}

preparation = PreparationConfig.from_params(sections["preparation"])
config = RhimeConfig(
    preparation=preparation,
    model=resolve_model_options(sections["model"], facts=preparation.requested_facts),
    sampler=resolve_sampler_options(sections.get("sampler", {})),
    output=RhimeOutputSpec.from_params(sections.get("output", {})),
)
result = run_standard(config=config)
```

The legacy INI adapter preserves supported aliases, duplicate-name precedence
and override precedence, then groups recognized options by their concrete owners.
Select a phase's values before normalizing them: calling today's complete
normalizer first would still reject irrelevant model/output settings. Unknown
names remain errors; known unrelated groups in a shared file remain inactive.
Explicit wrong-phase command overrides fail. A future section-preserving INI or
TOML frontend can supply the same mappings, without being required by this work.

Priors belong in model configuration, independently of preparation choices.
Prior-derived numerical preparation still depends on those model inputs as
described in the reuse table below.
Species, domain, requested dates and prepared source layout belong to preparation.
Final-product naming belongs to output; saved-stage destinations are explicit
arguments. Translate historically shared keys such as `use_bc` deliberately:
preparation determines available BC inputs, while inference chooses whether to use
them. Do not create independently writable copies of a preparation fact. Model
choices requiring absent BC/minimum-error arrays fail at scientific composition.
If a caller repeats a preparation fact at sampling, reject a conflicting claim;
inactive filters or basis-writing destinations do not become equality gates.

For standard and multisector, use the following binding rules. Omission means
absence of the key in the raw model mapping, not a prematurely filled model default.

| Choice | Preparation owner | Full and saved model binding |
| --- | --- | --- |
| `use_bc` | `PreparationConfig.use_bc` retains the existing default `True` and controls acquiring/constructing BC inputs. | An omitted model choice inherits the resolved preparation choice. Explicit `False` may ignore available BC inputs; explicit `True` requires them. |
| `bc_freq` | `PreparationConfig.bc_freq` retains the existing default `None` and controls the temporal expansion baked into the inputs. | Omission uses the saved/resolved preparation fact. A repeated equivalent value is a compatibility claim; a conflicting value fails, including when `use_bc=False`. No model default re-expands the data. |

The legacy flat `use_bc` option resolves once at its preparation owner and also
supplies the full run's model choice, preserving existing full-run behavior.
An explicitly supplied model group may choose `False` while preparation has
`True`; `True` with preparation `False` fails early for full configuration and
before model construction for saved sampling. Save the resolved preparation
BC choice and temporal policy as facts, and check the required arrays separately;
a missing array must not silently change a requested `True` into `False`.
`bc_freq=None` is a concrete existing temporal policy, not a marker for an omitted
model key. No duplicate default table or generic routing registry is required.
These rules cover standard/multisector; CO2 retains its family-specific defaults
and coherent preparation compatibility rules.

Full resolution can bind model defaults against resolved requested facts before
acquisition. Saved sampling binds the model section against prepared facts and
the associated run context, including the original date anchor, then aligns per-site options to retained
labels. The request and saved arrays are never mutated by this binding.

### 2. Live inference accepts an existing sampler or `"default"`

Use the public spelling `sampler`, not a new `sampler_options` object. Public
live entry points offering default sampler selection, including existing
prepared-input runners, accept `RhimeSampler | Literal["default"]`, defaulting
to `"default"`. A caller need not import `RhimeSampler` to request ordinary defaults.
Resolve that literal once at the concrete recipe's inference boundary into a fresh
sampler with that recipe's supported defaults. A supplied sampler retains its
explicit settings, subject to the recipe's existing compatibility checks. A full
configured run consumes `config.sampler`; it must not overwrite those choices
because a separate sampler argument was omitted. Keep the existing rejection of
a complete resolved configuration combined with additional raw overrides.

```python
def sample(prepared_manifest, *, model, sampler="default", output_dir):
    # Load prepared inputs and bind the scientific choices before inference.
    ...

sample(prepared_manifest, model=model_a, output_dir="runs/a")
sample(prepared_manifest, model=model_b, sampler="default", output_dir="runs/b")
# Advanced callers can instead pass their configured RhimeSampler.
```

`"default"` requests live behavior; it is not recorded as the settings actually
used. Save the resolved invocation's provenance. Do not use truthiness or
`sampler or RhimeSampler()` to resolve this choice. Every live sampling boundary rejects
`None` and unknown strings before sampling or artifact writes; `None` is not a
request to skip inference. Remove existing `None`-means-default adapters in the
same migration. Calls omitting the optional argument keep selecting defaults;
calls explicitly passing `None` must omit it or pass `"default"` instead, with a
clear migration error. Update signatures, docstrings, examples and tests together.
Lower-level operations that require an already resolved sampler can retain their
object-only argument rather than gain another default-selection boundary.
For cached CO2, defaults and explicit samplers still enter its matched construction,
step and prediction operation, never a generic sampler substitution.

These meanings are consistent even though each interface accepts only its
applicable subset:

| Value | Meaning |
| --- | --- |
| A supplied `RhimeSampler` | Use or retain that configured sampler |
| `"default"` | Request a fresh sampler; for standard execution, `RhimeSampler()` |
| `None` | No live sampler object is present; valid on results, invalid when sampling is requested |

There is no requirement for a three-way argument or a new "skip sampling" mode.
Preparation and replay do not select a sampler. A replay result with `None` can
still describe samples produced earlier through its saved provenance.

Take an invocation-local copy of the existing sampler and its owned keyword
containers. Preserve supported list/tuple/NumPy values and isolate overrides.
Do not indiscriminately deep-copy models, steps or borrowed scientific arrays.
Reject `idata_kwargs.coords` and `dims` once when constructing the sampler,
including through configuration parsing; model registration remains authoritative.
Do not repeat those checks during selection or sampling to detect later mutation.
Leave model-dependent argument validation to PyMC. Configuration may hold the existing
unbound sampler: strict separation into several settings/runtime classes is not
needed to achieve these guarantees.

A result is a record of what happened, not a request to execute defaults:

```python
# Proposed field in the existing result type; no default factory.
sampler: RhimeSampler | None = None

# Live result: snapshot from the invocation actually used.
result.output_metadata["sampler"] = describe_sampler(invocation_sampler)
# Replay result: sampler stays None; use the recorded description.
result.output_metadata["sampler"] = saved_sampler_metadata
```

Keep live sampler access available for existing direct callers. Shared outputs
consume the descriptive metadata in either route. `RhimeSampler` owns the small
explicit description of its actual settings; save that snapshot and reuse it.
Missing descriptive metadata remains unavailable rather than acquiring today's
defaults. It does not gate scientific postprocessing, and needs no strict-metadata
flag or separate parser for hypothetical future sampler options. Result/replay
construction never resolves `"default"` or performs inference. Ordinary parsing of
the associated run configuration is allowed; forbidding configuration-object
construction is not an acceptance criterion. Remove the existing `RhimeResult`
sampler default factory and adapt
the concrete result factories and output adapter together.
The old factory only constructs an unbound object; it does not sample or build a
model. Its problem is that omitting a result's sampler manufactures default
settings that need not describe the saved inference. A factory on a requested
configuration may still be appropriate because that object requests future work.

### 3. Keep scientific order visible; save only at explicit boundaries

Both full and saved routes call the same scientific owners. Small repeated
orchestration is preferable to a forwarding wrapper introduced only to give every
family identical stage names. The following sketch consumes #834/#837/#838/#839;
its helper names must be reconciled with those deliveries rather than implemented
again here:

```python
merged = acquire(explicit_acquisition_options)
with_errors = derive_observation_errors(merged, error_options)
filtered = filter_observations(with_errors, filters)
runtime_flux = resolve_runtime_flux(filtered, source_options)
basis = prepare_basis(
    selected_operator_path=basis_path,
    runtime_flux=runtime_flux,
    support_maps=support_maps,
)
site_data = build_sensitivities(filtered, basis, ...)
prepared = assemble_inputs(filtered, basis, site_data, ...)
# Only saved preparation adds this file boundary.
prepared_manifest = save_preparation(prepared, requested_facts, destination)
```

Preparation aligns metadata to retained observations; inference aligns its own
numeric per-site choices. Validate every supplied value, including dropped-site
entries, before selecting retained labels. Empty retained sets fail. Preserve
custom errors and derive missing errors before filtering/aggregation; nested
preparation does this for both domains before alignment, without promising new
nested checkpoint support.

Resolution remains data-free. Numerical inputs are borrowed and potentially lazy;
related arrays cross an explicit joint assembly, model-input, serialization or
eager-kernel boundary when that operation needs them. Consume #837's gathered-input
boundary rather than promising all prepared arrays remain lazy or moving all
materialization into model construction. Custom builders receive the supported
prepared representation and own any additional backend conversion they need.

A full runner continues visibly from preparation through construction, inference,
result reconstruction and products, without invoking file-handoff helpers. Saved
sampling uses the same concrete scientific operations:

```python
def sample(prepared_manifest, *, model, sampler="default", output_dir, run_context=None):
    record, paths, context = load_preparation_record(prepared_manifest, run_context)
    prepared = load_prepared(paths.prepared)
    resolved_model = resolve_model_options(model, facts=context.preparation)
    invocation = resolve_invocation_sampler(sampler, recipe="standard")
    # Construction combines the prepared arrays with the selected model choices.
    built = build_standard_model(prepared, resolved_model)
    posterior = sample_standard_model(built, invocation)
    return save_samples(
        output_dir, prepared_manifest, posterior, built.output_contract,
        run_config=paths.run_config,
        actual_inference=describe_invocation(resolved_model, invocation),
    )
```

Validate declared model coordinates/output roles even when no products are requested.
A custom model supporting only `none` is valid; it cannot claim unsupported replay
products. Product-specific requirements are checked when products are requested.
Use the model's existing variable roles/output contract as the shared source for
predictive sampling and postprocessing, rather than have the sampler infer a
second concentration role from variable names. Species-dependent mass conversion
belongs to postprocessing and uses the associated run context.

### 4. A saved handoff locates the run's existing inputs

Preparation and sampling return paths to completed records that locate their
numerical inputs. The associated run configuration and reporting context may be
referenced alongside those records or supplied explicitly; neither every file
nor every stage record must be self-contained. Use the configuration from the
completed run when interpreting its samples. Current output choices select
products and destinations, and cannot redefine the sampled science. A configured
full request can supply its preparation record; a preparation-only file can resolve
that record independently:

```python
preparation = PreparationConfig.from_params(sections["preparation"])
prepared = standard_stages.prepare(preparation, output_dir="runs/prepared")
a = standard_stages.sample(prepared, model=model_a, output_dir="runs/a")
b = standard_stages.sample(prepared, model=model_b, output_dir="runs/b")
result = standard_stages.postprocess(
    a, run_context=context,
    output=RhimeOutputSpec(output_format="basic", output_path="runs/products"),
)
```

The `model` argument here contains decoded model choices; the sampling boundary
resolves them against saved facts. Existing direct prepared-input scientific APIs
continue to accept their established resolved specifications. Do not hide an
ambiguous raw-or-resolved configuration union behind the staged argument.

In these sketches, `run_context` denotes an existing configuration path or mapping
plus the recipe's explicit reporting inputs, not a new universal context class.
Records store plain references and small descriptions, never runtime objects.

A small CLI selection table chooses the concrete family once from the run context
or saved record. There is no generic phase registry.
Keep existing staged command names where applicable, coordinating the full-run
entry point with #836. One shared file or phase-only files are both supported.
Prior prediction needs preparation, model and check choices; diagnosis remains
independently callable with posterior, optional supported sample record and check
options. Partial families need not implement unrelated operations.

### 5. Save numerical artifacts, then write the manifest last

`save_preparation` and `save_samples` are ordinary save functions around existing
numerical codecs. They replace the earlier `publish_*` terminology; there is no
publisher abstraction or remote publication. The final manifest marks completion.

```python
def save_samples(
    directory, prepared_manifest, posterior, contract,
    *, run_config, actual_inference,
):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    save_trace(posterior, directory / "posterior.nc")
    record = sample_record(
        preparation=str(Path(prepared_manifest).resolve()),
        posterior="posterior.nc",
        output_contract=contract.to_dict(),
        run_config=str(run_config) if run_config is not None else None,
        actual_inference=actual_inference,
    )
    write_json(directory / ".sample-manifest.tmp", record)
    os.replace(directory / ".sample-manifest.tmp", directory / "sample-manifest.json")
    return directory / "sample-manifest.json"
```

Use a fresh operation destination so sampling cannot silently overwrite a completed
run or its prepared inputs. Write the final manifest after numerical saves succeed;
a temporary JSON file and rename are enough. Do not add transaction machinery or
recursive symlink scans.

The optional `run_config` above is a reference to an existing configuration file.
If context is supplied only in memory, the caller supplies it again for later
postprocessing; saving does not serialize that runtime object.

Resolve relative references from their record's directory. Explicit absolute paths
are useful for shared preparation caches. Use those selected paths directly rather
than search for replacement files through ambient state. Several samples may refer
to the same preparation without copying it. Portable relocation of arbitrary file
trees and detection of changes during inference are outside this delivery.

The initial examples use existing NetCDF numerical codecs and JSON records. Existing
numerical Zarr support remains available; this change adds no directory hashing,
conversion framework or new codec. Loaders check the current format and required
contents once. Hashes are not mandatory stage gates or full-configuration reuse
keys; existing scientific affine associations remain owned by their numerical
loader/composition boundary.

### 6. Record only the facts and provenance that outputs need

| Location | Authoritative contents |
| --- | --- |
| Prepared numerical artifact | Retained observations/site metadata, coordinates, units, basis/operator and retained flux; existing numerical codec |
| Preparation record and associated configuration | Family and numerical references; original requested dates, species/domain and preparation choices at their existing owner |
| Sample record | Preparation and posterior references, embedded `OutputContract`, actual inference/sampler description and reference to the run context |
| Run context | Configuration used for the run and any required reporting/reconstruction inputs; may be retained beside records or supplied explicitly |
| Current output policy | Supported products, reporting maps/options, names and destinations |

Do not duplicate numerical coordinate arrays or conflicting preparation facts in
every record. Keep existing configuration files and numerical artifacts together
instead of inventing an export codec for every runtime object. Declare the small
record fields explicitly; do not infer a schema from `__slots__`. Callable identity
and supported arguments are descriptive, never imports or executable instructions.
Encoding numerical provenance is an explicit serialization boundary, not a hidden
computation performed by configuration access or a compatibility hash.

The current output-consumer inventory establishes the standard slice's boundary:

| Consumer | Saved information it needs |
| --- | --- |
| Run metadata and PARIS date presentation | Original requested start/end, not observation extrema |
| Output view/PARIS | Retained sites, averaging, footprint/basis provenance from prepared data; species/domain from facts |
| Reconstruction | Existing `OutputContract` roles and state mapping |
| Legacy products/attributes | `use_bc`, actual prior descriptions, `sigma_per_site`, sampler burn/tune/chains and related existing descriptive fields |
| Sector attribution | Recorded sector/source descriptions, retaining source order |
| Sampler summary | Actual draws, burn, tune, chains and backend, plus supported invocation provenance |

Use this inventory to pass existing run context to each supported product. Preserve
available timing metadata, package version and Git revision when available; an
unknown revision does not prevent execution. CO2 keeps its trace/affine meaning.

```python
def postprocess(sample_manifest, *, output, run_context=None):
    sample, paths, context = load_sample_record(sample_manifest, run_context)
    prepared = load_prepared(paths.prepared)
    posterior = load_trace(paths.posterior)
    contract = decode_output_contract(sample.output_contract)
    validate_requested_products(output, prepared, contract)
    preflight_product_destinations(output)
    result = make_standard_result(
        prepared=prepared, posterior=posterior, contract=contract,
        requested_window=context.requested_window,
        inference_metadata=sample.inference_metadata,
        sampler=None, output=output,
    )
    make_standard_outputs(result, prepared)
    return result
```

The loader owns supported-format and required-content checks. Existing scientific
composition owns label, unit and reconstruction compatibility; later code trusts
its results. Replay does not build a model, sample, or execute recorded callables.
It may parse the original run configuration without treating its provenance as a
new scientific request. Shared result/product owners accept that context; remove
the competing stage-owned `basic.nc` write and post-write artifact discovery.
One output policy and one product-writing owner govern both live and saved routes.
In-memory products remain available without a destination when no writing is requested.

### 7. Reuse stops at choices already encoded by preparation

| Later choice | Reuse? |
| --- | --- |
| Standard/multisector graph-time scaling, BC or offset priors | Yes, when required arrays/layout exist |
| Supported likelihood, available minimum-error input, sampler settings | Yes, after the selected component's scientific checks |
| Observations, filters, averaging, native inventory, basis/sensitivities, `bc_freq` or baked error values | New preparation required |
| CO2 independent BC/offset priors or supported mismatch settings | Yes, preserving the coherent reduced flux calculation |
| Defining CO2 native prior/covariance, operator or projection | Newly coherent reduction and matching affine companion required |
| Cached covariance/eigenbasis inputs | Matching or rebuilt runtime cache required |
| Product naming, destination or compatible reporting map | Yes, using the saved inference |

#838/#839 distinguish operator reuse for new preparation from full retained-object
restoration for replay. Extend #840's operator/current-flux test through the actual
saved workflow, without reimplementing its basis policy:

```python
prepared_B = prepare(current_flux=B, selected_operator=saved_operator_A)
assert prepared_B.retained_flux == B
saved_B = save_preparation(prepared_B, requested_facts, destination)
# Reload after disabling acquisition; same state gives the same flux based on B.
assert reconstruct(load_preparation(saved_B), state) == reconstruct(prepared_B, state)
```

Cover first-source weighting separately from all-source retained flux, source order
and signed flux. CO2's native prior moments, effective operator, affine intercept
and unresolved covariance remain one coherent preparation; replacing one array
with matching labels cannot establish compatibility.

## Migration Plan

Four implementation slices of OPE-207 are planned in [tasks.md](tasks.md): a
complete standard proof, multisector adoption, ordinary CO2 adoption, and cached
CO2 adoption. Review the standard working route before family expansion;
multisector and ordinary CO2 can then proceed independently, with cached adoption
following the ordinary handoff conventions. Each slice includes writer, reader,
callers, CLI/examples, documentation and validation. Do not split out a speculative
persistence framework or preliminary cleanup project as a prerequisite.

#808 coordinates removal of representation-sensitive gates and the corresponding
consumer fixes within these implementation slices. #838/#839
keep basis/path/flux ownership; #829 remains deferred. OPE-164 is completed initial
CO2 staging evidence; new CO2 work is adoption of these replacement handoffs.
OPE-165 retains linked CO2/O2 integration and depends on the common interfaces it
actually consumes, not completion of every OPE-207 family adoption. Complete
[#769](https://github.com/openghg/openghg_inversions/issues/769)'s production
predictive integration using [#795](https://github.com/openghg/openghg_inversions/pull/795)
before adding new cached predictive-stage plumbing. It replaces manual joint
prediction with the standard PyMC route while preserving the efficient cached
state step. Configuration, preparation reuse, ordinary sampling, diagnostics and
postprocessing can proceed in parallel. Do not extend a temporary prediction path
that this integration will remove. Nested and MAP remain future examples:
load the appropriate preparation, call that family's science, and save only when
that result has its own explicit replay contract.

For new CO2 preparation, use the existing numerical save/load and affine association
operations, then save the run record last. Keep those scientific associations at
their existing boundary instead of building another chain of digest gates. Obtain
original requested dates from the run context where outputs need them, rather than
inferring the inversion window from surviving observations. The configured full
CO2 product entry point must be named and tested, while raw runners retain their
numerical returns. Cached adoption tests its matched step order, accepted-cache
updates, conditional predictions and target density against scientific references.

Use 0.7.x to consume 0.7.x staged artifacts; 0.8.x does not promise compatibility.
No historical hash emulation, migration layer or setup aliases are needed solely
for retired staged contracts. Reject unsupported formats clearly.
Preserve numerical codecs, scientific customization and product schemas; coordinate
independent keyword/CLI transitions with their owners. Diagnosis's historical sample
support is separate from scientific replay. Readiness retains its family-specific
catch boundaries and fail/pass/unknown behavior. Announce breaking staged changes
with implementation release notes, not with a planning-only completion claim.

## Risks / Trade-offs

- Grouped configuration can duplicate defaults: move preparation ownership rather
  than keeping flat and nested settings in parallel; test full/phase equivalence.
- Defaults can hide invocation state: resolve `"default"` freshly in the live
  recipe and save actual settings; replay does not substitute current defaults.
- Saved metadata can grow into another configuration system: keep the run's
  existing configuration available and save only small explicit descriptions.
- Shared files can disappear: report missing inputs through their loader; use the
  selected run context without implicit copies or fallback search.
- A broad plan can repeat #798: require real standard scientific evidence and
  a readability review before adoption by other families.
