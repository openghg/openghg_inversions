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

**Goals:** Readable scientific order, one path per saved handoff, compatible
inference reuse, and products reconstructed from saved meaning. Follow the
[RHIME development rules](../../../docs/development/rhime_model_development.rst)
and [array ownership rules](../../../docs/plans/numerical_data_ownership_and_execution_boundaries.md).

**Non-goals:** A workflow engine, phase inheritance, publisher/storage framework,
new numerical codecs, new inference algorithms or mandatory acquisition persistence.
TOML support, nested/linked checkpoint implementation and MAP remain separate.
Graph-free replay forbids graph construction, not imports of modules using PyMC.

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

Species, domain, requested dates and prepared source layout belong to preparation.
Final-product naming belongs to output; saved-stage destinations are explicit
arguments. Translate historically shared keys such as `use_bc` deliberately:
preparation determines available BC inputs, while inference chooses whether to use
them. Do not create independently writable copies of a preparation fact. Model
choices requiring absent BC/minimum-error arrays fail at scientific composition.
If a caller repeats a preparation fact at sampling, reject a conflicting claim;
inactive filters or basis-writing destinations do not become equality gates.

Full resolution can bind model defaults against resolved requested facts before
acquisition. Saved sampling binds the model section against authenticated prepared
facts, including the original date anchor, then aligns per-site options to retained
labels. The request and saved arrays are never mutated by this binding.

### 2. Live inference accepts an existing sampler or `"default"`

Use the public spelling `sampler`, not a new `sampler_options` object. The new
live sampling entry point accepts `RhimeSampler | Literal["default"]`, defaulting
to `"default"`. A caller need not import `RhimeSampler` to request ordinary defaults.
Resolve that literal once at the concrete recipe's inference boundary into a fresh
sampler with that recipe's supported defaults. A supplied sampler retains its
explicit settings, subject to the recipe's existing compatibility checks. A full
configured run consumes `config.sampler`; it must not overwrite those choices
because a separate sampler argument was omitted. Keep the existing rejection of
a complete resolved configuration combined with additional raw overrides.

```python
def sample(prepared_manifest, *, model, sampler="default", output_dir):
    # Authentication and scientific binding precede the inference shown below.
    ...

sample(prepared_manifest, model=model_a, output_dir="runs/a")
sample(prepared_manifest, model=model_b, sampler="default", output_dir="runs/b")
# Advanced callers can instead pass their configured RhimeSampler.
```

`"default"` requests live behavior; it is not recorded as the settings actually
used. Save the resolved invocation's provenance. Do not use truthiness or
`sampler or RhimeSampler()` to resolve this choice. The new live boundary rejects
`None` and unknown strings before sampling or artifact writes; `None` is not a
request to skip inference. Established direct scientific APIs that already use
`None` for default selection can retain that spelling in their outer adapters.
For cached CO2, defaults and explicit samplers still enter its matched construction,
step and prediction operation, never a generic sampler substitution.

Take an invocation-local copy of the existing sampler and its owned keyword
containers. Preserve supported list/tuple/NumPy values and isolate overrides.
Do not indiscriminately deep-copy models, steps or borrowed scientific arrays.
Reject `idata_kwargs.coords` and `dims` at resolution and direct sampling;
model registration remains authoritative. Configuration may hold the existing
unbound sampler: strict separation into several settings/runtime classes is not
needed to achieve these guarantees.

A result is a record of what happened, not a request to execute defaults:

```python
# Proposed field in the existing result type; no default factory.
sampler: RhimeSampler | None = None

# Live result: snapshot from the invocation actually used.
result.output_metadata["sampler"] = describe_sampler(invocation_sampler)
# Replay result: sampler stays None; use validated recorded values.
result.output_metadata["sampler"] = saved_sampler_metadata
```

Keep live sampler access available for existing direct callers. Shared outputs
consume the descriptive metadata in either route. Result/replay construction never
resolves `"default"`, creates a sampler, or treats recorded provenance as executable
configuration. Remove the existing `RhimeResult` sampler default factory and adapt
the concrete result factories and output adapter together.

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
def sample(prepared_manifest, *, model, sampler="default", output_dir):
    record, paths, preparation_identity = verify_preparation(prepared_manifest)
    prepared = load_prepared(paths)
    resolved_model = resolve_model_options(model, facts=record.facts)
    check_scientific_requirements(prepared, resolved_model, record.baked_policy)
    invocation = resolve_invocation_sampler(sampler, recipe="standard")
    built = build_standard_model(prepared, resolved_model)
    posterior = sample_standard_model(built, invocation)
    validate_output_meaning(prepared, posterior, built.output_contract)
    return save_samples(
        output_dir, prepared_manifest, posterior, built.output_contract,
        preparation_identity=preparation_identity,
        actual_inference=describe_invocation(resolved_model, invocation),
    )
```

Validate declared model coordinates/output roles even when no products are requested.
A custom model supporting only `none` is valid; it cannot claim unsupported replay
products. Product-specific requirements are checked when products are requested.

### 4. A saved handoff is one manifest path

Preparation and sampling return absolute paths to completed manifests. Their callers
do not repeat numerical paths or retain the original configuration. A configured
full request can supply its preparation record; a preparation-only file can resolve
that record independently:

```python
preparation = PreparationConfig.from_params(sections["preparation"])
prepared = standard_stages.prepare(preparation, output_dir="runs/prepared")
a = standard_stages.sample(prepared, model=model_a, output_dir="runs/a")
b = standard_stages.sample(prepared, model=model_b, output_dir="runs/b")
result = standard_stages.postprocess(
    a, output=RhimeOutputSpec(output_format="basic", output_path="runs/products")
)
```

The `model` argument here contains decoded model choices; the sampling boundary
resolves them against saved facts. Existing direct prepared-input scientific APIs
continue to accept their established resolved specifications. Do not hide an
ambiguous raw-or-resolved configuration union behind the staged argument.

A small CLI selection table chooses the concrete family once. Sampling/replay infer
family from a checked supported manifest; an explicitly selected family must agree.
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
    *, preparation_identity, actual_inference,
):
    directory = Path(directory).resolve()
    check_destination_and_dependencies(directory, prepared_manifest)
    directory.mkdir(parents=True, exist_ok=False)
    save_trace(posterior, directory / "posterior.nc")
    record = sample_record(
        preparation=relative_reference(
            prepared_manifest, directory, digest=preparation_identity,
        ),
        posterior=relative_reference_with_digest(directory / "posterior.nc", directory),
        output_contract=contract.to_dict(),
        actual_inference=actual_inference,
    )
    verify_required_dependencies(record, directory)
    encoded = encode_record(record)
    write_exclusive(directory / ".sample-manifest.tmp", encoded)
    os.replace(directory / ".sample-manifest.tmp", directory / "sample-manifest.json")
    return directory / "sample-manifest.json"
```

Use a fresh operation destination; reject existing destinations and any write
that would replace a dependency. Writer-owned files stay under that destination.
References never authorize writes through dependencies. An interrupted save may
leave incomplete files, but no completed manifest. A temporary file and rename
prevent readers seeing partial final JSON; this is not rollback, resumable
transactions, concurrent-writer coordination or a power-loss durability promise.

References are relative to their declaring manifest, including `../prepared/...`.
Authenticate every required transitive dependency; never consult CWD, `RUN_ROOT`,
absolute saved paths, URIs or environment placeholders. Several samples reference
one preparation without copying it. Moving an intact dependency tree works; a
sample directory alone need not be self-contained. Hash exact bytes: re-encoding
creates a new artifact, while unchanged relocated content retains its identity.
Digests check content association, not author identity or scientific equivalence.
Sampling retains the preparation identity verified before inference and passes it
to saving. Before completing the sample manifest, verify the current preparation
and its required dependencies against that retained identity; fail if they have
changed. Never replace the retained identity with a new digest computed after
inference. This preserves the association with the inputs actually selected for
sampling without adding a transaction framework.

### 6. Record only the facts and provenance that outputs need

| Location | Authoritative contents |
| --- | --- |
| Prepared numerical artifact | Retained observations/site metadata, coordinates, units, basis/operator and retained flux; existing numerical codec |
| Preparation manifest | Envelope/family, numerical references/digests, original requested dates, species/domain and missing production/baked-policy facts; requested choices as provenance |
| Sample manifest | Envelope/family/MCMC kind, exact preparation-manifest reference/digest, posterior reference/digest, embedded `OutputContract`, actual inference/sampler provenance and bounded output context |
| Current output policy | Supported products, reporting maps/options, names and destinations |

Do not duplicate numerical coordinate arrays or conflicting preparation facts in
every record. A sample refers to preparation transitively. Do not serialize the
runtime dataclass layout of `RhimeConfig`, `RhimeModelSpec` or `RhimeSampler` as a
replay schema. Declare the small metadata encoding explicitly; callable identity
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

Verify this inventory against each supported product rather than reconstructing
all current configuration classes to obtain defaults. Preserve available timing
metadata. CO2 keeps its authenticated trace/affine representation instead of being
forced into standard multiplicative output meaning.

```python
def postprocess(sample_manifest, *, output):
    sample, preparation, paths = verify_complete_dependency_tree(sample_manifest)
    prepared = load_prepared(paths.prepared)
    posterior = load_trace(paths.posterior)
    contract = decode_output_contract(sample.output_contract)
    validate_output_associations(prepared, posterior, contract)
    validate_requested_products(output, prepared, contract)
    preflight_product_destinations(output)
    result = make_standard_result(
        prepared=prepared, posterior=posterior, contract=contract,
        requested_window=preparation.requested_window,
        inference_metadata=sample.inference_metadata,
        sampler=None, output=output,
    )
    make_standard_outputs(result, prepared)
    return result
```

Supported versions, reference structure, exact associations and all required
digests are checked before posterior loading or product destination creation.
After loading, validate scientific labels, units and roles before writes. Matching
digests do not replace these checks. Replay never validates a newly reconstructed
model configuration, instantiates a sampler, imports recorded callables or recovers
a graph. Shared result/product owners accept explicit saved provenance; remove
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

Four implementation children of OPE-207 are planned in [tasks.md](tasks.md): a
complete standard proof, multisector adoption, ordinary CO2 adoption, and cached
CO2 adoption. Review the standard working route before family expansion;
multisector and ordinary CO2 can then proceed independently, with cached adoption
following the ordinary handoff conventions. Each slice includes writer, reader,
callers, CLI/examples, documentation and validation. Do not split out a speculative
persistence framework as a prerequisite. Child issue identifiers are assigned when
the issues are created; this document does not claim their creation.

#808 keeps its bounded hash/compatibility contract and consumer fixes. #838/#839
keep basis/path/flux ownership; #829 remains deferred. OPE-164 is completed initial
CO2 staging evidence; new CO2 work is adoption of these replacement handoffs.
OPE-165 retains linked CO2/O2 integration. Nested and MAP remain future examples:
load the appropriate preparation, call that family's science, and save only when
that result has its own explicit replay contract.

For new CO2 preparation, save numerical inputs, obtain their exact identity,
bind/save the affine companion, validate their association, then save the complete
manifest last. Imported pairs retain their exact binding. Require authoritative
original requested dates for the supported output path; missing required facts
fail rather than being inferred from surviving observations. The configured full
CO2 product entry point must be named and tested, while raw runners retain their
numerical returns. Cached adoption tests its matched step order, accepted-cache
updates, conditional predictions and target density against scientific references.

Use 0.7.x to consume 0.7.x staged artifacts; 0.8.x does not promise compatibility.
No historical hash emulation, migration layer or setup aliases are needed solely
for retired staged contracts. Reject unsupported/down-labelled envelopes clearly.
Preserve numerical codecs, scientific customization and product schemas; coordinate
independent keyword/CLI transitions with their owners. Diagnosis's historical sample
support is separate from scientific replay. Readiness retains its family-specific
catch boundaries and fail/pass/unknown behavior. Announce breaking staged changes
with implementation release notes, not with a planning-only completion claim.

## Risks / Trade-offs

- Grouped configuration can duplicate defaults: move preparation ownership rather
  than keeping flat and nested settings in parallel; test full/phase equivalence.
- Defaults can hide invocation state: resolve `"default"` freshly in the live
  recipe and save actual settings; replay never reconstructs them.
- Saved metadata can grow into executable configuration: admit fields by their
  output consumer and test with configuration/model/sampler construction disabled.
- Shared files can disappear: fail on missing/changed dependencies and explain
  relocation; no implicit copy or fallback search.
- A broad plan can repeat #798: require real standard scientific evidence and
  a readability review before adoption by other families.
