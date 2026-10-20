# Design

## Context

The maintainer's 20 October 2026 amendment is delivered as OPE-219 → OPE-220 →
OPE-221. It supersedes earlier cached public-runner preservation: one configured
`run_co2` returns `Co2Output`, and the scientific fixed-OU/inferred-amplitude choice
selects the matched cached sampler internally. There is no public cached variant,
algorithm switch, silent fallback or CO2 compatibility alias requirement.
Existing other likelihoods retain their appropriate inference operations.

OPE-219 first constructs a supported independent positive amplitude prior once
in the scientific graph. The conditional amplitude update consumes that graph's
prior density with its Jacobian and physical-amplitude transform. This avoids
duplicating a HalfNormal distribution or assuming every sampling coordinate is a
log amplitude. Initialization uses the prior's support point or explicit physical
values and must be valid in physical and transformed coordinates. The covariance
target and accepted-cache lifecycle remain unchanged.

OPE-220 supplies shared scientific inference selection and configured full
execution; OPE-221 migrates saved phases onto the ordinary CO2 handoff/product
owners. The complete trace and bound affine reconstruction already fit the merged
`Co2Output`, so the 17 October common-carrier proposal does not require another
carrier migration. Linked numerical callers follow shared prior changes;
linked saved workflows and programme closeout retain their separate owners.

See [proposal.md](proposal.md) for the replacement's motivation. The inspected
landed base is `devel` at `7269f50f` (12 October 2026). PRs #773/#774 and #822–#825
provide acquisition, prepared-value, inference and concrete configuration owners.
This plan consumes that groundwork; #798 remains unmerged prototype evidence.

Merged #826/#830 remove the obsolete checkpoint/cache workflow and provide dataset
basis inputs. Merged #832 isolates legacy utilities; #834 changes
observation-error ordering; #835/#836 provide terminology/public entry points;
#837 provides joint input materialization. These deliveries do not complete this
change. PR #840 implements #838's explicit
basis paths and support maps. PR #841 implements #839's native runtime-flux
resolution and operator-only loading and remains under review. Recheck their
merged revisions before implementation.
Acquisition replay is deferred to #829 and does not block prepared/sample handoffs.

The [configuration capability](../../specs/rhime-configuration/spec.md) records
grouped configuration and independent phase resolution delivered by merged #866.
This handoff change consumes that baseline without a duplicate configuration delta. Its three
execution/handoff deltas remain pending, and completed baseline tasks do not
establish their implementation.

Merged #854 supplies current [model-selection](../../specs/rhime-model-selection/spec.md)
and [model-composition](../../specs/rhime-model-composition/spec.md) baselines.
Preserve the Python selector and installed full-run model choices, argument
forwarding and established aliases. The ordinary CO2 selection amendment approved on 16 October 2026 routes both
Python and CLI selection to `run_co2`, starting from coherent prepared inputs.
It removes the redundant ordinary raw runner and bundled builder; trace-only
callers compose the existing numerical operations. Cached/linked prepared-input
selection remains unchanged. No CO2 acquisition/native-preparation route is added;
that separate scientific work is tracked in #890.

Construction reuses `CoordRegistry` and the existing custom-likelihood seam,
including canonical `y`/`epsilon`, explicit kwargs, conflict precedence and
bounded callable provenance. Sharing declared output roles with prediction does
not relax that seam or turn callable metadata into executable instructions.
Complete-model builders retain their separately declared roles. The registry's
missing-auxiliary-value limitation remains tracked by #856; this plan does not
claim that fix as delivered or add repeated downstream coordinate checks.

Read the [execution](specs/recipe-execution/spec.md),
[preparation](specs/prepared-recipe-handoff/spec.md) and
[sample](specs/sampled-recipe-handoff/spec.md) contracts for normative acceptance.

## Goals / Non-Goals

**Goals:** Readable scientific order, reusable merged/prepared data, compatible
inference reuse, and repeatable products from a completed run. Follow the
[RHIME development rules](../../../docs/development/rhime_model_development.rst)
and [array ownership rules](../../../docs/development/numerical_array_ownership.rst).

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
existing acquisition/preparation fields/defaults into `RhimeConfig.data: DataConfig`;
do not keep another authoritative flat copy or parallel default table. Keep `SiteOptions`,
`RhimeModelSpec`, `RhimeSampler` and `RhimeOutputSpec` as their existing owners.
One data record has a concrete purpose: it can be resolved and consumed
without constructing a model, sampler or output request. No replacement sampler
settings class, `ModelChoices` class or generic phase registry is required.
The record covers acquisition as well as preparation. Use `data` for both the
complete configuration field and the grouped mapping key. The runner explicitly
selects acquisition arguments and preparation arguments from that record;
sharing it does not combine the operations. The proposed `preparation` field/key
has not shipped and needs no compatibility alias.

Illustrative pseudocode below describes the intended interfaces, not installed
API names. Decoded sections are ordinary mappings, not resolved records:

```python
sections = group_params(read_rhime_ini("run.ini"))
# {"data": {...}, "model": {...}, "sampler": {...}, "output": {...}}
config = RhimeConfig.from_params(sections, multisector=False)
result = run_standard(config=config)
```

The legacy INI adapter preserves supported aliases, duplicate-name precedence
and override precedence, then groups recognized options by their concrete owners.
Select a phase's values before normalizing them: calling today's complete
normalizer first would still reject irrelevant model/output settings. Unknown
names remain errors; known unrelated groups in a shared file remain inactive.
Explicit wrong-phase command overrides fail. A future section-preserving INI or
TOML frontend can supply the same mappings, without being required by this work.
Use the same alias precedence across full and saved requests, including a
canonical file option combined with a deprecated override. Report all unsupported
and wrong-phase override keys together before scientific work.

Resource-path interpretation and runner-owned optional basis saving follow
the [delivered resource-I/O contract](../../specs/rhime-resource-io/spec.md)
from #868. Manifest-relative dependency references retain their handoff owner.
Reuse `resolve_basis_inputs` and the existing codecs; this plan consumes those
operations rather than adding another resource or persistence abstraction.

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
| `use_bc` | `DataConfig.use_bc` retains the existing default `True` and controls acquiring/constructing BC inputs. | An omitted model choice inherits the resolved preparation choice. Explicit `False` may ignore available BC inputs; explicit `True` requires them. |
| `bc_freq` | `DataConfig.bc_freq` retains the existing default `None` and controls the temporal expansion baked into the inputs. | Omission uses the saved/resolved preparation fact. A repeated equivalent value is a compatibility claim; a conflicting value fails, including when `use_bc=False`. No model default re-expands the data. |

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
def sample(preparation_manifest, *, model, sampler="default", output_dir):
    prepared, data_config, generation = load_preparation(preparation_manifest)
    model_spec = resolve_model_options(model, data_config=data_config)
    invocation = resolve_sampler(sampler)
    inputs, priors = materialize_standard_inputs(prepared, model_spec)
    built = build_standard_model(inputs, model_spec, priors=priors)
    posterior = sample_rhime_model(built, invocation)
    # Record constructed priors and callable provenance once on the trace.
    return save_samples(..., preparation_generation=generation)
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
The 20 October amendment above supersedes explicit sampler selection for configured
CO2 fixed-OU inference. In the earlier cached interface, defaults and explicit samplers enter its matched construction,
step and prediction operation, never a generic sampler substitution.

These meanings are consistent even though each interface accepts only its
applicable subset:

| Value | Meaning |
| --- | --- |
| A supplied `RhimeSampler` | Use that configured sampler for live execution |
| `"default"` | Request a fresh sampler; for standard execution, `RhimeSampler()` |
| `None` | No sampler supplied; invalid when sampling is requested |

There is no requirement for a three-way argument or a new "skip sampling" mode.
Preparation and replay do not select a sampler. Scientific results have no live
sampler field and describe completed inference through saved provenance.

Take an invocation-local copy of the existing sampler and its owned keyword
containers. Preserve supported list/tuple/NumPy values and isolate overrides.
Do not indiscriminately deep-copy models, steps or borrowed scientific arrays.
Reject `idata_kwargs.coords` and `dims` once when constructing the sampler,
including through configuration parsing; model registration remains authoritative.
Do not repeat those checks during selection or sampling to detect later mutation.
Leave model-dependent argument validation to PyMC. Configuration may hold the existing
unbound sampler: strict separation into several settings/runtime classes is not
needed to achieve these guarantees.

Standard and multisector results use the existing `InversionOutput` scientific view, including
when no files are requested. Its inference description records what happened;
no live model/sampler or output policy is retained. Model inspection uses the
eager builder directly. Nested uses `NestedOutput`, and ordinary and cached CO2
use `Co2Output`. This supersedes the earlier
proposal to preserve the standard result envelope or a deprecated `run_rhime`
wrapper. The Python `run_rhime` name is removed in 0.8 with import guidance;
`run-rhime` remains the unchanged CLI spelling and calls `run_standard`.

The maintainer's 16 October multisector correction also retires the remaining
`run_rhime_from_prepared_inputs` route and multisector build/result/output
adapters. Prepared callers use `run_multisector(prepared=..., config=...)`;
complete custom models compose ordinary materialization, construction, sampling
and output operations directly. Multisector does not retain a `RhimeRunSpec`
complete-builder context or a `RhimeResult` compatibility route. After nested and
cached CO2 adoption, the unused shared envelope types and nested validation
context are removed.

### 3. Keep scientific order visible; save only at explicit boundaries

Both full and saved routes call the same scientific owners. Small repeated
orchestration is preferable to a forwarding wrapper introduced only to give every
family identical stage names. The following sketch consumes #834/#837/#838/#839;
its helper names must be reconciled with those deliveries and #868 rather than
implemented again here. Acquisition and preparation receive separate explicit
selections from `config.data`; resource paths/maps have already been resolved:

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
# Optional export is a runner operation using the scientific result.
if basis_output_path is not None:
    basis.save(basis_output_path)
# Only saved preparation adds this file boundary.
prepared_manifest = save_preparation(prepared, requested_facts, destination)
```

Acquisition or supplied-data selection stays visible in the runner. A shared
scientific preparation function accepts the acquired handoff and never retrieves
it internally. Optional basis saving follows the resource-I/O contract. When the
scientific calls are grouped in a preparation function, the runner obtains the
basis from its returned prepared handoff and uses the existing save codec.

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
entry points call the same step functions; they differ only by loading a checkpoint
before and saving one after (see decision 8). Saved sampling uses the same
concrete scientific operations:

```python
def sample(preparation_manifest, *, model, sampler="default", output_dir):
    prepared, data_config, generation = load_preparation(preparation_manifest)
    model_spec = resolve_model_options(model, data_config=data_config)
    inputs, priors = materialize_standard_inputs(prepared, model_spec)
    built = build_standard_model(inputs, model_spec, priors=priors)
    posterior = sample_rhime_model(built, resolve_sampler(sampler))
    return save_samples(..., preparation_generation=generation)
```

Validate declared model coordinates/output roles even when no products are requested.
Default traces retain scientific output states such as `x`, including when they
are named deterministics derived from `_latent` variables. Posterior `var_names` overrides are
rejected at the sampler boundary; intentional predictive selection remains
supported. General reconstruction remains with #850.
A retained multisector custom model supporting only `none` is valid; it cannot claim unsupported replay
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
full request can supply its data record; a preparation-only invocation can resolve
the same `data` mapping independently:

```python
data_config = DataConfig.from_params(sections["data"])
prepared = standard_stages.prepare(data_config, output_dir="runs/prepared")
a = standard_stages.sample(prepared, model=model_a, output_dir="runs/a")
b = standard_stages.sample(prepared, model=model_b, output_dir="runs/b")
result = standard_stages.postprocess(
    a,
    output=RhimeOutputSpec(
        output_format="paris", output_path="runs/products", country_file="countries.nc",
    ),
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
entry point delivered by #836 and specified in the model-selection baseline.
One shared file or phase-only files are both supported for these phase commands.
Prior prediction needs preparation, model and check choices; diagnosis remains
independently callable with posterior, optional supported sample record and check
options. Partial families need not implement unrelated operations.

### 5. Save numerical artifacts, then write the manifest last

`save_preparation` and `save_samples` are ordinary save functions around existing
numerical codecs. They replace the earlier `publish_*` terminology; there is no
publisher abstraction or remote publication. The final manifest marks completion.

```python
prepared, data_config, generation = load_preparation(preparation_manifest)
# Build/sample with these scientific inputs, then save using their generation.
sample_manifest = save_samples(
    preparation_manifest, posterior,
    preparation_generation=generation,
    output_contract=built.output_contract, model=model_spec,
    actual_inference=sampler.describe(), output_dir=sample_directory,
)
```

Let callers/orchestrators choose destinations and manage collision avoidance and
run isolation; existing directories are accepted. Write the final manifest after
numerical saves succeed;
a temporary JSON file and rename are enough. Do not add transaction machinery or
recursive symlink scans.
Checkpoint containment applies to checkpoint-owned files. Separately requested
basis exports follow the resource-I/O contract; this handoff design does not
introduce a new export restriction.

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
| Sample record | Preparation/posterior references, actual inference/sampler description and bounded run context; standard/multisector embed `OutputContract`, while CO2 retains trace roles and affine associations |
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
inference descriptions, package version and Git revision when available; an
unknown revision does not prevent execution. CO2 keeps its trace/affine meaning.

```python
def postprocess(sample_manifest, *, output):
    validate_standard_output_resources(output)
    inv_out = load_samples(sample_manifest)  # decode the borrowed scientific view
    postprocess_standard_outputs(inv_out, output)
    save_inversion_output(inv_out, output)
    return inv_out
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
reloaded_B, data_config, generation = load_preparation(saved_B)
assert reconstruct(reloaded_B, state) == reconstruct(prepared_B, state)
```

Cover first-source weighting separately from all-source retained flux, source order
and signed flux. CO2's native prior moments, effective operator, affine intercept
and unresolved covariance remain one coherent preparation; replacing one array
with matching labels cannot establish compatibility.

### 8. Standard runners visibly compose the same scientific operations

Revised on 15 October 2026 following the eager-builder and standard-result
maintainer decisions. This supersedes the earlier three-builder chain,
`RhimeResult` return/compatibility-wrapper recommendations and replay sketch.

Full and staged execution share preparation, one eager construction operation,
inference and scientific output construction. Callers explicitly materialize
selected inputs and enabled prior parameters together where needed; saved
preparation is already eager, but direct Python priors may remain lazy until this
call. No mandatory prior-materialization stage or new CLI command is introduced.

```python
validate_standard_output_resources(output_spec)
if supplied_prepared is None:
    basis_inputs = resolve_basis_inputs(data_config)
    bc_basis = (
        load_bc_basis(data_config.bc_basis_paths)
        if data_config.use_bc and data_config.bc_basis_paths is not None else None
    )
    if merged_data is None:
        merged_data = acquire_merged_data(data_config)
    else:
        merged_data.validate_for_preparation(
            **data_config.select("species", "domain", "start_date", "end_date", "split_by_sectors"),
        )
    prepared = prepare_standard(
        merged_data, basis_inputs=basis_inputs, bc_basis=bc_basis,
        **data_config.select(
            "domain", "start_date", "flux_sources", "averaging_error", "filters",
            "basis_algorithm", "nbasis", "fix_basis_outer_regions", "use_bc", "bc_freq",
            "min_error", "min_error_options",
        ),
    )
else:
    prepared = supplied_prepared.validated()

# Optional basis export saves this value without rewriting its source metadata.
inputs, priors = materialize_standard_inputs(prepared, model_spec)
built = build_standard_model(inputs, model_spec, priors=priors, likelihood_builder=...)
built.output_contract.validate_requested_output(output_spec.output_format)
posterior = sample_rhime_model(built, sampler)
annotate_sample_trace(built, posterior, likelihood_builder=...)
inv_out = make_model_output(...)
postprocess_standard_outputs(inv_out, output_spec)  # compute and write derived products
save_inversion_output(inv_out, output_spec)  # export scientific result and trace
return inv_out
```

- `prepare` resolves resources, acquires or validates supplied data, calls
  `prepare_standard`, then `save_preparation`. These steps remain visible in
  the phase runner, as in full execution; no helper combines acquisition and
  preparation. `acquire_merged_data` only acquires data.
  `sample` loads preparation, materializes where needed, builds, samples and saves.
  `postprocess` uses one `load_samples` decoder returning the scientific view and
  writes requested products/exports. Standard prepared-input execution uses the
  same post-preparation operations through `run_standard(prepared=..., config=...)`.
  The former dual-mode `run_rhime_from_prepared_inputs` standard branch is removed;
  its remaining signature and result are multisector-only.
- `InversionOutput` is returned even with `output_format="none"`; no files are
  written in that case. Its output metadata holds only the inference description.
  In-memory products can be recomputed through existing product functions.
- The runner calls `postprocess_standard_outputs` for PARIS/legacy generation
  and writing, then `save_inversion_output` for optional self-contained
  inversion-output and trace exports. Standard PARIS/legacy requests require
  a destination before execution; in-memory reporting uses product functions
  directly. Optional basis export remains visible beside preparation.
  Existing CLI export defaults, filenames, overwrite behaviour and logs remain.
  Deprecation of the self-contained file export remains a separate proposal.
- Preparation and sample persistence use their existing codecs and explicit
  records. Required manifest fields are exactly the fields used by their reader;
  unused build/sample timing fields are not written. Preparation references are
  stored relative to the declaring sample manifest. Preparation generation identifiers detect replacement at the same
  reference, without claiming payload integrity. Manifest-last completion and
  concrete-target checks apply without fresh-directory or recursive-scan policy.
- Prior scalars remain inline; array parameters use registered model data saved
  with the trace. Metadata writing does not execute borrowed prior graphs.
- Configuration helpers use `load_params`, `group_params` and `option_groups`.
  Configuration paths are invocation-relative; checkpoint references remain
  manifest-relative. `data_config` names the owner, separate from numerical data.
- `MergedData` remains acquired-only. Deferred families retain their interfaces
  and results until their adoption slices; this is not a family-wide migration.
- Human readability acceptance remains failed/pending renewed review. Passing
  tests or this amended plan do not mark task 1.11 complete.

## Migration Plan

Four implementation slices of OPE-207 are planned in [tasks.md](tasks.md): a
complete standard proof, multisector adoption, ordinary CO2 adoption, and cached
CO2 adoption. Review the standard working route before family expansion;
multisector and ordinary CO2 can then proceed independently, with cached adoption
following the ordinary handoff conventions. Each slice includes writer, reader,
callers, CLI/examples, documentation and validation. Do not split out a speculative
persistence framework or preliminary cleanup project as a prerequisite. The
concrete exception agreed during review is #868: land resource-path resolution
and runner-owned optional basis saves before #866, then consume those existing
operations in the new handoff routes. Its implementation includes affected callers
and API documentation; it does not implement the grouped configuration or handoffs.

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


### 9. Ordinary CO2 review clarifications — 16 October 2026

The maintainer approved the review corrections and explicitly selected one
ordinary product runner without an additional trace-only runner. `run_co2` owns
configured prepared-input product execution and is selected by both Python and
CLI. Retire ordinary `run_rhime_co2`, the bundled `build_rhime_co2` and the
unshipped `run_co2_from_config` rather than keeping naming/return aliases.
Direct scientific composition remains possible through materialize, eager-build,
sample and annotation operations. Cached/linked runners are outside this reset.

Ordinary CO2 does not have configured native acquisition/preparation. Its CLI
`prepare` validates/references an existing coherent prepared/affine pair; Python
can produce a new pair from canonical inputs and an explicit coherent reduction.
Native sensitivities, defining covariance and retained projection need their own
scientific integration in [#890](https://github.com/openghg/openghg_inversions/issues/890),
consuming #493/#566. Do not reinterpret basis-reduced standard `H` as a native
operator or invent native priors from a gas name. Document this prepared-input
starting point rather than implying acquisition parity with standard.

CO2 has one parser for each owned section. Section-level overrides preserve
unspecified choices; affine reconstruction references belong to `prepared_inputs`.
Saved sampling/readiness accept decoded model/likelihood choices and resolve them
against saved numerical facts. Explicit requested dates/domain may fill missing
imported facts but cannot replace conflicting recorded facts. Record the original
aggregation-error rank request in numerical preparation provenance, alongside
existing actual-rank/approximation diagnostics, without a parallel manifest truth.

CO2's saved scientific roles and affine associations satisfy the sampled-handoff
contract without forcing the generic `OutputContract` representation. Construction
owns bounded actual model metadata, registered prior descriptions and numerical
parameter references; missing imported descriptions remain unavailable. The CO2
external-trace validator stays beside the scientific output owner and checks roles,
dimensions, units and alignment even when products are disabled. Generic contract
validation does not replace these family-specific checks. Named preparation fields
remain a concrete handoff. Scientific output views have no live sampler field;
pure metadata readers normalize the available base inference description and
CO2 algorithm/tuning details without
creating an inference object or filling missing values from current defaults.
Cached output provenance comes from the sample manifest. Replay omits the former
zero-valued `build_and_sample_seconds` placeholder because it cannot measure the
earlier inference duration.

Full/saved routes call one trace-annotation operation and one preparation-record
finisher. They share output policy/resource preflight; calculation trusts that
preflight. Live planned prior generation and replay's actual saved prior draws
remain distinct checks. Inference loads coherent prepared inputs without eagerly
loading its unused affine companion; creation/import and replay retain the
scientific affine binding checks. CO2 readiness restores the configured optional
seed without requiring MCMC settings, and preserves the already specified execution-
error propagation. Returned invalid predictive evidence still produces a failed check.

### 10. One full flux state in standard and multisector — 16 October 2026

The maintainer approved implementing the explored shared-state formulation on
the combined multisector/ordinary CO2 delivery. This is an amendment to the
scientific construction and output contract, not a claim that the resulting
code has received readability acceptance or completed its validation gates.

Standard and multisector use one full labelled physical scaling state `x`, one
two-dimensional model sensitivity `hx` and one total flux concentration
contribution `mu`. The scientific calculation is `H @ x`; registered `hx` may
omit structurally zero columns, with the shared linear operation selecting the
matching full-state entries. Standard retains its existing state labels.
Multisector gathers source-local state blocks onto one source-labelled index;
rectangular input sensitivities are packed for construction only, and existing
gathered input indexes are preserved. Preparation and retained basis artifacts
stay borrowed and unchanged. State order comes from the scientific input labels,
not the order of sector declarations or backend variable suffixes.
The model-only state dimension may be renamed when its name conflicts with
another component's input or prior metadata; its scientific index values, level
names and order remain unchanged.

Independent sector priors may have different distributions, array parameters,
reparameterizations and active/fixed states. Their private prior variables and
construction-owned records may retain sector keys such as `x_<suffix>`, while
the complete physical state and forward calculation are shared. All fixed and
zero-effect entries remain in `x`. New graphs do not add per-sector full-state,
sensitivity or contribution aliases solely for output compatibility. One nearby
concrete flux-construction operation serves standard and multisector; this is
demonstrated reuse of the same equations, not a generic model or stage engine.

Both recipes declare `flux_scale: x`, `emissions_sensitivity: hx` and
`flux_contribution: mu`. Multisector product selection uses saved sector-to-source
associations to select a source block and restore the local basis state labels.
The output boundary may read existing current-format built-in traces with
`x_<suffix>` state names when their saved contract lacks the new canonical role.
An explicit canonical role remains authoritative: missing or incompatible canonical state
must fail rather than fall back. This bounded reader compatibility neither emits
new alias variables nor restores historical staged-envelope/schema recovery.
Generation checks and graph-free replay remain unchanged.

Ordinary CO2 already applies one full correlated retained state to its effective
sensitivity. Sharing the state representation does not change its defining
prior/covariance, affine concentration intercept or native reconstruction
`F [m + U* (alpha - alpha_ref)]`. CO2 native source and sector products are formed
after this reconstruction, retaining their conditional uncertainty scope. Its
scientific trace roles and affine owner remain authoritative; no generic
multiplicative output adapter is introduced.

Validation must compare the actual forward calculation and joint log density,
including heterogeneous priors and ragged/fixed states, and preserve signed
source/total flux products and current-format replay. Existing source coverage,
array ownership, minimum-error and baked BC checks remain at their owners.
Documentation/examples and the bounded tasks in group 6 follow this amendment;
multisector, ordinary CO2 and integrated acceptance gates remain separate.

## OPE-214 review amendment — 2026-10-20

The maintainer reconfirmed removal of the configured CO2 backend selector.
One decision on the resolved scientific model selects inference; full and saved
execution consume the same sampling controls. Saved Python execution accepts a
sampling mapping rather than a second sampler override. This keeps the change
local to CO2 and does not introduce a sampler registry or change linked APIs.

The explicit policy is NumPyro for additive sigma, PyMC for site sigma,
scalar sigma and fixed-amplitude OU, and matched cached PyMC for inferred OU.
Site/scalar choices are conservative defaults, not inherent restrictions;
fixed OU currently has a custom operation without a JAX implementation.
This is a new policy, not preservation of every historical template edit.
The normative per-likelihood table is in the recipe-execution delta.

Matched target-accept controls belong to sampling. Construction determines
whether any flux, boundary or offset states exist; the sampling boundary records
the actual algorithm and effective controls in the trace and manifest. Replay
reads those plain fields without constructing a sampler. `rhime_recipe="co2"`
continues to identify the scientific recipe independently of its update.

Explicit LogNormal reparameterisation requests warn and retain the physical prior
while recording the direct representation required by the matched sampler.
Both public independence checks remain. Float32/64 dense-reference tests cover
the model's own transforms; the cache is not forced to a different precision.
