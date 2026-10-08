# Design

## Context

See [proposal.md](proposal.md) and the
[behavioral contract](specs/rhime-configuration/spec.md). The inspected `devel`
revision is `771d1175`, including #773/#774/#807. `rhime.params` already separates
raw INI loading from option resolution, but `RhimeRunnerSetup.data_args` retains
site shorthand. Acquisition constructs `_SiteOptions`, unpacks it into a loader
and repeats expansion. Existing `RhimeRunSpec` documents retained sites, although
current setup constructs it before preparation and later replaces those fields.

Acquisition owns `_SiteOptions` and `RhimeMergedData`; `inversion_data.prepared_inputs`
owns `RhimePreparedInputs`; `inference.sampling` owns `RhimeSampler`. The sampler
stores settings and executes only when supplied a model through `.sample(...)`;
constructing it acquires no model, posterior or scientific data. Preserve these
owners and established compatibility exports.

## Goals / Non-Goals

**Goals:** One complete resolved request before acquisition, distinct requested
and retained site metadata, explicit record roles and ordinary procedural
forwarding into scientific functions.

**Non-goals:** A universal configuration framework, another sampling settings
class, new file formats, cross-family configuration unification, phase-only file
semantics, scientific/model-component relocation, hash/cache policy or numerical
execution cleanup.

## Decisions

### 1. Construct the complete request immediately after overrides

`RhimeConfig` is the in-memory representation of the resolved requested run. It
is independent of the file syntax used to express that request. The ordinary
runner's input boundary is:

```text
INI decoder --> raw mapping --> supported overrides --> semantic resolver
Python mapping --------------------------^                   |
                                                            v
                                                       RhimeConfig
                                                            |
                                                  acquisition/preparation
                                                            |
                                                   retained RhimeRunSpec
```

Proposed API sketch; the new types/functions are not installed yet:

```python
def load_rhime_config(path: str | Path) -> Mapping[str, object]: ...
def resolve_rhime_config(
    params: Mapping[str, object], *, multisector: bool
) -> RhimeConfig: ...

@dataclass(frozen=True)
class RhimeConfig:
    preparation: RhimePreparationConfig
    model: RhimeModelSpec
    output: RhimeOutputSpec
    sampler: RhimeSampler

@dataclass(frozen=True)
class RhimePreparationConfig:
    site_options: _SiteOptions
    species: str
    domain: str
    start_date: str
    end_date: str
    flux_sources: tuple[str, ...]
    split_by_sectors: bool
    use_bc: bool
    output_name: str
    # Remaining explicit fields are inventoried in decision 3.
```

Use the existing INI decoder for the loader. Its raw mapping is an intermediate,
not `RhimeConfig`. Merge supported overrides before resolving the effective
options: changing the site list must change the length used for broadcasting.
Resolve aliases, configuration-only defaults, sector/source routing, likelihood
selection and every supported site shorthand before returning the config. No
raw spellings or scalar site selectors are stored for later interpretation.
`None` remains a resolved choice where it means unspecified or a disabled
component; data-dependent interpretation remains at the owning scientific phase.

Scalar broadcasting is a shared external-input convention, not an INI-parser
responsibility. File-derived and Python mappings use the same resolver; a later
frontend could decode to the same mapping without changing the config contract.
INI remains the only new loader covered here. Avoid a parser registry or another
file frontend. Keep resolution in the existing parameter owner and reuse its
alias, required-option and rejection rules.

### 2. Give each record a distinct role

| Record | Role and contents | Creation and consumer |
| --- | --- | --- |
| `RhimeConfig` (new) | Complete resolved request, grouping preparation, model, output and sampler choices. No retained-run specification or scientific data. | Constructed at the input boundary; consumed by ordinary standard/multisector runners. |
| `RhimePreparationConfig` (new) | Requested data selection and the choices currently needed to transform it into model inputs. No priors, likelihood object, posterior or final-output policy. | Constructed during resolution; consumed by acquisition and named preparation stages. Independently usable by preparation adapters. |
| `_SiteOptions` (existing) | Complete ordered site labels and aligned averaging/selector tuples. Its meaning follows its holder: requested in configuration, retained in merged data. | Request constructed once with acquisition's `from_inputs`; retained records selected by label. |
| `RhimeModelSpec` (existing) | Scientific model choices: sectors, priors, likelihood, BC scaling, offsets, aggregation-error representation and state activity. Existing species/domain metadata remains for compatibility and outputs. | Constructed during resolution; consumed by concrete model recipes and outputs. |
| `SectorSpec` (existing) | One scientific flux sector: name, source routing, scaling prior, variable suffix and optional state activity. | Contained in the model specification; consumed by model construction and reconstruction. |
| Built-in likelihood settings (existing) | Resolved choices for the selected error model; `None` retains the custom-likelihood meaning. | Contained in the model specification; consumed by the selected likelihood component. |
| `RhimeOutputSpec` (existing) | Final product formats, destinations, naming and save choices. | Constructed during resolution; consumed by execution/output handling. |
| `RhimeSampler` (existing class, not a dataclass) | Sampling settings plus the existing execution method. Contains no bound model, acquired inputs or posterior. | Constructed during resolution; called with the completed model at inference. |
| `RhimeMergedData` (existing) | Acquired/reloaded scientific data and its authoritative retained site-options record. | Returned by acquisition and passed through filtering; never held in config. |
| `RhimePreparedInputs` (existing) | Durable labelled model inputs, basis and retained site metadata. | Returned by preparation; consumed by model construction and prepared-input runners. |
| `RhimeRunSpec` (existing) | Execution description: requested date bounds, retained sites/averaging periods, prepared sector-layout flag, model and output specifications. | Constructed after preparation in ordinary configured runs; consumed by builders, execution and output provenance. |
| `RhimeRunnerSetup` (existing compatibility record) | Legacy projection containing a run-spec-shaped setup, sampler and preparation dictionary. | Produced only where established helper/runner consumers require it; does not define canonical config semantics. |
| `RhimeResult` (existing) | Execution result with retained run/model/output descriptions, numerical inputs, posterior and output metadata. | Returned after execution; never used to represent an unresolved request. |

Ordinary canonical runs do not construct a requested-site `RhimeRunSpec` or store
one in `RhimeConfig`. Once preparation completes, construct it from:

- dates from `config.preparation`;
- sites and averaging periods from `prepared.sites` and `prepared.averaging_period`;
- the prepared layout corresponding to the selected runner;
- `config.model` and `config.output`.

For a request of TAC and MHD that retains TAC, configuration continues to name
both requested sites; the run specification names only TAC. Existing direct
prepared-input APIs and compatibility helpers retain their signatures and
alignment behavior, including `with_prepared_rhime_sites`. A legacy helper's
preparation-time run-spec projection is a compatibility exception, not the
meaning of the canonical configuration or a new public contract.

### 3. Explain shared facts without adding a context hierarchy

The existing model specification and new preparation record have some shared
facts because both phases need them. Resolve these once and forward the same
resolved choices while constructing the records. There is no second parse,
default policy or consistency-validation pass between trusted records.

| Shared fact | Why preparation needs it | Other use |
| --- | --- | --- |
| Species/domain | Store queries, basis construction and input metadata. | Existing model metadata and final output naming. |
| Flux sources/sector routing | Retrieve fluxes and construct source-labelled sensitivities. | `SectorSpec` identifies the corresponding scientific scaling states. Preserve the current one-to-one routing and ordering rules. |
| `use_bc` | Acquire BC data and construct BC sensitivities. | The model includes BC scaling terms when enabled. |
| Sector layout | Select the current preparation mode. | The retained run records the prepared layout; the model uses its sector specification. |
| Dates | Select observations and construct temporal inputs. | The retained run records the requested inversion bounds. |
| Output name | Name a requested basis artifact using current behavior. | `RhimeOutputSpec` names final products. |
| Sites/averaging periods | Describe the complete requested selectors. | Merged/prepared handoffs and the run specification describe retained values derived from them. These may differ after selection. |

Keep this small explicit duplication instead of introducing a shared context
object, inheriting preparation from the full model or moving established public
fields. Model and preparation remain separately usable. The runner passes named
values to the existing procedural functions; neither record is an ambient
context threaded through all scientific components.

The preparation inventory follows the explicit `RHIME_PREPARATION_OPTION_NAMES`
and current defaults:

| Fields | Representation and meaning |
| --- | --- |
| Sites, averaging, inlet, footprint height, instrument, platform, observation level, meteorological model, maximum level, time resolution | One existing `_SiteOptions` record of complete aligned tuples. |
| Species/domain, dates, output name, flux sources, sector layout, BC use | Named string, tuple and boolean fields; shared facts explained above. |
| BC/observation/footprint/emissions stores, emissions domain, footprint model/species, calibration scale, BC input | Existing typed string/optional-string choices. |
| Footprint and BC basis cases/directories, country directory, outer-regions path, basis algorithm/count, fixed outer regions | Existing path/string/integer/boolean choices. |
| Filters, averaging error, BC frequency, minimum error and minimum-error options | Existing supported filter contract, boolean/frequency choices and `MinErrorConfig`/normalized error options. |
| Reload/save merged data, merged directory/name, basis destination, non-finite flux check | Existing path/string/boolean choices and `FluxNonFiniteCheck`. |

Declare every field explicitly; the table does not imply a dynamic schema or
`extra_options` field. BC frequency and minimum-error settings are scientifically
model choices, but current preparation materializes them into labelled inputs.
Their presence here records the existing execution boundary; moving them to model
components is a separate scientific change. Basis naming is similarly preparation
artifact policy, distinct from final-output policy.

### 4. Reuse the sampler directly

Store the existing `RhimeSampler` in `RhimeConfig`. Its constructor already owns
sampling normalization and copies its keyword dictionaries. Preserve current
configured `draws`, `burn`, `tune`, `chains`, `nuts_sampler`, `progressbar`,
`sample_kwargs` and `posterior_predictive_kwargs`, their defaults, and the
sampler's existing predictive defaults. Do not add predictive configuration
switches or a second settings record. Only `.sample(model, ...)` executes
inference, after model construction. Preserve the inference owner and public
RHIME sampler aliases.

The new aggregate/preparation records are frozen, but existing dictionaries and
the sampler remain mutable objects. Do not claim recursive immutability. Own
supported configuration containers at translation boundaries so resolution and
execution do not mutate caller inputs or the established request. Preserve
supported opaque keyword values without copying or computing scientific arrays.
No generic freezing, serialization or object-graph copying framework is needed.

### 5. Expand once, then retain by label

Reuse acquisition's `_SiteOptions.from_inputs` and shared expansion helpers when
constructing configuration. Uppercase site labels, reject empty/duplicate
requests, broadcast scalars and require explicit sequences to match the effective
requested count. For two sites, `"1h"` and `["1h", "1h"]` yield identical period
tuples; `["1h"]` is invalid. Preserve optional values, supported inlet slices,
integer levels and `time_resolved=None` without inventing metadata defaults.

Canonical acquisition accepts the complete site record and forwards resolved
entries into the retrieval body. Keep the public low-level shorthand API as an
adapter to the same translation/body. Direct public preparation resolves its
applicable preparation choices without requiring model, output or sampler
settings. Converting tuples to lists for an external API is representation
conversion, not another scalar-expansion pass.

Acquisition, compatible reload and filtering retain existing label-selection
mechanics and validation boundaries. Ignore unused legacy returned metadata
lists; select requested options by the returned retained labels. A valid supplied
merged handoff retains its authoritative site record and performs no acquisition.
After a site drop, select every applicable option together and leave the original
request intact. Do not resolve external shorthand again against retained sites.

### 6. Keep compatibility and adjacent work bounded

Preserve `params_from_config` and its default normalized-dictionary return,
`resolve_rhime_options`/`RhimeRunnerSetup` consumers, nested RHIME, the HBMCMC shim
and composition examples through projections from the same resolution rules.
Only compatibility projections construct the legacy requested run specification.
Independent builders and prepared-input runners do not require `RhimeConfig`.

Keep flat INI section interpretation, alias warnings, canonical-name precedence,
existing rejection order, ordinary cwd-relative paths and staged source-relative
paths. Existing staged JSON input remains supported transport. #807 consumes
omitted/false `use_tracer` and rejects effective true at resolution and relevant
direct public boundaries, including supplied merged data. No tracer field is
stored in config.

CO2's TOML loader already separates decoding from recipe resolution, but its
current configuration describes prepared-input replay. Expansion onto an actual
observation axis requires prepared labels and remains at that boundary. This
change does not replace CO2's recipe records or add a cross-family config. Future
site shorthand can reuse the same early-resolution rule without this change
owning that future work.

Equal resolved scalar/list requests do not promise stable historical hashes.
[#808](https://github.com/openghg/openghg_inversions/issues/808) and
[#802](https://github.com/openghg/openghg_inversions/pull/802) own representation-sensitive
gates, handoff authentication and staged policy. The
[staged guide](../../../docs/usage/staged_workflow.rst) declares that 0.8 does not
support 0.7 staged artifacts after the tracer-default removal changed hashes.
Honor that boundary. Do not add old-spelling fields, hash methods, a preservation
layer or an identity protocol. Handle any affected currently supported staged
encoding explicitly without silently redefining its contract.

## Risks / Trade-offs

- Public setup consumers still expect a pre-preparation run spec -> isolate that
  legacy projection; document retained semantics for ordinary canonical runs.
- Shared facts could acquire independent defaults -> construct both phase views
  from one resolver and forward values explicitly, without a new hierarchy.
- Frozen aggregates contain mutable existing objects -> document ownership and
  check caller/request non-mutation instead of adding a generic freeze system.
- Some scientific choices still execute in preparation -> explain current
  placement and preserve numerical behavior; defer component relocation.
- Staged consumers encode current setup representations -> inspect integration
  consumers while leaving broader manifest policy to its separate work.

## Migration Plan

Review this revised planning change before generating tasks. On the landed
#773/#774/#807 foundations, add the aggregate/preparation records and resolver,
switch ordinary consumers to canonical values, derive retained run specifications
and adapt supported entry points together. Reuse existing site/drop/reload,
legacy-metadata and tracer tests. Replace the test expecting raw scalar periods
to reach acquisition; add override-before-expansion, file/Python equivalence,
caller/request non-mutation and retained-run checks. Reuse existing sampler and
facade coverage, plus #811's consolidated sampling coverage; configuration
checks need no new stochastic sampling or repeated CLI subprocesses. Run focused
and relevant broader coverage, update configuration guidance and add
`newsfragments/804.feature.md` during implementation. This draft changes no
runtime paths and needs planning validation only.
