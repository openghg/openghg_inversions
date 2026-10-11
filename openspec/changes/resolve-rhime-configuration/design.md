# Design

## Context

See [proposal.md](proposal.md) and the [behavioral contract](specs/rhime-configuration/spec.md).
PR #809 is merged. PR #813 originally implemented its aggregate configuration:
`RhimeConfig(preparation=RhimePreparationConfig(...), model=..., output=..., sampler=...)`.
This specification supersedes that design. Implementation and validation
progress is tracked in [tasks.md](tasks.md).

The accepted conversation made `RhimeConfig` the in-memory equivalent of resolved
configuration-file options, independent of file format. The previous design
instead kept a second configuration record and required callers to use
`config.preparation` for data selectors and preparation choices. Its compatibility
section also required historical encoding preservation while prohibiting a hash
preservation layer. Those requirements are superseded here.

The shared `inversion_data._site_options` module owns public `SiteOptions`;
acquisition owns `RhimeMergedData`;
`inversion_data.prepared_inputs` owns `RhimePreparedInputs`; `inference.sampling`
owns `RhimeSampler`. Keep those owners and the procedural scientific recipes.

## Goals / Non-Goals

**Goals:** One complete request configuration, early shorthand resolution, clear
contracts for configuration versus scientific data and execution descriptions,
and neutral acquisition naming for surface and column observations.

**Non-goals:** Flatten every established model/output/sampler type, add another
settings hierarchy, change equations or array execution, introduce a frontend
registry, unify CO2 configuration, or design manifest/hash migration.
Configuration export and an INI writer are deferred to #814.

## Decisions

### 1. One request configuration, with preparation fields directly on it

`RhimeConfig` directly owns requested data selection, dates, stores, basis,
filter/error preparation and preparation-artifact choices. There is no
`RhimePreparationConfig`, `config.preparation` or preparation-config projection
method. Existing model/output/sampler objects remain composed values because
these already have independent consumers and contain coherent choices.

```text
read_rhime_ini --> decoded options
                          |
Python options -----------+--> winning overrides / recipe extraction
                                         |
                              RhimeConfig.from_params
                         (translate aliases, then resolve)
                                         |
                                    RhimeConfig
                                         |
                      explicit selected values into scientific stages
                                         |
                         acquired / filtered / prepared data
                                         |
                           retained execution RhimeRunSpec
```

Target API outline:

```python
def read_rhime_ini(path: str | Path) -> dict[str, object]: ...

@dataclass(frozen=True, kw_only=True)
class RhimeConfig:
    site_options: SiteOptions
    species: str
    domain: str
    start_date: str
    end_date: str
    flux_sources: tuple[str, ...]
    split_by_sectors: bool
    use_bc: bool
    output_name: str
    # Remaining explicit acquisition/preparation fields inventoried below.
    model: RhimeModelSpec
    output: RhimeOutputSpec
    sampler: RhimeSampler

    @classmethod
    def from_params(cls, params: Mapping[str, object], *, multisector: bool) -> RhimeConfig: ...

    def select(self, *names: str) -> dict[str, object]: ...
```

The configuration is equivalent to fully resolved options, not necessarily a
flat copy of INI spelling. Cohesive site/model/output/sampler values are allowed;
a newly invented preparation configuration is not. Do not mirror complete prior,
likelihood, output or sampler settings alongside their existing owner. The few
shared facts listed below, including output naming for preparation artifacts,
are explicit exceptions needed by existing independent contracts.
No raw options mapping or acquired arrays are stored as a second source of truth.

The existing shared species/domain/BC/source/output-name facts may also appear
in independently usable model/output specifications. Construct these views from
one resolution of each option. They do not own another default policy or require
repeated consistency checks. Model specifications describe scientific choices;
they are not constructed PyMC models. Requested source ordering and resolved
sector routing keep their existing meanings.

### 2. Contracts and adopted names

The table describes the target after reconciliation. Names marked rename are
adopted names for this revision; other established names are retained.

| Name | Role | Input / output and boundary contract |
| --- | --- | --- |
| `read_rhime_ini` (replace `load_rhime_config` introduced in #813) | INI configuration frontend. | INI path -> decoded options. Owns INI syntax and existing section interpretation; runners apply overrides, extract their own options and resolve the final request. |
| `resolve_rhime_config` (remove) | Former construction wrapper. | All callers use `RhimeConfig.from_params`; no compatibility alias is retained. |
| `RhimeConfig` (keep; revise contents) | Complete resolved requested configuration. | Direct acquisition/preparation fields plus existing model/output/sampler values. Intended to be serializable as settings; export/INI-writing deferred to #814. No scientific data, retained run or historical identity contract. |
| `SiteOptions` (promote existing `_SiteOptions`; export from `inversion_data`) | Public cohesive aligned selector record. | Complete ordered site/period/inlet/platform/etc. tuples. `from_inputs` normalizes external shorthand; direct construction accepts resolved aligned values. Requested when held by config; authoritative retained values when held by merged data. Selection creates a new complete record. |
| `RhimeModelSpec` / `SectorSpec` (keep) | Scientific recipe choices and individual flux-sector definitions. | Priors, likelihood/component choices and source routing; no PyMC graph or acquired arrays. Independently usable by existing builders. |
| Built-in likelihood settings (keep) | Choices for the selected observation-error component. | Contained in the model specification. `None` retains the custom-likelihood meaning; do not create another configuration vocabulary. |
| `RhimeOutputSpec` (keep) | Final-product policy. | Formats, naming, destinations and saving. Independently usable by existing output/execution APIs. |
| `RhimeSampler` (keep) | Existing sampling settings and execution method. | Construction stores settings; `.sample(model, ...)` performs inference. No bound model, posterior or data in the configuration. No duplicate sampler-options class. |
| `retrieve_inversion_data` (rename public `data_processing_surface_notracer`) | Fresh acquisition and merge, covering surface and column data. | Existing shorthand arguments -> existing six-tuple of merged data and retained metadata lists. Includes existing observation-error construction and optional merged saving; no reload, basis construction or inference. |
| `_retrieve_inversion_data_from_options` (rename private canonical body) | Fresh acquisition with resolved selectors. | Complete site-options record plus explicit non-site arguments -> same six-tuple. No second shorthand expansion. |
| `data_processing_surface_notracer` (deprecated compatibility wrapper) | Preserve existing public calls/imports. | Same established signature, shorthand, six-tuple return and errors; issue `DeprecationWarning` naming `retrieve_inversion_data`, then delegate to the same body. No duplicate acquisition implementation. |
| `RhimeMergedData.from_options` / `.load` (#815 split) | Distinct fresh-acquisition and strict-cache factories. | Resolved selectors and explicit acquisition or cache arguments -> `RhimeMergedData`. Fresh acquisition retains optional saving; loading requires caller selectors and validates layout/time-resolution with no fallback. Recipes reuse supplied compatible data before selecting a factory. |
| `RhimeMergedData` (keep) | Acquired/reloaded numerical handoff. | Merged scientific datasets plus authoritative retained site options. Borrowed, potentially lazy arrays; no hidden materialization. |
| `prepare_rhime_inputs` (deprecate; retain adapter) | Acquisition-and-preparation convenience entry point. | Preserves applicable arguments and `RhimePreparedInputs` return, warns with the explicit loading/preparation replacements, and delegates to the same named scientific stages. Resolves applicable shorthand without a complete model/output/sampler request. |
| `filter_rhime_observations`, `build_rhime_basis`, `build_rhime_sensitivities`, `assemble_rhime_inputs` (keep) | Named scientific stages in the ordinary recipe. | Borrowed numerical handoffs and explicit resolved values -> filtered data, basis, sensitivities and assembled inputs. Named keyword options with required scientific identity/source inputs; remove the former positional `data_args` adapter. No phase-config class, reparsing or generic request context threaded through scientific components. |
| `RhimePreparedInputs` (keep) | Labelled prepared model-input handoff. | Numerical inputs, basis and retained metadata; independent of full requested configuration. |
| `RhimeRunSpec` (keep) | Execution description. | Constructed after ordinary preparation from requested dates, retained sites/periods, prepared layout and resolved model/output choices. No acquisition options or sampler execution. |
| `RhimeRunnerSetup` / `make_rhime_runner_setup` / `resolve_rhime_options` (remove) | Redundant internal setup bundle and its constructors. | Migrate ordinary, nested, staged, shim and example consumers to `RhimeConfig.from_params`; construct retained run descriptions only after preparation. Do not retain a renamed bundle or compatibility projection. |
| `RhimeResult` (keep) | Completed execution result. | Retained descriptions, prepared numerical inputs, posterior and outputs. Never represents an unresolved request. |
| `run_rhime` / `run_rhime_multisector` (keep) | Ordinary procedural orchestration. | Existing file-plus-keyword inputs -> result. Read/resolve the effective request, prepare, build, sample and output in visible order. Decode options, combine overrides and extract recipe choices before canonical construction; reuse an already-resolved request when supplied. |

The deprecated `params_from_config` adapter lives with HBMCMC compatibility
and keeps its normalized-dictionary default, `normalise=False` behavior and
supported overrides. It calls the same decoding reader. Modern runners no longer
need this adapter to access unresolved options.

Rename the existing acquisition-owned record to `SiteOptions` and export it
from `inversion_data`; update public configuration and merged-data annotations,
imports and documentation. Its previous privacy was an internal-helper detail;
the type now participates in public handoffs. Keep one record and owner rather
than adding a parallel public wrapper. `from_inputs` owns shorthand expansion,
label normalization and selector validation; direct construction accepts already
resolved aligned values and checks structural invariants without another
shorthand pass. Selection returns a complete new record.

For the low-level acquisition rename, preserve the six-tuple explicitly. Do not
quietly replace it with `RhimeMergedData`: that handoff belongs to the higher
retrieval/reload boundary. The canonical body shares scientific work with both
public names. Internal calls use the neutral name and must not emit deprecated
wrapper warnings.

The approved #815 split gives the handoff distinct `from_options` and `load`
class methods. The recipe reuses compatible supplied data unchanged, otherwise
selects strict cache loading or fresh acquisition explicitly. Preserve
selector/layout checks, retained-option selection and numerical normalization
at the owning boundaries. Missing directories, missing/corrupt artifacts and
load `ValueError` propagate without fallback (#806). Fresh saving remains
opt-in; neither supplied-data reuse nor loading saves. The current codec stays
in place; dataset-only snapshots are a separate #815 stack change.

### 3. Resolve after overrides and before any scientific phase

`read_rhime_ini` owns file access and the existing INI decoding rules, returning
a flat dictionary without semantic resolution. The runner applies overrides and
extracts recipe-specific options before calling `RhimeConfig.from_params`.
Shorthand remains available for those edits. Preserve section flattening and
first-occurrence precedence for now; redesigning the INI template or model-type
selection is outside this cleanup.

`RhimeConfig.from_params` is the single construction entry point. It calls the
alias translator from `hbmcmc.compatibility`, then performs modern coercion,
validation and existing default resolution. Translation emits `DeprecationWarning`
only when replacing or removing deprecated spellings, including old output-format
values; canonical spellings win when both are present. Canonical options need no
deprecation warning. Remove the `resolve_rhime_config` wrapper and repeated
translation calls in runners. Fixedbasis scientific behavior and the deprecated
dictionary reader adapter remain in the compatibility module. The adapter
retains its intentional `rhime` package export; old-location translator aliases
in `rhime.params` and `hbmcmc.run_hbmcmc` are removed. Do not import the executable
HBMCMC runner into modern configuration code.

Apply overrides before resolving defaults or site shorthand, including a changed
site list or date bound. Expand scalar site selectors to the effective site
count; explicit lists/tuples must already match it. For two sites,
`"1h"` and `["1h", "1h"]` resolve identically, while `["1h"]` fails early.
Reuse `SiteOptions.from_inputs` and its small alignment helpers for this common
external-input convention. Keep scalar broadcasting and aligned-sequence checks
independent of INI syntax so direct adapters and other parsers can share them.
Extract an ordinary helper only where needed; do not add a parser framework or
duplicate the existing normalization algorithm. Canonical scientific consumers
receive complete aligned values and do not perform this expansion again.

Supported names and defaults follow their consumers: configuration fields and
site inputs, likelihood settings, output policy, and the sampler's supported
configuration subset. Required raw inputs derive from those declarations;
composed and recipe-derived values are not required external parameters. Keep
aliases and external spelling translation at the input boundary. Do not infer
configuration choices from scientific callable signatures or introduce a registry.

Standard and multisector runners accept a complete `config=` as an alternative
to an INI path and raw options. They reuse it without another resolution pass;
combining these input modes is an error. The compatibility shim retains its
preflight validation before file-copy side effects and passes that resolved
request into the standard runner.

`RhimeMergedData.save` delegates to the existing merged-data serializer at an
explicit write boundary. Existing fresh-retrieval save flags remain optional
and default to false; supplied data and successful reloads do not trigger them.
This does not add configuration export or a new cache format.

The shared resolver's responsibilities are:

| Step | Work and dependency |
| --- | --- |
| Normalize and validate effective options | Translate deprecated spellings once in the classmethod, then reject removed/unknown options and check required fields and value types. Alias definitions belong to the compatibility module; file section interpretation belongs to the frontend. |
| Resolve sources and recipe choices | Normalize sources, enforce standard/multisector requirements and establish sector/source routing. |
| Resolve scientific settings | Resolve priors, likelihood and active BC/offset settings; check incompatible choices. Date-dependent defaults use the effective overridden dates. |
| Resolve sampler and output policy | Construct existing sampler/output values with their defaults and configuration checks; do not execute sampling or output. |
| Resolve acquisition/preparation settings | Apply defaults and construct complete `SiteOptions` through shared shorthand helpers. Overrides have already established the final requested sites and selectors. |
| Assemble the requested configuration | Own supported ordinary containers and construct `RhimeConfig` directly, without scientific data access or an intermediate preparation configuration. |

Consolidate repeated checks at the owning input boundary rather than validating
locally constructed settings again. Shared diagnostics identify RHIME options;
INI-specific syntax advice belongs to the INI frontend.

Reuse existing required-option, rejection, site expansion and likelihood
rules, retaining alias definitions in the compatibility module. Preserve case
normalization, optional selectors, inlet slices and
`time_resolved=None`. Omitted/false `use_tracer` is consumed; effective true fails
early, including direct preparation and supplied-data routes. No tracer field
is retained. Keep established custom-likelihood conflict ordering.

The resolver declares every acquisition/preparation field explicitly. The
following inventory describes fields on `RhimeConfig`, not another class:

| Direct fields | Representation / meaning |
| --- | --- |
| `site_options` | Public `SiteOptions` containing complete aligned site-selector tuples. |
| Species/domain, date bounds, flux sources, layout, BC use and basis output name | Requested/resolved strings, tuples and booleans; forwarded consistently to existing model/output views. |
| BC/observation/footprint/emissions stores, emissions domain, footprint model/species, calibration scale and BC input | Existing supported typed selectors. |
| Footprint/BC basis cases and directories, country directory, outer-regions path, basis algorithm/count and fixed outer regions | Existing supported path/string/integer/boolean choices. |
| Filters, averaging error, BC frequency, minimum error and normalized error options | Existing contracts; data-dependent materialization remains in preparation. |
| Reload/save merged data, merged directory/name, basis destination and non-finite flux check | Existing preparation-artifact policy and flux-check choices. |

The runner forwards needed named values at each scientific call, directly or
with `**config.select("name", ...)`. `select` returns a fresh shallow dictionary
of explicitly named attributes; values remain borrowed and missing attributes
raise `AttributeError`. It does not infer consumer signatures, copy nested values,
resolve defaults or export a durable settings schema. A private orchestration
helper may accept the resolved config while composing several stages; scientific
components continue to accept their own named arguments. No dynamic option
schema or second request/preparation record is introduced.

Raw overrides are applied after decoding and before `from_params`.
`dataclasses.replace` is for already coherent resolved changes, not for
recomputing dependent defaults or shared model/output choices. Direct retrieval
and deprecated preparation adapters resolve only applicable inputs and share the
scientific bodies with ordinary configured runs. Use `SiteOptions.from_inputs`
for public selector construction. The internal
`inversion_data._site_options.convert_to_list` helper retains its calling and
list-return contract without a warning; its old `get_data` import alias is
removed.
The preparation adapter must
preserve canonical footprint provenance instead of assembling a second result.

### 4. Requested options and retained data have different meanings

Configuration describes what was requested. Acquisition, compatible reload and
filtering select every aligned selector together by retained label. They leave
configuration unchanged, ignore unused redundant legacy metadata, and do not
expand shorthand against a reduced site count. Supplied compatible merged data
keeps its authoritative site record and performs no acquisition/reload.

After preparation, compose `RhimeRunSpec` with `config.start_date`,
`config.end_date`, `prepared.sites`, `prepared.averaging_period`, the prepared
layout, `config.model` and `config.output`. A TAC/MHD request retaining TAC stays
a TAC/MHD configuration; the execution description contains TAC. Remove the
pre-preparation run-shaped setup exception. Nested, staged and example recipes
also start from `RhimeConfig`; local option overrides do not require a second
setup record. Independent prepared-input runners and alignment helpers retain
their scientific contracts.

A frozen configuration is not recursively immutable: existing mappings and the
sampler remain mutable. Own ordinary supported containers at resolution;
scientific arrays and opaque keyword values remain borrowed. Do not copy,
compute, persist, densify or rechunk data during configuration construction or
hide those operations in accessors. Preserve numerical execution boundaries.

### 5. Compatibility follows public behavior, not historical encodings

Preserve supported scientific entry points, shorthand, result shapes,
scientific behavior and sampling defaults. The internal setup dataclass and
constructor/helper return shapes are deliberately removed; migrate in-repository
consumers, imports, tests and documentation together. Exported names alone do
not justify retaining redundant orchestration types. This does not require restoring raw
scalar/list/None spellings or sparse default-prior metadata after resolution.
Remove #813's compatibility logic whose sole purpose is unchanged staged hashes.
Do not add old-spelling fields, identity methods or hash migration bridges.

[#808](https://github.com/openghg/openghg_inversions/issues/808) and
[#802](https://github.com/openghg/openghg_inversions/pull/802) own staged encoding,
manifest and identity contracts. The [staged guide](../../../docs/usage/staged_workflow.rst)
requires 0.7 artifacts to be consumed with 0.7; 0.8 does not promise compatibility.
Any change affecting a currently supported staged contract must be identified
explicitly and reconciled with that work, not silently assigned the old contract.
Historical hash equality is not an acceptance criterion for this change.

Keep current flat INI interpretation, canonical-alias precedence and warnings,
ordinary cwd-relative paths and staged source-relative paths. CO2 TOML recipe
configuration and its prepared-axis interpretation remain separate.

### 6. Serializable configuration is a deferred follow-up

`RhimeConfig` should be serializable as resolved values, including sampler
settings rather than execution methods. This is distinct from a stable manifest
schema or historical identity. Keep existing composed model/output/sampler
choices; do not add duplicate settings classes to anticipate export. The precise
representation, supported-value encodings, round-trip contract and writer API
belong to [#814](https://github.com/openghg/openghg_inversions/issues/814),
including a writer for the existing RHIME INI format.

The HBMCMC shim prints extracted parameters before full resolution; staged
workflows construct effective-configuration mappings for manifests. Neither is
a direct `RhimeConfig` serialization contract. The follow-up may reuse a
configuration-only representation for resolved logging without computing
scientific data through general artifact serializers. No serialization or new
logging behavior is required to complete #804.

## Risks / Trade-offs

- The former implementation/docs need reconciliation with this plan: verify
  affected tasks before claiming completion of the current stack revision.
- Public acquisition names have imports, monkeypatch seams and tuple consumers:
  preserve the deprecated wrapper, share the body and update internal callers.
- Model/output views repeat a few shared facts: resolve once and forward values;
  do not create independent defaults or validation authorities.
- Nested/staged/example consumers use the removed setup bundle: migrate them to
  one request type and identify supported persisted-contract effects explicitly.
- A resolved request has many options: group the documentation by purpose rather
  than adding another phase configuration class.

## Migration Plan

Deliver the reconciled #813 implementation through the #815 stack, with
configuration in #824. Remove the preparation class and its plumbing,
remove the internal setup bundle, migrate ordinary/nested/staged/shim/example
consumers, implement the decoding-only INI reader, promote/export `SiteOptions`,
implement the neutral acquisition names and separate fresh acquisition and strict cache loading into
`RhimeMergedData.from_options` and `.load`. Update documentation/exports. Reuse the existing equivalence, override,
site/drop/reload, ownership, nested/shim and sampler coverage. Add focused checks
for direct configuration access, decoded reader results and subsequent
construction, public site-options
construction and deprecated-wrapper forwarding/warnings;
preserve scientific output checks. Run relevant broader coverage before handoff.

The previous locked Python 3.12/3.13 and documentation passes validate the old
implementation only. They do not establish conformance to this revised design.
Track the remaining reconciliation in [tasks.md](tasks.md).
