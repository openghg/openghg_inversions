# Design

## Context

See [proposal.md](proposal.md) and the [behavioral contract](specs/rhime-configuration/spec.md).
PR #809 is merged. PR #813 currently implements its aggregate configuration:
`RhimeConfig(preparation=RhimePreparationConfig(...), model=..., output=..., sampler=...)`.
This revision corrects that design; it does not describe the current Python code
as already conforming.

The accepted conversation made `RhimeConfig` the in-memory equivalent of resolved
configuration-file options, independent of file format. The previous design
instead kept a second configuration record and required callers to use
`config.preparation` for data selectors and preparation choices. Its compatibility
section also required historical encoding preservation while prohibiting a hash
preservation layer. Those requirements are superseded here.

Acquisition owns the existing `_SiteOptions` and `RhimeMergedData`;
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
INI --> raw options --> winning overrides --> resolve_rhime_config
Python options -----------------^                    |
                                                     v
                                                RhimeConfig
                                                     |
                      explicit resolved values into scientific stages
                                                     |
                                   acquired / filtered / prepared data
                                                     |
                                     retained execution RhimeRunSpec
```

Proposed API outline; this is a plan for revising #813, not installed behavior:

```python
def read_rhime_ini(path: str | Path) -> Mapping[str, object]: ...
def resolve_rhime_config(
    params: Mapping[str, object], *, multisector: bool
) -> RhimeConfig: ...

@dataclass(frozen=True)
class RhimeConfig:
    site_options: _SiteOptions
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

### 2. Contracts and recommended names

The table describes the target after reconciliation. Names marked rename are
recommendations adopted for this revision; other established names are retained.

| Name | Role | Input / output and boundary contract |
| --- | --- | --- |
| `read_rhime_ini` (rename `load_rhime_config` introduced in #813) | Format-specific decoder. | INI path -> raw option mapping. File I/O only; no aliases, defaults, overrides or site expansion. The name does not promise a `RhimeConfig` return. |
| `resolve_rhime_config` (keep) | Format-neutral semantic boundary. | Effective mapping after winning overrides -> complete `RhimeConfig`. Resolves supported aliases/defaults/shorthand and fails before scientific work. |
| `RhimeConfig` (keep; revise contents) | Complete resolved requested configuration. | Direct acquisition/preparation fields plus existing model/output/sampler values. Intended to be serializable as settings; export/INI-writing deferred to #814. No scientific data, retained run or historical identity contract. |
| `_SiteOptions` (keep existing type) | Cohesive aligned selector record. | Complete ordered site/period/inlet/platform/etc. tuples. Requested when held by config; authoritative retained values when held by merged data. Selection creates a new complete record. |
| `RhimeModelSpec` / `SectorSpec` (keep) | Scientific recipe choices and individual flux-sector definitions. | Priors, likelihood/component choices and source routing; no PyMC graph or acquired arrays. Independently usable by existing builders. |
| Built-in likelihood settings (keep) | Choices for the selected observation-error component. | Contained in the model specification. `None` retains the custom-likelihood meaning; do not create another configuration vocabulary. |
| `RhimeOutputSpec` (keep) | Final-product policy. | Formats, naming, destinations and saving. Independently usable by existing output/execution APIs. |
| `RhimeSampler` (keep) | Existing sampling settings and execution method. | Construction stores settings; `.sample(model, ...)` performs inference. No bound model, posterior or data in the configuration. No duplicate sampler-options class. |
| `retrieve_inversion_data` (rename public `data_processing_surface_notracer`) | Fresh acquisition and merge, covering surface and column data. | Existing shorthand arguments -> existing six-tuple of merged data and retained metadata lists. Includes existing observation-error construction and optional merged saving; no reload, basis construction or inference. |
| `_retrieve_inversion_data_from_options` (rename private canonical body) | Fresh acquisition with resolved selectors. | Complete site-options record plus explicit non-site arguments -> same six-tuple. No second shorthand expansion. |
| `data_processing_surface_notracer` (deprecated compatibility wrapper) | Preserve existing public calls/imports. | Same established signature, shorthand, six-tuple return and errors; issue `DeprecationWarning` naming `retrieve_inversion_data`, then delegate to the same body. No duplicate acquisition implementation. |
| `load_rhime_data` (replace retrieval/reload forwarding layers) | One shared data-loading boundary. | Resolved selectors and applicable choices, plus optional supplied data -> `RhimeMergedData`. Handles cache loading/fallback, selector/layout checks and retained-site alignment. A valid supplied handoff is returned unchanged and bypasses I/O. |
| `RhimeMergedData` (keep) | Acquired/reloaded numerical handoff. | Merged scientific datasets plus authoritative retained site options. Borrowed, potentially lazy arrays; no hidden materialization. |
| `prepare_rhime_inputs` (keep) | Independent preparation entry point. | Applicable preparation arguments -> `RhimePreparedInputs`; does not require a complete model/output/sampler request. Resolves applicable shorthand at its input boundary. |
| `filter_rhime_observations`, `build_rhime_basis`, `build_rhime_sensitivities`, `assemble_rhime_inputs` (keep) | Named scientific stages in the ordinary recipe. | Borrowed numerical handoffs and explicit resolved values -> filtered data, basis, sensitivities and assembled inputs. No phase-config class, reparsing or generic request context threaded through components. |
| `RhimePreparedInputs` (keep) | Labelled prepared model-input handoff. | Numerical inputs, basis and retained metadata; independent of full requested configuration. |
| `RhimeRunSpec` (keep) | Execution description. | Constructed after ordinary preparation from requested dates, retained sites/periods, prepared layout and resolved model/output choices. No acquisition options or sampler execution. |
| `RhimeRunnerSetup` / `make_rhime_runner_setup` / `resolve_rhime_options` (remove) | Redundant internal setup bundle and its constructors. | Migrate ordinary, nested, staged, shim and example consumers to `RhimeConfig` / `resolve_rhime_config`; construct retained run descriptions only after preparation. Do not retain a renamed bundle or compatibility projection. |
| `RhimeResult` (keep) | Completed execution result. | Retained descriptions, prepared numerical inputs, posterior and outputs. Never represents an unresolved request. |
| `run_rhime` / `run_rhime_multisector` (keep) | Ordinary procedural orchestration. | Existing file-plus-keyword inputs -> result. Decode, override, resolve, prepare, build, sample and output in visible order. |

`params_from_config` keeps its established normalized-dictionary default and
supported overrides. It delegates INI decoding to the clearly named decoder;
normalization is not full configuration resolution. `_SiteOptions` remains an
existing alignment helper, not a new public schema. Promoting it to a new public
`SiteOptions` API is not required for this correction.

For the low-level acquisition rename, preserve the six-tuple explicitly. Do not
quietly replace it with `RhimeMergedData`: that handoff belongs to the higher
retrieval/reload boundary. The canonical body shares scientific work with both
public names. Internal calls use the neutral name and must not emit deprecated
wrapper warnings.

`load_rhime_data` remains a function because cache loading is shared substantive
work. Keep the existing load-failure fallback, time-resolution and sector-layout
checks, retained-option selection and numerical normalization at their owning
boundary. Collapse the forwarding layers in `rhime.preparation` and acquisition;
do not replace each with another alias or `from_options` wrapper. Fresh retrieval
and the deprecated public wrapper remain distinct because their six-tuple
contract differs from the higher-level numerical handoff.

### 3. Resolve after overrides and before any scientific phase

`read_rhime_ini` returns raw values. Apply supported overrides before the semantic
resolver, including a changed site list. Expand scalar site selectors to that
effective site count; explicit lists/tuples must already match it. For two sites,
`"1h"` and `["1h", "1h"]` resolve identically, while `["1h"]` fails early.
This is an external-input convention shared by file and Python inputs; it is
not deferred to acquisition or tied to INI parser syntax.

Reuse existing alias, required-option, rejection, site expansion and likelihood
rules. Preserve case normalization, optional selectors, inlet slices and
`time_resolved=None`. Omitted/false `use_tracer` is consumed; effective true fails
early, including direct preparation and supplied-data routes. No tracer field
is retained. Keep established custom-likelihood conflict ordering.

The resolver declares every acquisition/preparation field explicitly. The
following inventory describes fields on `RhimeConfig`, not another class:

| Direct fields | Representation / meaning |
| --- | --- |
| `site_options` | Existing complete aligned site-selector tuples. |
| Species/domain, date bounds, flux sources, layout, BC use and basis output name | Requested/resolved strings, tuples and booleans; forwarded consistently to existing model/output views. |
| BC/observation/footprint/emissions stores, emissions domain, footprint model/species, calibration scale and BC input | Existing supported typed selectors. |
| Footprint/BC basis cases and directories, country directory, outer-regions path, basis algorithm/count and fixed outer regions | Existing supported path/string/integer/boolean choices. |
| Filters, averaging error, BC frequency, minimum error and normalized error options | Existing contracts; data-dependent materialization remains in preparation. |
| Reload/save merged data, merged directory/name, basis destination and non-finite flux check | Existing preparation-artifact policy and flux-check choices. |

The runner forwards needed named values at each scientific call. A small shallow
keyword mapping at an existing compatibility boundary is acceptable; it does not
justify a generic `as_data_args()` method, dynamic option schema or another
request/preparation record. Direct retrieval/preparation adapters resolve only
applicable inputs and share the scientific bodies with ordinary configured runs.

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

- The current implementation/docs no longer match this plan: reopen affected
  tasks and reconcile them before claiming completion or marking #813 ready.
- Public acquisition names have imports, monkeypatch seams and tuple consumers:
  preserve the deprecated wrapper, share the body and update internal callers.
- Model/output views repeat a few shared facts: resolve once and forward values;
  do not create independent defaults or validation authorities.
- Nested/staged/example consumers use the removed setup bundle: migrate them to
  one request type and identify supported persisted-contract effects explicitly.
- A resolved request has many options: group the documentation by purpose rather
  than adding another phase configuration class.

## Migration Plan

Update the existing proposal, design, behavioral spec and tasks on #813 first.
Then use the apply workflow to remove the preparation class and its plumbing,
remove the internal setup bundle, migrate ordinary/nested/staged/shim/example
consumers, implement the neutral names and consolidate loading into
`load_rhime_data`. Update documentation/exports. Reuse the existing equivalence, override,
site/drop/reload, ownership, nested/shim and sampler coverage. Add focused checks
for direct configuration access and deprecated-wrapper forwarding/warnings;
preserve scientific output checks. Run relevant broader coverage before handoff.

The previous locked Python 3.12/3.13 and documentation passes validate the old
implementation only. They do not establish conformance to this revised design.
Track the remaining reconciliation in [tasks.md](tasks.md).
