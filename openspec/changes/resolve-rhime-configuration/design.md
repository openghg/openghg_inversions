# Design

## Context

See [proposal.md](proposal.md) and the [behavioral contract](specs/rhime-configuration/spec.md).
The inspected checkout is `9b8cac81`. Its `rhime.params` already separates
`params_from_config(..., normalise=False)` from `resolve_rhime_options`, but
`RhimeRunnerSetup` contains a runtime sampler and partly raw `data_args`.
`inversion_data.preparation._SiteOptions` owns aligned selectors; acquisition
unpacks them into the legacy loader, which expands them again.

Use the ownership resulting from #773/#774 when implementing. PR #807 consumes
unsupported tracer requests at resolution; preserve that behavior rather than
restoring a `use_tracer=False` field merely for hashing. PR #802 is the proposed
replacement staged design; #798 supplies prototype evidence only.

## Goals / Non-Goals

**Goals:** One readable translation boundary, ordinary typed values, reused site
alignment, and explicit forwarding into the existing procedural scientific flow.

**Non-goals:** A universal configuration/schema framework, phase-only file
semantics, new file formats, sampler lifecycle redesign, stage ownership changes,
cache policy, numerical execution cleanup or new scientific equations.

## Decisions

### 1. Keep loading and resolution separate

Proposed API sketch; these functions/types are not installed yet:

```python
def load_rhime_config(path: str | Path) -> Mapping[str, object]: ...
def resolve_rhime_config(
    params: Mapping[str, object], *, multisector: bool
) -> RhimeConfig: ...

class RhimeConfig:
    run_spec: RhimeRunSpec
    preparation: RhimePreparationConfig
    sampling: RhimeSamplingOptions

class RhimePreparationConfig:
    site_options: _SiteOptions
    species: str
    domain: str
    start_date: str
    end_date: str
    flux_sources: tuple[str, ...]
    # Other existing preparation choices are named typed fields.
```

Implement the loader with the existing INI parser. It performs no alias/default
resolution; a Python mapping needs no loader. Merge file values and supported
overrides before invoking the resolver. Keep `params_from_config` as the existing
compatibility wrapper, including its default normalized-dictionary return.

The resolver remains in the existing parameter owner and reuses its alias,
required-option, sector/source and likelihood rules. The full-run bundle is
specific to standard/multisector RHIME; its values are independently usable by
later phase interfaces. Avoid a cross-family recipe container or new registry.
Retaining only a typed dictionary would still leave internal consumers unpacking
an unstructured preparation contract.

### 2. Reuse existing specifications and explicitly own missing fields

Reuse `RhimeRunSpec`, `RhimeModelSpec`, `RhimeOutputSpec`, `SectorSpec` and built-in
likelihood settings. Copy established configuration-only defaults for active
priors from their current constants; retain builders' direct-API fallback behavior.
Site-dependent prior conversion and numerical policies remain at their scientific
boundaries. The requested run specification is a pre-preparation view; derive a
separate retained run specification after preparation without modifying the config.

Preparation field inventory follows the existing explicit
`RHIME_PREPARATION_OPTION_NAMES` and defaults, reconciled with landed changes:

| Fields | Representation |
| --- | --- |
| Sites, averaging, inlet, height, instrument, platform, observation level, meteorological model, maximum level, time resolution | Existing complete `_SiteOptions` record |
| Species/domain, dates, output name, flux sources, sector layout | Named strings, source tuple and boolean |
| Store selectors, emissions domain, footprint model/species, calibration, BC input | Existing string/optional-string choices |
| Basis cases/directories, country directory, outer regions, algorithm/count, fixed outer regions | Existing typed path/string/integer/boolean choices |
| Filters, averaging error, BC frequency, minimum error and its options | Existing supported component values; reuse `MinErrorConfig` and established filter contract |
| Reload/save merged data, merged directory/name, basis destination, non-finite flux mode | Existing path/string/boolean choices and `FluxNonFiniteCheck` |

All fields are declared explicitly; this table is an inventory, not a dynamic
schema builder or an `extra_options` escape hatch. Keep shared facts consistent
when constructing the records instead of adding repeated consistency validators.

`RhimeSamplingOptions` is a small frozen values record with the currently accepted
`draws`, `burn`, `tune`, `chains`, `nuts_sampler`, `progressbar`, `sample_kwargs`
and `posterior_predictive_kwargs`. Preserve existing defaults and keyword value
types. The runner creates the existing `RhimeSampler` explicitly from those
values after resolution; its algorithms, predictive defaults and direct API stay
unchanged. Do not expose additional predictive switches through configuration.

Use frozen records for stable choices. Copy ordinary containers where translation
needs ownership; never deep-copy scientific handoffs or build a general recursive
freezing system. Keep supported opaque sampler keyword values compatible.

### 3. Resolve site shorthand once, then select by label

Reuse `_SiteOptions.from_inputs`, promoting/moving the existing record into its
shared owner only if required to avoid import cycles. It already uppercases site
labels, expands scalars, validates sequence lengths and supports complete-record
selection. Preserve `time_resolved=None`, supported inlet slices, integer levels
and all optional string selectors; do not invent metadata defaults.

Canonical acquisition accepts that record and forwards its entries directly to
the retrieval body. Keep the public low-level loader's shorthand signature as an
adapter using the shared site translation before entering the same body. The
public preparation adapter resolves applicable preparation choices without
requiring likelihood, model-output or sampler settings. Neither route should run
full-run resolution for choices it cannot use. Tuple/list conversion needed by
an external API is explicit representation conversion, not another parsing pass.

Acquisition, compatible reload and filtering keep their existing label-selection
mechanics. Supplied merged data retains its own authoritative site record.
Requested choices describe input intent; merged/prepared handoffs describe what
was retained. No second requested-option resolver is introduced after a site drop.

### 4. Keep public adaptation small

Ordinary full runners use the concrete config directly. Preserve public
`resolve_rhime_options`/setup consumers, including nested RHIME, the HBMCMC shim
and composition examples, through adapters to the same semantic resolution.
Compatibility projection must not own a second alias/default/validation policy.
Independent model builders and prepared-input runners remain independent.

Keep flat INI section interpretation, alias warnings and canonical-name
precedence. Preserve ordinary cwd-relative paths and existing staged
source-relative paths at their adapters. Existing staged JSON parameter input is
an existing transport contract; this work adds no new file frontend and does not
remove that interface. Phase-only resolution is subsequent #802 work.

### 5. Keep hash/manifest policy outside this configuration change

Equivalent scalar/sequence inputs yield equal canonical values; this is not a
new promise about historical hashes. Do not add hash methods, serialization
protocols, migration machinery or raw-spelling fields to the configuration.
[#808](https://github.com/openghg/openghg_inversions/issues/808) collects the
representation-sensitive gates; [#802](https://github.com/openghg/openghg_inversions/pull/802)
owns reusable handoffs, authentication and explicit staged contract transitions.
If an existing supported staged encoding is affected during integration, handle
that consumer explicitly; never silently redefine its declared version. This
does not require completing the wider hashing work before reviewing this spec.

## Risks / Trade-offs

- Public helpers have consumers beyond the ordinary runners -> inspect those
  adapters and reuse existing nested/shim/composition coverage.
- Configuration defaults can still hide in builders -> resolve only known
  configuration defaults; retain data-dependent interpretation where data exists.
- Module ownership is changing -> implement against the landed acquisition stack,
  without copying the old and new owners into parallel paths.
- Frozen records can suggest deeper immutability than they provide -> document
  ownership and preserve input non-mutation without a generic freeze framework.

## Migration Plan

Review this bounded spec before generating implementation tasks. Implement after
#773/#774, reconciling #807's tracer rejection. Add the resolver/value records,
switch ordinary consumers and adapt supported entry points together. Reuse
existing site/drop/reload checks, replacing the test that expects raw scalar
periods to reach acquisition. Add file/Python equivalence, early-failure and
caller-non-mutation checks; run relevant broader coverage rather than a new test
harness. Update configuration guidance and add `newsfragments/804.feature.md` at
implementation time. No release note or runtime test run is needed for this
planning-only draft.
