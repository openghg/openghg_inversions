# Design

## Accepted target and delivery status — 23 October 2026

The maintainer-approved #944/#961 boundary and subsequent #972 review amendments
require modern execution without HBMCMC or temporary runtime option translation.
The desired contracts do not retain old interfaces merely because transitional
code implemented them. The #719 implementation deletes those runtime surfaces;
revision-specific acceptance is recorded in tasks and the PR, with no release
readiness claim.

The finite [CLI map](tasks.md#cli-command-name-amendment--23-october-2026) targets
one `rhime` executable and canonical `run --model` commands. The
[five-name Python map](tasks.md#named-python-alias-amendment--23-october-2026)
targets `run_multisector`, `run_nested`, `MergedData`, `PreparedInputs` and
`assemble_inputs`, without naming-only forwarders. Preserve canonical callable
and type behavior, constructors, results and borrowed/lazy execution boundaries.
Supported format/version acceptance, conversion or rejection belongs to artifact
owners; this naming cleanup neither mechanically renames scientific contents nor
promises historical-format support.

The [migration checklist](tasks.md#migration-evidence-checklist-and-delivery-order)
owns the pinned #960 comparison, affected-entry evidence, scientific test migration,
current owners and delivery order. Final version/public names stay with #945,
independent basis decisions with #951, and broader Python naming with #663/#945
and the #776 revival. Finished converters/launchers, maintenance publishing and
integrated acceptance remain M2 release gates rather than prerequisites for
bounded development retirement.

## Implementation decisions

1. Delete HBMCMC's wrapper, adapters and legacy utilities without recreating their
   namespace through imports, forwarding shims or console scripts. Replace normal
   callers with the existing canonical interfaces; do not make modern execution
   invoke a converter or historical translator.
2. Remove temporary spelling/output-format/flux translation from modern
   frontends. Validate supported names and applicable values generically; defer
   values in known inactive groups until their consuming phase. Keep replacement
   advice in migration documentation/tooling, without legacy-name tables or
   historical-token checks in runner validation.
   Remove every `use_tracer` field, translation branch and special error:
   every explicit occurrence, regardless of value, is unsupported; omission is
   valid without tracer execution. Cover file, override, saved-phase, direct and
   supplied-data routes before scientific work.
3. Keep independent modern INI decoding, configuration groups, exact resource
   selections, invocation-relative paths, BC loading and packaged defaults.
   Existing path conversion uses owning declarations/type annotations and a
   small standard-library converter; retaining that helper implementation is not
   a public behavioral requirement. Check file-derived and explicit override
   paths against the same invocation working directory; manifest dependency
   references retain their own declaring-record base.
4. Port scientific tests before removing adapter-only plumbing. Canonical full,
   prepared, saved and output routes retain their equations, labelled inputs,
   signed/current-run flux, sampling defaults, all retained chains and explicit
   array materialization boundaries. Ordinary supported basis representations,
   `output_format="legacy"`, `save_inversion_output` and exact export destinations
   remain independent of executor retirement.
5. Canonical standard/multisector execution returns `InversionOutput` and uses
   `run_standard`/`run_multisector`, their eager builders and the custom-likelihood
   seam. The older prepared runner, multisector build/result adapters and
   `RhimeResult`/`RhimeRunSpec` envelopes have no target runtime compatibility
   contract. Do not condition the desired interface on migration of a historical
   cached consumer or preserve bespoke import errors as permanent requirements.
6. Keep public `SiteOptions.from_inputs` shorthand/alignment behavior. Private
   `convert_to_list` helper selection, constructor refactors, ordinary forwarding
   and test/example updates are implementation choices, not public contracts.

## Scientific-input coordination

The focused scientific-input amendment is owned by #876 and
[PR #964](https://github.com/openghg/openghg_inversions/pull/964). It owns the
retained observation population, missing-error derivation/imputation and nested
alignment. Configuration owns option resolution and passes explicit scientific
choices; it must not establish a conflicting acquired-population order.
Provider `get_obs_surface(..., average=...)` resamples before returning acquired
observations, with repeatability and variability propagated separately. Inversion
preparation combines the returned components, preserving supplied `mf_error`.
The selection population used for missing errors is a distinct scientific choice.

The #971 retirement covers exactly `daily_median` and `six_hr_mean` and their
registered names. Their historical post-error aggregation order is source evidence,
not a target requirement. Filters select observations; any aggregation is an
explicit scientific operation with its own uncertainty propagation. These records
do not prescribe error combination before provider resampling or implement the
separate scientific change. #719, including tracer cleanup, precedes #964; #971
is a separate prerequisite for #964, not for #719.

## Historical isolation and transition evidence

At PR #869 revision `e41e2838`, modern code imported `hbmcmc.compatibility`, while
`run_hbmcmc.main` called fixedbasis translation without the required resource
translation. The 0.8 isolation delivery corrected this with one-way dependencies
and legacy-owned discovery. CI passed at `7260b43d`; resource I/O was promoted and
[archived](../archive/2026-10-13-move-rhime-resource-io-to-runners/tasks.md).
The checked tasks retain that historical validation and former 0.9 schedule.

The removed transition operated after winning overrides. HBMCMC's
`params_from_config(normalise=False)` returned decoded overridden historical
values; normalization translated names/discovery and produced canonical absolute
paths. Historical NESW directories warned without lookup, and only winning
resource selections were discovered. Ordinary parameter aliases favored a
canonical spelling; old/new flux selectors conflicted even for equal values.
Those mechanics remain source evidence for explicit conversion, not modern
request requirements.

Historical HBMCMC wrapper scientific checks rejected `no_model_error=True` and
the pollution-event combination with observation-derived error and offsets;
private likelihood, unused-sigma, minimum-error and chain-selection controls have
already been removed from modern construction. Preserve their source observations
without restoring historical graphs. Historical equations and chain-0 summaries
require matching old software; modern products retain all-chain semantics.

The six-tuple `retrieve_inversion_data` and deprecated
`data_processing_surface_notracer` forwarding wrapper live in HBMCMC's
`legacy_data` at the pinned pre-retirement revision. Their error preparation,
cache behavior, signatures, warnings and
legacy-container returns are historical evidence covered by namespace retirement,
not an independent modern API promise. Port relevant scientific coverage to
canonical `MergedData`/`PreparedInputs` without creating another wrapper.

## Risks and validation

A compatibility name can identify a retained scientific or artifact capability.
Review actual consumers before deleting anything outside the finite maps; keep
independent basis decisions with #951. Verify modern imports/execution without
HBMCMC, wheel/sdist contents and rejection of retired access paths. Use focused
canonical-input scientific regressions and distinguish inspected source from
executed checks at the implementation revision. Past isolation CI and #972
contract checks do not validate this runtime deletion.

## Runtime retirement implementation — 23 October 2026

The base is merged #972, `17e28ed081954b4155c9d85b66eaeef2ad81853c`, with
#966 already landed. The dated #960 comparison in
`docs/development/migration_960.rst` and plain
`tests/fixtures/migration_960/retirements/runtime_719_2026_10_23.json` records
new discoveries before namespace removal, exact maps, dispositions and owners.
Original archived inputs and expectations are preserved.

Runner requests now group only canonical names; unsupported options fail at the
owning request boundary. Pure INI decoding continues to return original values.
Every explicit `use_tracer` value follows ordinary unsupported-option validation.
Acquisition accepts only canonical flux selectors. Independent basis keyword
compatibility remains with #951 rather than being removed by runtime cleanup.
The CLI retains generic `run --model` dispatch, including distinct CO2 forwarding,
and staged/product operations; only the approved command map is retired. Only
the five approved Python alias assignments/re-exports are removed.

Scientific regressions formerly using dictionary/six-tuple adapters move to
canonical `MergedData`, resolved `DataConfig` acquisition and explicit error
preparation. Resource tests use exact canonical selections; historical discovery
expectations remain plain converter evidence. Canonical explicit settings retain
sampler meaning, priors, offset/fixed-error semantics and all-chain summaries.
Historical wrapper defaults and config-copy origin behavior remain archived,
without becoming default policies of modern execution. No aggregation retirement,
retained-population imputation or nested-alignment science is imported from the
isolated #971/#964/#973 drafts.


## Clarified validation and rebase follow-up — 23 October 2026

Maintainer review [5634401507](https://github.com/openghg/openghg_inversions/pull/974#pullrequestreview-5634401507)
authorizes generic supported-name/applicable-value validation and corrects the
contradictory every-output-alias-occurrence promise. Unknown names fail even in
inactive groups; known inactive values remain deferred. A shared preparation
file can therefore contain an invalid output format without resolving it.
The eventual full/output consumer applies ordinary supported-format validation.
No special `hbmcmc` token check is added. Explicit `use_tracer` remains an unknown
option, regardless of value.

Delete `_request_validation.py` and its removed-name/resource diagnostic tables.
Remove nested's dedicated historical inner-selector branches and those names
from its extraction set, leaving generic owner validation to report unknowns.
Migration advice stays in the guide and later #949 tooling. Independently
supported basis adapters retain #951 ownership.

Rebase onto `b288ac2530ad93ae8f67c5f1d47ee8ee7b666000` retains #975's modern
Zarr compatibility fixes. Its historical cache writer changes do not restore the
retired namespace. The dated `review_974_2026_10_23.json` records the new source
hashes, ZIP expectations, pre-removal diagnostics and phase-value cases before
this follow-up. Original evidence and earlier revision-specific results remain
unchanged; fresh validation belongs to the resulting implementation revision.


## Approved factory retirement and validation cleanup — 24 October 2026

[Approval 6380451664](https://github.com/openghg/openghg_inversions/pull/974#issuecomment-6380451664)
adds only `PreparedInputs.from_legacy_inputs` to this retirement, separately from
the five aliases. Preserve canonical construction, schema/version acceptance,
codec behavior and scientific contents. The dated #960 fixture
`factory_and_cleanup_2026_10_24.json` captures the exact pre-removal signature,
original two cases, source hashes and provenance/coordinate rules at `e1d020f0`
before deletion. Its factory body is unchanged from pinned `0a32d73a` and base
`b288ac25`. #945/#663 own the bounded constructor disposition; #951 does not.

The sole production caller, xarray preparation, already owns paired stationary
and moving release-coordinate normalization. Construct site metadata with the
existing `_make_site_metadata`, preserve the basis's source/path metadata through
`with_metadata` and call `PreparedInputs` directly. Preserve namespaced and plain
fallback keys, string normalization, nullable periods and retained site order.
An unannotated supplied basis must continue to report `"supplied"`, not fall
through to the prepared property's `"generated"` default. Do not mutate the input
basis or numerical payloads, compute Dask work or add the old scalar-coordinate
arguments to the constructor. Replace factory-only coverage with canonical
coordinate expectations and xarray provenance/nonmutation regressions.

For [review 5637015484](https://github.com/openghg/openghg_inversions/pull/974#pullrequestreview-5637015484),
delete migration-only `rhime.__getattr__` and let removed names be ordinarily
absent. Reject resolved configuration combined with raw sources before parsing
the raw request; retain the nested custom-likelihood conflict check at the
resolved owner only. These are invalid requests already, so neither scientific
inputs nor extra configuration values need resolving.

CO2 outer request loaders own whole-request name validation once; they keep
unknown names in inactive groups unsupported while leaving known inactive values
deferred. Preparation/model/seed/sampler/output component parsers consume their
applicable sections and validate their values, without rescanning unrelated
sections. Raw handoff mappings enter the outer loader once; loaded CLI paths
forward resolved component choices. A small traversal may remain local to this
boundary; no marker envelope or repository-wide schema framework is introduced.

Keep the private sentinel and retained `emissions_name`/`flux_sources` resolution
inline in the independent basis adapter, removing the root generic helper.
Retain omitted/explicit-null/conflicting-selector meanings. Remove
the unapproved fixed 0.9 warning date; #945/#951 own any later alias decision.
Remove dead HBMCMC Jinja branches and historical vocabulary from active templates;
INI/preparation docstrings describe canonical decoding/resources. Retained legacy
output and basis/artifact capabilities are unchanged. Fresh validation belongs
to the resulting revision; earlier complete matrices do not validate this slice.
