# Validation prototype follow-up

**Date:** 20 October 2026

**Status:** research and experimental implementation scope; not an adopted capability or serialization policy

**Baseline:** `devel` at `b87f315fb29b80085ea3c5358787a0a2e93c2ff2`

This note keeps the research, prototype boundaries and related issue priorities
together. It records the choices to exercise in two prototypes, rather than
commissioning a general validation framework or claiming that the outstanding
issues are implemented. The earlier [validation-library research](validation_library_research.md)
contains the package comparison and its limitations; it remains dated research.
Maintained guidance is [validation and xarray](../development/validation_and_xarray.rst),
[numerical array ownership](../development/numerical_array_ownership.rst) and
[RHIME model development](../development/rhime_model_development.rst).

## Landed baseline

The baseline includes merged PRs [#887](https://github.com/openghg/openghg_inversions/pull/887),
[#886](https://github.com/openghg/openghg_inversions/pull/886),
[#891](https://github.com/openghg/openghg_inversions/pull/891),
[#898](https://github.com/openghg/openghg_inversions/pull/898),
[#900](https://github.com/openghg/openghg_inversions/pull/900) and
[#899](https://github.com/openghg/openghg_inversions/pull/899).

- Standard and multisector execution have concrete preparation/sample handoffs
  and return `InversionOutput`. Preserve preparation generations and the
  operational checks against replaced or aliased artifacts. Do not reinstate
  retired result envelopes, output-binding artifacts or configuration hashes
  to implement schema validation. See the [sample writer and loader](https://github.com/openghg/openghg_inversions/blob/b87f315fb29b80085ea3c5358787a0a2e93c2ff2/openghg_inversions/rhime/multisector_stages.py#L211).
- Preparation owns species/domain. Standard saved model records omit sector
  sources; multisector records retain explicit source associations and order.
  Adapt the earlier [#874 Pydantic prototype](https://github.com/openghg/openghg_inversions/pull/874)
  to the [current model record](https://github.com/openghg/openghg_inversions/blob/b87f315fb29b80085ea3c5358787a0a2e93c2ff2/openghg_inversions/rhime/specs.py#L318),
  rather than copying its older envelope fields.
- Repeated `add_model_data` registration is validated after #891. #899 fixes
  the immediate flux/BC coordinate collision by qualifying BC coordinates;
  shared scalars must agree and same-owner conflicts still fail. The wider
  source/sector naming question remains [#893](https://github.com/openghg/openghg_inversions/issues/893).
  Preserve the [component-coordinate mechanics](https://github.com/openghg/openghg_inversions/blob/b87f315fb29b80085ea3c5358787a0a2e93c2ff2/openghg_inversions/models/coords.py#L26)
  and current one-source-per-sector restriction.
- [#844](https://github.com/openghg/openghg_inversions/issues/844) is closed and
  its delivered metadata fixes are baseline behavior to preserve. Older
  unchecked items and pre-merge comments are historical evidence, not a fresh
  implementation checklist. Publishing the official policy remains #785.

## Two bounded prototypes

### Custom merged-data schemas

Exercise `xarray-validate` at the boundary accepting independently supplied
`RhimeMergedData`. Declare local Dataset/DataArray structure near that owner:
required named variables, dimensions, dtypes and meaningful attributes. Preserve
explicit role and recipe choices rather than inferring scientific ownership
from dimension order, suffixes or the existence of a gathered level.

Keep cross-array label relationships, units normalization and scientific
invariants as ordinary named operations. Use Pint for dimensional compatibility
and scale checks. The merged-data prototype will treat absent error, response
and BC units as inheriting the `mf` convention without mutating attributes;
explicit units must express the same scale. Flux units remain a separate
physical role. This is an explicit prototype choice, not an upstream #1721
requirement. Physical signals, observation counts, diagnostic uncertainty
components and dimensionless metadata have different requirements; do not
require units or finite values on every variable indiscriminately. In
particular, absent, NaN and zero uncertainty components must retain their
existing preparation meaning.

Preserve the existing [merged-data boundary](https://github.com/openghg/openghg_inversions/blob/b87f315fb29b80085ea3c5358787a0a2e93c2ff2/openghg_inversions/inversion_data/acquisition.py#L96):
Dataset-only inputs, site/selector agreement, provenance inventory checks, and
separate request compatibility against known acquisition facts. Unknown facts
remain unknown; they do not establish compatibility. Missing provenance is
valid and must not be fabricated from a requested version or current package.

Construction and structural validation borrow the numerical payloads. Validate
already eager indexes directly; treat auxiliary coordinates as potentially
lazy. No properties, schema callbacks or scalar coercions may hide computation,
copying, densification or rechunking. Value-wide checks belong at a named
execution boundary, with related Dask arrays materialized together.

### Pydantic constructors, configuration and records

Use Pydantic for small scalar settings, selectors, acquisition facts, provenance,
output contracts and current handoff records. Pressure-test direct constructors
and configuration resolution with the same rules, including unknown keys and
likelihood-irrelevant options. Put likelihood settings and record conversion
beside their components; retain procedural scientific recipes and explicit
forwarding of resolved values.

The experiment may use validated dataclass construction to compare with the
passive-dataclass/explicit-resolver direction in #665. That comparison does not
silently change the production contract or deliver all of #661. Borrowed prior
arrays need an explicit numerical boundary; `arbitrary_types_allowed` alone is
not scientific validation. Reject opaque or lazy values in scalar fields before
numeric coercion can execute them.

Use current `make_model_output` and `InversionOutput` for standard/multisector
in-memory and saved outputs. CO2 and nested recipes retain their own `Co2Output`
and `NestedOutput` types. Prior metadata is descriptive: retain nullable scalars, registered
array references and named/opaque symbolic descriptions without evaluating a
graph. Executable reconstruction is a separate operation and must reject
unresolved dependencies when it needs their values. Do not restore arbitrary
PyMC graphs or require executable priors merely to inspect saved provenance.

## Contract choices to preserve or make explicit

| Boundary | Prototype constraint |
| --- | --- |
| JSON provenance/configuration | A deliberate readable projection of supported values; finite JSON, useful errors for unsupported values, no catch-all stringification or array-graph encoding. State temporal precision/units and sequence normalization rather than promising arbitrary Python-object restoration. |
| Typed scientific labels | Reuse the existing tagged label codec where exact scalar/tuple types matter; do not merge it into a lossy provenance projection. |
| NetCDF/Zarr attributes | Preserve the paired product-specific writer/reader, supported arrays and missingness; a JSON mapping schema is not a storage codec. |
| Record shape | Reject unknown structural fields and Boolean schema versions. Extensible provenance mappings remain finite JSON. Current manifest readers accept extra envelope fields, so stricter manifest schemas are an explicit prototype change. |
| Absence and malformed input | Preserve each owner's contract. Standard/multisector sample records require inference and provenance keys; malformed present sampler descriptions remain unavailable as `{}`, while provenance is a strict finite-JSON mapping. Ordinary and cached CO2 intentionally tolerate missing, null or malformed historical descriptive metadata as `{}`, without constructing fresh defaults; their description keys are optional. Strict product metadata still rejects malformed present values. |
| Provenance omission | In-memory `provenance=None` collects installed provenance; explicit `{}` remains empty. Saved replay must preserve recorded provenance rather than stamp the current environment. See [output binding](https://github.com/openghg/openghg_inversions/blob/b87f315fb29b80085ea3c5358787a0a2e93c2ff2/openghg_inversions/rhime/outputs.py#L478). |
| Constructed priors | Required `rhime_priors` metadata may contain `{}` when construction used no priors; a missing record is an error. Verify registered references without reading their payload. Preserve only constructed priors, not unused caller arrays. See [trace prior reading](https://github.com/openghg/openghg_inversions/blob/b87f315fb29b80085ea3c5358787a0a2e93c2ff2/openghg_inversions/models/priors.py#L259). |
| Defaults and replay | Resolve fresh configuration defaults at its owning boundary, persist actual choices, and avoid resolving a new sampler during replay. Preserve omission versus explicit `None`/false where they have different meanings. Do not silently reinterpret preparations with baked alignment/error policies. |

These are prototype constraints and policy questions to carry into #785, not a
replacement for its reviewed specification. Larger changes to dimensional
prior defaults, coordinate naming or retained scientific information need an
explicit decision in their owning issues.

## Related issue priorities

| Issue | Relationship and timing |
| --- | --- |
| [#665](https://github.com/openghg/openghg_inversions/issues/665) | Governs boundary ownership and no-hidden-execution checks in both prototypes. Compare library approaches without a universal schema registry. |
| [#789](https://github.com/openghg/openghg_inversions/issues/789) | Review repeated prepared/basis/CO2 save-time validation alongside adoption. A new schema must not add another reconstruction/materialization pass. |
| [#818](https://github.com/openghg/openghg_inversions/issues/818) | Focused adjacent fix: invalid filters must fail without the current blanket eager retry. Do not mistake successful structural validation for a pipeline-wide laziness guarantee. |
| [#785](https://github.com/openghg/openghg_inversions/issues/785) | Decide accepted values, restoration and compatibility before widespread format changes. Owns the three official metadata contracts and publication; this note and prototype do not close it. |
| [#814](https://github.com/openghg/openghg_inversions/issues/814) | Follow the constructor prototype with configuration-only export and a real write/read/resolve round trip. No acquisition, graph construction or numerical materialization during export. |
| [#661](https://github.com/openghg/openghg_inversions/issues/661) | Umbrella for real recipe-local typed settings and defaults. Exercise a bounded current recipe first; wider external formats and linked configurations are separate delivery. |
| [#877](https://github.com/openghg/openghg_inversions/issues/877) | Record a small durable-owner matrix before deleting data routes or claiming general reconstruction. Compare prepared arrays, `constant_data` and compact descriptors by scientific role. |
| [#625](https://github.com/openghg/openghg_inversions/issues/625) | Decide observation-error retention before moving its durable owner. Preserve resolved `mf_error` and component meanings; postprocessing must not recombine them silently. |
| [#876](https://github.com/openghg/openghg_inversions/issues/876) | Later coordinated migration of model-owned alignment/error construction. Current preparation still bakes BC periods and minimum error; schema adoption does not make those policies reusable under arbitrary new choices. |
| [#878](https://github.com/openghg/openghg_inversions/issues/878) | Follow #876/#877/#625 decisions when rerouting output consumers. Delete fallbacks only after every supported recipe supplies the chosen representation. |
| [#893](https://github.com/openghg/openghg_inversions/issues/893) | Preserve merged BC-only ownership now; wider flux/source-versus-sector semantics remain open. Do not add aggregation, automatic coordinate allocation or prefix guessing. |
| [#890](https://github.com/openghg/openghg_inversions/issues/890) | Later configured native-input preparation for ordinary CO2. It needs a scientific acquisition/reduction owner; custom merged-data schemas do not substitute for native inputs or reconstruct them from reduced H. |
| [#844](https://github.com/openghg/openghg_inversions/issues/844) | Completed baseline, not new prototype scope. Preserve its public writer/reader and metadata regression behavior. |

Older acquisition-cache wording in [#803](https://github.com/openghg/openghg_inversions/issues/803),
[#815](https://github.com/openghg/openghg_inversions/issues/815) and
[#259](https://github.com/openghg/openghg_inversions/issues/259) must be read against
the modern dataset-only merged-data boundary and deferred acquisition replay
in [#829](https://github.com/openghg/openghg_inversions/issues/829). Observation-error
regressions and auditability remain useful; they do not require reviving an
obsolete acquisition artifact. Likewise, the pre-merge comment on #893 is
historical; its updated delivery status distinguishes the merged fix from the
remaining design work.

## Upstream evidence and adoption checks

As checked on 20 October 2026:

- OpenGHG [#1720](https://github.com/openghg/openghg/pull/1720) is merged. Its
  `xarray-validate` refactor preserves the existing API: required named variables,
  dimension membership rather than fixed order, dtype checks and permitted
  extras. It does not deliver the later units/attributes/JSON policies.
- [#1721](https://github.com/openghg/openghg/pull/1721),
  [#1722](https://github.com/openghg/openghg/pull/1722) and
  [#1723](https://github.com/openghg/openghg/pull/1723) are open drafts for unit
  requirements, stable attributes and JSON schema declarations respectively.
- [#1768](https://github.com/openghg/openghg/pull/1768) is open and proposes shared
  xarray provenance utilities. Do not assume its API is installed or adopted.

Evaluate the prototypes independently. Test useful multi-field errors, current
scientific values/labels/units, unknown/nullable metadata, caller non-mutation,
preserved Dask identity/chunks and zero execution on invalid structural/scalar
input. Use actual public NetCDF/Zarr writer/reader paths for any persistence
claim. Keep numerical relationships, coordinate restoration and scientific
policy explicit, and compare removed validation branches with added adapters
before deciding whether either dependency earns production adoption. This note
records no test or performance result for the new prototypes.
