# Research: simplifying xarray validation

**Status:** research recommendation; no implementation proposed in this change

**Date:** 2026-08-13

**Motivating example:** [PR #583, `covariance_products.py`](https://github.com/openghg/openghg_inversions/pull/583/changes#diff-65be4811d0f555a30d02e82331cd44c6e0f2ae718917450e0a66a76062c2d33f)

## Executive recommendation

Do not adopt `xarray-schema`. If we trial an xarray-specific validation
dependency, trial **`xarray-validate`** for declarative validation of individual
`DataArray` and `Dataset` structure. Use **Pydantic v2 only for small scalar and
metadata models**, where its type-driven parsing and aggregated errors are a
natural fit. Keep scientific, numerical, cross-array, and persistence-transform
invariants as ordinary project functions.

The intended architecture is therefore layered:

1. Pydantic validates scalar options and decoded metadata.
2. `xarray-validate` validates local xarray structure: required variables,
   dimensions, shapes, dtypes, coordinates, attributes, and finite-value checks.
3. OpenGHG Inversions functions validate relationships: exact coordinate
   equality, row/column binding, MultiIndex semantics, covariance identities,
   rank, symmetry, and positive definiteness.
4. Serialization code performs reversible transformations such as coordinate
   namespacing and MultiIndex encoding; it is not treated as validation.

This is deliberately not a recommendation to wrap every xarray object in a
Pydantic model. Doing that would relocate the existing imperative checks into
custom Pydantic validators without removing their complexity.

Before adding either dependency to the project, run a small branch-only pilot
against `NativeCovarianceProducts.from_dataset`. Adopt the approach only if it
substantially reduces project-owned branching while preserving the current
error specificity, exact label semantics, and all tamper tests.

## Why PR #583 feels difficult

The cited PR adds a single 1,890-line module with 78 explicit `ValueError`
sites. This is not one kind of validation repeated 78 times. It combines five
different responsibilities:

| Responsibility | Examples in PR #583 | Best owner |
|---|---|---|
| Local xarray structure | required variables, dimension order, square shapes, finite real values | `xarray-validate` candidate |
| Scalar/metadata parsing | schema version, view literal, SHA-256 syntax, JSON string mappings | Pydantic candidate |
| Cross-array label relationships | matching retained-state labels, mirrored matrix-column coordinates, shared observation axes | project code |
| Scientific/numerical invariants | full rank, symmetry, positive definiteness, `B Pi.T = U_* C_alpha` | project code |
| Persistence transforms and integrity | collision-safe coordinate names, MultiIndex restoration, source/configuration digests | project code |

The largest concentration is deserialization. `NativeCovarianceProducts.from_dataset`
validates root metadata, six required product variables, shared identities and
strategy attributes, coordinate-namespace ownership, array dimensions and
shapes, finite real values, restored MultiIndexes, and cross-array labels. The
construction path separately validates projection strategy output, exact native
coordinates, covariance action results, and numerical identities.

Schema libraries can make the first row concise and the second row clearer.
They do not remove the domain logic in the remaining rows.

There is also validation duplication at an ownership boundary:
`_validated_sensitivity` checks dimensions, coordinates, numeric/finite values,
then calls `covariance.apply`, whose own matrix validator repeats much of that
work. An xarray schema does not resolve that duplication. The pilot should also
decide which side of each protocol boundary owns which preconditions.

The PR has strong focused coverage (30 covariance-product tests plus three
multisource integration tests), but the API has no RHIME/model/postprocessing
production caller yet. That makes this a good point to simplify the boundary
without needing a broad caller migration, while still treating the publicly
re-exported API and serialized version as compatibility constraints.

## Library assessment

### Pydantic v2

Pydantic is mature, actively maintained, and strong at validating Python data
models from type hints. Its reusable `Annotated` validators, field/model
validators, `Literal` support, strict mode, structured `ValidationError`, and
`TypeAdapter` are useful here. Pydantic's documentation explicitly supports
custom validation for arbitrary third-party types, but those integrations still
require us to write the xarray-specific rules.

Good fits in this repository would be:

- schema/version metadata;
- `Literal["dense", "diagonal"]` and positive batch-size options;
- non-empty identifiers and 64-character lowercase SHA-256 strings;
- JSON-decoded `dict[str, str]`, namespace mappings, and MultiIndex dimension
  declarations;
- configuration/provenance records whose source form is ordinary Python or
  JSON-compatible data.

Poor fits would be:

- expressing a `DataArray` shape whose dimension names are runtime parameters;
- exact equality of xarray indexes and auxiliary coordinates across arrays;
- checking covariance actions or matrix identities;
- encoding/restoring namespaced coordinates or MultiIndexes;
- using `arbitrary_types_allowed=True` and assuming that it validates an
  `xr.DataArray`. It checks the instance type only unless custom validators are
  supplied.

Pydantic should therefore sit at the metadata boundary, not become the owner of
the numerical object model. If introduced, use its public validators and
`TypeAdapter`; avoid coupling this project to `pydantic-core` schema internals.

### xarray-schema

`xarray-schema` has the right conceptual API: `DataArraySchema`,
`DatasetSchema`, and component schemas for dimensions, shapes, coordinates,
attributes, chunks, names, and dtypes. It can serialize schema descriptions and
supports custom checks.

It is not a good new production dependency:

- its own documentation labels it experimental;
- PyPI labels it alpha;
- the latest release is 0.0.3 from April 2022;
- the project describes itself as a very early prototype;
- validation raises eagerly rather than aggregating errors.

The API is useful prior art, but its maintenance and compatibility risk outweigh
the modest amount of local structural validation it would replace.

### xarray-validate

The likely package intended by “xarray-validator” is
[`xarray-validate`](https://github.com/leroyvn/xarray-validate) (imported as
`xarray_validate`). It is a maintained refactor/fork of `xarray-schema`, with a
0.0.5 release in January 2026 and a beta classifier. Its required dependencies
are only `attrs`, `numpy`, and `xarray`; YAML, Dask, Pint unit support, and their
dependencies are optional.

Compared with `xarray-schema`, it adds or documents:

- eager or lazy validation, with lazy mode collecting errors and their paths;
- nested `DataArray`, `Dataset`, coordinate, and attribute schemas;
- arbitrary callable checks on arrays and datasets;
- schema construction from existing xarray objects;
- schema serialization/deserialization and optional YAML loading;
- exact, glob, and regular-expression matching for variables, coordinates, and
  attributes;
- optional Pint-backed unit validation.

This is the best match for replacing the mechanical parts of
`_validate_serialized_product_arrays`. In particular, a product dataset schema
could declare the six required arrays and their local dimension/shape patterns,
then attach a reusable “finite real” check. Lazy mode should improve load errors
by reporting several malformed variables at once.

Important limitations remain:

- version 0.0.x and one maintainer make it a higher-risk dependency than
  Pydantic;
- the schema still needs custom callables for value-level rules;
- dynamic names such as `state_dim`, native dimensions, and collision-safe
  column dimensions mean schemas may need to be built by a factory per artifact;
- local coordinate schemas do not establish equality between different arrays;
- its schema serialization is not a reason to replace this project's artifact
  schema or digest logic; the project explicitly says JSON round-trip is not
  guaranteed;
- its latest package metadata does not declare xarray upper/lower bounds, so
  compatibility with this project's `xarray>=2025.06.0` must be tested rather
  than inferred.

## Recommended boundary for PR #583

### Move to an xarray schema

The following rules are repetitive and local to one loaded dataset or array:

- all six product variables are present;
- each array has the expected number and order of dimensions;
- state and observation covariance arrays have square or diagonal shape as
  selected by the view;
- arrays contain finite, real values;
- required local attributes exist and have basic scalar types;
- required dimension coordinates exist.

These checks should be defined once, close to the artifact schema version, and
run in lazy/aggregate mode.

### Move to Pydantic only if it earns its dependency

The following metadata is a clean Pydantic model candidate:

- root `schema`, `schema_version`, `strategy`, and observation-covariance view;
- source, view, and configuration identities;
- decoded `basis_provenance`;
- decoded coordinate namespaces and MultiIndex dimension declarations.

Pydantic would consolidate repeated `isinstance`, membership, JSON shape, and
string-format branches and return path-aware errors. However, this metadata is
small enough that a stdlib dataclass plus explicit parser could also remain
reasonable. Do not add Pydantic solely to replace a handful of checks; first
verify whether it is already a required transitive dependency across supported
OpenGHG environments, then decide whether to make it direct.

### Keep explicit project validators

These rules carry OpenGHG Inversions semantics and should remain named Python
functions, even when called from a schema's `checks` hook:

- exact native, retained-state, and observation coordinate equality;
- equality of all mirrored auxiliary coordinates on matrix column axes;
- restored pandas `MultiIndex` presence and level names;
- namespace ownership, reversibility, and collision safety;
- derived view identity and covariance-configuration digest binding;
- projection rank and retained-state consistency;
- symmetry and positive-definiteness diagnostics;
- `B Pi.T = U_* C_alpha` and other covariance-specific identities;
- compatibility across the covariance `apply`/`solve` protocol boundary.

Keeping these functions explicit makes the scientific contract searchable,
directly testable, and independent of a third-party validation framework.

## Proposed internal design

The artifact boundary should have four visible phases:

1. **Decode:** restore MultiIndexes and decode namespaced coordinates and JSON
   metadata. Transformations happen here.
2. **Validate local structure:** run the version-specific xarray schema and the
   scalar metadata parser. No scientific computation occurs here.
3. **Validate relationships:** compare labels, coordinate mirrors, identities,
   digests, and view-dependent relationships.
4. **Construct:** create `NativeCovarianceProducts` only from the validated,
   decoded values.

This separation is more important than the particular library. It prevents
load-time transformation, generic schema validation, and scientific validation
from growing together in one method again.

Schemas should be version-specific and internal, for example one factory for
native-covariance-products schema version 2. Runtime dimension names and the
selected dense/diagonal view are inputs to that factory. Public APIs should
continue to raise a stable project-level load error (with the third-party error
as its cause or formatted detail), so replacing a validation dependency does
not become a user-facing exception compatibility break.

## Pilot and decision gates

Run a no-commit or experimental-branch spike with the following scope:

1. Re-express only the local structural portion of
   `NativeCovarianceProducts.from_dataset` using `xarray-validate` 0.0.5.
2. Optionally model decoded root metadata with Pydantic v2; measure it as an
   independent choice.
3. Preserve every current invalid/tamper test and add focused cases for wrong
   schema version, missing variables, invalid view/batch options, colliding
   dimensions, and missing observation coordinates.
4. Test against the repository's current, previous, and development OpenGHG
   environments, especially their resolved xarray versions.
5. Compare project-owned lines, branches, error messages, import time, and
   dependency resolution before and after.

Adopt `xarray-validate` only if the pilot:

- deletes substantially more custom validation than it adds in schema adapters;
- produces errors at least as precise as the current tests require;
- preserves exact label and MultiIndex semantics;
- does not coerce or mutate scientific arrays;
- passes the full compatibility matrix;
- leaves scientific invariants as clear named functions.

Adopt Pydantic independently only if metadata parsing is reused by more than
this one artifact or if its structured errors materially simplify the loader.

If either gate fails, retain project-owned validation but still apply the
four-phase boundary above. A small internal validation toolkit for required
variables, exact indexes, finite-real arrays, square labelled matrices, and
column-coordinate mirroring may deliver most of the readability benefit with
less dependency risk.

## Decisions and open questions

### Recommended decisions

- Reject `xarray-schema` as a new dependency.
- Evaluate `xarray-validate`, not both xarray packages.
- Treat Pydantic and `xarray-validate` as complementary and independently
  optional choices.
- Do not encode numerical/scientific invariants into opaque framework hooks.
- Do not change the serialized artifact format merely to match a library's
  schema serialization.
- Do not introduce unit validation in this pilot; product unit algebra is a
  separate scientific design question.

### Open questions for the pilot

- Does `xarray-validate` preserve useful error paths when checks depend on a
  dynamically generated schema?
- Can it validate dimension coordinates and auxiliary coordinates without
  accidentally accepting xarray alignment or reordering?
- Does lazy validation touch `.values` in a way that changes eager/Dask cost?
- Is Pydantic already consistently installed through supported OpenGHG/PyMC
  dependency sets, and would relying on that transitive dependency be unstable?
- Should direct construction of `NativeCovarianceProducts` remain permissive,
  or should validated construction become the only supported internal path?
- Which exception type and message stability are part of the public loading
  contract?

## Sources

- [Pydantic: validators](https://docs.pydantic.dev/latest/concepts/validators/)
- [Pydantic: custom and arbitrary types](https://docs.pydantic.dev/latest/concepts/types/)
- [xarray-schema documentation](https://xarray-schema.readthedocs.io/en/latest/)
- [xarray-schema repository](https://github.com/xarray-contrib/xarray-schema)
- [xarray-schema on PyPI](https://pypi.org/project/xarray-schema/)
- [xarray-validate documentation](https://xarray-validate.readthedocs.io/en/latest/getting_started.html)
- [xarray-validate API](https://xarray-validate.readthedocs.io/en/latest/api.html)
- [xarray-validate repository](https://github.com/leroyvn/xarray-validate)
- [xarray-validate on PyPI](https://pypi.org/project/xarray-validate/)
- [OpenGHG Inversions PR #583](https://github.com/openghg/openghg_inversions/pull/583)
