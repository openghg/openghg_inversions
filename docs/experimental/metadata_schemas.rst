Declarative metadata schemas prototype
======================================

This deliberately unmerged experiment for `issue 785
<https://github.com/openghg/openghg_inversions/issues/785#issuecomment-6174790340>`_
uses Pydantic for real standard prepare/sample manifests, ``OutputContract``,
nested RHIME model records and flux sanitation metadata, with a separate
msgspec comparison. It also tries
``xarray-validate`` at the CO2 prepared-input boundary. It is evidence for a
future decision, not the official metadata policy or a migration proposal.

Run the comparison
-------------------

Install the optional comparison dependencies alongside the runtime prototype:

.. code-block:: bash

   uv sync --extra schema-prototype
   uv run --extra schema-prototype pytest \
       tests/test_pydantic_handoff_records.py tests/test_model_record_schema.py \
       tests/test_msgspec_metadata_records.py tests/test_co2_schema_validation.py \
       tests/test_flux_metadata_schema.py tests/test_flux_sanitization.py \
       tests/models/test_likelihood_settings.py

The tests create manifests through the public ``save_preparation`` and
``save_samples`` writers. They independently validate their JSON using generated
JSON Schema, exercise saved replay, and check malformed fields, numerical data
references, disk round trips and lazy execution.

For an existing standard sample manifest, Pydantic exposes the complete nested
record and its schema directly:

.. code-block:: python

   import json
   from pathlib import Path
   from openghg_inversions.rhime._manifest_records import SampleManifest

   saved = Path("sample-manifest.json")
   record = SampleManifest.load(saved)
   print(record.model.likelihood)
   print(record.model_dump_json(by_alias=True, indent=2))
   Path("sample.schema.json").write_text(
       json.dumps(SampleManifest.model_json_schema(), indent=2) + "\n"
   )

The alternative msgspec implementation reads the same file:

.. code-block:: bash

   uv run --extra schema-prototype python -m \
       openghg_inversions.experimental.metadata_records sample-manifest.json
   uv run --extra schema-prototype python -m \
       openghg_inversions.experimental.metadata_records sample-manifest.json --schema

What changed
-------------

The Pydantic integration uses required typed fields, literals, forbidden extra
fields and a likelihood union discriminated by ``kind``. Standard manifest
classes now own ``load(path)`` and atomic ``save(directory)`` methods. Writers
validate before writing numerical artifacts and publish a completed JSON
manifest last. Loading saved numerical data and orchestrating scientific steps
remain procedural in the runners.

``OutputContract`` is now one Pydantic model, carrying its scientific methods,
validation, JSON encoding and schema. There is no second contract-record class
or dataclass conversion layer. ``create(...)`` supplies defaults for new
contracts; saved contracts must supply every declared field. ``with_updates``
validates a replacement and owns its JSON containers. Pydantic's native
``model_copy(update=...)`` does not validate updates, so it is not used for that
public operation. Replay reuses the validated contract directly.

.. code-block:: python

   from openghg_inversions.postprocessing.contracts import OutputContract

   contract = OutputContract.create(
       variable_roles={"flux_scale": "x"},
       supported_output_formats=("none", "inv_out"),
   )
   updated = contract.with_updates(metadata={"inventory": "example"})
   print(updated.model_dump_json())
   print(OutputContract.model_json_schema())

Likelihood settings and persisted options now live beside their scientific
components in ``models/pollution_event.py``, ``models/additive_sigma.py`` and
``models/fixed_error.py``. Pydantic dataclasses retain ``dataclasses.replace``
while validating scalar choices and rejecting unknown options. Component
methods own conversion to/from their records; a discriminated adapter selects
the requested likelihood instead of central parallel dispatch tables.
``DataConfig`` retains ownership of preparation option semantics, and
``RhimeModelSpec`` remains an ordinary dataclass.

Runtime likelihood settings can borrow numerical prior values. Their stored
options instead carry construction-owned scalars or references, so those are
still distinct types with an explicit scientific persistence boundary.
``PriorRecord`` owns restoring its references. Selecting a schema does not
construct a model or compute a prior array.

``FluxNonFiniteMetadata`` is also one Pydantic model. Its attribute reader and
JSON writer use native validation and serialization instead of manually
casting/reconstructing every field. The schema version is now required;
optional provenance defaults remain explicit. A missing attribute returns
``None``; malformed-present values fail with field paths. NetCDF and Zarr
tests exercise the public flux sanitation writer and metadata reader.

For example, ``schema_version=true`` fails at ``schema_version`` instead of
passing Python's ``True == 1`` comparison. A ``sigma_per_site`` option on a
``fixed_error`` likelihood fails at
``model.likelihood.fixed_error.options.sigma_per_site``. Missing recorded
scientific fields are errors; they are not reconstructed from today's defaults.

The first pass at ``9a9710b4`` added 137 physical source lines across the
contract/stage/model-record integration while reducing counted
``if``/``for``/``try``/``match`` statements from 57 to 45. The revised prototype
removes the duplicate contract representation and moves component-specific
declarations to their owners; schema declarations still cost code. The separate
msgspec alternative remains 206 lines and does not include a second runtime
reconstruction implementation. Its direct ``Struct`` constructor is unchecked,
so callers must use the supplied boundary helpers. These observations are not
performance benchmarks or a claim of total project code reduction.

Scientific ownership and compatibility
---------------------------------------

* The wire format remains ordinary named JSON, without Python module paths.
  Standalone nested model records inherit version ownership from their manifest;
  ``OutputContract`` has its own versioned envelope. Both implementations reject
  malformed versions and unknown fields. Existing writer-produced version-1
  records retain their fields; stricter acceptance is intentional in this
  prototype. No migration or historical-hash compatibility is promised.
* JSON provenance is a one-way projection. Existing helpers normalize NumPy
  scalars, paths and dates/durations; temporal strings retain nanosecond text,
  and ordinary sequences become JSON lists. Record validation is separate from
  that projection. Exact typed labels keep their existing tagged codec, and CF
  attributes keep their existing paired NetCDF/Zarr storage transformations.
* Prior arrays use the existing constructed-prior records from PR 873:
  ``{"data": "x_prior_mu"}`` refers to saved ``constant_data``. Required missing
  references fail during scientific reconstruction. The prototype never walks
  unused configured prior arrays. Symbolic ``variable``/``symbolic`` prior
  descriptors are intentionally unsupported; arbitrary graph restoration is
  outside this experiment.
* Pydantic rejects nested nonfinite JSON values. Strict float validation alone
  can still call ``__float__`` on array-like objects. The shared ``JsonFloat``
  field type requires an ordinary JSON number before conversion, preventing
  hidden execution of scalar Dask arrays. NumPy normalization remains an
  explicit provenance projection. Msgspec needs the explicit finite
  boundary helper: its native encoder would otherwise turn NaN/infinity into
  null. Both reject lazy arrays in these small-record fields without executing
  their graphs. Large arrays remain NetCDF/Zarr payloads.
* Open metadata mappings have intentionally open JSON Schema values. Runtime
  finite-JSON validation still applies; an exported schema cannot describe all
  scientific relationships or replace the product's loader.

CO2 dimensions and ``None``
----------------------------

``Co2PreparedInputs.dataset_schema()`` declares six required variables, ordered
dimensions and floating dtypes. In an unbound schema, ``None`` matches one
arbitrary dimension **name** at that position. For example,
``dims=("nmeasure", None)`` requires exactly two axes: ``nmeasure`` first and any
second-axis name. It does not mean an optional axis, a missing value or a size
constraint. Separate wildcards are independent; two ``None`` entries do not
require matching names or lengths.

.. code-block:: python

   from openghg_inversions.rhime.co2 import Co2PreparedInputs

   portable = Co2PreparedInputs.dataset_schema()
   retained = Co2PreparedInputs.dataset_schema("retained")

The bound schema requires ``H(nmeasure, retained)``,
``alpha_prior_mean(retained)`` and
``alpha_prior_covariance(retained, retained_cov)``. Construction reads H's
second-axis name and uses one bound schema pass; scientific datasets may use
``region``, ``retained`` or another meaningful state name. Missing H or H with
the wrong rank uses the unbound schema for required-variable/rank diagnostics.

Scientific coordinate equality/order, covariance squareness, numeric unit
scale and aggregation representations retain explicit owning checks. NetCDF
and Zarr tests verify round trips, loader rejection and non-mutation with a
nonstandard retained-state name; Dask tests verify no payload or auxiliary
coordinate computation. This prototype adds a floating-dtype restriction and
changes structural error text.

``xarray-validate==0.0.5`` cannot faithfully export this schema: generic dtype
constraints are silently omitted by ``serialize()``. The unbound declaration
does not infer state relationships; the product supplies explicit dimension
names for that purpose. This CO2
structural schema runs on construction/load. Saving a subsequently mutated
dataset does not rerun it, although other RHIME/aggregation checks still apply.

Assessment
-----------

Pydantic is the stronger candidate here for actionable nested errors and
removing routine handwritten validation. Msgspec remains attractive for small
manifest records, with a stricter need to funnel construction through boundary
helpers. Neither replaces scientific codecs. The CO2 schema is clearer when
its owner binds the state dimensions explicitly, although incomplete schema
export remains a limitation for wider adoption.

`OpenGHG PR 1768 <https://github.com/openghg/openghg/pull/1768>`_ was reviewed:
its proposed generic finite metadata/history utilities overlap common storage
mechanics, not these application-specific record schemas. It remains proposed;
this prototype neither assumes that API exists nor introduces a competing
shared provenance format. A real rollout still needs issue 785's policy,
version/compatibility decisions and a decision about shared ownership.

Library references: `Pydantic strict mode
<https://docs.pydantic.dev/latest/concepts/strict_mode/>`_, `Pydantic validators
<https://docs.pydantic.dev/latest/concepts/validators/>`_, `msgspec structs
<https://msgspec.dev/structs>`_, `msgspec supported types
<https://msgspec.dev/supported-types>`_, and `xarray-validate documentation
<https://xarray-validate.readthedocs.io/>`_.
