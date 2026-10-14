Declarative metadata schemas prototype
======================================

This deliberately unmerged experiment for `issue 785
<https://github.com/openghg/openghg_inversions/issues/785#issuecomment-6174790340>`_
compares Pydantic and msgspec on real standard prepare/sample manifests,
``OutputContract`` and nested RHIME model records. It also tries
``xarray-validate`` at the CO2 prepared-input boundary. It is evidence for a
future decision, not the official metadata policy or a migration proposal.

Run the comparison
-------------------

Install the optional comparison dependencies alongside the runtime prototype:

.. code-block:: bash

   uv sync --extra schema-prototype
   uv run --extra schema-prototype pytest \
       tests/test_pydantic_handoff_records.py tests/test_model_record_schema.py \
       tests/test_msgspec_metadata_records.py tests/test_co2_schema_validation.py

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
   record = SampleManifest.model_validate_json(saved.read_bytes())
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
loaders now return typed records; writers validate before writing numerical
artifacts and publish a completed JSON manifest last. Runtime configuration
dataclasses remain ordinary dataclasses, and ``dataclasses.replace`` still works.
``DataConfig`` retains ownership of preparation option semantics.

For example, ``schema_version=true`` fails at ``schema_version`` instead of
passing Python's ``True == 1`` comparison. A ``sigma_per_site`` option on a
``fixed_error`` likelihood fails at
``model.likelihood.fixed_error.options.sigma_per_site``. Missing recorded
scientific fields are errors; they are not reconstructed from today's defaults.

The comparison deliberately contains both implementations. These physical
source-line measurements exclude tests, documentation, dependencies and the
alternative msgspec implementation, and use baseline ``f8695d52``:

.. list-table:: Pydantic integration size
   :header-rows: 1
   :widths: 60 20 20

   * - Source
     - Before
     - After
   * - Output contracts
     - 179
     - 182
   * - Standard stages
     - 387
     - 364
   * - Runtime model specifications
     - 581
     - 521
   * - New shared, manifest and model schemas
     - 0
     - 217

Thus this pilot adds 137 source lines overall in those files. It replaces
manual field checks and reconstruction with explicit schemas and better nested
errors, but does **not** demonstrate a total line-count reduction. Counting
``if``/``for``/``try``/``match`` AST statements across those same files gives
57 before and 45 after. The separate msgspec alternative is 206 lines;
it does not include a second runtime reconstruction implementation. Its direct
``Struct`` constructor is unchecked, so callers must use the supplied boundary
helpers. These measurements are not performance benchmarks.

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
* Pydantic rejects nested nonfinite JSON values. Msgspec needs the explicit finite
  boundary helper: its native encoder would otherwise turn NaN/infinity into
  null. Both reject lazy arrays in these small-record fields without executing
  their graphs. Large arrays remain NetCDF/Zarr payloads.
* Open metadata mappings have intentionally open JSON Schema values. Runtime
  finite-JSON validation still applies; an exported schema cannot describe all
  scientific relationships or replace the product's loader.

The CO2 ``xarray-validate`` pilot declares six required variables, fixed local
dimension order and floating dtypes. Scientific coordinate equality/order,
dynamic state relationships, covariance shape, numeric unit scale and
aggregation representations retain explicit owning checks. NetCDF and Zarr
tests verify round trips and loader rejection; construction/Dask tests verify
non-mutation and no payload or auxiliary-coordinate computation.
This adds a floating-dtype restriction and
changes structural error text. Its source diff is 16 additions and 17 deletions.

``xarray-validate==0.0.5`` cannot faithfully export this schema: generic dtype
constraints are silently omitted by ``serialize()``. Its built-in dimension
schemas cannot bind dynamic state dimension names across variables; arbitrary
custom checks could retain those existing scientific checks. This CO2
structural schema runs on construction/load. Saving a subsequently mutated
dataset does not rerun it, although other RHIME/aggregation checks still apply.

Assessment
-----------

Pydantic is the stronger candidate here for actionable nested errors and
removing routine handwritten validation. Msgspec remains attractive for small
manifest records, with a stricter need to funnel construction through boundary
helpers. Neither replaces scientific codecs. The xarray pilot's one-line net
reduction and incomplete schema export do not yet justify a broad migration.

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
