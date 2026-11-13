Pydantic BC configuration prototype
===================================

This experiment adds validation, portable JSON and schema generation to the
nested ``BcBasisOptions`` dataclass. The same class remains the value used by
``DataConfig``, the INI option-group adapter and scientific preparation.
It explores a small configuration boundary; it introduces no checkpoint format
or change to the BC basis calculation.

Construct, export and read options
----------------------------------

.. code-block:: python

   from pathlib import Path
   from openghg_inversions.rhime.params import BcBasisOptions

   options = BcBasisOptions(algorithm="twelve-region", split_height=2500.0)
   encoded = options.to_json()
   restored = BcBasisOptions.from_json(encoded, active=True)
   assert restored == options

   selected_file = BcBasisOptions(paths=(Path("/data/bc.nc"),))
   assert isinstance(selected_file.to_params()["paths"][0], Path)
   # JSON uses strings for paths; reading it does not open the files.
   assert '"/data/bc.nc"' in selected_file.to_json()

Pydantic validates direct construction and ``dataclasses.replace``. Unknown
fields and incorrect field types report structured locations such as
``("split_height",)``. Numeric strings, booleans, arrays and arbitrary scalar
conversion objects are rejected as heights. Known eager NumPy numeric scalars
are unboxed; validation never computes Dask values. Direct construction expects
resolved ``Path`` objects and path tuples. The existing frontend still resolves
external strings relative to the invocation directory.

The dataclass remains frozen. Its field declarations drive constructor checks,
Python projection, JSON projection and schemas. ``to_params()`` retains Python
paths and tuples; ``to_json()`` produces portable strings and arrays. JSON
reading permits omitted fields to use their configuration defaults. This is a
configuration value, rather than a versioned scientific artifact.

Keep applicability with the consuming phase
-------------------------------------------

Field types are checked at construction, including for inactive selections.
This is deliberately stricter than the earlier unchecked direct constructor.
Scientific applicability still uses the explicit ``active`` argument:

.. code-block:: python

   inactive = BcBasisOptions(algorithm="not-selected")
   assert BcBasisOptions.from_params(inactive, active=False) is inactive
   # Selecting files also leaves algorithm and split-height applicability inactive.
   files = BcBasisOptions(paths=(Path("/missing/bc.nc"),), algorithm="not-selected")
   BcBasisOptions.from_json(files.to_json(), active=True)

An active built-in algorithm is checked by ``from_params`` or ``from_json``.
The twelve-region builder continues to own its finite split-height rule.
Inactive nonfinite numeric heights can remain in runtime options, but portable
JSON export and loading reject them. They are never silently replaced by null.
The constructor does not read files, and an absent selected file is reported
only by the existing resource-loading boundary.

INI flat spelling, deprecation warnings for flat Python/JSON requests, nested
overrides, resource resolution and preparation manifest versions retain their
existing owners. See :doc:`/usage/rhime_configuration` for those interfaces.

Inspect generated schemas
-------------------------

``BcBasisOptions.json_schema()`` describes the nested ``bc_basis`` value.
Unknown fields are forbidden. Algorithm names remain strings because an
inactive name need not be a supported built-in algorithm; the explicit active
check establishes that stronger requirement. Schema validation alone therefore
does not replace phase resolution.

Run the executable example:

.. code-block:: console

   python -m examples.pydantic_bc_options /tmp/pydantic-bc-options

It writes ``bc-basis.json`` and ``bc-basis.schema.json`` and demonstrates a
constructor error. The schema describes the nested BC value, rather than an
entire request or an INI file.

This prototype covers BC options only. A complete ``DataConfig`` schema also
needs an explicit representation of the Python slices accepted by
``SiteOptions``. Other configuration classes, opaque runtime options and
numerical serialization retain their existing behavior.
