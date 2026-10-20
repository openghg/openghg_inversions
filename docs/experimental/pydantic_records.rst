Pydantic configuration and checkpoint prototype
================================================

This prototype explores validation and generated JSON Schemas on the current
standard, multisector and ordinary CO2 workflows. The scientific runners still
read from top to bottom: resolve choices, prepare, materialize selected arrays,
build, sample, reconstruct and write. It changes metadata ownership and errors;
it does not change the model equations. The APIs on this page are experimental.

Run the example
---------------

From an installed prototype checkout, run:

.. code-block:: console

   python -m examples.pydantic_records /tmp/pydantic-records

The example writes component and checkpoint schemas, a resolved data snapshot
and an output contract. It roundtrips a path and an inlet slice without opening
scientific resources, acquiring observations, constructing a graph or sampling.
The schema files can be inspected by editors and external JSON Schema tools.
For real numerical checkpoint workflows, use :doc:`/usage/staged_workflow`.

Construct and inspect configuration
-----------------------------------

``RhimeConfig``, ``DataConfig``, ``RhimeModelSpec``, ``SectorSpec``, likelihood
settings, ``RhimeOutputSpec`` and ``SiteOptions`` remain dataclasses. Pydantic
validates their constructors, including reconstruction by ``dataclasses.replace``.
For example:

.. code-block:: python

   from pydantic import ValidationError
   from openghg_inversions.rhime.specs import PollutionEventSettings

   try:
       PollutionEventSettings(sigma_per_site="yes")
   except ValidationError as error:
       assert error.errors()[0]["loc"] == ("sigma_per_site",)

Unknown constructor options and invalid scalar types fail with field paths.
Ordinary prior and keyword containers are owned; numerical leaves remain
borrowed. A prior containing a lazy DataArray is retained without executing it.
A lazy value supplied as a Boolean, numeric power or sampler draw count is
rejected without converting it to a scalar.

Direct constructors validate supplied values and their declared defaults.
``RhimeConfig.from_params`` and the component resolvers still resolve request
choices: for example, an omitted likelihood period anchor uses the winning
requested start date, and enabled components receive their owning prior
defaults. File frontends decode syntax and apply overrides before these same
resolvers. See :doc:`/usage/rhime_configuration` for the configuration groups.

One output contract
--------------------

``OutputContract`` is one Pydantic model owning validation, serialization,
schema generation and scientific checks. ``OutputContract.create(...)`` supplies
defaults for a new runtime value. Reading a saved contract requires every
declared field; the loader never fills missing scientific state with defaults.

Its variable roles map scientific meanings to concrete array names. Supported
formats declare which products a builder can supply, including ``none``. The
optional state-dimension mapping identifies the posterior and basis axes used
by a selected scientific view. The contract checks required reconstruction
states without computing posterior payloads.

``with_updates(...)`` constructs a validated view. ``to_dict()`` validates current
Python containers before JSON conversion, so mutation cannot silently turn
distinct integer and string keys into the same JSON key. JSON metadata permits
finite scalars and ordinary containers; arrays and callables do not belong there.

Saved meaning and current defaults
----------------------------------

Standard and multisector manifests contain typed model descriptions with tagged
likelihood options. All saved scientific choices are required. A null prior or
power description records a prior that construction did not use; it never asks
for today's default. Registered array references and named or symbolic
dependencies remain descriptions. Saved output readers validate them and bind
the preparation facts directly to ``InversionOutput`` without reconstructing
executable model settings, a model graph or a live sampler.

Native sources and sector identities stay distinct. Multisector descriptions
retain their explicit source routing, while the posterior retains one labelled
flux state. Array loaders, coordinate registration and product reconstruction
continue to own scientific dimensions, source labels, units and alignment.

New checkpoint writers require finite sampler and provenance mappings. Historical
standard and multisector sampler descriptions remain unavailable when malformed,
and partial descriptions stay partial. Their recorded provenance must be a valid
mapping. Ordinary CO2 readers preserve their deliberately forgiving policy for
both descriptive mappings: missing or malformed descriptions remain unavailable.
Neither reader constructs a sampler or substitutes current inference defaults.
The cached CO2 stage/hash format is outside this prototype's manifest changes.

Schemas and operational guards
-------------------------------

Generated schemas express field types, required fields, tagged choices and
unknown-field policy. They do not establish cross-artifact compatibility or
scientific numerical invariants. The preparation generation check, target path
and alias checks, old completion-marker removal and publication of the new
manifest after successful payload writes keep their existing procedural owners.

Metadata projections explicitly encode supported paths, dates and inlet slices.
Raw mapping keys are checked before such conversion. Numerical materialization
remains at the named scientific or serialization boundary; Pydantic validators
do not copy or compute borrowed array payloads. This experiment does not define
a universal JSON or backend-attribute policy for every scientific artifact.
