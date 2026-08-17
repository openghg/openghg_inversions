Customising a RHIME likelihood
==============================

Preferred: pass a Python likelihood
-----------------------------------

For an ordinary likelihood variation, keep RHIME's complete
acquisition-to-output workflow and pass one direct-Python callable to
``run_rhime``. The callable is deliberately unavailable to INI files: it is
not imported from configuration or stored in a run or model specification.

The complete integration is one named argument:

.. code-block:: python

   from my_project.likelihoods import likelihood_builder
   from openghg_inversions.rhime import run_rhime

   result = run_rhime(
       config_file="config.ini",
       likelihood_builder=likelihood_builder,
   )

A builder is called as ``likelihood_builder(context)`` while the PyMC model is
active and returns ``RhimeLikelihoodResult``. Its semantic roles drive
predictive sampling, its supported output formats are validated before
sampling, and its metadata must be JSON-compatible. The ordinary caller does
not construct the context, a specification, or a role/output manifest.

Editable likelihood
~~~~~~~~~~~~~~~~~~~

The tested project-owned example replaces RHIME's Gaussian observation
distribution with Student-t while reusing RHIME's current mean and error
construction:

.. literalinclude:: ../../examples/rhime_customisation/likelihoods.py
   :language: python
   :linenos:

The callable's safe module and qualified name, together with its
JSON-compatible likelihood metadata, are recorded in result and serialized
output provenance. Executable Python code is not serialized.

The example makes those owned invariants explicit: ``student_y`` is declared
as the ``concentration`` role, RHIME's ``epsilon`` remains the ``model_error``
role, only ``none`` and ``inv_out`` outputs are declared compatible, and the
Student-t family and degrees of freedom are recorded as JSON metadata. It
rejects dense and low-rank aggregation covariance because the example uses an
independent Student-t distribution; supporting those modes would require a
multivariate likelihood rather than a hidden approximation.

Optional project CLI
~~~~~~~~~~~~~~~~~~~~

The short wrapper below packages the same one-call integration as a reusable
Python function and command-line entry point:

.. literalinclude:: ../../examples/rhime_customisation/run_with_likelihood.py
   :language: python
   :linenos:

Run it with a normal RHIME configuration and optional JSON overrides::

   python -m package_name.run_with_likelihood config.ini \
       --kwargs '{"output_path": "outputs", "output_format": "inv_out"}'

``run_rhime_multisector`` accepts the same Python-only builder contract. The
standard multi-sector model retains sector flux components and roles, then
passes their combined observation mean to the likelihood, so no special case
or semantic compromise is required.

Advanced: copy the complete runner
----------------------------------

The copied runner below is an advanced, version-coupled escape hatch for
scientific changes beyond the likelihood seam. It makes the major stages
visible while importing their implementations from the supported public RHIME
API. Prefer ``run_rhime_from_prepared_inputs`` when replaying prepared inputs,
replacing the complete model, or deliberately starting from a different
preparation graph.

The deliberate change is the likelihood passed to
``build_standard_rhime_model``: the example selects the same project-owned
Student-t builder as the preferred form. Acquisition, filtering, basis
construction, labelled input assembly, the eager PyMC boundary, sampling,
predictive selection, filenames, and output handling remain library-owned.

In a project created with the `OpenGHG project cookiecutter
<https://github.com/openghg/openghg-project-cookiecutter>`_, copy the preferred
modules to ``src/<package_name>/likelihoods.py`` and
``src/<package_name>/run_with_likelihood.py``. Copy the complete module below
to ``src/<package_name>/rhime_runner.py`` only when the project needs to own a
deeper orchestration change. Import scientific stage implementations from
``openghg_inversions.rhime`` rather than copying those implementations.

Run it with a normal RHIME configuration and optional overrides::

   python -m package_name.rhime_runner config.ini \
       --start-date 2020-01-01 --end-date 2020-02-01 \
       --output-path outputs --draws 1000 --tune 1000 --chains 4

Less common Python/config options can be supplied as a JSON object::

   python -m package_name.rhime_runner config.ini \
       --kwargs '{"reload_merged_data": true, "output_format": "inv_out"}'

The test suite imports all three sources, exercises both runners, and validates
the likelihood contract directly, so the documentation and runnable examples
cannot drift apart.

.. literalinclude:: ../../examples/rhime_customisation/runner.py
   :language: python
   :linenos:

Advanced: compose a custom basis stage
--------------------------------------

The second complete runner replaces one call in the preparation spine:
``build_project_basis`` replaces ``build_rhime_basis``. The project function
composes public basis primitives instead of selecting a built-in basis
algorithm. It:

* derives and normalizes a two-dimensional weight field with
  ``basis_weights_from_fp_all``;
* loads the public country grid and reduces positive country codes to ``land``
  and the remaining cells to ``ocean``;
* creates physical north-south/east-west coordinates with
  ``LatLonGridGeometry``;
* uses balanced inertial splits, decomposes every proposed child into
  four-neighbour connected components, and rejects splits whose children
  exceed the configured PCA eccentricity guard;
* generates class-safe labels with ``region_constrained_basis``; and
* wraps the flat labels and current run flux in retained ``BasisFunctions``
  with ``basis_functions_from_fp_all_flat_basis``.

The nested split strategy follows the latest selected-country guarded-basis
variant in the ``verification-games`` project. That variant gives the UK,
Ireland, France, Germany, Italy, Belgium, and the Netherlands separate classes,
while remaining land and ocean form two more. This smaller example deliberately
uses the land/ocean class variant so the composition stays readable. To adopt
the selected-country policy, replace ``_land_ocean_classes`` with a project
function that maps the loaded integer country codes or country names to those
classes. Keep this classification outside OpenGHG Inversions unless it becomes
a broadly supported policy.

The weighting is deliberately not identical to that later verification-games
preparation step. Verification-games sums absolute cached ``fp_x_flux`` after
those sensitivities exist. At this earlier visible-runner basis boundary they
have not been constructed yet, so the example uses the public
``basis_weights_from_fp_all`` field while preserving the guarded split strategy
and class composition.

The flat labels and retained object record namespaced provenance for the class
policy, normalized weight source, connected balanced-inertial strategy,
connectivity, and eccentricity threshold. Those fields travel with a saved
``BasisFunctions`` artifact and make the project choice inspectable later.

Acquisition, filtering, sensitivity construction, labelled input assembly,
the standard likelihood and model, the eager PyMC boundary, sampling,
predictive selection, filenames, and output handling remain library-owned.

The project owns the scientific validity of its classification, coverage,
region count, eccentricity threshold, and split policy. ``BasisFunctions`` and
the unchanged downstream stages validate the grid, coordinates, sources, site
alignment, state layout, and model inputs they consume; they cannot certify
that the project's scientific partition is appropriate. Malformed artifacts
fail while loading; structurally valid artifacts fail later if they violate a
downstream alignment contract.

The example exposes ``project_basis_path`` separately from standard RHIME
options. Point it at a self-contained ``.nc`` or ``.zarr`` artifact written by
``BasisFunctions.save`` to bypass fitting::

   python -m package_name.custom_basis_runner config.ini \
       --project-basis-path cache/project-basis.zarr

The eccentricity guard is also a visible project-owned option, rather than an
opaque library default. It defaults to ``10`` and is removed before standard
RHIME option resolution::

   python -m package_name.custom_basis_runner config.ini \
       --max-child-pca-eccentricity 10

``BasisFunctions.load`` eagerly loads the saved operator, metadata, and flux,
then closes the artifact. The serialized flux is deliberately retained. Use
the standard public ``load_basis_functions`` helper instead when loading a
named RHIME basis cache that should take its retained flux from the current
``fp_all`` acquisition.

Without an artifact, basis generation is a named eager boundary: the public
weight adapter eagerly materializes a derived two-dimensional weight field
from borrowed inputs; normalization, geometry construction, country-class
loading, and guarded region splitting then compute eager class and label
fields. Before that boundary, the filtered xarray objects are borrowed and may
be Dask-backed. The compatibility adapter retains the current run's flux with
the generated labels at this visible runner boundary, and the resulting
``BasisFunctions`` object flows unchanged through sensitivity construction and
labelled assembly. Model arrays are materialized separately and explicitly by
``materialize_pymc_inputs``.

Copy ``examples/rhime_customisation/custom_basis_runner.py`` to
``src/<package_name>/custom_basis_runner.py`` and keep the project basis rule
beside the copied orchestration spine. This source is imported and executed by
integration tests and rendered here, so the documentation and runnable example
cannot drift apart.

This guarded composition is intentionally project-specific and is not yet a
common, stable strategy that warrants another lower-ceremony ``run_rhime``
option.

.. literalinclude:: ../../examples/rhime_customisation/custom_basis_runner.py
   :language: python
   :linenos:
