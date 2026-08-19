Customising RHIME
=================

RHIME is built from reusable functions for data preparation, basis
construction, model construction, sampling, and output. This is a deliberate
design choice: when the standard workflow does not express the science you
need, you should be able to start from a working example and replace the
smallest relevant part.

The examples on this page use a procedural style. Data is passed from one
function to the next in the order that the inversion runs, so readers do not
need to design a new class hierarchy or framework before changing a model.
Some library functions return structured objects that group related values,
but the examples show how to use those objects directly.

Reusing the components does not make every custom workflow automatic. A
replacement function is responsible for the scientific choices it introduces,
and a deeper change may also need different data preparation. The remaining
RHIME stages can only validate the inputs and assumptions covered by their
documented interfaces.

Choose the smallest starting point that fits the change:

* To change only the likelihood, pass a Python function to ``run_rhime``.
* To resume from externally supplied merged observations, footprints, and
  fluxes, pass a borrowed ``RhimeMergedData`` object as ``merged_data``.
* To change a preparation stage such as basis construction, copy the visible
  runner and replace that stage.
* To start from prepared data or replace the complete model, use
  ``run_rhime_from_prepared_inputs``.

Resume from cached or external scientific data
----------------------------------------------

``run_rhime`` and ``run_rhime_multisector`` accept ``merged_data`` as a
Python-only handoff.  It bypasses OpenGHG acquisition and merged-cache I/O,
checks the single- or multi-sector layout, and then re-enters the visible
recipe at filtering::

   result = run_rhime(
       config_file="config.ini",
       merged_data=my_merged_data,
   )

The supplied ``RhimeMergedData`` and its xarray or Dask arrays remain borrowed.
Filtering returns a replacement handoff when it changes observations; basis
construction and labelled assembly consume that result without mutating the
external object.  A normal ``reload_merged_data`` request instead belongs to
the same retrieval stage and may read a configured artifact from disk.

Change the likelihood with a Python function
--------------------------------------------

For an ordinary likelihood variation, keep RHIME's complete
acquisition-to-output workflow and pass one Python function to ``run_rhime``.
The function is deliberately unavailable to INI files: it is
not imported from configuration or stored in a run or model specification.

The complete integration is one named argument:

.. code-block:: python

   from my_project.likelihoods import likelihood_builder
   from openghg_inversions.rhime import run_rhime

   result = run_rhime(
       config_file="config.ini",
       likelihood_builder=likelihood_builder,
   )

RHIME calls the function with explicit keyword arguments while constructing
the PyMC model: the prepared observations, completed forward-model mean,
pollution contribution, pollution-event baseline, sigma alignment and prior, and
the selected error policies. The function adds ``epsilon`` and the canonical
observed variable ``y`` to the active model and returns ``y``. There is no
framework context or likelihood-result record to construct.

Editable likelihood
~~~~~~~~~~~~~~~~~~~

The following example replaces RHIME's Gaussian observation distribution with
a Student-t distribution while reusing RHIME's current mean and error
construction:

.. literalinclude:: ../../examples/rhime_customisation/likelihoods.py
   :language: python
   :linenos:

RHIME records the function's module and name in the result and in saved output.
It does not copy the Python source code into the output, so a project should
keep the source and environment used for an inversion. Ordinary likelihood
builders retain the canonical ``y`` and ``epsilon`` names used by sampling and
output code.

The example rejects dense and low-rank aggregation covariance because it uses
an independent Student-t distribution. Supporting those aggregation-error
modes would require a multivariate likelihood.

The same example module also contains
``additive_sigma_likelihood_builder``. This is the small RHIME adapter from
``rhime.likelihoods``; it delegates to the installed
``models.additive_sigma.add_additive_sigma_gaussian_likelihood`` component,
which adds ``sigma**2`` directly to the reported observation-error variance
rather than multiplying sigma by a pollution event. Explicitly selected
aggregation covariance is supported::

   from my_project.likelihoods import additive_sigma_likelihood_builder

   result = run_rhime(
       config_file="config.ini",
       likelihood_builder=additive_sigma_likelihood_builder,
   )

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

Use the seam from a generated project
-------------------------------------

Create a normal downstream package with the current `OpenGHG project
cookiecutter <https://github.com/openghg/openghg-project-cookiecutter>`_::

   uvx cookiecutter gh:openghg/openghg-project-cookiecutter

The template already declares ``openghg`` and ``openghg_inversions`` as
dependencies and creates a ``src`` layout. Add the project-owned files beside
the generated package's existing modules, without copying any OpenGHG
Inversions implementation::

   src/my_inversion/
     __init__.py
     likelihoods.py
     runner.py
   tests/
     test_runner.py

Put only the scientific change in ``likelihoods.py``:

.. literalinclude:: ../../examples/rhime_cookiecutter/my_inversion/likelihoods.py
   :language: python
   :linenos:

Keep project-level invocation in ``runner.py``. Its one library call owns no
retrieval, filtering, basis, input assembly, sampling, predictive selection,
or output implementation:

.. literalinclude:: ../../examples/rhime_cookiecutter/my_inversion/runner.py
   :language: python
   :linenos:

After ``uv sync --extra dev``, run the module with a normal RHIME INI file::

   uv run python -m my_inversion.runner inversion.ini \
       --kwargs '{"output_path": "outputs", "output_format": "inv_out"}'

An optional console command can point directly at the same ``main`` function.
Add this table to the generated ``pyproject.toml``::

   [project.scripts]
   my-inversion = "my_inversion.runner:main"

Then the equivalent command is::

   uv run my-inversion inversion.ini \
       --kwargs '{"output_path": "outputs", "output_format": "inv_out"}'

The generated runner uses documented names from ``openghg_inversions.rhime``.
Its likelihood module imports reusable components from their documented owner
modules: ``models.pollution_event``, ``observation_error``, and ``sigma``.
The dependency direction is therefore the generated project to OpenGHG
Inversions; OpenGHG Inversions does not import the consumer package. Pin the
release or commit used for a scientific run in the downstream project's
lockfile. An optional RHIME recipe or generated-project CI in the generic
cookiecutter would be a separate cross-repository change and is not required
for this workflow.

Copy the complete runner
------------------------

The copied runner below is an advanced, version-coupled escape hatch for
scientific changes beyond the likelihood seam. It makes the major stages
visible while importing their implementations from the supported public RHIME
API. Prefer ``run_rhime_from_prepared_inputs`` when replaying prepared inputs,
replacing the complete model, or deliberately starting from a different
preparation graph.

The deliberate change is the likelihood passed to
``build_standard_rhime_model_result``: the example selects the same project-owned
Student-t builder as the preferred form. Acquisition, filtering, basis
construction, labelled input assembly, conversion of delayed arrays for PyMC,
sampling, predictive selection, filenames, and output handling remain
library-owned.

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

Compose a custom basis stage
----------------------------

The second complete runner replaces one call in the preparation spine:
``build_project_basis`` replaces ``build_rhime_basis``. The project function
composes public basis primitives instead of selecting a built-in basis
algorithm. It:

* derives and normalises a two-dimensional weight field with
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
policy, normalised weight source, connected balanced-inertial strategy,
connectivity, and eccentricity threshold. Those fields travel with a saved
``BasisFunctions`` artifact and make the project choice inspectable later.

Acquisition, filtering, sensitivity construction, labelled input assembly,
the standard likelihood and model, conversion of delayed arrays for PyMC,
sampling, predictive selection, filenames, and output handling remain
library-owned.

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

``BasisFunctions.load`` loads the saved operator, metadata, and flux into
memory, then closes the artifact. The saved flux is deliberately retained. Use
the standard public ``load_basis_functions`` helper instead when loading a
named RHIME basis cache that should take its retained flux from the current
``fp_all`` acquisition.

Without an artifact, the basis-building function computes the arrays needed by
the splitting algorithm. Before that point, the filtered xarray objects may
still defer their calculations with Dask. The function derives and normalises
the two-dimensional weights, loads the country classes, constructs the
geometry, and creates the region labels. It then combines the labels with the
current run's flux in a ``BasisFunctions`` object. That object flows unchanged
through sensitivity construction and labelled assembly. The arrays needed by
PyMC are converted separately by ``materialize_pymc_inputs`` immediately
before model construction.

The customisation is concentrated in ``_guarded_basis``. The runner's one
deliberate substitution is marked by an inline comment where
``build_project_basis`` replaces the standard ``build_rhime_basis`` call. The
complete source remains below because the test suite executes the same file
that the documentation displays.

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
