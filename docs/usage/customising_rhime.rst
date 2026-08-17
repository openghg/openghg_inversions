Customising a RHIME likelihood
==============================

Preferred: pass a Python likelihood
-----------------------------------

For an ordinary likelihood variation, keep RHIME's complete
acquisition-to-output workflow and pass one direct-Python callable to
``run_rhime``. The callable is deliberately unavailable to INI files: it is
not imported from configuration or stored in a run or model specification.

This example selects the absolute-sigma Gaussian likelihood in place of
RHIME's default pollution-event-scaled Gaussian. The likelihood is the same
one used by the complete copied runner below, so the difference between the
examples is orchestration ceremony rather than scientific behaviour.

The project-owned likelihood selection is one small module:

.. literalinclude:: ../../examples/rhime_customisation/likelihoods.py
   :language: python
   :linenos:

The preferred runner keeps the full standard pipeline in one ``run_rhime``
call:

.. literalinclude:: ../../examples/rhime_customisation/run_with_likelihood.py
   :language: python
   :linenos:

Run the short form with a normal RHIME configuration and optional JSON
overrides::

   python -m package_name.run_with_likelihood config.ini \
       --kwargs '{"output_path": "outputs", "output_format": "inv_out"}'

The callable's safe module and qualified name, together with its
JSON-compatible likelihood metadata, are recorded in result and serialized
output provenance. Executable Python code is not serialized.

A builder is called as ``likelihood_builder(context)`` while the PyMC model is
active and returns ``RhimeLikelihoodResult``. Its semantic roles drive
predictive sampling, its supported output formats are validated before
sampling, and its metadata must be JSON-compatible. The ordinary caller does
not construct the context, a specification, or a role/output manifest.

``run_rhime_multisector`` accepts the same Python-only builder contract. The
standard multi-sector model retains sector flux components and roles, then
passes their combined observation mean to the likelihood, so no special case
or semantic compromise is required.

Advanced: copy the complete runner
----------------------------------

The copied runner below is an advanced, version-coupled escape hatch for
scientific changes beyond the likelihood seam. It makes the major stages
visible while importing their implementations from the
supported public RHIME API. Prefer ``run_rhime_from_prepared_inputs`` when
replaying prepared inputs, replacing the complete model, or deliberately
starting from a different preparation graph.

The deliberate change is the likelihood passed to
``build_standard_rhime_model``: the example selects
``build_absolute_sigma_gaussian_likelihood`` in place of RHIME's default
pollution-event-scaled Gaussian. Acquisition, filtering, basis construction,
labelled input assembly, the eager PyMC boundary, sampling, predictive
selection, filenames, and output handling remain library-owned.

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

All three sources are imported and executed by the integration test, so the
documentation and runnable examples cannot drift apart.

.. literalinclude:: ../../examples/rhime_customisation/runner.py
   :language: python
   :linenos:
