Customising the complete RHIME workflow
=======================================

The copied runner below is an advanced, version-coupled escape hatch for
scientific changes that do not fit an ordinary ``run_rhime`` option. It makes
the major stages visible while importing their implementations from the
supported public RHIME API. Prefer the standard runner when configuration is
enough, and prefer ``run_rhime_from_prepared_inputs`` when replaying prepared
inputs or deliberately starting from a different preparation graph.

The deliberate change is the likelihood passed to
``build_standard_rhime_model``: the example selects
``build_absolute_sigma_gaussian_likelihood`` in place of RHIME's default
pollution-event-scaled Gaussian. Acquisition, filtering, basis construction,
labelled input assembly, the eager PyMC boundary, sampling, predictive
selection, filenames, and output handling remain library-owned.

In a project created with the `OpenGHG project cookiecutter
<https://github.com/openghg/openghg-project-cookiecutter>`_, copy this exact
module to ``src/<package_name>/rhime_runner.py``. Keep the short orchestration
spine in the project and import the scientific stage implementations from
``openghg_inversions.rhime``; do not copy those implementations into the
project.

Run it with a normal RHIME configuration and optional overrides::

   python -m package_name.rhime_runner config.ini \
       --start-date 2020-01-01 --end-date 2020-02-01 \
       --output-path outputs --draws 1000 --tune 1000 --chains 4

Less common Python/config options can be supplied as a JSON object::

   python -m package_name.rhime_runner config.ini \
       --kwargs '{"reload_merged_data": true, "output_format": "inv_out"}'

The source below is also imported and executed by the integration test, so the
documentation and runnable example cannot drift apart.

.. literalinclude:: ../../examples/rhime_customisation/runner.py
   :language: python
   :linenos:
