Standard RHIME tutorial
=======================

This tutorial runs the supported one-sector :func:`openghg_inversions.rhime.run_rhime`
recipe, inspects its labelled result, and reloads its durable inversion-output
product. It is a basic production workflow, not a Verification Games adapter.

Prerequisites and data
----------------------

Install the package as described in :doc:`installation`, then populate the
companion OpenGHG store from the versioned `v1.0.0 data release
<https://github.com/openghg/openghg_inversions_tutorial_data/tree/v1.0.0>`_.
The repository is currently private, so the release is available only to
collaborators with repository access. Making the repository public will change
who can clone it, not the pinned release used by this tutorial:

.. code-block:: console

   $ git clone https://github.com/openghg/openghg_inversions_tutorial_data.git
   $ cd openghg_inversions_tutorial_data
   $ git checkout v1.0.0
   $ git lfs pull
   $ python scripts/populate_store.py

The population command verifies the LFS files and registers the resulting
store as ``inversions_tutorial_data``. It contains real January 2020 CH4 data
for Mace Head (``MHD``) and Tacolneston (``TAC``): observations, matching NAME
footprints, EDGAR v8 anthropogenic flux, WetCHARTs v1.3.1 wetlands flux, and
CAMS v22r2 daily boundary conditions. MHD's observation inlet is ``24m`` and
its footprint release height is ``10m``; TAC uses ``185m`` for both.

The release's `manifest.toml
<https://github.com/openghg/openghg_inversions_tutorial_data/blob/v1.0.0/manifest.toml>`_
records each file's hash, source record, transformations, and expected OpenGHG
search. `DATA_LICENSES.md
<https://github.com/openghg/openghg_inversions_tutorial_data/blob/v1.0.0/DATA_LICENSES.md>`_
records the upstream licences, attribution, and scientific citations. The
bundle's MIT licence covers only repository-authored software and
documentation; cite the upstream scientific datasets when publishing results.

The configured quick run covers the first week, from ``2020-01-01`` inclusive
to ``2020-01-08`` exclusive, and averages both sites to four hours. Change only
``end_date`` to ``2020-02-01`` to use the supplied full month. The ``NESW``
boundary-condition basis is constructed by OpenGHG Inversions and is not part
of the companion data bundle. The boundary term is important for an absolute
atmospheric CH4 inversion: without it, the emissions contribution would be
asked to explain the background concentration as well as enhancements.

Configuration
-------------

The packaged example is a complete production-shape configuration validated by
the test suite. It is runnable after populating the named companion store:

.. literalinclude:: ../../openghg_inversions/rhime/config/standard_tutorial.ini
   :language: ini

``quadtree`` constructs four flux-scaling regions from the retrieved footprint
and prior flux. The 50 tuning and 50 retained draws are only an end-to-end
smoke test. A scientific inversion needs enough chains and draws for stable
diagnostics and must not disable convergence checks.

Run it
------

From a source checkout, the supported CLI route is:

.. code-block:: console

   $ pixi run -e dev openghg-inversions run-rhime \
       --config openghg_inversions/rhime/config/standard_tutorial.ini \
       --output-path outputs

The :doc:`installation` page documents the equivalent ``uv run`` command for
an existing compatible uv environment.

The equivalent supported Python entry point is:

.. code-block:: python

   from importlib.resources import as_file, files
   from pathlib import Path

   from openghg_inversions.rhime import run_rhime

   resource = files("openghg_inversions.rhime").joinpath("config/standard_tutorial.ini")
   with as_file(resource) as config:
       result = run_rhime(config_file=config, output_path="outputs")

Configuration is resolved once at the runner boundary. Explicit Python or CLI
values such as ``output_path`` override the INI file.

Inspect and interpret the result
--------------------------------

``RhimeResult`` keeps the resolved scientific and output specifications,
canonical labelled arrays, the ArviZ trace, and generated products together:

.. code-block:: python

   print(result.run_spec)
   print(result.model_spec.sectors)
   print(result.inv_inputs[["mf", "mf_error", "H"]])
   print(result.inv_inputs["H"].dims)
   print(result.inv_inputs.indexes["nmeasure"])
   print(result.idata.posterior["x"].mean(("chain", "draw")))
   print(result.model_build_result.variable_roles)
   print(result.output_metadata)

``mf(nmeasure)`` is the observed mole fraction and ``mf_error(nmeasure)`` is
its supplied observation uncertainty. ``H(region, nmeasure)`` is the labelled
sensitivity of each observation to each basis region. ``x(region)`` is the
posterior multiplicative scaling of the prior flux: values above one increase
that region's flux relative to the prior and values below one decrease it.
Inspect the ``site`` and ``time`` levels of ``nmeasure`` rather than relying on
array position. The complete role and model-variable contract is described in
:doc:`concrete_rhime_model`.

Inspect diagnostics before interpreting any posterior quantity:

.. code-block:: python

   import arviz as az

   print(az.summary(result.idata, var_names=["x", "bc", "sigma"])))
   if "sample_stats" in result.idata.groups():
       print("divergences:", int(result.idata.sample_stats["diverging"].sum()))

The 50-draw tutorial result is too short for scientific interpretation.
Increase chains, tuning, and retained draws; require acceptable R-hat,
effective sample sizes, trace behaviour, divergences, and posterior-predictive
fit before drawing conclusions.

The configured ``inv_out`` product is available in memory and on disk:

.. code-block:: python

   from openghg_inversions.postprocessing.inversion_output import InversionOutput

   inv_out = result.outputs["inversion_output"]
   print(inv_out.provenance)
   print(inv_out.run_metadata)
   saved = Path(result.output_metadata["inversion_output_path"])
   reloaded = InversionOutput.load(saved)
   print(reloaded.trace.posterior["x"])

The file retains the trace, canonical inputs, basis functions, run/model/output
metadata, and provenance needed by supported postprocessing. It is not a
legacy fixedbasis NetCDF file. ``provenance["contract"]`` identifies this
modern product, while ``run_metadata["basis_artifact_source"]`` records how the
basis entered preparation and ``output_metadata["sampler"]`` records the
resolved sampling options. It does not snapshot the OpenGHG object store or
guarantee that the original acquisition query can be replayed unchanged.
The companion-data tag is likewise not added to this output automatically:
record ``v1.0.0`` with the run so the file-level manifest can be matched to the
inversion.

Controlled clean-checkout task test
-----------------------------------

Repository contributors can exercise the whole documented preparation, model,
result, and persistence route without downloading the Git LFS bundle:

.. code-block:: console

   $ uv run pytest tests/test_rhime_tutorials.py -q

The test standardises maintained ``tests/data`` files into a temporary
``inversions_tests`` store and overrides store/output paths. It uses a
deterministic one-draw sampler so it validates mechanics and labels, not NUTS
quality or scientific conclusions. Running the packaged config normally uses
the configured real PyMC sampler.

Common failures
---------------

``Search found no data`` or a missing-input error
   Rerun ``python scripts/populate_store.py --verify-only`` in the companion
   checkout, then check the configured store name and exact time interval.

Coordinate or dimension mismatch
   Observations and footprints must describe the same site/time sampling, and
   flux and footprint grids must align on the configured domain. Do not strip
   labels or reorder NumPy arrays to silence this error.

Unsupported or misspelled option
   RHIME rejects unknown configuration keys. Start from the packaged file and
   use the snake-case names in :doc:`rhime`.

Missing output path or unexpected product
   ``output_format = "inv_out"`` with ``save_inversion_output = True`` needs
   ``output_path``. Other derived products have additional input requirements;
   select them only after consulting the output guidance in :doc:`rhime`.

Poor convergence or divergences
   Do not interpret the posterior. Check prior scales, model-data mismatch,
   influential observations and identifiability, then increase tuning/draws
   and rerun. More samples do not repair a scientifically misspecified model.
