Standard RHIME tutorial
=======================

This tutorial runs the supported one-sector :func:`openghg_inversions.rhime.run_rhime`
recipe, inspects its labelled result, and reloads its durable inversion-output
product.

The code and recorded outputs form one stateful session. You can
:jupyter-download-notebook:`download it as a Jupyter notebook <rhime_standard_tutorial>`
to rerun or modify locally.

.. jupyter-kernel:: python3
   :id: rhime_standard_tutorial

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
of the companion data bundle.

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

.. jupyter-input::

   from importlib.resources import as_file, files
   import os
   from pathlib import Path

   from openghg_inversions.rhime import run_rhime

   tutorial_output_path = Path(os.environ.get("OPENGHG_TUTORIAL_OUTPUT_PATH", "outputs"))
   resource = files("openghg_inversions.rhime").joinpath("config/standard_tutorial.ini")
   with as_file(resource) as config:
       result = run_rhime(config_file=config, output_path=tutorial_output_path)

   {
       "OpenGHG Inversions commit": os.environ.get(
           "OPENGHG_TUTORIAL_CODE_REF", "local checkout"
       ),
       "tutorial data": os.environ.get("OPENGHG_TUTORIAL_DATA_TAG", "v1.0.0"),
       "sites": list(result.run_spec.sites),
       "observations": result.inv_inputs.sizes["nmeasure"],
       "posterior samples": {
           name: result.idata.posterior.sizes[name] for name in ("chain", "draw")
       },
   }

.. jupyter-output::

   Run ``docs-tutorials-record`` to refresh this output.

Configuration is resolved once at the runner boundary. Explicit Python or CLI
values such as ``output_path`` override the INI file.

Inspect and interpret the result
--------------------------------

``RhimeResult`` keeps the resolved scientific and output specifications,
canonical labelled arrays, the ArviZ trace, and generated products together:

.. jupyter-input::

   measurement_index = result.inv_inputs.indexes["nmeasure"]
   {
       "period": (result.run_spec.start_date, result.run_spec.end_date),
       "sectors": [
           (sector.name, sector.flux_source) for sector in result.model_spec.sectors
       ],
       "H dimensions": result.inv_inputs["H"].dims,
       "input sizes": dict(result.inv_inputs.sizes),
       "measurement sites": measurement_index.get_level_values("site").unique().tolist(),
       "x dimensions": result.idata.posterior["x"].dims,
       "variable roles": result.model_build_result.variable_roles,
       "output products": sorted(result.outputs),
   }

.. jupyter-output::

   Run ``docs-tutorials-record`` to refresh this output.

``mf(nmeasure)`` is the observed mole fraction and ``mf_error(nmeasure)`` is
its supplied observation uncertainty. ``H(region, nmeasure)`` is the labelled
sensitivity of each observation to each basis region. ``x(region)`` is the
posterior multiplicative scaling of the prior flux: values above one increase
that region's flux relative to the prior and values below one decrease it.
Inspect the ``site`` and ``time`` levels of ``nmeasure`` rather than relying on
array position. The complete role and model-variable contract is described in
:doc:`concrete_rhime_model`.

Inspect diagnostics before interpreting any posterior quantity:

.. jupyter-input::

   import arviz as az

   summary = az.summary(
       result.idata,
       var_names=["x", "bc", "sigma"],
       kind="diagnostics",
   )
   summary["divergences"] = int(result.idata.sample_stats["diverging"].sum())
   summary.round(2)

.. jupyter-output::

   Run ``docs-tutorials-record`` to refresh this output.

The 50-draw tutorial result is too short for scientific interpretation.
Increase chains, tuning, and retained draws; require acceptable R-hat,
effective sample sizes, trace behaviour, divergences, and posterior-predictive
fit before drawing conclusions.

The configured ``inv_out`` product is available in memory and on disk:

.. jupyter-input::

   from openghg_inversions.postprocessing.inversion_output import InversionOutput

   inv_out = result.outputs["inversion_output"]
   saved = Path(result.output_metadata["inversion_output_path"])
   reloaded = InversionOutput.load(saved)
   {
       "provenance contract": reloaded.provenance["contract"],
       "basis artifact source": reloaded.run_metadata["basis_artifact_source"],
       "split by sectors": reloaded.run_metadata["split_by_sectors"],
       "saved file": saved.name,
       "posterior variables": sorted(reloaded.trace.posterior.data_vars),
       "posterior sizes": dict(reloaded.trace.posterior.sizes),
       "sampler fields": sorted(reloaded.output_metadata["sampler"]),
   }

.. jupyter-output::

   Run ``docs-tutorials-record`` to refresh this output.

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

Refreshing the recorded outputs
-------------------------------

Maintainers intentionally refresh the committed ``jupyter-output`` blocks
after changing either tutorial, its dependencies, or the companion data. From
a clean source checkout, run:

.. code-block:: console

   $ pixi run -e dev docs-tutorials-record

This opt-in command clones the pinned ``v1.0.0`` companion release under the
ignored ``build`` directory, verifies its Git LFS files against the manifest,
populates the named OpenGHG store, executes both downloadable notebooks, and
updates only their paired output blocks. It records the current clean
OpenGHG Inversions commit and the data tag, then rebuilds the rendered pages.
Review and commit the resulting RST changes. Ordinary previews, documentation
CI, and ``tox -e docs`` never acquire data or execute these tutorial inputs;
they render the committed outputs offline.

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
