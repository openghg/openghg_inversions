Running RHIME from the command line
===================================

The ``rhime`` command is the recommended entry point for running
RHIME from a terminal or a batch scheduler. It is installed with
``openghg_inversions``, so a run does not need to know where the package source
code is located.

Standard, multisector, and nested runs
--------------------------------------

Use ``rhime run`` for a standard inversion (the default). Select
``--model multisector`` for shared-basis multisector inversion or
``--model nested`` to combine independent outer and inner spatial grids:

.. code-block:: console

   $ rhime run 2019-01-01 2019-02-01 \
       --config /path/to/rhime.ini \
       --output-path /path/to/outputs

   $ rhime run --model multisector 2019-01-01 2019-02-01 \
       --config /path/to/rhime_multisector.ini \
       --output-path /path/to/outputs

   $ rhime run --model nested 2019-01-01 2019-02-01 \
       --config /path/to/rhime_nested.ini

``--config`` (or ``-c``) is required. The start and end dates are optional
positional arguments; when supplied, they override ``start_date`` and
``end_date`` in the INI file. Likewise, ``--output-path`` overrides the
configured output directory. Other RHIME keyword arguments can be overridden
with a JSON object passed to ``--kwargs``:

.. code-block:: console

   $ rhime run -c rhime.ini \
       --kwargs '{"draws": 2000, "tune": 1000, "chains": 4}'

Keep the JSON in single quotes so the shell passes it as one argument. Run
``rhime run --help`` or
``rhime run --model multisector --help`` for the complete ordinary
command syntax. See the :doc:`nested-domain model family
<nested_domain_model_family>` for its support boundary and configuration
guide. New configuration files should use the RHIME vocabulary documented in
:doc:`rhime`; the packaged starting point is
``openghg_inversions/config/templates/rhime_template.ini``. Complete,
validated production-shape examples are used by
:doc:`rhime_standard_tutorial` and :doc:`rhime_multisector_tutorial`.

CO₂ runs
------------------

Use ``rhime run --model co2`` with a CO₂ TOML configuration and prepared
scientific inputs:

.. code-block:: console

   $ rhime run --model co2 --config co2.toml --output-path run/full

The configured model, sampler and output choices govern execution and requested
products. Original requested dates and domain must be supplied explicitly or
retained in preparation provenance. See :doc:`co2_model_family` for its scientific prerequisites.

Saved standard and multisector phases
--------------------------------------

``rhime prepare`` resolves preparation choices and prints one manifest path.
``rhime sample --preparation-manifest ...`` accepts current model/sampler
choices; ``rhime postprocess --sample-manifest ...`` accepts current output
choices. Sampling and postprocessing do not require a full configuration file.
Select ``--model multisector`` for the same one-manifest command forms with
sector-labelled inputs. See :doc:`staged_workflow` for the complete
preparation/two-model/two-output sequence and the separate CO2 interface.

.. _rhime-cli-command-retirement:

Removed CLI commands
--------------------

Modern development uses the following finite command-name migration. The old
executable and full-run aliases are unavailable. Update existing scripts to the
canonical routes.

.. list-table:: Accepted command replacements
   :header-rows: 1

   * - Retire
     - Canonical replacement
   * - ``openghg-inversions`` executable
     - The same subcommands under ``rhime``
   * - ``run-rhime`` under either executable
     - ``rhime run --model standard`` or ``rhime run``
   * - ``run-rhime-multisector`` under either executable
     - ``rhime run --model multisector``
   * - ``run-rhime-nested`` under either executable
     - ``rhime run --model nested``
   * - ``run-co2`` under either executable
     - ``rhime run --model co2``

The package has one installed executable, ``rhime``. Positional dates,
``--kwargs``, effective configuration, date/output overrides, scientific
execution, outputs and artifact contracts retain their meanings. No replacement
argument syntax is agreed: positional dates and ``--kwargs`` remain open for
discussion and possible revision in a separate argument-design change. Their
current preservation applies to this command-name migration, not a final syntax
decision. The Python package, distribution and installation
names do not change. See :doc:`legacy_and_migration` for the evidence gate;
broader Python symbols/imports require their own explicit replacement maps.

Python model selection and compatibility
----------------------------------------

The public Python selector is :func:`openghg_inversions.rhime.run`:

.. code-block:: python

   from openghg_inversions.rhime import run

   result = run(config_file="rhime.ini")  # model="standard" by default
   result = run(model="multisector", config_file="rhime_multisector.ini")
   result = run(model="co2", config_file="co2.toml")

Python ``run(model="co2")`` and CLI ``rhime run --model co2`` both use
:func:`openghg_inversions.rhime.co2.run_co2`, consuming a configured product
request and returning ``Co2Output`` in Python. CO₂ uses prepared
scientific inputs and explicit requested dates/domain; it has no configured
OpenGHG acquisition or native-reduction preparation route. The
``co2_o2`` and ``co2_o2_cached_sigma`` Python choices retain
their existing ``prepared_inputs`` interfaces and trace returns. Linked recipes are not choices for the configuration-driven ``rhime run`` command.
See :doc:`co2_model_family` for the TOML configuration and handoff commands.

The finite CLI names listed above are absent from the interface under the
:doc:`agreed development boundary <legacy_and_migration>`. The separately
approved five-name Python map below defines the required alias removals;
broader Python naming/types/module moves need separately agreed maps.

.. _rhime-python-alias-retirement:

Removed Python aliases
~~~~~~~~~~~~~~~~~~~~~~

Use the canonical names in new Python calls and imports. These five aliases
are unavailable, including naming-only forwarding aliases. Their canonical
replacements remain supported.

.. list-table:: Accepted Python alias replacements
   :header-rows: 1

   * - Retire
     - Canonical name
   * - ``run_rhime_multisector``
     - ``run_multisector``
   * - ``run_rhime_nested``
     - ``run_nested``
   * - ``RhimeMergedData``
     - ``MergedData``
   * - ``RhimePreparedInputs``
     - ``PreparedInputs``
   * - ``assemble_rhime_inputs``
     - ``assemble_inputs``

The canonical callable/class objects, constructors, argument/default meanings,
result contents, scientific behavior and borrowed/lazy execution guarantees are
preserved. Import-name retirement must not silently reinterpret persisted
scientific contents or incidentally rename scientific variables and coordinates.
Supported artifact formats/versions and explicit conversion or rejection of
unsupported inputs are specified independently; this naming change promises no
automatic compatibility with 0.7 artifacts. Broader public names remain outside
this finite map. The separate existing ``run_rhime``, ``RhimeRunSpec`` and ``RhimeResult``
migrations retain their own contracts; there is no blanket result-wrapper exemption.

The obsolete ``with_prepared_sites`` and ``with_prepared_rhime_sites`` run-spec
helpers are removed; use the retained sites on the prepared handoff directly.

Package naming, the recipe-module relocation proposed in #764/#776 and complete
workflow parity remain separate changes.

Translating the older batch example
-----------------------------------

The older documentation launched an internal Python file directly:

.. code-block:: bash

   INI_FILE=/user/home/example/my_inversions/my_hbmcmc_inputs.ini
   python /user/home/example/openghg_inversions/openghg_inversions/hbmcmc/run_hbmcmc.py -c "$INI_FILE"

With a modern RHIME config, replace those two lines with the installed CLI.
The following updated version uses the repository's Pixi environment. Pixi is
recommended for inversion jobs that read NetCDF/HDF5 data because the workspace
keeps the compiled HDF5 and NetCDF stack together on conda-forge; see
:doc:`installation` for the package constraints and smoke check.

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=my_inv
   #SBATCH --output=openghg_inversions.out
   #SBATCH --error=openghg_inversions.err
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=4
   #SBATCH --time=04:00:00
   #SBATCH --mem=30gb
   #SBATCH --account=dept123456

   module --force purge
   module load git/2.45.1

   REPOSITORY=/user/home/example/openghg_inversions
   cd "$REPOSITORY"

   INI_FILE=/user/home/example/my_inversions/rhime.ini
   OUTPUT_DIR=/user/home/example/my_inversions/outputs

   pixi run --locked -e dev rhime run \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

Submit the saved script in the same way as before, for example
``sbatch my_inversion_script.sh``. ``pixi run --locked`` checks that
``pixi.lock`` agrees with the workspace and installs the selected environment
when necessary. Install Pixi and create the environment on the login node
before the first submission if compute nodes do not have network access:

.. code-block:: console

   $ cd /user/home/example/openghg_inversions
   $ pixi install --locked -e dev

Alternative environment blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the repository was installed with ``uv``, replace the ``pixi run ...`` line
with the following command, still running it from ``$REPOSITORY``:

.. code-block:: bash

   uv run --locked rhime run \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

Prepare the environment on the login node with ``uv sync --locked`` when
compute nodes cannot download packages. ``uv`` uses the repository's
``uv.lock``, but its PyPI wheels do not provide the same single conda-forge
HDF5/NetCDF stack as Pixi. Prefer Pixi if a ``uv`` environment reports HDF5,
``h5py``, ``h5netcdf``, or ``netCDF4`` binary errors.

An existing conda environment remains usable too. Replace the Pixi setup and
command with:

.. code-block:: bash

   eval "$(conda shell.bash hook)"
   conda activate pymc_env

   rhime run \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

In every case, invoke the installed command rather than an internal
``openghg_inversions/hbmcmc/run_hbmcmc.py`` path.

For a multisector batch run, only the config and subcommand need to change:

.. code-block:: bash

   INI_FILE=/user/home/example/my_inversions/rhime_multisector.ini
   OUTPUT_DIR=/user/home/example/my_inversions/outputs

   pixi run --locked -e dev rhime run --model multisector \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

The historical ``run_hbmcmc.py`` entry point is removed. Convert fixedbasis-style
settings explicitly or retain matching older software. For modern batch jobs,
start from the RHIME template, use ``flux_sources`` for standard runs, and
configure the sector sources described in :doc:`rhime` before selecting
``rhime run --model multisector``.

Merging PARIS outputs
---------------------

Merge sequential annual or sub-annual PARIS NetCDF files with the installed
CLI. The command detects legacy and latest PARIS concentration and flux
templates from their schema:

.. code-block:: console

   $ rhime merge-paris-outputs \
       SF6_EUROPE_PARIS_flux_2019-01-01.nc \
       SF6_EUROPE_PARIS_flux_2020-01-01.nc \
       --output SF6_EUROPE_PARIS_flux_2019-2020.nc

Use ``--type flux`` or ``--type concentration`` (also accepted as ``conc``) to
select one product when a broad input glob matches both. Inputs selected for
one invocation must use the same template version; run the command separately
for legacy and latest products because their variable contracts differ.

Staged execution
----------------

``prepare``, ``prior-predictive``, ``sample``, ``diagnose`` and ``postprocess``
separate a run into persisted handoffs. Select ``--model standard``,
``--model multisector``, ``--model co2`` or ``--model co2_o2`` explicitly on
scientific stages; ``diagnose`` reads the posterior independently of the recipe.
See
:doc:`staged_workflow` for the shared stage interface and
:doc:`co2_model_family` for CO2-family TOML examples, including inferred fixed OU
and the same-unit joint linked route. CO₂ and linked CO₂/O₂ sampling consume
one preparation manifest; replay consumes only its sample manifest plus current
output choices. Saved scientific meaning and provenance drive replay without
constructing a graph. Git revision
and checkout status are recorded when available, with unavailable values left
explicitly unknown.
