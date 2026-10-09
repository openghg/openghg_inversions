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

Python model selection and compatibility
----------------------------------------

The public Python selector is :func:`openghg_inversions.rhime.run`:

.. code-block:: python

   from openghg_inversions.rhime import run

   result = run(config_file="rhime.ini")  # model="standard" by default
   result = run(model="multisector", config_file="rhime_multisector.ini")

The ``co2``, ``co2_cached_sigma``, ``co2_o2`` and ``co2_o2_cached_sigma`` Python
choices call their existing prepared-input recipes. They require the appropriate
``prepared_inputs`` object and preserve each recipe's arguments and return type;
selection does not provide a new acquisition or checkpoint route. These families
are therefore not choices for the configuration-driven ``rhime run`` command.
Existing CO2 staged commands retain their current support boundaries.

``openghg-inversions`` and its ``run-rhime``, ``run-rhime-multisector`` and
``run-rhime-nested`` subcommands remain supported through the 0.9 compatibility
cycle. Both executables expose the same command set. The existing runner imports and
shared handoff/assembly names listed here remain aliases to their implementations: ``run_rhime`` is
``run_standard``, ``run_rhime_multisector`` is ``run_multisector``, and
``run_rhime_nested`` is ``run_nested``. No argument conversion or warning is
added to these aliases. Shared ``MergedData`` and ``PreparedInputs`` retain the
``RhimeMergedData`` and ``RhimePreparedInputs`` aliases and unchanged serialized
schemas. ``assemble_inputs`` and ``with_prepared_sites`` retain their former
``assemble_rhime_inputs`` and ``with_prepared_rhime_sites`` imports.

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

The historical ``run_hbmcmc.py`` entry point remains a compatibility wrapper
for supported older fixedbasis-style INI files. It does not turn such a file
into a multisector configuration. For new batch jobs, start from the RHIME
template, use ``flux_sources`` for standard runs, and configure the sector
sources described in :doc:`rhime` before selecting
``run-rhime-multisector``.

Merging PARIS outputs
---------------------

Merge sequential annual or sub-annual PARIS NetCDF files with the installed
CLI. The command detects legacy and latest PARIS concentration and flux
templates from their schema:

.. code-block:: console

   $ openghg-inversions merge-paris-outputs \
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
``--model multisector`` or ``--model co2`` explicitly on scientific stages;
``diagnose`` reads the posterior independently of the recipe. See
:doc:`staged_workflow` for the shared stage interface and
:doc:`co2_model_family` for ordinary and cached fixed-OU CO2 TOML examples.
The CO2 route starts from a saved coherent prepared-input artifact and requires
an identifiable installed Git revision; it uses its own constrained output
contract.
