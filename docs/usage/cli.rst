Running RHIME from the command line
===================================

The ``openghg-inversions`` command is the recommended entry point for running
RHIME from a terminal or a batch scheduler. It is installed with
``openghg_inversions``, so a run does not need to know where the package source
code is located.

Standard and multisector runs
-----------------------------

Use ``run-rhime`` for a standard inversion and
``run-rhime-multisector`` for a shared-basis multisector inversion:

.. code-block:: console

   $ openghg-inversions run-rhime 2019-01-01 2019-02-01 \
       --config /path/to/rhime.ini \
       --output-path /path/to/outputs

   $ openghg-inversions run-rhime-multisector 2019-01-01 2019-02-01 \
       --config /path/to/rhime_multisector.ini \
       --output-path /path/to/outputs

``--config`` (or ``-c``) is required. The start and end dates are optional
positional arguments; when supplied, they override ``start_date`` and
``end_date`` in the INI file. Likewise, ``--output-path`` overrides the
configured output directory. Other RHIME keyword arguments can be overridden
with a JSON object passed to ``--kwargs``:

.. code-block:: console

   $ openghg-inversions run-rhime -c rhime.ini \
       --kwargs '{"draws": 2000, "tune": 1000, "chains": 4}'

Keep the JSON in single quotes so the shell passes it as one argument. Run
``openghg-inversions run-rhime --help`` or
``openghg-inversions run-rhime-multisector --help`` for the complete command
syntax. New configuration files should use the RHIME vocabulary documented in
:doc:`rhime`; the packaged starting point is
``openghg_inversions/config/templates/rhime_template.ini``.

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

   pixi run --locked -e dev openghg-inversions run-rhime \
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

   uv run --locked openghg-inversions run-rhime \
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

   openghg-inversions run-rhime \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

In every case, invoke the installed command rather than an internal
``openghg_inversions/hbmcmc/run_hbmcmc.py`` path.

For a multisector batch run, only the config and subcommand need to change:

.. code-block:: bash

   INI_FILE=/user/home/example/my_inversions/rhime_multisector.ini
   OUTPUT_DIR=/user/home/example/my_inversions/outputs

   pixi run --locked -e dev openghg-inversions run-rhime-multisector \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

The historical ``run_hbmcmc.py`` entry point remains a compatibility wrapper
for supported older fixedbasis-style INI files. It does not turn such a file
into a multisector configuration. For new batch jobs, start from the RHIME
template, use ``flux_sources`` for standard runs, and configure the sector
sources described in :doc:`rhime` before selecting
``run-rhime-multisector``.
