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
The rest of the environment setup and the SLURM resource requests can remain
the same:

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
   eval "$(conda shell.bash hook)"
   conda activate pymc_env

   INI_FILE=/user/home/example/my_inversions/rhime.ini
   OUTPUT_DIR=/user/home/example/my_inversions/outputs

   openghg-inversions run-rhime \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

Submit the saved script in the same way as before, for example
``sbatch my_inversion_script.sh``. Because the command is installed in
``pymc_env``, there is no hard-coded checkout path to update when the package
location changes.

For a multisector batch run, only the config and subcommand need to change:

.. code-block:: bash

   INI_FILE=/user/home/example/my_inversions/rhime_multisector.ini
   OUTPUT_DIR=/user/home/example/my_inversions/outputs

   openghg-inversions run-rhime-multisector \
       2019-01-01 2019-02-01 \
       --config "$INI_FILE" \
       --output-path "$OUTPUT_DIR"

The historical ``run_hbmcmc.py`` entry point remains a compatibility wrapper
for supported older fixedbasis-style INI files. It does not turn such a file
into a multisector configuration. For new batch jobs, start from the RHIME
template, use ``flux_sources`` for standard runs, and configure the sector
sources described in :doc:`rhime` before selecting
``run-rhime-multisector``.
