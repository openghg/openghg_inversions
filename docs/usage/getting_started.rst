Getting started with OpenGHG Inversions
=======================================

This page introduces the data and supported RHIME entry points used by a
regional inversion. If atmospheric inversions are new to you, first read
:doc:`conceptual_inversion` and then use the
:doc:`model recipe chooser <model_recipes>` to select a workflow.

Required data
-------------

Before running an inversion, choose the gas species, measurement sites, time
period, and domain. A standard inversion retrieves the following products from
OpenGHG:

- observations for each site;
- flux data;
- boundary-condition data; and
- footprints for each site.

Use OpenGHG's ``search_surface``, ``search_flux``, ``search_bc``, and
``search_footprints`` functions to confirm that the products are available.
Flux and boundary-condition products are normally domain-specific rather than
site-specific, so a multisite inversion will commonly reuse the same products.

An inversion may also need files outside the OpenGHG object store:

- a country file, such as ``country_EUROPE.nc``; and
- predefined flux or boundary-condition basis functions, when the selected
  basis is not constructed during preparation.

Flux basis functions can be created with supported algorithms such as
``quadtree`` and ``weighted`` or loaded from a saved basis artifact. The Python
basis API also supports region-constrained generation when the caller supplies
an already loaded ``region_classes`` ``DataArray``. See
:doc:`shared_scientific_concepts` and the basis API reference for the current
options and limits.

Run a standard inversion
------------------------

Python scripts and notebooks should call ``run_rhime``::

   from openghg_inversions.rhime import run_rhime

   result = run_rhime(
       species="ch4",
       sites=["TAC"],
       averaging_period=["1h"],
       domain="EUROPE",
       start_date="2019-01-01",
       end_date="2019-02-01",
       flux_sources=["total-ukghg-edgar7"],
       output_path="outputs",
       output_name="ch4_TAC",
   )

For an INI configuration, copy
``openghg_inversions/config/templates/rhime_template.ini`` and use the
installed command::

   openghg-inversions run-rhime \
       2019-01-01 2019-02-01 -c rhime.ini --output-path outputs

The :doc:`RHIME guide <rhime>` describes the full configuration vocabulary,
likelihoods, outputs, and Python result object. For source-resolved inversions,
use :doc:`standard_model_family`; for CO2-only and linked CO2/O2 models, use
:doc:`co2_model_family`.

Run on a SLURM cluster
----------------------

An inversion can be submitted as a serial SLURM job. Request one task and set
``--cpus-per-task`` to the number of chains when chains are run in parallel.
The exact account, module, memory, and time values are cluster-specific. A
minimal batch script is::

   #!/bin/sh
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
   # Load the Python environment modules used by your cluster.
   source /path/to/venv/bin/activate

   openghg-inversions run-rhime \
       2019-01-01 2019-02-01 \
       -c /path/to/rhime.ini \
       --output-path /path/to/outputs

Submit it with ``sbatch my_inversion_script.sh`` and inspect jobs with the
cluster's normal ``sacct`` or ``squeue`` workflow. For repeated periods, use a
SLURM array and override dates or output paths per array element.

On systems with networked home and work storage, put environments, prepared
data, traces, and outputs in the larger work filesystem. Sampling and NetCDF
outputs can be substantially larger than source code or configuration files.

Migrate an old fixedbasis workflow
----------------------------------

The direct ``fixedbasisMCMC`` and ``inferpymc`` implementation has been
removed. Existing supported fixedbasis-style INI files can temporarily use
``python -m openghg_inversions.hbmcmc.run_hbmcmc``; the wrapper translates
them and always runs RHIME. It does not support ``--legacy-fixedbasis`` or
generate new legacy templates.

See :doc:`legacy_and_migration` for parameter mappings, batch-script changes,
return types, legacy-format output, and interfaces with no current equivalent.
