Getting started with OpenGHG Inversions
=======================================

This is an overview of what OpenGHG Inversions does, and how to use it.

Overview
--------

- Countries are required to create “bottom-up” inventories of emissions.
  These may be totals for a country, or may be a map of estimated
  emissions for a given time period.
- To check these inventories, we create “top-down” constraints by
  passing emissions/flux maps through a physical model and comparing the
  result with observations. We use a Bayesian model to update the flux
  maps using the given observation data.
- The model is roughly
  :math:`\mathrm{obs} \approx \mathrm{sensitivities} \times \mathrm{flux} + \mathrm{baseline} + \mathrm{error}`
- The baseline is calculated by multiplying the flux at the boundaries
  by a sensivity map for each boundary “curtain” (NESW).
- Disturbances from the baseline are calculated by multiplying a
  “footprint” (sensitivities for fluxes) times a flux map.
- The sensitivities are considered deterministic, and the fluxes and
  boundary conditions are modelled as random quantities
- We place prior distributions on the fluxes and boundary conditions and
  use the observation data and MCMC to sample from their posterior
  distributions. (These are specified by the ``xprior`` and ``bcprior``
  variables in the .ini file below.)
- Roughly, an inversion attempts to solve
  :math:`\mathrm{obs} - \mathrm{baseline} \approx \mathrm{sensitivities} \times \mathrm{flux}`;
  the sensitivity matrix is not invertible, so a method like
  least-squares is necessary. We use a hierarchical Bayesian regression
  approach, which estimates uncertainties in a natural way.
- The output of an inversion contains prior and posterior: modelled
  observations (“ :math:`Y` ” variables), fluxes, and boundary
  conditions.

What do you need to run an inversion?
---------------------------

For an inversion, you need to decide the:

- gas species (currently you can only choose one)
- sites (how many, which sites)
- time period (specified using ``start_date``, ``end_date``)
- domain for inversion (for sites in the UK, this is usually
  ``'EUROPE'``)

Once you know the rough outline of your inversion, you need to make sure
the right data is stored in OpenGHG, and you will need a few extra files
not stored in OpenGHG.

Data stored in OpenGHG
~~~~~~~~~~~~~~~~~~~~~~

- Observation data: stored using ``standardise_surface``; look for it
  using ``search_surface``. You need observation data for each site.
- Flux data: stored using ``standardise_flux``; look for it using
  ``search_flux``.
- Boundary conditions data: stored using ``standardise_bc``; look for it
  using ``search_bc``.
- Footprints: stored using ``standardise_footprint``; look for it using
  ``search_footprints`` (NOTE: footprint(s) is plural for search,
  singular for standardise). You need separate footprints for each site.

NOTE: flux and boundary conditions data usually has a specific *domain*,
but it is not specific to any one measurement site, so for multiple
sites, you will usually use the same flux and boundary conditions.

Data not stored in OpenGHG
~~~~~~~~~~~~~~~~~~~~~~~~~~

- countries file

  - e.g. ``country_EUROPE.nc``
  - default location is ``openghg_inversions/countries``

- basis functions

  - “basis function” usually means a netCDF file with latitude and
    longitude coordinates, with integer values. So all of the
    coordinates with value 1 are in region 1, and so on.
  - basis functions for fluxes are created on the fly by a “quadtree
    basis” algorithm. You can also read in pre-defined basis functions.
    Default location: ``openghg_inversions/basis_functions``
  - basis functions for the boundary conditions have a similar format.
    Defaul location ``openghg_inversions/bc_basis_functions``

Summary

- basis functions (for flux) are created on the fly, or you can provide
  them; you need to provide a country file and ``bc_basis_functions``
  files.
- all of these files can be found in ``/group/chemistry/acrg/LPDM``.

How do you run an inversion?
----------------------------

Method 1: python script or notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming you have the necessary data, you just need to run the
``fixedbasisMCMC`` function from ``hbmcmc.py`` in
``openghg_inversions.hbmbmc``.

Method 2: ini file
~~~~~~~~~~~~~~~~~~

- Create a ``my_inversion.ini`` file based on the template in
  ``openghg_inversions/hbmcmc/config/``.
- Activate a conda or venv environment with inversions installed, and
  call
  ``python <path to openghg_inversions>/openghg_inversions/hbmcmc/run_hbmcmc.py -c my_inversion.ini``
- A sample ``.ini`` script is at the bottom of this document.

Method 3: as a job on Blue Pebble
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a ``.sh`` file to run a `batch job using
  SLURM <https://www.acrc.bris.ac.uk/protected/hpc-docs/job_types/serial.html>`__
  using ``sbatch``.
- This file will specify:

  - the name of the job
  - resources required (number of nodes, number of tasks per node,
    number of cpus per task, memory required, compute time required).
    Typically, you need 1 node, 1 task, and as many cpus per task as
    chains in your inversion (each chain is an indepedent MCMC process;
    using 4 is best for checking convergence)
  - where terminal output and errors should be written
  - shell commands to run on the compute node:

    - commands to activate your virtual environment
    - commands to run an inversion (usually method 2, but you could use
      method 1 in a python script)

- On the blue pebble login node, run ``sbatch my_inversion_script.sh``
  (replace ``'my_inversion_script'`` with whatever your script is
  called).
- Use the command ``sacct`` to see what batch jobs you have running.

Note: if you are running multiple inversions with a similar ini file (for instance,
several years of yearly inversions), then you should use a SLURM array job.

Getting set up on Blue Pebble
-----------------------------

This is assuming you can ssh into blue pebble, and are able to modify
files and run commands there (either on the terminal or through VS code
or similar).

(This section is specific to the ACRG group at Bristol, but the parts
about running jobs on SLURM could apply elsewhere.)

Virtual environment
~~~~~~~~~~~~~~~~~~~

TODO: update conda instructions

- Blue pebble has pre-installed software called “modules”.

  - ``module avail`` shows available modules
  - ``module load ...`` will load a module

- Typically you will use the latest anaconda module:
  ``module load lang/python/anaconda``.
- To make your own environment for ``openghg_inversions``, you should:

  1. make a conda env ``conda create --name inv_env numpy`` (note:
     installing ``numpy`` from ``conda`` will install ``openblas``,
     which is a fast linear algebra library; these libraries are in
     non-standard locations on Blue Pebble, and ``pip install numpy``
     will not find them.)
  2. clone openghg_inversions:
     ``git clone https://github.com/openghg/openghg_inversions.git``
  3. ``pip install openghg_inversions`` (in the same directory where you
     just cloned ``openghg_inversions``)

Example batch job script
~~~~~~~~~~~~~~~~~~~~~~~~

This script assumes that you have already created a conda env called
``pymc_env`` and you have an ``.ini`` file in a folder called
``my_inversions``.

.. code:: bash

   #!/bin/sh
   # ****************************************************************************
   # Wrapper script for submitting jobs on ACRC HPC
   # docs: https://www.acrc.bris.ac.uk/protected/hpc-docs/index.html
   # ****************************************************************************
   #SBATCH --job-name=my_inv
   #SBATCH --output=openghg_inversions.out
   #SBATCH --error=openghg_inversions.err
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --cpus-per-task=4
   #SBATCH --time=04:00:00
   #SBATCH --mem=30gb
   #SBATCH --account=dept123456


   # Set up Python environment
   module --force purge
   module load git/2.45.1
   eval "$(conda shell.bash hook)"
   conda activate pymc_env

   #conda info

   # run inversion script
   INI_FILE=/user/home/bm13805/my_inversions/my_hbmcmc_inputs.ini
   python /user/home/bm13805/openghg_inversions/openghg_inversions/hbmcmc/run_hbmcmc.py -c $INI_FILE

   # check numpy config
   # python -c "import numpy as np; np.show_config()"
   # python -c "import pymc"

If this script is saved as ``my_inversions_script.sh``, you would run it
with ``sbatch my_inversion_script.sh``.

If you comment out the lines under ``# run inversion script`` and
uncomment the lines under ``# check numpy config``, you can check if
numpy is using a fast linear algebra library. If you are **not** using a
fast linear algebra library, then in the ``.err`` output file you will
see the warning:

::

   WARNING (pytensor): Using NumPy C-API based implementation for BLAS functions.

Your inversion will run much slower if this is the case; try using conda
to install numpy before installing ``openghg_inversions``, as mentioned
above.

Work vs. Home directory
~~~~~~~~~~~~~~~~~~~~~~~

- Your home directory ``/user/home/ab12345`` (replace ab12345 with your
  using name) has a storage quota of 25GB.
- Check you usage with command ``user-quota``.
- Your work directory ``/user/work/ab12345`` has a storage quota of
  1000GB.
- Conda environments and the inversion output files can be quite large,
  so if you are doing many inversions, you should put the output
  directory in your work directory. (And if you are making multiple
  conda environments, you should put them in your work directory as
  well.)

Sample .ini file
----------------

The following file, ``my_hbmcmc_inputs.ini`` can be used to run an

.. code:: ini

   ; Configuration file for HBMCMC code
   ; Required inputs are marked as such.
   ; All other inputs are optional (defaults will be used)

   [INPUT.MEASUREMENTS]
   ; Input values for extracting observations
   ; species (str) - species name,  e.g. "ch4", "co2", "n2o", etc.
   ; sites (list) - site codes as a list, e.g. ["TAC", "MHD"]
   ; averaging_period (list) - Time periods for to average the measurements to (can be None and must match length of sites)
   ; start_date (str) - Start of observations to extract (format YYYY-MM-DD)
   ; end_date (str) - End of observations to extract (format YYYY-MM-DD) (non-inclusive)
   ; inlet (list/None) - Specific inlet height for the site (list - must match number of sites)
   ; instrument (list/None) - Specific instrument for the site (list - must match number of sites)

   species     = 'ch4'   ; (required)
   sites       = ['TAC'] ; (required)
   averaging_period = ['1H']  ; (required)
   start_date  = '2019-01-01'      ; (required)
   end_date    = '2019-02-01'      ; (required)
   inlet         = ["185m"]
   instrument    = ["picarro"]


   [INPUT.PRIORS]
   ; Input values for extracting footprints, emissions and boundary conditions files (also uses values from INPUT.MEASUREMENTS)
   ; domain (str) - Name of inversion spatial domain
   ; fp_height (list) - Release height for footprints (must match number of sites).
   ; emissions_name (list/None) - Name for specific emissions source.

   domain = 'EUROPE'  ; (required)
   fp_height = ["185m"]  ; typically the same as inlet, but may differ slightly (e.g. if instrument moved to 180m, for instance)
   fp_model = "NAME"  ; LPDM model, usually NAME
   emissions_name = ["total-ukghg-edgar7"]  ; total = all emissions sources; agric-ukghg-edgar7 would be agricultural sources only
   met_model = 'UKV'  ; or None if not specified, check the metadata for your footprint

   [INPUT.BASIS_CASE]
   ; Input values to extract the basis cases to use within the inversion for boundary conditions nd emissions
   ; basis_algorithm (str): Choice of basis function algorithm to use. One of "quadtree" or "weighted"
   ; bc_basis_case (str): Boundary conditions basis, defaults to "NESW" (looks for file format {bc_basis_case}_{domain}_*.nc)
   ; bc_basis_directory (str/None): Directory for bc_basis functions. If None provided, creates new folder in openghg_inversions expecting to find bc_basis_function files there.
   ; fp_basis_case (str/None): Emissions bases:
   ; - if specified, looks for file format {fp_basis_case}_{domain}_*.nc
   ; - if None, creates basis function using algorithm specified and associated parameters
   ; nbasis: Number of basis functions to use for algorithm-specified basis function (rounded to %4) in domain
   ; basis_directory (str/None): Directory containing the basis functions (with domain name as subdirectories)
   ; country_file (str/None): Directory with filename  containing the indices of country boundaries in domain

   basis_algorithm = "quadtree"
   bc_basis_case = "NESW"
   fp_basis_case = None
   nbasis = 50
   basis_directory = "/group/chemistry/acrg/LPDM/basis_functions/"
   bc_basis_directory = None
   country_file = None

   [MCMC.TYPE]
   ; Which MCMC setup to use. This defines the function which will be called and the expected inputs.
   ; Options include: "fixed_basis"

   mcmc_type = "fixed_basis"


   [MCMC.PDF]
   ; Definitions of PDF shape and parameters for inputs
   ; - xprior (dict) - emissions
   ; - bcprior (dict) - boundary conditions
   ; - sigprior (dict) - model error

   ; Each of these inputs should be dictionary with the name of probability distribution and shape parameters.
   ; See https://docs.pymc.io/api/distributions/continuous.html
   ; Check openghg_inversions.hbmcmc.inversion_pymc for options.

   xprior   = {"pdf":"lognormal", "stdev":1}  ; lognormal with mean = 1, stdev = 1
   bcprior  = {"pdf":"truncatednormal", "mu":1.0, "sigma":0.02}  ; truncated normal with mean = 1, stdev 0.02
   sigprior = {"pdf":"uniform", "lower":0.5, "upper":10}
   ;add_offset = False
   ;offsetprior = {"pdf": "normal"}
   ;offset_args = {"drop_first": False}  ;; set to True if you want first site to have offset 0.0

   [MCMC.BC_SPLIT]
   ; Boundary conditions setup
   ; - bc_freq - The period over which the baseline is estimated. e.g.
   ;  - None - one scaling for the whole inversion
   ;  - "monthly" - per calendar monthly
   ;  - "*D" (e.g. "30D") - per number of days (e.g. 30 days)

   bc_freq    = "monthly"
   sigma_freq = None
   sigma_per_site = True


   [MCMC.ITERATIONS]
   ; Iteration parameters
   ; nit (int) - Number of iterations for MCMC
   ; burn (int) - Number of iterations to burn in MCMC
   ; tune (int) - Number of iterations to use to tune step size

   nit  = 5000
   burn = 1000
   tune = 2000


   [MCMC.NCHAIN]
   ; Number of chains to run simultaneously. Must be at least 2 to allow convergence to be checked.

   nchain = 4

   [MCMC.OPTIONS]
   ; averaging_error (bool): Add variability in averaging period to the measurement error (Note: currently this
   ;                         doesn't work correctly)
   ; min_error (float): value specifying a lower bound for the model-measurement mismatch error (i.e. the error on
   ;                    (y - y_mod)). Ignored if compute_min_error = True.
   ; fix_basis_outer_regions (bool): If True, the "outer regions" of the domain use basis regions specified by a
   ;                                 file provided by the Met Office (from their "InTem" model), and the "inner
   ;                                 region", which includes the UK, is fit using our basis algorithms.
   ; use_bc (bool): defaults to True. If False, no boundary conditions will be used in the inversion. This
   ;                implicitly assumes that contributions from the boundary have been subtracted from the
   ;                observations.
   ; nuts_sampler (str): defaults to "pymc". The other option is "numpyro", which will the JAX accelerated sampler
   ;                     from Numpyro; this tends to be significantly faster than the NUTS sampler built into PyMC.
   ; save_trace (bool): If True, the arviz InferenceData output from sampling will be saved to the output path of
   ;                    the inversion, with a file name of the form f"{outputname}{start_data}_trace.nc.
   ;                    Alternatively, you can pass a path (including filename), and that path will be used.
   ; calculate_min_error: computes min_error on the fly using the "residual error method" or a method based on percentiles.
   ;                      values can be: "residual", "percentile", None. If value is None, then value passed to `min_error`
   ;                      is used.
   ; pollution_events_from_obs (bool): Determines whether the model error is calculated as a fraction of:
   ;                                   - the measured enhancement above the modelled baseline (if True)
   ;                                   - the prior modelled enhancement (if False)
   ; reparameterise_log_normal (bool): If True, rewrite log normal prior samples as a function of standard normal samples.
   ;                                   This may reduce divergences when sampling.
   ; sampler_kwargs (dict): Kwargs to pass to the sampler (e.g. sampler_kwargs = {'target_accept': 0.99})

   averaging_error = True
   min_error = 0.0
   fix_basis_outer_regions = False
   use_bc = True
   nuts_sampler = "numpyro"
   save_trace = True
   min_error_options = {"by_site": True}  ; options to pass to function used to compute min error
   pollution_events_from_obs = True
   no_model_error = False
   reparameterise_log_normal = False

   [MCMC.OUTPUT]
   ; Format of output
   ; See docs for openghg_inversions.hbmcmc.hbmcmc.fixedbasisMCMC
   ; for full list of options
   output_format = "hbmcmc"  ; default option, but "paris" tends to be used more

   ; Details of where to write the output
   ; outputpath (str) - directory to write output
   ; outputname (str) - unique identifier for output/run name.

   outputpath = '/user/work/ab12345/my_inversions'  ; (required)
   outputname = 'ch4_TAC_test'  ; (required)

Description of HBMCMC output file
---------------------------------

The output of ``run_hbmcmc`` (and ``fixedbasisMCMC``) is an xarray
Dataset with the following variables and attributes:

HMCMC output data variables:

- ``Y``: Measurements used in inversion
- ``Yerror``: Measurement error
- ``Ytime``: Measurement times
- ``Yapriori``: Modelled measurements using a priori emissions
- ``Ymod``: Posterior modelled mean measurements
- ``Ymod95``: Posterior modelled 95% measurement uncertainty (2.5% and
  97.5%)
- ``Ymod68``: Posterior modelled 68% measurement uncertainty (16% and
  84%)
- ``YmodBC``: Posterior modelled mean boundary conditions
- ``YaprioriBC``: Modelled BCs using a priori BCs
- ``xtrace``: MCMC chain for basis functions
- ``bctrace``: MCMC chain for boundary conditions
- ``sigtrace``: MCMC chain for model error (currently iid)
- ``siteindicator``: numerical value corresponding to each site, same
  size as Y.
- ``sitename``: name of site (in order or siteindicators)
- ``site_lon``: Measurement site longitude
- ``site_lat``: Measurement site latitude
- ``aprioriflux``: Mean a priori flux for whole domain at NAME
  resolution. E.g.., if emissions file is monthly, and you are running
  for April and May it will take the mean emissions weighted by number
  of days in each month.
- ``meanflux``: Mean posterior flux for whole domain at NAME resolution
- ``meanscaling``: Mean posterior scaling for whole domain at NAME
  resolution
- ``basis_functions``: Basis functions used at NAME resolution
- ``countrytotals``: Mean posterior total emissions for every country in
  domain country file
- ``country68``: 68% uncertainty for country totals (16% and 84%)
- ``country95``: 95% uncertainty for country totals (2.5% and 97.5%)
- ``countrysd``: standard deviation for country totals

Attributes:

- Units for all variables that need them
- Start date of inversion period
- End date of inversion period
- Type(s) of MCMC sampler used
- Prior PDF for emissions scaling
- Prior PDF for BC scaling
- Prior PDF for model error
- Creator
- Date created
- Convergence: Passed/Failed based on multiple chains (min 2) having
  Gelman-Rubin diagnostic < 1.05
