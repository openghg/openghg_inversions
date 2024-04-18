
# Table of Contents

1.  [Overview](#orgf3238fd)
2.  [What do you need to run an inversion?](#org4510696)
    1.  [Data stored in OpenGHG](#orge9dc720)
    2.  [Data not stored in OpenGHG](#org35e2355)
3.  [How do you run an inversion?](#orgb430906)
    1.  [Method 1: python script or notebook](#org42edfde)
    2.  [Method 2: ini file](#org6a59239)
    3.  [Method 3: as a job on Blue Pebble](#org11c7799)
4.  [Getting set up on Blue Pebble](#org16ffe70)
    1.  [Virtual environment](#orgc7be83e)
    2.  [Example batch job script](#orgc73302c)
    3.  [Work vs. Home directory](#orgd09b62d)
5.  [Sample .ini file](#orge4d8500)
6.  [Description of HBMCMC output file](#org3f637ce)



<a id="orgf3238fd"></a>

# Overview

-   Countries are required to create &ldquo;bottom-up&rdquo; inventories of emissions. These may be totals for a country, or may be a map of estimated emissions for a given time period.
-   To check these inventories, we create &ldquo;top-down&rdquo; constraints by passing emissions/flux maps through a physical model and comparing the result with observations. We use a Bayesian model to update the flux maps using the given observation data.
-   The model is roughly $\mathrm{obs} \approx \mathrm{sensitivities} \times \mathrm{flux} + \mathrm{baseline} + \mathrm{error}$
-   The baseline is calculated by multiplying the flux at the boundaries by a sensivity map for each boundary &ldquo;curtain&rdquo; (NESW).
-   Disturbances from the baseline are calculated by multiplying a &ldquo;footprint&rdquo; (sensitivities for fluxes) times a flux map.
-   The sensitivities are considered deterministic, and the fluxes and boundary conditions are modelled as random quantities
-   We place prior distributions on the fluxes and boundary conditions and use the observation data and MCMC to sample from their posterior distributions. (These are specified by the `xprior` and `bcprior` variables in the .ini file below.)
-   Roughly, an inversion attempts to solve $\mathrm{obs} - \mathrm{baseline} \approx \mathrm{sensitivities} \times \mathrm{flux}$; the sensitivity matrix is not invertible, so a method like least-squares is necessary. We use a hierarchical Bayesian regression approach, which estimates uncertainties in a natural way.
-   The output of an inversion contains prior and posterior: modelled observations (&ldquo;$Y$&rdquo; variables), fluxes, and boundary conditions.


<a id="org4510696"></a>

# What do you need to run an inversion?

For an inversion, you need to decide the:

-   gas species (currently you can only choose one)
-   sites (how many, which sites)
-   time period (specified using `start_date`, `end_date`)
-   domain for inversion (for sites in the UK, this is usually `'EUROPE'`)

Once you know the rough outline of your inversion, you need to make sure the right data is stored in OpenGHG, and you will need a few extra files not stored in OpenGHG.


<a id="orge9dc720"></a>

## Data stored in OpenGHG

-   Observation data: stored using `standardise_surface`; look for it using `search_surface`. You need observation data for each site.
-   Flux data: stored using `standardise_flux`; look for it using `search_flux`.
-   Boundary conditions data: stored using `standardise_bc`; look for it using `search_bc`.
-   Footprints: stored using `standardise_footprint`; look for it using `search_footprints` (NOTE: footprint(s) is plural for search, singular for standardise). You need separate footprints for each site.

NOTE: flux and boundary conditions data usually has a specific *domain*, but it is not specific to any one measurement site, so for multiple sites, you will usually use the same flux and boundary conditions.


<a id="org35e2355"></a>

## Data not stored in OpenGHG

-   countries file
    -   e.g. `country_EUROPE.nc`
    -   default location is `openghg_inversions/countries`
-   basis functions
    -   &ldquo;basis function&rdquo; usually means a netCDF file with latitude and longitude coordinates, with integer values. So all of the coordinates with value 1 are in region 1, and so on.
    -   basis functions for fluxes are created on the fly by a &ldquo;quadtree basis&rdquo; algorithm. You can also read in pre-defined basis functions. Default location: `openghg_inversions/basis_functions`
    -   basis functions for the boundary conditions have a similar format. Defaul location `openghg_inversions/bc_basis_functions`

Summary

-   basis functions (for flux) are created on the fly, or you can provide them; you need to provide a country file and `bc_basis_functions` files.
-   all of these files can be found in `/group/chemistry/acrg/LPDM`.


<a id="orgb430906"></a>

# How do you run an inversion?


<a id="org42edfde"></a>

## Method 1: python script or notebook

Assuming you have the necessary data, you just need to run the `fixedbasisMCMC` function from `hbmcmc.py` in `openghg_inversions.hbmbmc`.


<a id="org6a59239"></a>

## Method 2: ini file

-   Create a `.ini` file based on the templates in `openghg_inversions/config/templates`. (NOTE: these need to be updated.)
-   Activate a conda or venv environment with inversions installed, and call `python <path to openghg_inversions>/openghg_inversions/hbmcmc/run_hbmcmc.py -c somefile.ini`
-   A sample `.ini` script is at the bottom of this document.


<a id="org11c7799"></a>

## Method 3: as a job on Blue Pebble

-   Create a `.sh` file to run a [batch job using SLURM](https://www.acrc.bris.ac.uk/protected/hpc-docs/job_types/serial.html) using `sbatch`.
-   This file will specify:
    -   the name of the job
    -   resources required (number of nodes, number of tasks per node, number of cpus per task, memory required, compute time required). Typically, you need 1 node, 1 task, and as many cpus per task as chains in your inversion (each chain is an indepedent MCMC process; using 4 is best for checking convergence)
    -   where terminal output and errors should be written
    -   shell commands to run on the compute node:
        -   commands to activate your virtual environment
        -   commands to run an inversion (usually method 2, but you could use method 1 in a python script)
-   On the blue pebble login node, run `sbatch my_inversion_script.sh` (replace `'my_inversion_script'` with whatever your script is called).
-   Use the command `sacct` to see what batch jobs you have running.


<a id="org16ffe70"></a>

# Getting set up on Blue Pebble

This is assuming you can ssh into blue pebble, and are able to modify files and run commands there (either on the terminal or through VS code or similar).


<a id="orgc7be83e"></a>

## Virtual environment

-   Blue pebble has pre-installed software called &ldquo;modules&rdquo;.
    -   `module avail` shows available modules
    -   `module load ...` will load a module
-   Typically you will use the latest anaconda module: `module load lang/python/anaconda`.
-   To make your own environment for `openghg_inversions`, you should:
    1.  make a conda env `conda create --name inv_env numpy` (note: installing `numpy` from `conda` will install `openblas`, which is a fast linear algebra library; these libraries are in non-standard locations on Blue Pebble, and `pip install numpy` will not find them.)
    2.  clone openghg<sub>inversions</sub>: `git clone https://github.com/openghg/openghg_inversions.git`
    3.  `pip install openghg_inversions` (in the same directory where you just cloned `openghg_inversions`)


<a id="orgc73302c"></a>

## Example batch job script

This script assumes that you have already created a conda env called `pymc_env` and you have an `.ini` file in a folder called `my_inversions`.

```bash
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
module load lang/python/anaconda
eval "$(conda shell.bash hook)"
conda activate pymc_env

#conda info

# run inversion script
INI_FILE=/user/home/bm13805/my_inversions/my_hbmcmc_inputs.ini
python /user/home/bm13805/openghg_inversions/openghg_inversions/hbmcmc/run_hbmcmc.py -c $INI_FILE

# check numpy config
# python -c "import numpy as np; np.show_config()"
# python -c "import pymc"
```

If this script is saved as `my_inversions_script.sh`, you would run it with `sbatch my_inversion_script.sh`.

If you comment out the lines under `# run inversion script` and uncomment the lines under `# check numpy config`, you can check if numpy is using a fast linear algebra library. If you are **not** using a fast linear algebra library, then in the `.err` output file you will see the warning:

    WARNING (pytensor): Using NumPy C-API based implementation for BLAS functions.

Your inversion will run much slower if this is the case; try using conda to install numpy before installing `openghg_inversions`, as mentioned above.


<a id="orgd09b62d"></a>

## Work vs. Home directory

-   Your home directory `/user/home/ab12345` (replace ab12345 with your using name) has a storage quota of 25GB.
-   Check you usage with command `user-quota`.
-   Your work directory `/user/work/ab12345` has a storage quota of 1000GB.
-   Conda environments and the inversion output files can be quite large, so if you are doing many inversions, you should put the output directory in your work directory. (And if you are making multiple conda environments, you should put them in your work directory as well.)


<a id="orge4d8500"></a>

# Sample .ini file

The following file, `my_hbmcmc_inputs.ini` can be used to run an

```ini
; Configuration file for HBMCMC code
; Required inputs are marked as such.
; All other inputs are optional (defaults will be used)

[INPUT.MEASUREMENTS]
; Input values for extracting observations
; species (str) - species name,  e.g. "ch4", "co2", "n2o", etc.
; sites (list) - site codes as a list, e.g. ["TAC", "MHD"]
; meas_period (list) - Time periods for measurements as a list (must match length of sites)
; start_date (str) - Start of observations to extract (format YYYY-MM-DD)
; end_date (str) - End of observations to extract (format YYYY-MM-DD) (non-inclusive)
; inlet (list/None) - Specific inlet height for the site (list - must match number of sites)
; instrument (list/None) - Specific instrument for the site (list - must match number of sites)

species     = 'ch4'   ; (required)
sites       = ['TAC'] ; (required)
meas_period = ['1H']  ; (required)
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
; Input values to extract the basis cases to use within the inversion for boundary conditions and emissions/fluxes
; bc_basis_case (str) - boundary conditions basis, defaults to "NESW" (looks for file format {bc_basis_case}_{domain}_*.nc)
; fp_basis_case (str/None) - emissions bases:
; - if specified, looks for file format {fp_basis_case}_{domain}_*.nc
; - if None, creates basis function using quadtree algorithm and associated parameters
;   - nbasis - Number of basis functions to use for quadtree derived basis function (rounded to %4)

bc_basis_case  = "NESW"
bc_basis_directory = "/group/chemistry/acrg/LPDM/bc_basis_functions/"  ; LPDM/bc_basis_functions is default
fp_basis_case  = None ;  do not read in a basis for footprint/flux
quadtree_basis = True ; create quadtree basis on the fly
nbasis         = 100 ; number of basis functions to use
basis_directory = "/group/chemistry/acrg/LPDM/basis_functions/"
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

xprior   = {"pdf":"lognormal", "mu":1, "sigma":1}  ; lognormal with mode = 1, mean = exp(1.5)
bcprior  = {"pdf":"lognormal", "mu":0.004, "sigma":0.02}  ; lognormal with mode = 1, mean = exp(0.006)
sigprior = {"pdf":"uniform", "lower":0.5, "upper":10}


[MCMC.BC_SPLIT]
; Boundary conditions setup
; - bc_freq - The period over which the baseline is estimated. e.g.
;  - None - one scaling for the whole inversion
;  - "monthly" - per calendar monthly
;  - "*D" (e.g. "30D") - per number of days (e.g. 30 days)

bc_freq    = "monthly"
sigma_freq = None


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

[MCMC.ADD_ERROR]
; Add variability in averaging period to the measurement error

averagingerror = False

[MCMC.OUTPUT]
; Details of where to write the output
; outputpath (str) - directory to write output
; outputname (str) - unique identifier for output/run name.

outputpath = '/user/work/ab12345/my_inversions'  ; (required)
outputname = 'ch4_TAC_test'  ; (required)
```

<a id="org3f637ce"></a>

# Description of HBMCMC output file

The output of `run_hbmcmc` (and `fixedbasisMCMC`) is an xarray Dataset with the following variables and attributes:

HMCMC output data variables:

-   `Y`: Measurements used in inversion
-   `Yerror`: Measurement error
-   `Ytime`: Measurement times
-   `Yapriori`: Modelled measurements using a priori emissions
-   `Ymod`: Posterior modelled mean measurements
-   `Ymod95`: Posterior modelled 95% measurement uncertainty (2.5% and 97.5%)
-   `Ymod68`: Posterior modelled 68% measurement uncertainty (16% and 84%)
-   `YmodBC`: Posterior modelled mean boundary conditions
-   `YaprioriBC`: Modelled BCs using a priori BCs
-   `xtrace`: MCMC chain for basis functions
-   `bctrace`: MCMC chain for boundary conditions
-   `sigtrace`: MCMC chain for model error (currently iid)
-   `siteindicator`: numerical value corresponding to each site, same size as Y.
-   `sitename`: name of site (in order or siteindicators)
-   `site_lon`: Measurement site longitude
-   `site_lat`: Measurement site latitude
-   `aprioriflux`: Mean a priori flux for whole domain at NAME resolution. E.g.., if emissions file is monthly, and you are running for April and May it will take the mean emissions weighted by number of days in each month.
-   `meanflux`: Mean posterior flux for whole domain at NAME resolution
-   `meanscaling`: Mean posterior scaling for whole domain at NAME resolution
-   `basis_functions`: Basis functions used at NAME resolution
-   `countrytotals`: Mean posterior total emissions for every country in domain country file
-   `country68`: 68% uncertainty for country totals (16% and 84%)
-   `country95`: 95% uncertainty for country totals (2.5% and 97.5%)
-   `countrysd`: standard deviation for country totals

Attributes:

-   Units for all variables that need them
-   Start date of inversion period
-   End date of inversion period
-   Type(s) of MCMC sampler used
-   Prior PDF for emissions scaling
-   Prior PDF for BC scaling
-   Prior PDF for model error
-   Creator
-   Date created
-   Convergence: Passed/Failed based on multiple chains (min 2) having Gelman-Rubin diagnostic < 1.05

