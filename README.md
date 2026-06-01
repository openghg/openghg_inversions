<img src="https://github.com/openghg/logo/raw/main/OpenGHG_Logo_Landscape.png" width="100">

# OpenGHG Inversions

OpenGHG Inversions is a Python package that is being developed as part of the [OpenGHG project](https://openghg.org) with the aim of merging the data-processing and simulation modelling capabilities of OpenGHG with the atmospheric Bayesian inverse models developed by the Atmospheric Chemistry Research Group (ACRG) at the University of Bristol, UK.

Currently, OpenGHG Inversions includes the following regional inversion models:
- Hierarchical Bayesian Markov Chain Monte Carlo (HBMCMC) model (as described in Ganesan et al., 2014, _ACP_)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10650595.svg)](https://doi.org/10.5281/zenodo.10650595)

## Installation

### Using Pixi (recommended for development)

OpenGHG Inversions reads and writes NetCDF/HDF5 data through OpenGHG,
`xarray`, `h5netcdf`, `h5py`, and `netcdf4`. If these packages
are installed from unrelated PyPI wheels, their bundled HDF5 libraries can
be incompatible. The Pixi environment in this repository installs the
compiled HDF5/NetCDF stack from conda-forge and installs
`openghg_inversions` in editable mode.

Install [Pixi](https://pixi.prefix.dev/latest/installation/), then run:

```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
pixi install -e dev
pixi run -e dev python -c "import openghg_inversions, h5py, h5netcdf, netCDF4"
```

Useful development commands:

```bash
pixi run -e dev test
pixi run -e dev lint
pixi run -e dev typecheck
pixi run -e dev tox
```

To run the optional real country-file HDF5 smoke check on a machine that
can access the ACRG country files, set the country directory and run the
Pixi task:

```bash
OPENGHG_COUNTRY_FILE_SMOKE_DIR=/group/chem/acrg/LPDM/countries pixi run -e dev country-file-smoke
```

The smoke check opens `country_EUROPE_EEZ_PARIS_gapfilled.nc` and
`country_EUROPE.nc` with xarray's default backend, `h5netcdf`, and
`netcdf4`, then exercises `openghg_inversions._country_file.load_country_dataset`.
It prints the `xarray`, `h5netcdf`, `h5py`, and `netCDF4` versions and the
per-engine result. Without `OPENGHG_COUNTRY_FILE_SMOKE_DIR`, the real-file
tests are skipped so uv/pip CI does not need access to cluster data.

To test against a local OpenGHG checkout without replacing the Pixi-managed
HDF5/NetCDF dependencies, install only the local package code:

```bash
pixi run -e dev python -m pip install --no-deps -e ~/Documents/openghg
```

Avoid running plain `pip install -U h5py h5netcdf netcdf4` inside the Pixi
environment, as that can reintroduce incompatible wheels.

### Using pip

```bash
pip install openghg-inversions
```

### Using uv (faster alternative)

```bash
uv pip install openghg-inversions
```

Or with uv's project management:

```bash
# Add to your project
uv add openghg-inversions

# Or install in a virtual environment
uv venv
uv pip install openghg-inversions
```

### Development Installation

If you want to contribute or modify the package:

**With Pixi (recommended when working with NetCDF/HDF5 data):**
```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
pixi install -e dev
```

**With uv:**
```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
uv sync --dev
```

**With pip:**
```bash
git clone https://github.com/openghg/openghg_inversions.git
cd openghg_inversions
pip install -e ".[dev]"
```

## Installation and Setup
As OpenGHG Inversions is dependent on OpenGHG, please ensure that when running locally you are using Python 3.10 or later on Linux or MacOS. Please see the [OpenGHG project](https://github.com/openghg/openghg/) for further installation instructions of OpenGHG and setting up an object store.

### Setup a virtual environment

Check that you have Python 3.10 or greater:
```bash
python --version
```
(Note for Bristol ACRG group: If you are on Blue Pebble, the default anaconda module `lang/python/anaconda` is Python 3.9. Use `module avail` to list other options; `lang/python/miniconda/3.10.10.cuda-12` or `lang/python/miniconda/3.12.2.inc-perl-5.30.0` will work.)

Make a virtual environment
```bash
python -m venv openghg_inv
```

Next activate the environment
```bash
source openghg_inv/bin/activate
```

### Installation using `pip`

First you'll need to clone the repository

```bash
git clone https://github.com/openghg/openghg_inversions.git
```

Next make sure `pip` and related install tools are up to date and then install OpenGHG Inversions using the editable install flag (`-e`)

```bash
pip install --upgrade pip setuptools wheel
pip install -e openghg_inversions
```

Optionally, install the developer requirements (there is more information about this in the "Contributing" section below):
``` bash
pip install -r requirements-dev.txt
```

### Verify that PyMC is using fast linear algebra libraries
At this point, run

``` bash
python -c "import pymc"
```
This should run without printing any messages.
If you receive a message about `pymc` or `pytensor` using the `numpy` C-API, then your inversions might run slowly because the fast linear algebra libraries used by `numpy` haven't been found.

Solutions to this are:
1. Use the Pixi development environment above, which installs `numpy` and the NetCDF/HDF5 stack from conda-forge.
2. Try `python -m pip install numpy` after upgrading `pip, setuptools, wheel`.
3. Create a `conda` env, install `numpy` using `conda`, then use `pip` to upgrade `pip, setuptools, wheel` and install `openghg_inversions`.


## Using OpenGHG Inversions

### Getting Started

For an overview of OpenGHG inversions, see this [primer](docs/getting_started.md).

### Modern RHIME entry points

New RHIME runs can be launched without calling an internal source file path:

```python
from openghg_inversions.rhime import run_rhime, run_rhime_multisector

result = run_rhime(
    species="ch4",
    sites=["TAC"],
    averaging_period=["1h"],
    domain="EUROPE",
    start_date="2019-01-01",
    end_date="2019-01-02",
    output_path="outputs",
    output_name="example",
    flux_sources=["total-ukghg-edgar7"],
)
```

For SLURM batch scripts and installed environments, use the console entry point:

```bash
openghg-inversions run-rhime 2019-01-01 2019-01-02 -c rhime.ini --output-path outputs
openghg-inversions run-rhime-multisector 2019-01-01 2019-01-02 -c rhime_multisector.ini
```

The new RHIME config template is available at
`openghg_inversions/config/templates/rhime_template.ini`. New configs should use
`flux_sources`; legacy `emissions_name` is accepted when `flux_sources` is absent.
See the [RHIME terminology and quickstart](docs/usage/rhime.rst) page for the
canonical config vocabulary.

RHIME terminology:

- `species`: primary gas or tracer name used for object-store lookup and output naming.
- `source`: OpenGHG metadata key used to retrieve flux data.
- `flux_sources`: RHIME field containing requested OpenGHG flux `source` values.
- `sector_sources`: optional mapping from RHIME sector names to OpenGHG flux `source` values.
- `sector`: model component optimized separately, usually backed by one flux `source`.
- `tracer`: additional species used to constrain the primary species through linked forward models.
- `emissions_name`: legacy compatibility spelling only; use `flux_sources` in new RHIME configs.

### Legacy HBMCMC Compatibility

New runs should use `openghg-inversions run-rhime` or the Python
`run_rhime(...)` API above. The historical `run_hbmcmc.py` script remains as a
compatibility wrapper for old fixedbasis-style INI files: it translates legacy
names such as `outputpath`, `outputname`, `nit`, `nchain`, `verbose`, and
`sampler_kwargs` to modern RHIME names and then calls `run_rhime(...)`.
This branch is no longer preserving the exact historical fixedbasisMCMC /
inferpymc passthrough behaviour. Use release `0.6` or earlier if you need the
old fixedbasis implementation.

The old output names `hbmcmc` and `hbmcmc_postprocessing` are deprecated
aliases for the modern `legacy` output format. The compatibility wrapper keeps
the old HBMCMC filename convention for these outputs; direct `run_rhime` calls
use RHIME filenames unless configured otherwise.

The compatibility entry point still accepts the old INI layout and command-line
overrides.

#### Ways of passing arguments to the inversion

##### Passing options in an `ini` file

Extra options can be added to an `ini` file in almost any location.
The [template ini file](openghg_inversions/hbmcmc/config/openghg_hbmcmc_input_template_example.ini) puts
these option under the heading `MCMC.OPTIONS`:

``` ini
[MCMC.OPTIONS]
averaging_error = True
fix_basis_outer_regions = True
use_bc = True
nuts_sampler = "numpyro"
save_trace = False
min_error = "percentile"
pollution_events_from_obs = True
reparameterise_log_normal = False
sampler_kwargs = {"target_accept": 0.99}
```

These options are read from the old file layout and translated where a modern
RHIME equivalent exists. Fixedbasis-only options that are enabled and no longer
have a RHIME equivalent raise a targeted error.

##### Passing options at the command line

When running inversions using the script `run_hbmcmc.py`, you must specify the start and end date of
the inversion period, and you pass an `ini` file using the flag `-c`.

In addition, you can pass the output path using the flag `--output-path`; this is useful if your SLURM script
uses different output locations for different array jobs.

You can also pass supported RHIME-compatible keyword arguments to `run_hbmcmc.py` using the `--kwargs` flag.
For instance:

``` bash
python run_hbmcmc.py "2019-01-01" "2019-02-01" -c "example.ini" --kwargs '{"averaging_error": true, "min_error": 20.0, "nuts_sampler": "numpyro"}'
```
It is crucial that you enclose the dictionary in single quotes, otherwise the command line will split the dictionary on white space.

Again, this can be used to change supported inversion arguments on the fly (say, in a SLURM script).
Unsupported fixedbasis-only options now raise targeted errors instead of being
passed through to `inferpymc`.

The format of the dictionary inside single quotes must be JSON, because the value of `kwargs` is parsed using `json.loads`.
Python translates JSON according to [this table](https://docs.python.org/3/library/json.html#encoders-and-decoders).
In particular, `"true"` in JSON translate to `True` in Python (but `"True"` will be translated as a string).

The parsing in our `ini` files is more flexible; in particular, values that are Python statements will be translated to Python, so you don't need to worry about translation.

#### What parameters can you set?

The following sections detail some parameters that enable/specify optional behaviour in the inversion.

##### Parameters for `fixedbasisMCMC`

These are compatibility-era notes for old fixedbasis-style workflows, not the
recommended interface for new runs. New configs should use the RHIME vocabulary
above. See the docstring for `fixedbasisMCMC` in the [hbmcmc module](openghg_inversions/hbmcmc/hbmcmc.py)
for the current compatibility arguments.


Arguments affecting the data using in the inversion:
- `sites`: a list of the sites to use in the inversion. Other information applied on a site-to-site basis that is presented in lists must be in the same order as used in the `sites` list.
- `inlet`: a list of inlets for each site. If only one inlet is available for a given site and species, then `None` may be used as the value for that site. If there are a range of inlet heights at a single site, and these should correspond to a single footprint release height, then you may use, for instance, `slice(140, 160)` to combine inlet heights between 140 and 160 meters into a single timeseries of observations.
-`instrument`, `fp_height`, `obs_data_level`, and `met_model` must either be lists of the same length as `sites`, or a single value may be supplied and will be converted to a list of the correct length.


Arguments affecting the inverse model:
- `averaging_error`: if `True`, the error from resampling to the given `averaging_period` will be added to the observation's error.
- `use_bc`: defaults to `True`. If `False`, no boundary conditions will be used in the inversion. This implicitly assumes that contributions from the boundary have been subtracted from the observations.
- `fix_basis_outer_regions`:
  - Default value is `False`
  - If `True`, the "outer regions" of the (`EUROPE`) domain use basis regions specified by a file provided by the Met Office (from their "InTem" model), and the "inner region", which includes the UK, is fit using our basis algorithms.
  - This option is only available for the `EUROPE` domain currently.
- `calculate_min_error`: calculate min_error (see below) on the fly using the "residual error method" or a method based on percentiles of observations. Available arguments:
  - `residual`: use "residual error method"
  - `percentile`: use method based on percentiles
  - `None`: in this case, you should pass in a value directly using (for instance) `min_error = 12.3`
- `min_error_options`: additional parameters to pass to the function that compute min error. This should be a dictionary, and the available options depend on the function used. (The functions to compute min. model error are in `model_error.py`).
  - If `calculate_min_error = "residual"`, then, for instance, you could use `min_error_options = {"robust": False, "by_site": True}`. (By default, `robust` is `False`, this is just to show the possibilities.)
- `filters`: filters to apply to data (after it is resampled and aligned)
  - `filters = None` will skip filtering
  - if `filters` is a list of filters (or a string containing a single filter name), those filters will be applied to all sites.
  - if `filters` is a dictionary with site codes as keys and lists of filters as values, then each site will have filters applied individually according to this dictionary. All sites must supplied; to skip a site, pass `None` instead of a list (or omit that site from the dictionary). For instance: `filters = {"MHD": ["pblh_inlet_diff", "pblh_min"], "JFJ": None}`.
  - the list of available filters can be found in the `filtering` function in the [utils module](openghg_inversions/utils.py).
  - Further parameters affecting the model are in the next subsection: they are passed to `inferpymc`.
  - `xprior` and `bcprior`: these should be a dictionary containing `"pdf": <distribution>` and the arguments that should be passed to the PyMC distribution with that name. `<distribution>`

Arguments affecting the output of the inversion:
- `save_trace`:
  - The default value is `False`.
  - If `True`, the arviz `InferenceData` output from sampling will be saved to the output path of the inversion, with a file name of the form `f"{outputname}{start_data}_trace.nc`. To load this trace into arviz, you need to use `InferenceData.from_netcdf`.
  - Alternatively, you can pass a path (including filename), and that path will be used.


##### Parameters for `inferpymc`

In release `0.6` and earlier, unrecognised `fixedbasisMCMC` keyword arguments
were passed through to `inferpymc`. Current compatibility paths validate
RHIME-compatible options instead. The argument routing design is being cleaned
up as part of the fixedbasis retirement work.

Historical inferpymc-era parameters included:
- `min_error`: a non-negative float value specifying a lower bound for the model-measurement mismatch error (i.e. the error on (y - y_mod)).
- `nuts_sampler`: a string, which defaults to `"pymc"`. The other option is `"numpyro"`, which will the [JAX](https://jax.readthedocs.io/en/latest/index.html) accelerated sampler from [Numpyro](https://num.pyro.ai/en/stable/index.html); this tends to be significantly faster than the NUTS sampler built into PyMC.
- `pollution_events_from_obs`: Determines whether the model error is calculated as a fraction of:
  - the measured enhancement above the modelled baseline (if `True`)
  - the prior modelled enhancement (if `False`)
- `no_model_error`: if `True`, only use obs error in likelihood (omitting min. model error and model error from scaling pollution events).
- `reparameterise_log_normal`: if `True`, then log normal priors will be sampled by transforming samples from standard normal random variable to samples from the appropriate log normal distribution.


### The output from inversions

The results of an inversions are returned as an xarray `Dataset`.

The dimension `nmeasure` consists of the time for each observation stacked into a single 1D array.

TODO: complete this part

- `Yerror`: obs. error used in the inversion; if `add_averaging` is True, this will contain the combined "repeatability" and "variability"; otherwise, it will just contain "repeatability", if it is available, or "variability"
- `Yerror_repeatablity`: obs. repeatability. If repeatability isn't available for some sites, then this is filled with zeros.
- `Yerror_variability`: obs. variability.



## Contributing

### Code quality tools

To contribute to `openghg_inversions`, you should also install the developer packages:
```bash
pip install -r requirements-dev.txt
```
This will install the packages `pytest`, `pytest-xdist`, `ruff`, `tox`, and `tox-uv`.

We use `ruff` to lint our code. To check for lint issues, run:
``` bash
ruff check openghg_inversions
```
in your `openghg_inversions` repository (with your virtual env activated).

To fix issues that Ruff can safely update, run:
``` bash
ruff check --fix openghg_inversions
```

You can run the tests using:
``` bash
pytest
```
in the `openghg_inversions` repository. (Make sure your virtual env is activated.)

### Using `tox` to check code

Alternatively, use `tox` to run tests and check the code format.
`tox` creates isolated environments to run the tests, which means it can test against different
versions of OpenGHG.
It does this automatically, so you don't need to manage pip or conda virtual environments to do this.

To install `tox` globally in a "safe" way, use:

```bash
uv tool install tox --with tox-uv
```
or, within a virtual environment, install `tox` and `tox-uv`.

Calling `tox -p` will run tests against OpenGHG devel and the last two releases of OpenGHG, and run Ruff lint checks.

To specify individual jobs, you can use, e.g.:

```bash
tox -e py310-openghgDev
```

to run the tests against the devel branch.

Use `tox -l` to list all options.

To pass arguments to pytest, Ruff, mypy, etc, you can use, e.g.

```bash
tox -- "openghg_inversions/hbmcmc"
```

which will pass the positional argument "openghg_inversions/hbmcmc" to the commands invoked by tox.

### Using branches

To contribute new code, make a branch off of the `devel` branch.
When your code is ready to be added, push it to github (`origin`).
You can then open a "pull request" on github and request a code review.
It's helpful to write a description of the changes made in your PR, as well as linking to any relevant issues.

Your code must past the tests and be reviewed before it can be merged.
After this, you can merge your branch and close it (it can always be recovered later if necessary).

## Citation and contributors

If you use this software, please cite the version-specific Zenodo DOI for the
release you used.

The recommended prose description is:

> We use RHIME, the Regional Hierarchical Inversion Modelling Environment,
> implemented in the `openghg_inversions` Python package.

The formal software citation lists the principal creators of the citable
software artifact. Additional code, testing, documentation, scientific, and
project contributions are recorded in the Zenodo metadata and GitHub history.

## References
Ganesan et al. (2014),_ACP_;

Western et al. (2021), _Enviro. Sci. Tech Lett._
