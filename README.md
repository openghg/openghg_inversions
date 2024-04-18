<img src="https://github.com/openghg/logo/raw/main/OpenGHG_Logo_Landscape.png" width="100">

# OpenGHG Inversions 

OpenGHG Inversions is a Python package that is being develoepd as part of the [OpenGHG project](https://openghg.org) with the aim of merging the data-processing and simulation modelling capabilities of OpenGHG with the atmospheric Bayesian inverse models developed by the Atmospheric Chemistry Research Group (ACRG) at the University of Bristol, UK. 

Currently, OpenGHG Inversions includes the following regional inversion models:
- Hierarchical Bayesian Markov Chain Monte Carlo (HBMCMC) model (as described in Ganesan et al., 2014, _ACP_)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10650596.svg)](https://doi.org/10.5281/zenodo.10650596)

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
1. try `python -m pip install numpy` after upgrading `pip, setuptools, wheel`
2. create a `conda` env, install `numpy` using `conda`, then use `pip` to upgrade  `pip, setuptools, wheel` and install `openghg_inversions` 


## Using OpenGHG Inversions

### Getting Started

For an overview of OpenGHG inversions, see this [primer](getting_started.md).

### Passing parameters to the inversion 

Keyword arguments are propagated as follows:
1. any key-value pair in an `ini` file or passed via the `--kwargs` flag is passed to the MCMC function as a keyword argument. (Currently, `fixedbasisMCMC` is the only available MCMC function)
2. any keyword argument not recognised by the MCMC function (i.e. `fixedbasisMCMC`) is passed to the function `inferpymc` in `hbmcmc.inversion_pymc`, which is the function that creates and samples from the RHIME model. 

Thus you can pass arguments to either `fixedbasisMCMC` or `inferpymc`, but all of these arguments will be specified in the `ini` file (or command line).

Let's look at these two steps in detail.

#### Ways of passing arguments to the inversion

##### Passing options in an `ini` file

Extra options can be added to an `ini` file in almost any location. 
The [template ini file](openghg_inversions/hbmcmc/config/openghg_hbmcmc_input_template_example.ini) puts
these option under the heading `MCMC.OPTIONS`:

``` ini
[MCMC.OPTIONS]
averaging_error = True
min_error = 20.0
fixed_basis_outer_regions = False
```

These will be passed to the MCMC function (e.g. `fixedbasisMCMC`) as keyword arguments.
Any argument in `fixedbasisMCMC` can be specified in an `ini` file this way.

##### Passing options at the command line

When running inversions using the script `run_hbmcmc.py`, you must specify the start and end date of
the inversion period, and you pass an `ini` file using the flag `-c`.

In addition, you can pass the output path using the flag `--output-path`; this is useful if your SLURM script
uses different output locations for different array jobs.

You can also pass arbitrary keyword arguments to `run_hbmcmc.py` using the `--kwargs` flag.
For instance:

``` bash
python run_hbmcmc.py "2019-01-01" "2019-02-01" -c "example.ini" --kwargs '{"averaging_error": true, "min_error": 20.0, "nuts_sampler": "numpyro"}'
```
It is crucial that you enclose the dictionary in single quotes, otherwise the command line will split the dictionary on white space.

Again, this can be used to change the arguments passed to an inversion on the fly (say, in a SLURM script).

The format of the dictionary inside single quotes must be JSON, because the value of `kwargs` is parsed using `json.loads`.
Python translates JSON according to [this table](https://docs.python.org/3/library/json.html#encoders-and-decoders).
In particular, `"true"` in JSON translate to `True` in Python (but `"True"` will be translated as a string).

The parsing in our `ini` files is more flexible; in particular, values that are Python statements will be translated to Python, so you don't need to worry about translation.

#### What parameters can you set?

The following sections detail some parameters that enable/specify optional behaviour in the inversion.

##### Parameters for `fixedbasisMCMC`

This is not a comprehensive list (see the docstring for `fixedbasisMCMC` in the [hbmcmc module](openghg_inversions/hbmcmc/hbmcmc.py) for more arguments).

- `save_trace`: 
  - The default value is `False`. 
  - If `True`, the arviz `InferenceData` output from sampling will be saved to the output path of the inversion, with a file name of the form `f"{outputname}{start_data}_trace.nc`. To load this trace into arviz, you need to use `InferenceData.from_netcdf`.
  - Alternatively, you can pass a path (including filename), and that path will be used.
- `averaging_error`: if `True`, the error from resampling to the given `averaging_period` will be added to the observation's error. (Note: currently this doesn't work correctly, see [GH issue #42](https://github.com/openghg/openghg_inversions/issues/42).)
- `use_bc`: defaults to `True`. If `False`, no boundary conditions will be used in the inversion. This implicitly assumes that contributions from the boundary have been subtracted from the observations.
- `fix_basis_outer_regions`:
  - default value is `False`
  - if `True`, the "outer regions" of the (`EUROPE`) domain use basis regions specified by a file provided by the Met Office (from their "InTem" model), and the "inner region", which includes the UK, is fit using our basis algorithms.


##### Parameters for `inferpymc`

As mentioned above, any keyword argument passed to `fixedbasisMCMC` (either by an `ini` file or from `--kwargs` on the command line) that is not recognised by `fixedbasisMCMC` is passed on to `inferpymc`.

These parameters include:
- `min_error`: a non-negative float value that is added to the model and observation error in the likelihood of the Bayesian model.
- `nuts_sampler`: a string, which defaults to `pymc`. The other option is `numpyro`, which will the [JAX](https://jax.readthedocs.io/en/latest/index.html) accelerated sampler from [Numpyro](https://num.pyro.ai/en/stable/index.html); this tends to be significantly faster than the NUTS sampler built into PyMC.

## Contributing

To contribute to `openghg_inversions`, you should also install the developer packages:
```bash
pip install -r requirements-dev.txt
```
This will install the packages `flake8, pytest, black`.

We use `black` to format our code. To check if your code needs reformatting, run:
``` bash
black --check openghg_inversions
```
in your `openghg_inversions` repository (with your virtual env activated).
If you replace the flag `--check` with `--diff`, you can see what will be changed.

To make these changes, run
``` bash
black openghg_inversions
```

We also recommend using `flake8` to check for code style issues, which you can run with:
``` bash
flake8 openghg_inversions
```

You can run the tests using:
``` bash
pytest
```
in the `openghg_inversions` repository. (Make sure your virtual env is activated.)


To contribute new code, make a branch off of the `devel` branch.
When your code is ready to be added, push it to github (`origin`).
You can then open a "pull request" on github and request a code review.
It's helpful to write a description of the changes made in your PR, as well as linking to any relevant issues.

Your code must past the tests and be reviewed before it can be merged.
After this, you can merge your branch and close it (it can always be recovered later if necessary).

## References
Ganesan et al. (2014),_ACP_; 

Western et al. (2021), _Enviro. Sci. Tech Lett._

