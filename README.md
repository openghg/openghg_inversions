<img src="https://github.com/openghg/logo/raw/main/OpenGHG_Logo_Landscape.png" width="100">

# OpenGHG Inversions 

OpenGHG Inversions is a Python package that is being develoepd as part of the [OpenGHG project](https://openghg.org) with the aim of merging the data-processing and simulation modelling capabilities of OpenGHG with the atmospheric Bayesian inverse models developed by the Atmospheric Chemistry Research Group (ACRG) at the University of Bristol, UK. 

Currently, OpenGHG Inversions includes the following regional inversion models:
- Hierarchical Bayesian Markov Chain Monte Carlo (HBMCMC) model (as described in Ganesan et al., 2014, _ACP_)

## Installation and Setup
As OpenGHG Inversions is dependent on OpenGHG, please ensure that when running locally you are using Python 3.8 or later on Linux or MacOS. Please see the [OpenGHG project](https://github.com/openghg/openghg/) for further installation instructions of OpenGHG and setting up an object store.

### Setup a virtual environment

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

At this point, run

``` bash
python -c "import pymc"
```
This should run without printing any messages.
If you receive a message about `pymc` or `pytensor` using the `numpy` C-API, then your inversions might run slowly because the fast linear algebra libraries used by `numpy` haven't been found.

Solutions to this are:
1. try `python -m pip install numpy` after upgrading `pip, setuptools, wheel`
2. create a `conda` env, install `numpy` using `conda`, then use `pip` to upgrade  `pip, setuptools, wheel` and install `openghg_inversions` 

### Setup

Once installed, ensure that your OpenGHG object store is configured and that you are comfortable with adding data to your object store. The HBMCMC inversion model assumes all necessary data required for the inversion run has already been added to the object store.  

## Getting Started
_We are currently writing documentation on using the HBMCMC inversion code. We thank you for your patience_

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
black --check openghg_inversions
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

