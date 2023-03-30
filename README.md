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

### Setup

Once installed, ensure that your OpenGHG object store is configured and that you are comfortable with adding data to your object store. The HBMCMC inversion model assumes all necessary data required for the inversion run has already been added to the object store.  

## Getting Started
_We are currently writing documentation on using the HBMCMC inversion code. We thank you for your patience_

## References
Ganesan et al. (2014),_ACP_; 

Western et al. (2021), _Enviro. Sci. Tech Lett._

