# OpenGHG Inversions

Dec. 2022

OpenGHG Inversions is a Python package that is being developed as a direct result of the OpenGHG project (https://openghg.org). Merging the capabilities of atmospheric trace gas processing in OpenGHG with Bayesian inversion models developed by the Atmospheric Chemistry Research Group ACRG) at the University of Bristol, we have developed a self-contained repository for performing regional-scale trace gas inversions. 

At present, OpenGHG Inversions only includes the ACRG Hierarchical Bayesian Markov Chain Monte Carlo (HBMCMC) model using invariant basis functions (e.g. Ganesan et al., 2014; Western et al., 2021). We hope to include additional inversion models in the near future. Watch this space! 

## Installation and Setup
As OpenGHG Inversions is dependent on OpenGHG, please ensure that when running locally you are using Python 3.8 or later on Linux or MacOS. Please see the OpenGHG project for further installation instructions of OpenGHG and setting up an object store.  

### Installation using `pip`
You can install OpenGHG Inversions using `pip` 
```bash
pip install openghg_inversions
```
It is recommended doing this in a new virtual environment.

### Setup
Once installed, ensure that your OpenGHG object store is configured and that you are comfortable with adding data to your object store. The HBMCMC inversion model assumes all necessary data required for the inversion run has already been added to the object store.  

## Getting Started
_We are currently writing documentation on using the HBMCMC inversion code. We thank you for your patience_

## References
Ganesan et al. (2014),_ACP_; 

Western et al. (2021), _Enviro. Sci. Tech Lett._
