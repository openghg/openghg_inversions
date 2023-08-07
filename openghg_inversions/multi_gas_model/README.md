## multiple_gas_inverse_model
A top-down inverse model for estimating sector-level greenhouse gas emissions using observations of secondary trace gases and their emission ratios to aid with source attribution.

### Overview

The model uses Bayesian statistical methods to produce a posterior (final) estimate of greenhouse gas emissions by comparing observed mole fraction concentrations to modelled concentrations, created from an inital estimate (prior) of the emissions. Modelled mole fractions are created by combining the a priori emissions with atmospheric transport footprints. A Markov Chain Monte Carlo (MCMC) process finds most the likely values for emissions, relative to both the observations and the priors, by testing the probability of a sampling of the prior against both datasets. This process is carried out iteratively to build up posterior distributions for each optimised parameter.

Emissions from n sectors are solved for directly, by giving the model an initial estimate of emission from each sector. Emissions from all sectors are then optimised simultaneously. 

This setup is modified further to include observations of one or more secondary gases and emission ratios relative to the primary gas. These emission ratios, specified by sector, are included as either fixed or variable parameters, and can be optimised on the same spatial and temporal scale as the emissions. This ensures that all uncertainty in these ratios are included in the posterior emissions estimates. 

Secondary gases can either be mole fraction or delta value isotope observations. Emission ratios for mole fraction secondary gases are treated similarly to emissions, where the model solves for posterior scaling factors of the a priori emission ratios. Delta values are used directly, not with a scaling factor. In the code documentation, the word 'ratio' is used to represent both mole fration ratios and delta values interchangeably, to refer to a more general relationship between the primary and secondary gases.

More information in how these different observations are included in the model is included in [this document](Mulitple_gas_inverse_model_description_and_equations.pdf). This file also includes details on how the model inputs are produced and step by step descriptions of the MCMC process for each variable.

The mode includes further options for modification, including: 
* A ‘pseudo data’ mode, where synthetic observations are created by combining transport footprints with the emissions prior.
* A model error hyperparameter, which can be set over different time periods (needs testing for non-continuous observations).
* Filtering of observations (needs to be tested for delta values).
* Customisable a priori emissions, basis function and boundary condition files.
* Options for uniform, truncated normal and normal prior PDFs for all parameters.
* See the .ini file for all input options.

The model can be run by specifying inputs in the .ini file (more information on all inputs is included in this file) and running using run_multi_gas_model.py. Outputs are saved as a dictionary in a netCDF file.

### Dependencies 

Various Python packages included in [environment.yaml](environment.yaml). The [model description document](Mulitple_gas_inverse_model_description_and_equations.pdf) includes more information on this.

The [ACRG repository](https://github.com/ACRG-Bristol/acrg) for use of functions in e.g. name/name.py and obs/read.py.
The [OpenGHG repository](https://github.com/openghg/openghg) if using the OpenGHG inputs option with an object store.

This version of the model was tested with ACRG repo commit bfeb92c79dbcc79c6b43ac23265f9e13a1684d3d (03/03/2023).
