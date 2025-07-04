# OpenGHG Inversions Change Log

# Unreleased

## Model updates

- Offsets can be applied to all but one site (ini option `offset_args = {"drop_first": True}`) or to all sites, which is the default option (ini option `offset_args = {"drop_first": False}`). [#PR 285](https://github.com/openghg/openghg_inversions/pull/285)

- Updated RHIME likelihood to use a power of 1.99 instead of 2. The power can be specified with the `power` argument an ini file. The value of `power` can be a float or a dict of prior args, which will create a hyperprior for `power`. [#PR 277](https://github.com/openghg/openghg_inversions/pull/277)

## Code changes

- Added offset to PARIS concentration outputs. [#PR 282](https://github.com/openghg/openghg_inversions/pull/282)
- Compression added for output PARIS netcdf files. Standard RHIME output now shuffles to save space.
- Fixed warning messages for zeros/NaNs in `mf_error`. [#PR 292](https://github.com/openghg/openghg_inversions/pull/292)

# Version 0.3.0

- Fixed bug due to wrong BC units. [#PR 249](https://github.com/openghg/openghg_inversions/pull/249)

- Merged functionality of `min_error` and `calculate_min_error` into a single variable (`min_error`). [#PR 240](https://github.com/openghg/openghg_inversions/pull/240)

- Tidied `get_data.py`, splitting it into several files. [#PR 237](https://github.com/openghg/openghg_inversions/pull/237) 

- Updated post-processing, including adding PARIS formatting option. [#PR 225](https://github.com/openghg/openghg_inversions/pull/225). This works for both the EUROPE domain and EASTASIA [#PR 242](https://github.com/openghg/openghg_inversions/pull/242) 

- Unpinned numpy now that pymc upgraded. [#PR 236](https://github.com/openghg/openghg_inversions/pull/236)

- Changed optimization in weighted basis function from recursion to loop. [#PR 224](https://github.com/openghg/openghg_inversions/pull/224)

- Updated and simplified `sparse_xr_dot`. The old version caused errors due to upstream changes. [#PR 231](https://github.com/openghg/openghg_inversions/pull/231) 

- Added MHD obs and footprint to test data. [#PR 209](https://github.com/openghg/openghg_inversions/pull/209)

- Fixed Github workflow so that the last two versions of OpenGHG are automatically selected. [#PR 216](https://github.com/openghg/openghg_inversions/pull/216)

- Added coordinates and deterministics to pymc model, moved "save trace" from `inferpymc` to `fixedbasisMCMC`, and renamed variables in pymc model in preparation for adding in PARIS formatting code. [#PR 204](https://github.com/openghg/openghg_inversions/pull/204)

- Added option to use 'weighted' algorithm to derive basis functions for EASTASIA domain [#PR 199](https://github.com/openghg/openghg_inversions/pull/199)

# Version 0.2.0

- Added option to pass "mean" and "stdev" as parameters for lognormal BC prior [#PR 190](https://github.com/openghg/openghg_inversions/pull/190)

- Pinned numpy to version < 2.0 since PyTensor hasn't updated to numpy >= 2.0 [#PR 148](https://github.com/openghg/openghg_inversions/pull/148)

- Updated filtering to handle case `inlet == "multiple"`. [#PR 189](https://github.com/openghg/openghg_inversions/pull/189) 

- Added option to store merged data in a zarr ZipStore, which is essentially just a zipped zarr store. This should reduce the number of files created when saving merged data. [#PR 185](https://github.com/openghg/openghg_inversions/pull/185)

- Fixed issue where missing footprints times were dropped from basis function calculations. [#PR 186](https://github.com/openghg/openghg_inversions/pull/186) 

- Made format for `filtering` in ini file allow for missing sites. Made `inlet`, `instrument`, `fp_height`, `obs_data_level`, and `met_model`
  accept a single string in the ini file, which will be converted to a list of the correct length.  [#PR 182](https://github.com/openghg/openghg_inversions/pull/182). Bug fix: [#PR 188](https://github.com/openghg/openghg_inversions/pull/188)

- Added code to look for older flux data if none is found between start and end dates [#PR 177](https://github.com/openghg/openghg_inversions/pull/177)

- Moved code related to basis functions from `utils.py` to `basis` submodule [#PR 162](https://github.com/openghg/openghg_inversions/pull/162) 

- Fixed bug in `filtering` function and updated tests to cover all filters [#PR 179](https://github.com/openghg/openghg_inversions/pull/179) 

- Updated all docstrings (various PRs)

- Cleaned up `utils.py`: adding typing, and updated docstrings [#PR 158](https://github.com/openghg/openghg_inversions/pull/158) 

- Refactored `filters.py` so filter functions aren't nested inside `filtering`. Added code to keep track of filter functions. Updated docstrings. [#PR 163](https://github.com/openghg/openghg_inversions/pull/163)

- Replaced `utils.combine_datasets` with (nearly) equivalent function from `openghg.analyse._scenario`. There is currently a thin wrapper to make sure that the second
  dataset is loaded into memory, since this change is only on the devel branch of OpenGHG [#PR 160](https://github.com/openghg/openghg_inversions/pull/160) 

- Moved `basis` and related functions from `utils.py` to `basis._functions.py` to make more consistent [#PR 162](https://github.com/openghg/openghg_inversions/pull/162)

- Moved filters from `utils.py` to new submodule `filters.py` [#PR 159](https://github.com/openghg/openghg_inversions/pull/159) 

- Removed `site_info.json` and `species_info.json` and replaced with calls to functions in `openghg.util`, which pull the same info from `openghg_defs`. [#PR 152](https://github.com/openghg/openghg_inversions/pull/152) 

- Removed unused functions from `convert.py` and updated docstrings. [#PR 151](https://github.com/openghg/openghg_inversions/pull/151) 

- Added new option for computing min. model error based on percentiles. [#PR 142](https://github.com/openghg/openghg_inversions/pull/142) 

- Update the docstrings of `openghg_inversions.basis` and `openghg_inversions.array_ops` [#PR 150](https://github.com/openghg/openghg_inversions/pull/150)

- Fixed "add averaging" functional, which adds the variability of obs over a resampling period to the measurement error (repeatability). This closes [Issue #42](https://github.com/openghg/openghg_inversions/issues/42) . [#PR 144](https://github.com/openghg/openghg_inversions/pull/144)

- Add option to pass the filters as dictionary (with the sites as keys). [#PR 135](https://github.com/openghg/openghg_inversions/pull/135)

- fixed issue with missing obs due to dropping NaNs from other variables in `fp_data` (e.g. `wind_speed`, etc). [#PR 132](https://github.com/openghg/openghg_inversions/pull/132)

- added option `no_model_error` to run inversions without model error (i.e. no min. model error and no pollution event scaling). [#PR 131](https://github.com/openghg/openghg_inversions/pull/131)

- added work-around for error in post-processing caused by the order of the flux dimensions deviating from 'lat', 'lon', 'time'. [#PR 128](https://github.com/openghg/openghg_inversions/pull/128)

- removed `julian2time` function from `convert.py` because it used code that was deprecated by `matplotlib`. This function is still available at `github.com/ACRG-Bristol/acrg/acrg/time/convert.py`. [#PR 129](https://github.com/openghg/openghg_inversions/pull/129)

- `met_model` is now used by `data_processing_surface_notracer`; it is an optional argument, passed as a list with the same length as the number of sites. [#PR 125](https://github.com/openghg/openghg_inversions/pull/125)

- Added option to pass "mean" and "stdev" to lognormal xpriors. Additionally, if `reparameterise_log_normal = True` is added to an ini file, then the 
  log normal prior will be sampled by transforming samples from standard normal random variable to samples from the appropriate log normal distribution. [#PR 107](https://github.com/openghg/openghg_inversions/pull/107)

- Updated `pblh` filter to work with new variable names in footprints. [#PR 101](https://github.com/openghg/openghg_inversions/pull/101)

- NaNs are filled before converting to numpy and passing data to the inversion. This partly addresses [Issue#97](https://github.com/openghg/openghg_inversions/issues/97).  [#PR 101](https://github.com/openghg/openghg_inversions/pull/101)

- add option to calculate an estimate of the minimum model error on the fly [#PR 101](https://github.com/openghg/openghg_inversions/pull/101)

- added documentation, including a "getting started" tutorial, as well as expanding the README file, and updating the example ini files. [#PR 101](https://github.com/openghg/openghg_inversions/pull/101)

- added land/sea mask file needed for `weighted` basis functions, and updated code to retrieve it [#PR 101](https://github.com/openghg/openghg_inversions/pull/101)

- restored option to save raw trace from inversion. [#PR 101](https://github.com/openghg/openghg_inversions/pull/101)

- added option to use Numpyro nuts sampler. [#PR 101](https://github.com/openghg/openghg_inversions/pull/101)

- fix for uncaught error when a filter removes all data from a site. The PBLH filter was also modified to return a value in all cases. [#PR 105](https://github.com/openghg/openghg_inversions/pull/105)

- unpinned OpenGHG (from v0.6.2) and made changes for compatibility with OpenGHG v0.8, which uses zarr as a backend. CI was updated to test against OpenGHG versions 0.7.1, 0.8, and the devel branch. Merged data has been changed from pickle files to either zarr or netCDF (if zarr is not available). [#PR 92](https://github.com/openghg/openghg_inversions/pull/92)

- updates to `hbmcmc_post_process.py`, including changes to `site_info.json` and `species_info.json` to remove dependencies on ACRG paths; updates to documentation; changed `fluxmean` to variable with default `fluxmode`; fixed bug in `set_cmap`, which would fail for datasets with many NaNs; no updates to DIC. [#PR 88](https://github.com/openghg/openghg_inversions/pull/88)

# Version 0.1.3

- reorganised basis functions code into its own submodule `openghg_inversions.basis`. This submodule contains the basis function algorithms, functions to call those algorithms, and the basis function wrapper that was previously in `get_data.py`. [#PR 87](https://github.com/openghg/openghg_inversions/pull/87)

- `combine_datasets` loads data before reindexing, to avoid a performance problem (
`reindex_like` is very slow if dataset not loaded into memory pydata/xarray#8945). Also, the default method has been set to `nearest` instead of `ffill`, since `ffill` tends to create NaNs in the first lat/lon coordinates. [#PR 87](https://github.com/openghg/openghg_inversions/pull/87)

- if the basis functions don't have a "region" dimension (which is the case for all of the basis functions created by our algorithms), then the projection to basis functions is done by creating a sparse matrix that maps from lat/lon to basis regions, and multiplies the footprint by this matrix. This requires the `sparse` package. [#PR 87](https://github.com/openghg/openghg_inversions/pull/87)

- the required version of python has been increased to 3.10. This is because changes in scipy forced changes in arviz, and these change in arviz were implemented at the same time that they increased the required version of python to 3.10. This isn't caught by pip, so we end up with an old version of arviz that is incompatible with the most recent version of scipy. On Blue Pebble, you can use load lang/python/miniconda/3.10.10.cuda-12 instead of load lang/python/anaconda to get Python 3.10 (lang/python/anaconda gives you Python 3.9, even though it says you get Python 3.10) [#PR 87](https://github.com/openghg/openghg_inversions/pull/87)

- Option added to use InTem outer regions for basis functions. This can be selected by using `fixed_outer_basis_regions = True` in an .ini file. [#PR 87](https://github.com/openghg/openghg_inversions/pull/87)

- Refactored basis functions so that they return an xr.Dataset, rather than writing to temporary files. If an output directory is specified, they will save the basis functions as a side effect.

- Added option to run an inversion without boundary conditions. This is specified by adding `use_bc = False` in an .ini file. This assumes that the baseline has already been factored into the observations.

- Added tests to test `get_data.py`, including creating, saving, and loading merged data. Refactored inversions tests to reload merged data, instead of creating merged data. 

# Version 0.1.2

- Bugfix: fixed problem with error handling in `config.version` caused inversions to fail if git wasn't loaded on Blue Pebble. [#PR 91](https://github.com/openghg/openghg_inversions/pull/91)


# Version 0.1.1

- Bug fix: typo (?) from previous merge conflicts resulted in data not being gathered if `use_merged_data` was `True`,
  but no merged data was found.

# Version 0.1

- Formatted code base using `black` with line length 110. Configuration files set up for `black` and `flake8` with line length 110.

- Updated model to scale sampled uncertainty by the size of pollution events, as well as adding an additive minimal model error.

- Separated function to create and load merged data

- Added "bucket basis function", which prevents basis functions from falling on both land and sea (?)

- Added tests that run a full inversion for a small number of iterations

- Added a fix for reading in the correct prior fluxes, when creating the posterior country fluxes and saving everything after the inversion. The prior fluxes are now read directly from the merged data object, and the correct monthly/annual flux is sliced from the full flux object. This includes taking an average flux across a range of months, if the inversion is across multiple months.

- Added a try/except loop which drops sites from the inversion if the data merge process doesn't work for that site (which normally happens if there's no obs).

- Added a print out of the number and % of obs that are removed by each filter, at each site.

- Fixes for saving and reading in the merged data object, including modifying the site variable (and associated heights etc.) if these aren't found in the merged data object.

- Some minor bug fixes, including some in the basis function creation process and some variable names.
