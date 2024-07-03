# OpenGHG Inversions Change Log

# Version 0.2.0

- Removed `site_info.json` and `species_info.json` and replaced with calls to functions in `openghg.util`, which pull the same info from `openghg_defs`. [#PR 152](https://github.com/openghg/openghg_inversions/pull/152) 

- Removed unused functions from `convert.py` and updated docstrings. [#PR 151](https://github.com/openghg/openghg_inversions/pull/151) 

- Added new option for computing min. model error based on percentiles. [#PR 142](https://github.com/openghg/openghg_inversions/pull/142) 

- Update the docstrings of openghg_inversions.basis and openghg_inversions.array_ops [#PR 150](https://github.com/openghg/openghg_inversions/pull/150)


- Fixed "add averaging" functional, which adds the variability of obs over a resampling period to the measurement error (repeatability). This closes [Issue #42](https://github.com/openghg/openghg_inversions/issues/42) . [#PR 144](https://github.com/openghg/openghg_inversions/pull/144)

- Add option to pass the filters as dictionary (with the sites as keys). [#PR 135](https://github.com/openghg/openghg_inversions/pull/135)

- fixed issue with missing obs due to dropping NaNs from other variables in `fp_data` (e.g. `wind_speed`, etc). [#PR 132](https://github.com/openghg/openghg_inversions/pull/132)

- added option `no_model_error` to run inversions without model error (i.e. no min. model error and no pollution event scaling). [#PR 131](https://github.com/openghg/openghg_inversions/pull/131)

- added work-around for error in post-processing caused by the order of the flux dimensions deviating from 'lat', 'lon', 'time'. [#PR 128](https://github.com/openghg/openghg_inversions/pull/128)

- removed `julian2time` function from `convert.py` because it used code that was deprecated by `matplotlib`. This function is still available at `github.com/ACRG-Bristol/acrg/acrg/time/convert.py`. [#PR 129](https://github.com/openghg/openghg_inversions/pull/129)

- `met_model` is now used by `data_processing_surface_notracer`; it is an optional argument, passed as a list with the same length as the number of sites. [#PR 125](https://github.com/openghg/openghg_inversions/pull/125)

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
