# OpenGHG Inversions Change Log

# Unreleased

## Code changes

- Made direct composition of concrete standard and multisector RHIME models
  the default builder strategy. The semantic flux-plan compiler remains
  available as an explicit `builder_strategy="compiled"` opt-in on
  `RhimeModelSpec` and in RHIME configuration, while both paths share the same
  source selection, gathered ragged-state handling, and sector-prior
  validation. The concrete builders are the readable reference
  implementations; compiler internals remain private, and unchanged graph
  components are continuously checked against that reference contract.
- Added a tox PyTensor compiler preflight that automatically loads
  `gcc/12.3.0-sknc` on Rocky Linux or recognized Blue Pebble hosts when the
  compiler setting is empty, supports configurable module/compiler overrides,
  and fails before pytest when `pytensor.config.cxx` remains empty instead of
  allowing extremely slow C++-free PyMC test runs.
- Added a shared RHIME flux-plan/compiler seam for standard and multisector
  builders, and routed explicit sector-to-source mappings plus complete
  per-sector priors through multisector preparation and model specifications.
  Source-specific ragged state blocks remain gathered over
  `(source, region_in_source)`, scalar source provenance remains single-sector,
  and rectangular multisource adaptation is confined to the legacy
  `fixedbasisMCMC` boundary.
  [#402](https://github.com/openghg/openghg_inversions/issues/402),
  [#403](https://github.com/openghg/openghg_inversions/issues/403),
  [PR #529](https://github.com/openghg/openghg_inversions/pull/529)

- Reset retained posterior draw labels after burn-in before attaching predictive
  groups in both modern RHIME and fixed-basis sampling, and preserve the
  discarded burn count through trace and `InversionOutput` round trips.
  Trace-group merging still explicitly retains outer alignment for genuinely
  unequal external groups, while multisector totals require a value from every
  sector so padded draws cannot be interpreted as zero flux. Single- and
  multisector PARIS country samples are now promoted to float64 before totals
  and uncertainty statistics are calculated, then cast at the template
  boundary, keeping posterior stdev and covariance calculations consistent.

- Added versioned NetCDF and Zarr persistence for ``RhimePreparedInputs``,
  including CF compression-by-gathering for canonical MultiIndex inversion
  inputs, labeled site metadata decoded by integer site indicators, and the
  retained operator-backed basis and reference flux. Static multisource bases
  now use an ordered xarray ``source`` coordinate; basis provenance remains
  owned by ``BasisFunctions``. Site indicators are regenerated from labeled
  measurement sites, avoiding a second user-maintained source of site truth.
  Repeated Zarr saves replace the previous artifact rather than retaining stale
  groups.
  Generic DataTree, InferenceData, and MultiIndex serialization helpers now
  have shared ownership outside postprocessing.

- Added `run_rhime_from_prepared_inputs` so modern standard and multisector
  RHIME models can run from an existing `RhimePreparedInputs` object without
  repeating OpenGHG-backed data preparation. Existing `run_rhime` entry points
  now share the same post-preparation execution path.
  [#509](https://github.com/openghg/openghg_inversions/issues/509)

- Made retained `BasisFunctions` / `BasisOperator` metadata the primary basis
  contract for RHIME preparation and modern postprocessing outputs. Derived
  flux, country, PARIS, and legacy-format products now record stable basis
  reconstruction metadata, retained basis artifacts record loaded/saved paths,
  and source-specific multisector flux reconstruction no longer reaches through
  the legacy flat-basis view. Legacy flat basis artifacts remain readable as an
  explicit compatibility fallback but are deprecated for new workflows.
  [#429](https://github.com/openghg/openghg_inversions/issues/429)
- Added a modern `output_format="legacy"` compatibility product, routed deprecated
  `hbmcmc` / `hbmcmc_postprocessing` output requests to it, and made
  `run_hbmcmc.py` translate fixedbasis-style configs into `run_rhime` calls while
  preserving legacy output filenames. The shim now validates translated arguments
  before copying configs, translates deprecated `calculate_min_error` and
  `reparameterise_log_normal` options where possible, old HBMCMC output attrs
  are produced from modern `InversionOutput`, legacy KDE mode statistics now
  handle all-NaN and partially-NaN rows without dropping every draw, derived
  RHIME products no longer save large `InversionOutput` sidecars unless
  `save_inversion_output` is requested, and user docs mark historical
  `fixedbasisMCMC` behavior as available from release 0.6 or earlier.
  Country-file loading in modern country and legacy-format
  postprocessing now falls back to direct HDF5 reads when h5netcdf dimension-scale
  decoding fails on cluster nodes, and floating legacy-format output variables
  are written as `float32` to avoid footprint-alignment upcasts. Modern PARIS
  compatibility outputs also cast floating data variables to `float32` to match
  the historical fixedbasis-style file contract. RHIME and the `run_hbmcmc.py`
  compatibility shim now emit grep-friendly `TIMING ... seconds=... maxrss_kb=...`
  lines for setup, preparation, sampling, sampler statistics, postprocessing,
  and output writes so batch logs can identify runtime regressions. Modern
  RHIME model imports also apply the same PyTensor `floatX=float32` default as
  the historical fixedbasis PyMC path, avoiding accidental float64 sampling
  after the `run_hbmcmc.py` route switch.
  [#416](https://github.com/openghg/openghg_inversions/issues/416)
- Routed modern RHIME and fixedbasis postprocessing through modern `InversionOutput` semantics, retained `BasisFunctions` / `BasisOperator` products, variable-role lookups, and product-local capability checks; removed the transitional postprocessing protocol/view layer and deleted `LegacyInversionOutput` plus the dead legacy inversion-output builder helpers. [#383](https://github.com/openghg/openghg_inversions/issues/383)
- Migrated standard RHIME `basic` and `paris` postprocessing toward modern `InversionOutput` as an intermediate step before the final #383 product-local postprocessing contract. [#435](https://github.com/openghg/openghg_inversions/issues/435)
- Introduced the temporary modern/legacy output split and modern `InversionOutput` serialization; the transitional `LegacyInversionOutput` carrier was removed by #383. [#401](https://github.com/openghg/openghg_inversions/issues/401)
- Moved public RHIME model-builder exports into `openghg_inversions.models` and shared data preparation between `fixedbasisMCMC`, `run_rhime`, and `run_rhime_multisector`. [#399](https://github.com/openghg/openghg_inversions/issues/399), [#425](https://github.com/openghg/openghg_inversions/issues/425)
- Retained `BasisFunctions` objects through shared inversion preparation, RHIME results, and opt-in `fixedbasisMCMC` debug output; DataTree basis artifacts are loaded when available while legacy flat basis artifacts remain supported. [#428](https://github.com/openghg/openghg_inversions/issues/428)
- Added modern `run_rhime` and shared-basis `run_rhime_multisector` pipelines, RHIME CLI entry points, RHIME config template, modern result/spec objects, and focused tests for the new public runners. [#398](https://github.com/openghg/openghg_inversions/issues/398)
- Made concat-gather handling of mismatched site data variables order-independent, added an opt-in drop policy used by `make_inv_inputs`, and added lightweight regression tests for issue #394. [#394](https://github.com/openghg/openghg_inversions/issues/394)
- Fix bug which was assigninig the wrong times to inversion flux outputs in non-standard cases, such as 3-monthly inversions. [#PR 387](https://github.com/openghg/openghg_inversions/pull/387)
- Fix small bug where postprocessing was failing if country codes in file didn't match exactly those in `paris_regions_dict`. [#PR 377](https://github.com/openghg/openghg_inversions/pull/377)
- More flexibility for new inversion domains. [#PR 333](https://github.com/openghg/openghg_inversions/pull/333)
- More flexibility for types of boundary condition basis functions. [#PR 333](https://github.com/openghg/openghg_inversions/pull/333)
- Bug fix for quadtree algorithm. [#PR 333](https://github.com/openghg/openghg_inversions/pull/333)
- Fixed minor bugs in code for storing merged data. Added option to change merged data format by including an extension in `merged_data_name`, e.g. `merged_data_name="my_merged_data.nc"` will save to netCDF, while `merged_data_name="my_merged_data"` will save to zipped Zarr. [#PR 345](https://github.com/openghg/openghg_inversions/pull/345) 
- Added the ability to process TCCON data, along with additional output variables `obs_prior_factor` and `obs_prior_upper_level_factor`. [PR #327](https://github.com/openghg/openghg_inversions/pull/327)
- Fixed bug introduced by PR 327, which caused "prior factor" variables filled with None values to be passed to post-processing. [PR #353](https://github.com/openghg/openghg_inversions/pull/353)
- Added new `inversion_inputs.py` with helper functions for creating the inputs to the PyMC code. Tests added to check compatibility with older `inversionsetup.py` helpers. [PR #356](https://github.com/openghg/openghg_inversions/pull/356)
- Added `BasisFunctions` object (backed by `BasisOperator` objects) and tests to confirm that these preserve existing behaviour when computing H matrices. These classes will be used to refactor the basis functions wrapper in a future PR. [PR #358](https://github.com/openghg/openghg_inversions/pull/358)
- Fixed issue that raised `IndexError` in `inferpymc` when a monthly of data was missing an `sigam_freq` is "monthly". (Code accidentally merged into devel instea of PR, so this is a placeholder PR)[PR #365](https://github.com/openghg/openghg_inversions/pull/365)
- Added opt-in `basis_functions_wrapper` support for returning `BasisFunctions` objects and saving basis artifacts in DataTree format while keeping legacy flat-basis output as the default. [PR #367](https://github.com/openghg/openghg_inversions/pull/367)
- Stage A of PyMC model refactor. Added regression tests for `inferpymc` and extracted function to build the PyMC model. [PR #378](https://github.com/openghg/openghg_inversions/pull/378)
- Stage B of PyMC model refactor. Updated `inferpymc` to accept current/legacy inputs as well as xarray `Dataset`. [PR #380](https://github.com/openghg/openghg_inversions/pull/380)
- Stage C of PyMC model refactor. Added function for building PyMC "model components". The model building code from Stage B is still used by default, but the new code can be selected by adding `model_builder="components"` to the .ini file. [PR #382](https://github.com/openghg/openghg_inversions/pull/382)
- Stage D of PyMC model refactor. Removed temporary scaffolding to preserve legacy model building code. `inferpymc` now only accepts inversion inputs as `xr.Dataset`, and `fixedbasisMCMC` has been updated to reflect this. [PR #389](https://github.com/openghg/openghg_inversions/pull/389)
- Neutral refactor of `fixedbasisMCMC` output handling to make the end-of-run logic clearer, whichis now split into explicit stages for artefact creation, `InversionOutput` construction, and output mode dispatch. [PR #390](https://github.com/openghg/openghg_inversions/pull/390)
- Stage E follow-up PyMC refactor tidy-up. `inferpymc` is now a thinner compatibility wrapper over model building, modern `InferenceData` sampling, and explicit legacy adaptation; legacy trace renaming moved out of model construction; `InversionOutput` no longer carries a PyMC model; and the current latent/step compatibility logic is more clearly isolated ahead of a future modern run-inversion path. [PR #391](https://github.com/openghg/openghg_inversions/pull/391)

# Version 0.6.0

## Model changes

- Fixed issue due to `met_model` being converted from `None` to `not_set` [#PR 341](https://github.com/openghg/openghg_inversions/pull/341)

## Code changes

- Added new release process using `uv`.

# Version 0.5.0

## Model updates
- Passing `platform="flask"` for obs data (same format as e.g. inlet or averaging period) will 1) prevent the data from being resampled, and 2) tell ModelScenario to align the footprints to the obs without resampling. If you want to resample flask data, use `platform=None`. [#PR 322](https://github.com/openghg/openghg_inversions/pull/322)

## Code changes

- Optimisations for speeding up data processing (also some improvements to memory usage during MCMC and postprocessing). [#PR 311](https://github.com/openghg/openghg_inversions/pull/311)
- Removed duplicate code for computing "fp x flux" and "bc sensitivity" matrices. This is done using `ModelScenario` now. This also means that units are aligned using `pint`. [#PR 305](https://github.com/openghg/openghg_inversions/pull/305)
- Removed threshold for filling missing obs. error. [#PR 306](https://github.com/openghg/openghg_inversions/pull/306)

# Version 0.4.0

## Model updates

- Offsets can be applied and solved for on a monthly basis, as well as for the entire inversion period (ini option `offset_args = {"offset_freq": "M"}` for monthly, although other frequencies can be passed). 

- Offsets can be applied to all but one site (ini option `offset_args = {"drop_first": True}`) or to all sites, which is the default option (ini option `offset_args = {"drop_first": False}`). [#PR 285](https://github.com/openghg/openghg_inversions/pull/285)

- Updated RHIME likelihood to use a power of 1.99 instead of 2. The power can be specified with the `power` argument an ini file. The value of `power` can be a float or a dict of prior args, which will create a hyperprior for `power`. [#PR 277](https://github.com/openghg/openghg_inversions/pull/277)

- Can now specify kwargs to pass to the sampler as a dictionary called sampler_kwargs in the .ini file (e.g. sampler_kwargs = {"target_accept": 0.99})

## Code changes

- Added offset to PARIS concentration outputs. [#PR 282](https://github.com/openghg/openghg_inversions/pull/282)
- Compression added for output PARIS netcdf files. Standard RHIME output now shuffles to save space.
- Fixed warning messages for zeros/NaNs in `mf_error`. [#PR 292](https://github.com/openghg/openghg_inversions/pull/292)
- `get_flux_data` tries to infer the "time period" of the flux, which is used to set the time offset for PARIS flux outputs. [#PR 302](https://github.com/openghg/openghg_inversions/pull/302)

# Version 0.3.0

- Fixed bug due to wrong BC units. [#PR 249](https://github.com/openghg/openghg_inversions/pull/249)

- Merged funactionality of `min_error` and `calculate_min_error` into a single variable (`min_error`). [#PR 240](https://github.com/openghg/openghg_inversions/pull/240)

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
