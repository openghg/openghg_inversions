# OpenGHG Inversions Change Log

# Version 0.0.3

- Formatted code base using `black` with line length 110. Configuration files set up for `black` and `flake8` with line length 110.

- Updated model to scale sampled uncertainty by the size of pollution events, as well as adding an additive minimal model error.

- Separated function to create and load merged data

- Added "bucket basis function", which prevents basis functions from falling on both land and sea (?)

- Added tests that run a full inversion for a small number of iterations

- Added a fix for reading in the correct prior fluxes, when creating the posterior country fluxes and saving everything after the inversion. The prior fluxes are now read directly from the merged data object, and the correct monthly/annual flux is sliced from the full flux object. This includes taking an average flux across a range of months, if the inversion is across mulitple months.

- Added a try/except loop which drops sites from the inversion if the data merge process doesn't work for that site (which normally happens if there's no obs).

- Added a print out of the number and % of obs that are removed by each filter, at each site.

- Fixes for saving and reading in the merged data object, including modifying the site variable (and associated heights etc.) if these aren't found in the merged data object.

- Some minor bug fixes, including some in the basis function creation process and some variable names.
