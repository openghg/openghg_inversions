.. openghg_inversions documentation master file, created by
   sphinx-quickstart on Thu Sep 25 09:28:38 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpenGHG Inversions Documentation
================================

..
   Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.

OpenGHG Inversions is a Python package that is being developed as part of the [OpenGHG project](https://openghg.org) with the aim of merging the data-processing and simulation modelling capabilities of OpenGHG with the atmospheric Bayesian inverse models developed by the Atmospheric Chemistry Research Group (ACRG) at the University of Bristol, UK.

OpenGHG Inversions uses RHIME for regional inversion workflows. Existing
fixedbasis-style INI files may use a transitional wrapper which translates
them to RHIME; the direct hierarchical Bayesian Markov chain Monte Carlo
(HBMCMC) implementation has been removed.

RHIME provides standard single-flux and multisector recipes with complete
acquisition-to-output runners. Its advanced CO₂ model family provides CO₂-only
and linked CO₂/O₂ recipes at prepared-input boundaries. Start with
:doc:`the model recipe chooser <usage/model_recipes>` to select a recipe by
model topology and workflow requirements rather than by gas name alone.

Releases are tagged with a `DOI <https://doi.org/10.5281/zenodo.10650595>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage/usage
   development/index
   experimental/index
   reference/index
