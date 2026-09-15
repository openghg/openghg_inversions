Standard RHIME model family
===========================

The standard RHIME family contains the complete one-component and multisector
recipes. Both acquire OpenGHG data, prepare labelled inversion inputs, build and
sample a PyMC model, and construct supported outputs. They can also run through
the staged command-line workflow.

Start with the standard tutorial when one flux component and one scaling state
fit the scientific question. Use the multisector tutorial when named sources
must retain separate states and diagnostics. Species selection does not choose
between these recipes.

.. toctree::
   :maxdepth: 1

   rhime_standard_tutorial
   rhime_multisector_tutorial
   concrete_rhime_model
   rhime
   cli
   customising_rhime
