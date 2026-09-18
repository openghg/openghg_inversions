Experimental code
=================

The :mod:`openghg_inversions.experimental` namespace contains runnable model
variants and integration prototypes that are useful for scientific comparison
and future API design, but are not part of the stable public API.

.. warning::

   Experimental modules may change or be removed without a deprecation period.
   Their inputs and results should be treated as research interfaces rather
   than production contracts. Importing the namespace does not retrieve data or
   start sampling.

Experimental code is kept separate from production model builders so that it
can:

* preserve scientifically useful historical behavior;
* test requirements for planned generic interfaces;
* use current preparation, coordinate, prior, and sampling infrastructure; and
* carry focused tests and validation evidence without changing stable results.

Available experiments
---------------------

The first experiment is the Ramsden et al. (2022) methane/ethane shared-state
model. It ports the paper-shaped two-gas likelihood to the modern RHIME stack
without restoring the obsolete historical loader, sampler, or output system.

.. toctree::
   :maxdepth: 2

   ramsden2022
   ramsden2022_validation
