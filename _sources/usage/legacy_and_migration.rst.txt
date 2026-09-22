Legacy interfaces and migration
===============================

The fixedbasis and hierarchical Bayesian Markov chain Monte Carlo (HBMCMC)
interfaces are compatibility paths for existing scripts, configuration files,
and historical outputs. New work should choose a current RHIME recipe and use
its documented Python or command-line entry point.

Chain handling in ``run_hbmcmc.py``
-----------------------------------

The ``run_hbmcmc.py`` compatibility command uses chain 0 for derived products
by default, matching its historical output behavior, and emits a warning when
it does so. Pass ``--all-chains`` to use every retained chain in the modern
derived ``basic`` and PARIS products, and in legacy-format summary fields.
This option cannot be combined with ``--legacy-fixedbasis``, which selects the
untranslated ``fixedbasisMCMC`` / ``inferpymc`` workflow rather than modern
RHIME postprocessing.

Selecting chain 0 affects only the compatibility command's derived product.
When requested with ``save_inversion_output``, the modern inversion-output
artifact retains the full-chain trace. A
legacy-format product keeps chain 0 in its historical trace variables but
calculates convergence and reports its chain-count metadata from the full
trace. Pooling all chains for a summary does not by itself establish that the
chains converged.

The existing getting-started page is retained at its original URL for users who
need its data overview and migration details.

.. toctree::
   :maxdepth: 1

   getting_started
