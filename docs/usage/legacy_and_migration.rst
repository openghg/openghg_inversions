Migrate from HBMCMC to RHIME
============================

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

The compatibility product retains familiar variables such as ``Y``,
``Yerror``, ``Ymod``, ``xtrace``, ``bctrace``, ``sigtrace``, ``meanflux``,
``meanscaling``, and country totals. It is a formatting compatibility promise,
not a promise to reproduce the removed executor's exact trace or all its
historical attributes.

Removed Python helper APIs
--------------------------

Code that imported fixed-basis preparation or model helpers should move to
the retained RHIME abstractions:

.. list-table:: Removed helper mappings
   :header-rows: 1
   :widths: 38 30 32

   * - Removed API
     - Current API
     - Migration note
   * - ``prepare_fixedbasis_inversion_data``
     - ``prepare_rhime_inputs``
     - Returns the inputs used by the standard RHIME recipe
   * - ``FixedBasisPreparedData``
     - ``RhimePreparedInputs``
     - Backend-neutral prepared observations and sensitivities
   * - ``basis_functions_wrapper``
     - ``make_basis_functions``
     - Call ``BasisFunctions.sensitivity`` when applying the retained basis;
       full workflows should normally use ``prepare_rhime_inputs``
   * - ``add_inferpymc_likelihood_component``
     - RHIME likelihood selection
     - Configure a built-in likelihood, or pass a ``likelihood_builder`` for a
       custom Python model; there is no direct inferpymc-component replacement

See :doc:`rhime` for preparation and recipe APIs and
:doc:`customising_rhime` for the custom-likelihood boundary.

Removed interfaces
------------------

The following interfaces have no supported current equivalent:

- direct ``fixedbasisMCMC`` and ``inferpymc`` execution;
- ``--legacy-fixedbasis`` and legacy config-template generation;
- legacy debug-return dictionaries and ``rerun_output``;
- direct ``inferpymc_postprocessouts`` calls; and
- plotting and file-analysis helpers from ``hbmcmc_post_process``.

If an archived analysis must execute those interfaces exactly, use a pinned
OpenGHG Inversions 0.7.x environment. Do not use that release line to start a
new workflow. Current RHIME preserves old INI translation and the modern legacy
output adapter, but does not promise bit-for-bit reproduction of the removed
sampling path.

Scientific compatibility
------------------------

The wrapper retains the translated pollution-event likelihood semantics used
by existing old-INI workflows. This is a compatibility boundary inside the
single RHIME implementation, not a second model executor. Migrating a config
to canonical RHIME names should therefore be separated from intentional
changes to priors, likelihood options, basis construction, or output format so
that scientific changes remain reviewable.
