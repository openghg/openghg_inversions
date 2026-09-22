Migrate from HBMCMC to RHIME
============================

RHIME is the only inversion implementation in current OpenGHG Inversions.
The direct ``fixedbasisMCMC`` and ``inferpymc`` implementation was removed in
0.8 after its remaining known active user agreed to migrate. The 0.7.x release
line is the last line containing that implementation and the
``--legacy-fixedbasis`` option. The ``openghg_inversions.hbmcmc`` namespace
remains to host the transitional ``run_hbmcmc`` compatibility wrapper.

This page is for users with fixedbasis-style Python calls, INI files, batch
scripts, or HBMCMC outputs. For a new inversion, start with the
:doc:`model recipe chooser <model_recipes>` and :doc:`RHIME guide <rhime>`.

Chain handling in ``run_hbmcmc.py``
-----------------------------------

The compatibility wrapper uses chain 0 for derived products by default,
matching its historical output behaviour, and emits a warning when it does so.
Pass ``--all-chains`` to use every retained chain in modern ``basic`` and PARIS
products and in legacy-format summary fields. The wrapper does not save the
modern inversion-output artifact by default. When requested with
``save_inversion_output``, that artifact retains the full-chain trace in either
case. Pooling all chains for a summary does not by itself establish that the
chains converged.

Choose a migration route
------------------------

Existing INI and batch workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Existing supported fixedbasis-style INI files may continue to use the
transitional wrapper::

   python -m openghg_inversions.hbmcmc.run_hbmcmc \
       2019-01-01 2019-02-01 -c my_inversion.ini

The wrapper translates the old vocabulary, copies the effective configuration
for provenance, and calls ``run_rhime``. It does not run the removed HBMCMC
implementation. ``--legacy-fixedbasis`` and ``--generate`` are no longer
accepted. Keep existing configuration files while migrating, but create new
ones from ``openghg_inversions/config/templates/rhime_template.ini`` and run
them with ``openghg-inversions run-rhime``.

Batch scripts need only replace a direct source-tree path with the module form
above. Start and end dates, ``-c``, ``--output-path``, ``--kwargs``, and
``--all-chains`` remain available. The JSON supplied to ``--kwargs`` must be
quoted as one shell argument, for example::

   python -m openghg_inversions.hbmcmc.run_hbmcmc \
       2019-01-01 2019-02-01 -c my_inversion.ini \
       --kwargs '{"min_error": 20.0, "nuts_sampler": "numpyro"}'

Python workflows
~~~~~~~~~~~~~~~~

Replace ``fixedbasisMCMC(...)`` with ``run_rhime(...)`` and use the names in
the table below::

   from openghg_inversions.rhime import run_rhime

   result = run_rhime(
       species="ch4",
       sites=["TAC"],
       averaging_period=["1h"],
       domain="EUROPE",
       start_date="2019-01-01",
       end_date="2019-02-01",
       flux_sources=["total-ukghg-edgar7"],
       draws=5_000,
       chains=4,
       output_path="outputs",
       output_name="ch4_TAC",
   )

``run_rhime`` returns a :class:`~openghg_inversions.rhime.RhimeResult`, not a
legacy tuple or sampler dictionary. Its principal attributes are ``idata``
(ArviZ ``InferenceData``), ``inv_inputs`` (the labelled model inputs),
``inv_out`` (the modern ``InversionOutput`` when constructed), and ``outputs``
(requested derived products).

Parameter mapping
-----------------

Use canonical RHIME names in new Python and configuration files. The wrapper
continues to translate the common old names.

.. list-table:: Fixedbasis-to-RHIME names
   :header-rows: 1
   :widths: 28 24 48

   * - Fixedbasis name
     - RHIME name
     - Migration note
   * - ``nit``
     - ``draws``
     - Number of retained draws
   * - ``nchain``
     - ``chains``
     - Number of sampler chains
   * - ``verbose``
     - ``progressbar``
     - Whether to show sampler progress
   * - ``sampler_kwargs``
     - ``sample_kwargs``
     - Keyword mapping passed to the sampler
   * - ``outputpath``
     - ``output_path``
     - Output directory
   * - ``outputname``
     - ``output_name``
     - Run/output stem
   * - ``xprior``
     - ``x_prior``
     - Flux-scaling prior mapping
   * - ``bcprior``
     - ``bc_prior``
     - Boundary-scaling prior mapping
   * - ``sigprior``
     - ``sigma_prior``
     - Model-error prior mapping
   * - ``offsetprior``
     - ``offset_prior``
     - Additive-offset prior mapping
   * - ``emissions_name``
     - ``flux_sources``
     - OpenGHG flux ``source`` values
   * - ``hbmcmc`` or ``hbmcmc_postprocessing``
     - ``legacy``
     - Modern HBMCMC-compatible NetCDF output
   * - ``calculate_min_error``
     - ``min_error``
     - Use ``"residual"`` or ``"percentile"``
   * - ``reparameterise_log_normal``
     - prior ``reparameterise``
     - Put ``reparameterise = true`` in each relevant lognormal prior mapping

The wrapper retains the historical output filename convention. Direct
``run_rhime`` calls use the RHIME filename convention unless explicitly
configured otherwise.

Legacy-format output
--------------------

``output_format="legacy"`` remains supported for standard single-sector
RHIME. It creates the HBMCMC-compatible NetCDF product from the modern
``InversionOutput``; it does not invoke ``fixedbasisMCMC`` or ``inferpymc``.
The deprecated output names ``hbmcmc`` and ``hbmcmc_postprocessing`` remain
aliases for ``legacy``.

The compatibility product uses variables such as ``Yobs``, ``Yerror``,
``Ymodmean``, ``Ymodmedian``, ``Ymodmode``, ``xtrace``, ``sigtrace``,
``fluxmode``, ``scalingmean``, ``scalingmode``, and country totals. When
boundary conditions are enabled, it also includes ``bctrace`` and the
``YmodmeanBC``, ``YmodmedianBC``, and ``YmodmodeBC`` summaries. This is a
formatting compatibility promise, not a promise to reproduce the removed
executor's exact trace or all its historical attributes.

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
