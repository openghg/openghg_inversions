Released 0.7 migration evidence
===============================

This page records historical evidence for `issue #960`_, part of `issue #944`_
and `issue #948`_. It is an evidence record, not a promised converter interface.
The comparison revision is compatibility-bearing devel
``0a32d73a5c4a72be0505a4ed682472e1613e49e9``. The parent migration ledger
records current-source cases and follow-up owners. A removal PR must refresh
evidence for affected files changed after that revision.

Pinned releases and publication evidence
----------------------------------------

The GitHub releases collection was queried on 22 October 2026. Each release
below has ``draft=false``, ``prerelease=false`` and the stated publication time;
local tag commits agree with the release's ``target_commitish``. These are
GitHub publication facts. This capture does not establish PyPI artifact identity
or reproduce an original publisher's environment.

.. list-table:: Published 0.7 GitHub releases
   :header-rows: 1
   :widths: 12 57 31

   * - Tag
     - Commit
     - Published (UTC)
   * - ``v0.7.0``
     - ``0a06f3d23e95f0d3548706887070694b64e19d82``
     - 2026-09-18 14:25:51
   * - ``v0.7.1``
     - ``1b86fd88e82d632e98a2e245c4a8444b9bdcba81``
     - 2026-09-18 18:09:19
   * - ``v0.7.2``
     - ``8e3749ca230604331e62d293b305b52b81877185``
     - 2026-09-21 08:45:13
   * - ``v0.7.3``
     - ``9c0d23c15000dc7a7e4c3306d3f1bfd4816bdb4f``
     - 2026-09-25 15:09:56
   * - ``v0.7.4``
     - ``0e5d6ef76eb724da8cee1b780633578bdc05f2b4``
     - 2026-10-01 09:47:31

The latest published baseline used for execution is ``v0.7.4``. Main at
``9a70c8753a0f8a50e50d91d0468ec4351a3228fe`` contains prepared version
``0.7.5``; the releases collection contains no published ``v0.7.5``. A version
string and assembled changelog therefore do not establish publication. See the
`releases collection <https://api.github.com/repos/openghg/openghg_inversions/releases?per_page=100>`_
and `published v0.7.4 <https://github.com/openghg/openghg_inversions/releases/tag/v0.7.4>`_.
PyPI JSON verification was unavailable here: the restricted request could not
resolve the host; the permitted network request failed certificate verification.
No certificate verification was disabled.

Source comparison found no changes from ``v0.7.0`` to ``v0.7.4`` in the
characterized CLI, INI decoder, RHIME parameters/standard runner/sampler,
HBMCMC wrapper/inferpymc, pollution/fixed-error implementation, or
``postprocessing/inversion_output.py``. This is inspection evidence about those
files; it is not an execution claim for every patch release or their outputs.

Observed checks and environment
-------------------------------

The fixtures live in ``tests/fixtures/migration_960/historical/``. The
hand-written ``expected.json`` is independent of the translator: numeric
defaults were read from function signatures and setup bodies, error strings
from explicit raises, and the scientific witnesses use elementary arithmetic.
``source_hashes.json`` pins twelve characterized source files to the exact tag.
The harness verifies those hashes before importing historical runtime modules;
the package version alone is insufficient because devel can retain that version.

On 22 October 2026, ``characterize.py`` passed using an isolated tag archive at
``/tmp/migration-960-v074`` with uv 0.8.13, CPython 3.12.11, and
``uv sync --frozen --no-default-groups --python 3.12``. Its released ``uv.lock``
SHA256 is
``d8f79c3d6d048343fb0b5687f430e819424a4153a5f0a0e65bcb27a34e6cf962``.
That lockfile, retained at the pinned Git commit, supplies the full dependency
resolution; 127 packages were installed, including:

.. list-table:: Selected installed historical dependencies
   :header-rows: 1

   * - Package
     - Version
   * - OpenGHG / OpenGHG Inversions
     - 0.16.0 / 0.7.4
   * - NumPy / SciPy / pandas
     - 2.2.6 / 1.17.0 / 2.3.3
   * - PyMC / PyTensor / ArviZ
     - 5.26.1 / 2.35.1 / 0.23.1
   * - xarray / Dask
     - 2026.1.0 / 2026.1.1
   * - Numba / sparse
     - 0.63.1 / 0.17.0

This is a newly reproduced locked environment, not a recovered user's original
environment. Modern production dependencies were unchanged. The harness
suppresses OpenGHG's unconditional home-directory file logger during import,
then restores normal logging; it does not change ``HOME``. The actual installed
executable's separate validation check used its ordinary logger.
The bounded harness also passed under the normal cluster user identity after
a sandbox-only import failure: ArviZ's daily-warning code tried to create a
directory under the sandbox identity's nonexistent home. This was an import
environment limitation, not a scientific failure; no home variable was changed.

The executed checks cover direct Python ``run_rhime``, CLI dispatch through
``cli.main``, ordinary wrapper dispatch, and explicit true fixedbasis dispatch.
All stop at the first acquisition boundary, or replace the final fixedbasis
call with a capture. Config-copy output is suppressed. No retrieval, scientific
graph evaluation, inference, or data acquisition was performed. A synthetic
trace replaces ``pm.sample`` to exercise the actual RHIME burn-slicing body;
another synthetic trace executes released chain-zero selection.

The installed ``.venv/bin/openghg-inversions run-rhime --help`` succeeded.
The actual installed executable, with ``rhime.ini`` and
``--kwargs '{"mismatch_model": null}'``, exited 1 with
``ValueError: A standard RHIME run requires a built-in or custom likelihood.``
at ``standard.py:528``, before retrieval. Thus the installed surface was checked
in addition to captured Python dispatch. There is no standalone installed
``run-rhime`` executable: ``run-rhime`` is an ``openghg-inversions`` subcommand.

To rerun from an evidence checkout, choose unused temporary directories and
retain the checkout path in ``migration_evidence_repo`` before changing directory:

.. code-block:: bash

   migration_evidence_repo="$PWD"
   migration_historical_dir=$(mktemp -d /tmp/migration-960-v074.XXXXXX)
   git archive 0e5d6ef76eb724da8cee1b780633578bdc05f2b4 |
       tar -x -C "$migration_historical_dir"
   cd "$migration_historical_dir"
   uv sync --frozen --no-default-groups --python 3.12
   export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
   export NUMBA_CACHE_DIR="$migration_historical_dir/numba-cache"
   export MPLCONFIGDIR="$migration_historical_dir/matplotlib-cache"
   # Preserve scientific/compiler flags, replacing only the cache location.
   migration_pytensor_flags=""
   IFS=',' read -r -a migration_inherited_flags <<< "${PYTENSOR_FLAGS:-}"
   for migration_flag in "${migration_inherited_flags[@]}"; do
       case "$migration_flag" in
           base_compiledir=*|"") ;;
           *) migration_pytensor_flags="${migration_pytensor_flags:+$migration_pytensor_flags,}$migration_flag" ;;
       esac
   done
   export PYTENSOR_FLAGS="${migration_pytensor_flags:+$migration_pytensor_flags,}base_compiledir=$migration_historical_dir/pytensor-cache"
   PYTHONPATH="$migration_historical_dir" .venv/bin/python \
       "$migration_evidence_repo/tests/fixtures/migration_960/historical/characterize.py"
   .venv/bin/openghg-inversions run-rhime --help

These are bounded checks in an isolated uv environment, not environment-creating
tox. Any later tox rerun that creates environments must follow the repository's
Slurm procedure. Do not use the example inference invocations in
``invocations.txt`` as a substitute for this bounded harness.

Source facts and scientific boundaries
--------------------------------------

The following are source findings unless explicitly described above as observed.
Their candidate replacements are decisions for `issue #949`_ / `issue #948`_;
`issue #719`_ owns preserving relevant scientific regressions during removal.

* **H1: Routes and omitted defaults.** Released Python/installed RHIME uses
  draws/burn/tune/chains ``1000/0/1000/4`` and requires an explicit built-in or
  custom likelihood. The ordinary wrapper chooses pollution-event mismatch
  itself, uses the same sampler defaults when omitted, and selects legacy
  filenames/output. True ``--legacy-fixedbasis`` dispatches untranslated legacy
  parameters and defaults to nit/burn/tune/nchain
  ``250000/50000/125000/2``. Its x prior is truncated-normal sigma 1 and BC prior
  truncated-normal sigma 0.1; direct RHIME defaults instead to a reparameterized
  lognormal x prior and BC sigma 0.05. The true fixedbasis sigma prior is uniform
  ``[0.1, 3]``; RHIME pollution-event's unresolved ``sigma_prior=None`` becomes
  uniform ``[0, 0.1]`` at model construction. ``None`` in the observed JSON is
  the resolved configuration value, not a claim that no sigma prior exists.
  Candidate conversion must materialize route-specific defaults explicitly.
  Disposition: mechanical settings capture, with supported source coverage
  unresolved in #948; #949 owns conversion of those explicit defaults.
  Shared omitted defaults are ``use_bc=True``, ``add_offset=False``, offset
  normal(0,1), ``sigma_per_site=True``, ``sigma_freq=None`` (one period),
  ``pollution_events_from_obs=False``, power 1.99, and ``min_error=0.0``.
  RHIME's resolved ``sigma_freq_anchor`` is the run start date, even when the
  one-period choice makes it immaterial. The lower-level direct ``inferpymc``
  API is another inspected signature: nit/burn/tune/nchain
  ``20000/10000/10000/4``, lognormal x, BC sigma 0.05, and uniform sigma
  ``[0.1,3]``. It must not supply defaults for an omitted fixedbasisMCMC call.

* **H2: Sampling meaning.** ``inferpymc`` passes ``draws=nit`` to its sampler;
  both released sampling bodies call ``pm.sample`` before slicing away ``burn``
  posterior draws. Tune remains separate. The captured explicit wrapper request
  nit 7, burn 2, tune 3 means five retained draws per chain, not seven plus burn,
  and not five minus tuning. This is a mechanical spelling change with preserved
  meaning. Only RHIME slicing was executed using a mocked sampling boundary;
  the true inferpymc body was inspected.
  Candidate: explicit ``draws``, ``burn`` and ``tune``; disposition: mechanical.
  #949 owns conversion and #952 owns later numerical validation.

* **H3: Pollution-event offsets.** With observed pollution scaling, the historical
  compatibility runner supplies only ``boundary_mean`` as the baseline. Modern
  construction includes the offset in the baseline. For Y=20, BC=10, offset=3,
  the old enhancement is 10, modern 7. Without BC, the old baseline is absent:
  it uses ``abs(Y) + 1e-6*mean(Y)`` (20.00002), while modern offset-aware scaling
  gives 17. This is a scientific change, not a mechanical rename. The likelihood
  otherwise forms ``sqrt(error**2 + (pollution_event*sigma)**power)`` with a
  minimum-error floor. The witnesses were derived from source, not evaluated as
  PyMC graphs.
  Candidate: explicitly choose the modern full-baseline pollution-event model
  or retain the original environment. Disposition: scientific change;
  #948 owns the choice, #949 its report, and #719 the affected regressions.

* **H4: Fixed-error exceptions.** Historical pollution-event ``no_model_error``
  retains a disconnected sigma variable and uses
  ``max(abs(error), 1e-12*nanmean(Y))`` without the prepared minimum-error floor.
  For Y=[20,20], error=[0,2], min_error=[5,5], epsilon=[2e-11,2]. Its wrapper's
  additive-sigma/no-model-error route separately preserves the minimum-error
  floor: error 2 and minimum 5 gives epsilon 5. Modern fixed error alone gives
  2. Choosing which route to preserve requires explicit scientific review;
  restoring disconnected historical sigma in modern models is not proposed.
  Candidate: modern ``mismatch_model='fixed_error'`` after an explicit choice,
  or original 0.7 execution. Disposition: scientific change / unresolved
  supported subset; #948/#949 own selection/reporting and #719 regressions.

* **H5: Chain summaries.** Released ``convert_idata_to_dataset`` discards all but
  chain zero, and true inferpymc's result adapter also selects chain zero before
  generating its arrays. Raw InferenceData can still contain every chain.
  The observed synthetic released conversion gives mean 2 for chains [1,3] and
  [9,11]. Current ``merge_trace_groups`` retains chains and stats combines
  chain/draw at reduction, giving mean 6. Output consumers must acknowledge
  this scientific change; the evidence does not reinstate a modern chain policy
  switch.
  The current test ``test_modern_summary_pools_the_archived_chain_witness``
  executes the modern reduction on the same plain witness. Candidate: modern
  all-chain summaries; disposition: scientific change. #948/#949 own the
  acknowledgement/report and #719 owns preservation of summary regressions.

* **H6: Output and Python consumers.** Direct released RHIME returns ``RhimeResult``
  with ArviZ ``idata``, model/specs, outputs and optional ``inv_out``. Ordinary
  wrapper ``main`` discards that return. True fixedbasis has format-dependent
  Dataset/dictionary/InversionOutput returns. Current full RHIME returns an
  ``InversionOutput`` with a DataTree trace, so a consumer such as
  ``result.idata.posterior`` needs an explicit rewrite. Released direct output
  defaults to ``inv_out`` with saving enabled; the wrapper defaults to
  ``legacy`` with ``save_inversion_output=False``. Trace saving defaults false.
  A true/path save setting can request writing independently of a derived
  output mode. Derived filename conventions are
  ``evidence_ch4_EUROPE_2019-01-01.nc`` for direct RHIME and
  ``CH4_EUROPE_evidence_2019-01-01.nc`` for the wrapper. Default trace and
  inversion-output filenames respectively use
  ``evidence2019-01-01_trace.nc`` and
  ``evidence2019-01-01_inversion_output.nc``. These filename/return bodies were
  inspected; no full output serialization or result-construction run occurred.
  Candidate: explicit output/save/filename settings and modern
  ``InversionOutput`` consumers. Disposition: mechanical settings with
  unresolved consumer automation; #945 owns the API, #949 output conversion,
  and #950 Python/batch launch adaptation.

Pinned source pointers
----------------------

At commit ``0e5d6ef76eb724da8cee1b780633578bdc05f2b4``:

* `RHIME setup/defaults <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/rhime/params.py#L713>`_
  and `priors <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/rhime/specs.py#L20>`_.
* `Wrapper dispatch and likelihood selection <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/hbmcmc/run_hbmcmc.py#L490>`_,
  `true fixedbasis signature <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/hbmcmc/hbmcmc.py#L674>`_.
* `inferpymc sampling/chain adapter <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/hbmcmc/inversion_pymc.py#L243>`_
  and `RHIME slicing <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/rhime/sampling.py#L249>`_.
* `Legacy baseline construction <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/rhime/standard.py#L249>`_,
  `pollution likelihood equation <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/models/pollution_event.py#L133>`_,
  `fixed-error compatibility dispatch <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/rhime/_model_building.py#L247>`_.
* `Chain-zero trace conversion <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/postprocessing/inversion_output.py#L142>`_,
  `result and filenames <https://github.com/openghg/openghg_inversions/blob/0e5d6ef76eb724da8cee1b780633578bdc05f2b4/openghg_inversions/rhime/outputs.py#L34>`_.

The science witnesses and original INIs remain plain data after HBMCMC removal.
The historical harness stays opt-in and requires the pinned isolated release;
it must not be added to modern default pytest imports. Existing staged/cache
artifacts, arbitrary dynamic Python builders and external user workflows have
not been migrated or validated by these checks. Those boundaries remain with
`issue #948`_, `issue #949`_, `issue #952`_ and historical 0.7 execution.

.. _issue #960: https://github.com/openghg/openghg_inversions/issues/960
.. _issue #944: https://github.com/openghg/openghg_inversions/issues/944
.. _issue #948: https://github.com/openghg/openghg_inversions/issues/948
.. _issue #949: https://github.com/openghg/openghg_inversions/issues/949
.. _issue #719: https://github.com/openghg/openghg_inversions/issues/719
.. _issue #952: https://github.com/openghg/openghg_inversions/issues/952
