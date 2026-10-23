Migration evidence before compatibility removal
===============================================

This is the bounded M1 evidence archive for `issue #960
<https://github.com/openghg/openghg_inversions/issues/960>`_, part of
`#944 <https://github.com/openghg/openghg_inversions/issues/944>`_ /
`#948 <https://github.com/openghg/openghg_inversions/issues/948>`_. It serves
maintainers removing compatibility and authors of the later explicit converter.
It records source behavior, not a new configuration contract or a promise of
numerical reproduction. Production code and behavioral specifications are
unchanged by this evidence PR.

Read the ledger together with the plain inputs and expected observations in
``tests/fixtures/migration_960/``. Those files do not import HBMCMC and remain
available after its deletion. Runtime characterization tests deliberately call
the existing boundary: deletion PRs must account for their affected cases and
retain the plain expectations for later converter checks. Runtime-only drivers
may retire with their adapters before the converter exists; scientific
regressions must be ported in the removal PR. Do not regenerate expected values
from the replacement translator.

.. toctree::
   :maxdepth: 1

   migration_960_historical
   migration_960_resources

Evidence vocabulary and pinned comparison
-----------------------------------------

* **Observed** means a bounded check was executed with acquisition and sampling
  absent or replaced by a capture stub. It establishes settings or errors, not
  a successful scientific inversion.
* **Inspected** means the pinned source body/signature establishes the statement;
  no execution of that scientific case is claimed.
* **Candidate** identifies a possible modern spelling or policy. It does not
  resolve the release's API, support window, or scientific choices.
* **Mechanical** preserves the represented setting; **scientific change** needs
  an explicit choice or acknowledgement; **unsupported** stays in its source
  environment; **unresolved** has an identified follow-up owner.

The verified compatibility-bearing development and comparison revision is
``0a32d73a5c4a72be0505a4ed682472e1613e49e9`` (``devel`` fetched on
22 October 2026). It still contains ``hbmcmc/compatibility.py`` and
``hbmcmc/run_hbmcmc.py``. This is also the production revision exercised by
this PR; the evidence-only commits change no production modules. The earlier
``6a0d6acf`` audit is not the baseline. A fresh diff from that commit found no
changes in the HBMCMC package, ``rhime/_runner_compat.py``, ``requests.py``,
``ini.py`` or ``config/config.py``; current ``cli.py`` and model code were also
read. See the historical page for exact released tag commits, publication
status and the separately prepared ``main`` version.

Future removal PRs must compare their base to this exact revision and refresh
any affected ledger entries, inputs, expected observations and tests. For
example::

   git diff 0a32d73a5c4a72be0505a4ed682472e1613e49e9..HEAD -- \
     openghg_inversions/hbmcmc openghg_inversions/rhime \
     openghg_inversions/config openghg_inversions/_flux_options.py \
     openghg_inversions/cli.py openghg_inversions/inference \
     openghg_inversions/models openghg_inversions/postprocessing

Current development environment
-------------------------------

The focused checks use a new worktree-local lean environment made with
``uv sync --locked``; no historical dependency was installed into it. Interpreter:
CPython 3.13.7, Clang 20.1.4, Linux. ``uv.lock`` at the pinned development ref has
SHA256 ``a60675ff1116d759773ad94a3f71d265c4ac330a232dc4f021891d12031acf3c``.
The installed project reports version ``0.7.4`` from development metadata;
that string does not identify the released v0.7.4 code.

Resolved versions used here: PyMC 6.3.2, PyTensor 3.3.2, ArviZ 1.3.0,
NumPy 2.2.6, SciPy 1.15.3, xarray 2025.9.1, OpenGHG 0.19.0,
Dask 2026.1.1, sparse 0.17.0, pytest 9.0.2 and Ruff 0.16.0. The pinned lockfile
provides the remaining dependency provenance. Historical provenance and
limitations are recorded separately; do not loosen modern requirements to run
old source.

Rerun the bounded checks
------------------------

Check out this evidence PR to obtain the fixtures/tests, keeping its pinned
production source, then create its lean environment::

   uv sync --locked
   export DASK_NUM_WORKERS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
   export MPLCONFIGDIR=/tmp/mpl-960
   IFS=',' read -r -a inherited_flags <<< "${PYTENSOR_FLAGS:-}"
   export PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor-960
   for flag in "${inherited_flags[@]}"; do
       case "${flag//[[:space:]]/}" in base_compiledir=*|'') continue ;; esac
       PYTENSOR_FLAGS+=",$flag"
   done
   .venv/bin/pytest tests/test_migration_960_requests.py tests/test_migration_960_resources.py
   .venv/bin/ruff check tests/test_migration_960_requests.py tests/test_migration_960_resources.py
   git diff --check

The tests use only small local files and synthetic arrays. They do not acquire
real data, sample a posterior, submit an inversion or validate arbitrary user
scripts. Use the historical page's isolated environment instructions for
released-source observations. Environment-creating tox runs on this cluster
must use ``sbatch scripts/slurm_tox.sh``; no such environment is needed for
these bounded direct checks.

Migration ledger: requests and execution
----------------------------------------

The following entries use the development revision above unless marked
``v0.7.4``. Detailed historical defaults and scientific entries live in
:doc:`migration_960_historical`; resource entries live in
:doc:`migration_960_resources`. Each entry separates the source meaning from
its candidate replacement and names the issue owning the next decision.
These owners identify workstreams, not accepted personal assignments.

.. list-table:: Route identity and dispositions
   :header-rows: 1
   :widths: 14 25 25 20 16

   * - ID / source route
     - Old spelling/default and effective meaning
     - Candidate / disposition
     - Evidence
     - Follow-up owner
   * - R1 / released RHIME Python and CLI
     - ``run_rhime`` is a Python result-wrapper API; ``run-rhime`` is an
       ``openghg-inversions`` subcommand, not a standalone executable.
     - ``run_standard`` and its ``InversionOutput`` consumer API; mechanical
       name adaptation plus unresolved result-consumer migration.
     - Historical page: exact signatures, installed CLI and result fields.
     - #945 public API; #950 launchers; #949 conversion.
   * - R2 / development Python and CLI
     - Python ``run_rhime`` raises ``ImportError``; CLI ``run-rhime`` calls
       ``run_standard``. A standard run requires an explicit built-in
       ``mismatch_model`` or custom likelihood; omission errors before
       acquisition. Vocabulary alone does not identify a route.
     - Explicit source version and route; unresolved final CLI aliases.
     - ``rhime/__init__.py::__getattr__``; ``cli.py::_run_rhime_command``;
       request characterization.
     - #945 / #950.
   * - R3 / ordinary HBMCMC wrapper
     - Omitted ``likelihood`` selects ``pollution_event``; the script calls
       RHIME, even with fixedbasis-style names. It has no direct legacy
       executor on development.
     - Explicit ``mismatch_model='pollution_event'``; mechanical for this
       model selection, scientific exceptions remain separate.
     - ``run_hbmcmc.py::main``; request fixtures and tests.
     - #949 conversion; #719 removal.
   * - R4 / true v0.7.4 fixedbasis
     - ``--legacy-fixedbasis`` selects a different executor with different
       defaults and historical equations; ordinary wrapper is not this route.
     - Source environment or reviewed modern scientific choice; scientific
       change / unresolved supported subset.
     - Historical page; original invocations and defaults fixture.
     - #948 scientific coverage; #946 environment.

.. list-table:: Input interpretation and dispositions
   :header-rows: 1
   :widths: 12 27 27 20 14

   * - ID / source
     - Old spelling/default and effective meaning
     - Candidate / disposition
     - Evidence
     - Follow-up owner
   * - I1 / INI, released and development
     - Case-preserved option names, inline comments, flattened sections,
       first occurrence and Python-value decoding are observable parser
       behavior. ``eval_check`` evaluates expressions as well as literals.
     - Preserve effective values, not section grouping; mechanical within
       captured syntax. Unknown/custom options need a decision.
     - ``config/config.py`` parser bodies; original INIs / expected values;
       historical and request checks.
     - #949 parser; #948 supported syntax.
   * - I2 / Python and CLI overrides
     - File values precede explicit overrides; date/output flags precede
       JSON kwargs. Alias normalization happens after these updates.
     - Materialize winning values with source-route metadata; mechanical.
     - ``cli.py::_command_kwargs``, wrapper ``main`` and ``requests.load_params``;
       capture tests.
     - #949 / #950.
   * - I3 / ordinary aliases
     - ``outputpath/outputname/xprior/bcprior/sigprior/offsetprior/``
       ``emissions_name/outer_region_definition_file`` map to canonical names;
       simultaneous canonical spelling wins, even from an earlier source.
     - ``output_path/output_name/x_prior/bc_prior/sigma_prior/offset_prior/``
       ``flux_sources/outer_regions_path``; mechanical with precedence retained.
     - Both alias tables and translation bodies; request fixtures.
     - #949; #719 runtime removal.
   * - I4 / flux selector aliases
     - ``emissions_store/domain`` and ``inner_emissions_store/domain`` map to
       corresponding ``flux_*``. Simultaneous old/new spelling raises even
       for equal values. ``emissions_name`` follows the ordinary rule above.
     - Canonical ``flux_*`` with explicit conflict reporting; mechanical.
     - ``_flux_options.py::resolve_deprecated_keyword``; request checks.
       Inner selectors are inspected only; executed conflict cases cover the
       ordinary store/domain selectors.
     - #949 / #719.
   * - I5 / wrapper sampling aliases
     - ``nit/nchain/verbose/sampler_kwargs`` mean
       ``draws/chains/progressbar/sample_kwargs``. ``burn`` remains a separate
       post-sampling discard, not part of ``tune`` or subtracted from draws
       during translation.
     - Explicit sampler settings; mechanical renaming. Preserve
       ``draws - burn`` per chain, as justified by both sampling bodies.
     - ``compatibility._translate_legacy_aliases``; historical sampling bodies;
       synthetic trace check.
     - #949; #952 numerical validation.
   * - I6 / scientific switches
     - ``calculate_min_error`` accepts residual/percentile when enabled;
       enabled ``reparameterise_log_normal`` updates only supplied lognormal
       emissions/BC priors. Removed switches are route dependent.
     - Explicit ``min_error`` and prior ``reparameterise`` fields; mechanical
       only when source-effective priors/policies are preserved.
     - ``compatibility._translate_legacy_options``; request cases; historical
       scientific ledger for non-equivalent cases.
     - #948 / #949.

Output and unsupported boundaries
---------------------------------

.. list-table:: Output intent and exclusions
   :header-rows: 1
   :widths: 12 27 27 20 14

   * - ID / source
     - Old spelling/default and effective meaning
     - Candidate / disposition
     - Evidence
     - Follow-up owner
   * - O1 / ordinary wrapper
     - Default ``legacy`` format, legacy filename convention, no
       inversion-output sidecar unless requested (``inv_out`` is exceptional).
       ``paris_postprocessing`` takes precedence over output format.
     - Explicit output format, save flag and naming convention; mechanical.
       Modern legacy codec is independently supported.
     - ``fixedbasis_params_to_rhime`` and output normalization body; request
       policy fixture and tests.
     - #949; #820 default policy.
   * - O2 / aliases and copied INI
     - ``hbmcmc/hbmcmc_postprocessing`` mean ``legacy``. Wrapper INI copy
       uses raw ``outputpath/outputname`` while translated output settings
       use winning canonical names; destinations can differ.
     - Preserve product intent, report conflicting copy/result destinations;
       unresolved converter handling of the copied INI.
     - ``run_hbmcmc._copy_config_file``; capture tests and expected filenames.
     - #948 / #949 / #950.
   * - O3 / development standard
     - ``RhimeOutputSpec`` defaults to ``inv_out``, save enabled and RHIME
       naming. ``basic`` no longer computes a discarded summary;
       ``run_standard`` returns ``InversionOutput``.
     - Set explicit source output/save policy; adapt Python consumers to
       modern ``trace``, labelled inputs, basis and metadata. Mechanical
       settings; unresolved consumer
       automation and default choices.
     - ``rhime/specs.py``, ``rhime/standard.py`` and migration guide;
       request defaults check; historical comparison.
     - #945 / #948 / #950; #820.
   * - U1 / historical staged and cache artifacts
     - Saved envelopes, acquisition caches and trace/basis files are not
       interchangeable with a complete modern inversion output.
     - Unsupported by this evidence/converter scope; retain original source
       environment or obtain a separately reviewed artifact migration.
     - :doc:`../usage/staged_workflow`; pinned source APIs; no artifact
       conversion or scientific reproduction check in this PR.
     - #948 / #952; #951 independent basis capabilities.
   * - U2 / custom or dynamic Python
     - Arbitrary functions, runtime-generated kwargs, custom likelihoods,
       local edits and unversioned scripts have no safely inferred static
       route or model intent.
     - Unsupported automatic rewriting; preserve source/version/route and
       manually port the scientific recipe.
     - Source signatures distinguish callables from serializable settings;
       this archive contains finite representative invocations only.
     - #950 launch scope; #948 coverage.

Removal and validation handoff
------------------------------

Before deleting a runtime behavior, #719 must cite its ledger entry, account
for source changes since the pinned commit, and port any scientific regression
which currently enters through that adapter. Passing translation tests alone
does not establish scientific parity. #961 owns compatibility-promise changes;
this page does not supersede the current specifications or announce a release.
#948 owns the final supported source versions and target choices; #949/#950
own conversion and launch tooling; #952 owns integrated scientific and packaging
acceptance. Independent basis readers/writers remain with #951.

The existing regression inventory includes ``tests/test_rhime_compatibility.py``,
``test_run_hbmcmc_shim.py``, ``test_rhime_configuration.py``,
``test_acquisition.py``, ``test_filters.py``, ``test_basis_paths.py`` and
``test_unsupported_tracer.py``. Shared ground/satellite request fixtures in
``tests/conftest.py`` still contain ``emissions_store`` and ``emissions_name``.
This is an inspection inventory, not a claim that all those scientific tests
were executed here or an authorization to delete them. Preserve scientific
expectations on canonical inputs in the affected removal PRs.

Observed validation on 22 October 2026
--------------------------------------

The new request/resource checks and the existing unsupported-tracer checks
passed: **56 tests**. The related compatibility, wrapper, configuration,
output-resource, BC-resource and resource-boundary modules passed:
**295 tests**. The broader invocation was::

   .venv/bin/pytest tests/test_rhime_compatibility.py \
     tests/test_run_hbmcmc_shim.py tests/test_rhime_configuration.py \
     tests/test_rhime_output_resources.py tests/test_bc_basis_resources.py \
     tests/test_rhime_resource_boundaries.py

Use the environment setup above for both commands. The new tests comprise
26 request checks and 14 resource checks; the modern chain-summary witness
uses only modern postprocessing and the archived plain values. Changed-path
Ruff and ``git diff --check`` passed. The historical page records its separate
successful harness and installed-command observations.

The three new pages were rendered with Sphinx 9.1.0, Docutils 0.22.4 and the
repository's PyData theme 0.21.0 in an existing documentation environment.
This bounded build copied only these pages, used ``migration_960`` as its root,
and treated warnings as errors; outgoing ``:doc:`` references to omitted pages
were checked against the full source tree separately. The generated headings,
ledger tables, navigation and code blocks were checked in the built HTML.
This does not claim a full API/tutorial documentation build.

No full scientific test suite, Python 3.12 modern matrix or CI-equivalent tox
run is claimed. Final cross-version, packaging and scientific acceptance remains
with #952. Source-inspected equations, historical resources, arbitrary external
scripts and the final supported patch-version subset retain their stated limits.

Aggregation filter retirement comparison, 23 October 2026
---------------------------------------------------------

This dated addendum records the removal of ``daily_median`` and ``six_hr_mean``
under `#954 <https://github.com/openghg/openghg_inversions/issues/954>`_. It
preserves the preceding evidence, historical rows, expected values and source
hashes. The source route is ``openghg_inversions.filters.filtering`` at pinned
revision ``0a32d73a5c4a72be0505a4ed682472e1613e49e9``. The affected filter
functions were unchanged from that source before retirement on comparison base
``3b27c542``. This comparison uses inspected source and independent arithmetic;
no historical execution of these aggregation cases is claimed.

The plain inputs and expectations are retained in
:download:`aggregation_filters_2026_10_23.json
<../../tests/fixtures/migration_960/retirements/aggregation_filters_2026_10_23.json>`.
For site ``TAC``, timestamps are 00:00, 01:00 and 02:00 on 1 January 2020,
``mf = [0, 1, 100]`` and ``mf_error = [1, 9, 3]``. All three timestamps fall
within one bin for either source operation, labelled 00:00 on that date.
With the default ``keep_missing=False``, both operations drop missing time
rows after aggregation; these inputs contain no missing values. The daily
median implementation also calls ``dataset.load()`` before resampling.

.. list-table:: Source aggregation and modern disposition
   :header-rows: 1
   :widths: 19 27 24 30

   * - Source filter
     - Inspected operation
     - Independent expected values
     - Modern disposition
   * - ``daily_median``
     - Daily median of dataset variables.
     - ``mf = 1``; ``mf_error = 3``.
     - Function and registry entry removed; named requests raise guided
       ``KeyError``.
   * - ``six_hr_mean``
     - Six-hour mean of dataset variables.
     - ``mf = 101 / 3``; ``mf_error = 13 / 3``.
     - Function and registry entry removed; named requests raise guided
       ``KeyError``.

The old operations aggregated ``mf_error`` generically as another dataset
variable. The archived arithmetic describes that behavior; it does not endorse
those mean or median values as propagated observation uncertainty. Modern
requests receive guidance to perform deliberate uncertainty-propagating
resampling before observation selection. There is no automatic mean or median
replacement.

The current behavior is covered by
``tests/test_filters.py::test_retired_resampling_filters_are_not_available`` and
``tests/test_filters.py::test_invalid_filter_requests_fail_before_any_data_execution``.
The latter covers scalar, list and site-specific requests through both
``filtering`` and the public observation-selection stage, including rejection
before lazy data execute. These tests own the runtime removal behavior; the
plain evidence file needs no duplicate runtime test.

Implementation belongs to #954, public API disposition to
`#945 <https://github.com/openghg/openghg_inversions/issues/945>`_, and the later
migration report to
`#949 <https://github.com/openghg/openghg_inversions/issues/949>`_. This addendum
does not define a replacement resampling algorithm or revise the historical
migration expectations above.
