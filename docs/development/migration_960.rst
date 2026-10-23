Migration evidence before compatibility removal
========================================

This is the bounded M1 evidence archive for `issue #960
<https://github.com/openghg/openghg_inversions/issues/960>`_, part of
`#944 <https://github.com/openghg/openghg_inversions/issues/944>`_ /
`#948 <https://github.com/openghg/openghg_inversions/issues/948>`_. It serves
maintainers removing compatibility and authors of the later explicit converter.
It records source behavior, not a new configuration contract or a promise of
numerical reproduction. Production code and behavioral specifications are
unchanged by this evidence PR.

Archived retirement records
----------------------------

The five dated #719 retirement records are historical review evidence, not
inputs consumed by pytest. They were removed from the current test tree on
26 October 2026; their original contents remain available at the full Git
revision ``6a1d22d10c07442514dd610e2ec62b7a2719e297``:

* `Initial runtime removal comparisons
  <https://github.com/openghg/openghg_inversions/blob/6a1d22d10c07442514dd610e2ec62b7a2719e297/tests/fixtures/migration_960/retirements/runtime_719_2026_10_23.json>`_.
* `Supplementary historical codec witnesses
  <https://github.com/openghg/openghg_inversions/blob/6a1d22d10c07442514dd610e2ec62b7a2719e297/tests/fixtures/migration_960/retirements/legacy_codec_witnesses_2026_10_23.json>`_.
* `23 October review and rebase comparisons
  <https://github.com/openghg/openghg_inversions/blob/6a1d22d10c07442514dd610e2ec62b7a2719e297/tests/fixtures/migration_960/retirements/review_974_2026_10_23.json>`_.
* `Prepared factory and cleanup comparisons
  <https://github.com/openghg/openghg_inversions/blob/6a1d22d10c07442514dd610e2ec62b7a2719e297/tests/fixtures/migration_960/retirements/factory_and_cleanup_2026_10_24.json>`_.
* `26 October BC, schema-name and contract-owner comparisons
  <https://github.com/openghg/openghg_inversions/blob/6a1d22d10c07442514dd610e2ec62b7a2719e297/tests/fixtures/migration_960/retirements/review_974_2026_10_26.json>`_.

These records preserve source revisions/hashes, original cases, dispositions and
follow-up owners for review and future explicit migration work. They are not
current expected behavior or proof of an executed scientific check. Dated paths
in the ledger below refer to this archive. Actual validation remains attributed
to its implementation revision in the task and PR records.

The original #960 plain inputs and expectations remain in
``tests/fixtures/migration_960/``. Current tests consume the applicable cases;
the historical cases also remain available for later explicit converter checks.
The retirement-record cleanup does not remove or rewrite those original files.

Original evidence scope and fixtures
-------------------------------------

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

Under the `#961 development boundary
<https://github.com/openghg/openghg_inversions/issues/961>`_, the
`retirement delivery checklist
<https://github.com/openghg/openghg_inversions/blob/devel/openspec/changes/isolate-and-retire-hbmcmc/tasks.md#migration-evidence-checklist-and-delivery-order>`_
records concrete owners, order and baseline checks. Each PR removing or
deliberately changing migration-relevant behavior must:

* Identify affected ledger entries and plain cases. Add bounded evidence for
  newly discovered behavior before removal, keeping source version and route explicit.
* Compare its base and affected source with pinned revision
  ``0a32d73a5c4a72be0505a4ed682472e1613e49e9``, accounting for intervening changes
  and relevant merged PRs.
* Record disposition and follow-up owner: retained, mechanically migrated,
  deliberate scientific change, unsupported or unresolved. Preserve historical
  inputs and expectations; add dated comparisons rather than replacing them
  with modern output.
* Preserve or port relevant scientific expectations before retiring adapter/test
  plumbing. Test unchanged behavior on canonical inputs and agreed scientific
  changes against their new semantics. Distinguish inspected evidence from
  executed checks and link validation to the implementation revision.

This remains a pinned source archive. Only affected evidence and live checks
need refreshing; neither a whole-audit rerun nor finished converter/launcher
tooling is required for every removal. Passing translation tests alone does not
establish scientific parity. The gate does not resolve pending interface or
scientific choices. #719 owns runtime removal and relevant regression migration;
#961 amends the compatibility promises without itself implementing removal.
See :doc:`/usage/legacy_and_migration` for the amended requirements and retained capabilities.
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


Dated retirement comparison: issue 719, 23 October 2026
-------------------------------------------------------

This addition records the affected retirement evidence without replacing any
original source observation above or any ``requests/``, ``resources/`` or
``historical/`` fixture. Plain discoveries and dispositions are captured in
``tests/fixtures/migration_960/retirements/runtime_719_2026_10_23.json``.
The PR base is ``17e28ed081954b4155c9d85b66eaeef2ad81853c``. It includes
merged #966 (evidence, merge ``3b27c542``) and #972 (approved desired contracts,
merge ``17e28ed0``). Contract/documentation checks from #972 do not validate
runtime removal. The comparisons below are source inspection; fresh executed
checks belong to the implementation revision's task and PR records.

Pinned-source comparison
~~~~~~~~~~~~~~~~~~~~~~~~

Comparing the affected sources at pinned ``0a32d73a`` with the PR base finds:

* No changes in ``hbmcmc/compatibility.py``, ``hbmcmc/run_hbmcmc.py``,
  ``hbmcmc/legacy_basis.py``, the runner alias/request/acquisition/nested sources,
  ``cli.py``, or the five Python alias assignments/re-exports.
* ``hbmcmc/legacy_data.py::make_combined_scenario`` gained explicit
  ``xr.concat(..., join="outer")`` through #956. Its intended source-union
  behavior must survive in canonical labeled-input regressions; deletion does
  not license changing source ordering, missing-time markers or signed fluxes.
* #956 also made modern trace/statistic merges explicitly use
  ``join="outer", compat="no_conflicts"`` and updated supported PyMC imports;
  #959/#969 fixed fixed-OU projection/roundoff, #963 calibrated posterior
  Monte Carlo checks, and #967 added compiled-backend checks. These intervening
  changes stay in the base; they are not retirement changes or claims of exact
  numerical equivalence with the pinned environment.
* ``pyproject.toml`` changed xarray's lower bound and test markers. Its two
  console-script declarations were unchanged. Removing the second executable
  therefore needs fresh built-distribution checks.

Affected entries, cases and dispositions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **R1/R2, I2:** the accepted command map replaces the second executable and
  four full-run aliases with ``rhime run --model standard|multisector|nested|co2``.
  Original ``invocations.txt`` files retain the historical commands. Dated case
  ``command-map`` records the new map and preserved dates/JSON/override/CO2/staged
  meanings. Disposition: mechanically migrated. #719 delivers removal; #945
  retains broader naming decisions and #950 owns launcher migration.
* **I3/I4/O2:** original request/precedence/conflict observations remain archived.
  ``runtime-options`` adds the exact eight parameter, four flux-selector and
  two output-format spellings. Every spelling now receives unsupported-option
  validation rather than translation, including old/new flux names supplied
  together. Canonical values remain supported. Disposition: mechanically
  migrated request vocabulary; #949 owns explicit conversion and conflict reports.
* **I6:** ``tracer`` preserves false-consumed/true-special-error source behavior.
  The approved modern outcome rejects every explicit value ordinarily before
  scientific work, with omission valid. Disposition: unsupported; #719 owns
  removal, #949 reporting. Draft #964/#973 overlap was inspected, not imported;
  their validation does not validate this revision. #971 is separate from #719.
* **R3/R4, I2/I5/I6, O1/O2:** ``namespace-wrapper`` retires the ordinary wrapper,
  dictionary normalization, config-copy behavior and module execution together.
  Original ``fixedbasis.ini``, ``explicit-wrapper.json``, defaults and sampling
  witnesses remain available. Disposition: unsupported execution on modern
  development; matching 0.7 execution belongs to #946/#947, finite conversion
  and launcher support to #948/#949/#950.
* **U1 and the resource ledger:** ``legacy-data`` and ``legacy-basis`` add the
  previously uncatalogued six-tuple retrieval/cache, dictionary/scenario/DataTree
  serialization, fp_all basis conversion and weight dispatch surfaces.
  Disposition: unsupported historical adapters/artifacts, not deletion of
  scientific representations. Original resource cases still describe sorted
  partitions, retained-artifact preference/ambiguity, NESW/no-BC, missing resources,
  conflicts and invocation-relative paths. #948/#949 own converter coverage;
  #951 retains independent basis discovery/read/write and flat partitions.
* **R2/U2:** ``python-aliases`` supplies a bounded source witness for the five
  approved aliases, absent from the original ledger. Their canonical functions
  and classes remain. Disposition: mechanically migrated; #950 consumes the map,
  while #945 owns all unresolved names. Artifact acceptance remains codec-owned.
* **O1/O3, H2-H6:** canonical sampling, offset/fixed-error semantics, all-chain
  summaries, modern ``legacy`` products, ``InversionOutput`` exports and exact
  paths remain independently supported. Disposition: retained. Ported scientific
  regressions use canonical labeled inputs; historical quirks are not restored.
  #948/#949 own historical scientific-choice reporting and #952 final acceptance.

The modern migration tests use canonical request settings and exact resource
paths. They no longer execute a removed adapter. The archived plain expectations
remain later converter inputs; checking their preservation is not historical
execution. No converter, launcher, new artifact format support, release number
or release-readiness decision is supplied here. Retained-population imputation
and nested alignment remain #876/#964; provider acquisition averaging precedes
inversion-side error combination, and aggregation is not selection.

Additional bounded discovery before the CO2 validation slice: known inactive
configuration sections in ``validate_co2_config`` / ``load_co2_o2_params`` did
not validate option names. An ``outputs.use_tracer=False`` could survive
preparation, and ``sampling.use_tracer`` could survive postprocessing. Dated
case ``co2-inactive-section-names`` preserves that inspected behavior. Ordinary
name-only validation now owns unsupported options in all groups and phases;
inactive scientific values and resource loading remain deferred. Disposition:
unsupported, #719 implementation and #949 migration reporting. This discovery
does not authorize resolving inactive values or changing CO2 science.

The affected ``rhime/co2/configuration.py``, ``co2_o2_configuration.py`` and
``execution.py`` have no changes between pinned ``0a32d73a`` and base
``17e28ed0``. Generic validation treats scientific mapping contents as opaque
and retains supported output fields, including ``source_to_sector``; it checks
option names without evaluating numerical arrays or resolving resources.

Supplementary codec witnesses after initial acceptance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tests/fixtures/migration_960/retirements/legacy_codec_witnesses_2026_10_23.json``
preserves ten concrete adapter-only input/expectation cases, with exact pinned
and PR-base blob hashes, SHA256 values and source links. They cover typed
metadata, dotted markers, source labels/order, borrowed serialization, native
source times, period fallback and lazy/all-NaN layouts. This source-inspected
supplement follows the initial retirement commit and expands the earlier
``legacy-data`` inventory; it is not historical execution or newly promised
artifact support. Disposition remains unsupported historical codecs under
U1, with coverage/reporting owned by #948/#949 and independent basis support
by #951. Original #960 inputs and observations remain unchanged.


Review follow-up and current-devel source refresh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maintainer review `5634401507
<https://github.com/openghg/openghg_inversions/pull/974#pullrequestreview-5634401507>`_
clarifies generic name validation and phase-owned effective-value validation.
The dated ``retirements/review_974_2026_10_23.json`` supplement preserves the
pre-change diagnostics and output-value cases alongside source hashes.
The earlier ``runtime-options`` comparison grouped removed option names and
output values together; the clarified contract rejects all unknown names but
keeps validation of known inactive values deferred. A preparation-only shared
request can contain an invalid output format, which is checked only by its
later consuming phase. No special historical-token rejection is introduced.
Every explicit ``use_tracer`` remains an unknown name regardless of its value.

Before removal, the reviewed ``_request_validation`` helper added replacement
advice after generic name rejection, and nested execution had dedicated errors
for three historical inner resource selectors. The follow-up removes those
diagnostic tables/branches and unsupported extraction names. Bounded before/after
request checks observed generic rejection with advice becoming generic rejection
alone, without acquisition or scientific work. Disposition: unsupported names
remain unsupported; #719 owns implementation and #949 replacement reporting.
Independent supported basis adapters remain with #951.

The new PR base is ``b288ac2530ad93ae8f67c5f1d47ee8ee7b666000``.
``requests.py``, ``_runner_compat.py`` and ``nested.py`` are unchanged from pinned
``0a32d73a`` through that base; ``_request_validation.py`` was introduced by the
retirement and is compared at reviewed ``ad293dac``. Intervening #975 changes
historical Zarr writing to format 2, backend-compatible compression and staged
ZIP creation; its borrowed-metadata tests add ZIP expectations. The supplement
records exact new-base source hashes and those expectations before resolving the
namespace deletion. Modern serializers retain #975's independently supported
fixes; historical cache support is not restored. Original eighteen #960 fixtures,
previous codec witnesses and dated validation results remain unchanged. Fresh
follow-up validation belongs to its resulting revision's task and PR records.


24 October addendum: prepared-input factory and review cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The maintainer's `approved factory disposition
<https://github.com/openghg/openghg_inversions/pull/974#issuecomment-6380451664>`_
and `analysis
<https://github.com/openghg/openghg_inversions/pull/974#issuecomment-6380430675>`_
authorize removing only ``PreparedInputs.from_legacy_inputs`` in addition to the
five aliases. The exact replacement is
``PreparedInputs(inv_inputs=..., basis_functions=..., site_metadata=...)``.
This is a compatibility-constructor disposition under #945/#663, not a sixth
alias or a #951 basis capability decision.

Before deletion, ``retirements/factory_and_cleanup_2026_10_24.json`` records
source revision ``e1d020f04640a7689079a57642952e324ded4eb5``, the exact factory
signature, original two test inputs/expectations, provenance overrides,
coordinate validation/precedence and sole modern caller. The factory body has
an identical source hash at pinned ``0a32d73a``, base ``b288ac25`` and this
pre-removal revision. The caller already normalizes stationary/mobile release
coordinates; it never supplies the factory's legacy per-site coordinate lists.
Its existing ``supplied`` provenance fallback, namespaced/plain source/path
metadata, site/period order and borrowed lazy arrays remain supported through
canonical construction. Persisted schema, variables, units and codecs do not
change. Historical scalar-coordinate adaptation remains captured rather than
being added to the canonical constructor. Disposition: unsupported compatibility
constructor; #719 delivers removal, #945/#663 own this bounded API decision.

The same dated fixture captures `review 5637015484
<https://github.com/openghg/openghg_inversions/pull/974#pullrequestreview-5637015484>`_'s
remaining cleanup before editing: bespoke removed-import errors, resolved/raw
request conflict checks preceded by unnecessary parsing, repeated CO2 whole-request
name scans, and the independent basis alias helper's root-module location and
unapproved 0.9 warning date. Pinned-to-base sources are unchanged except #975's
linked-CO2 string-coordinate fixes, which remain. Compared with that base,
earlier retirement introduced the repeated scans and source-conflict parsing;
finite Python-name updates left the factory's body unchanged.

Disposition: removed interfaces remain ordinarily unavailable; ambiguous sources
fail without parsing; one outer CO2 request owner validates names while consuming
components validate applicable values; supported basis alias behavior remains
with #951 while its private sentinel/resolution logic becomes inline in the
sole consumer, the root helper is removed and the unapproved schedule is dropped. Known inactive values stay deferred, and unknown inactive
names still fail at the outer boundary. Replacement advice remains migration
material. These are inspected source/disposition records; fresh executed checks
belong to the new implementation revision. Earlier dated validation and all
original eighteen #960 fixtures remain unchanged.

One additional source-conflict witness was inspected after its test assertion
was ported: ``test_standard_prepared_run_rejects_complete_model_builder`` at
``e1d020f0`` combined resolved configuration/prepared inputs with a
``model_builder`` callback that must not execute. Its original expected error
named the unsupported builder option; the modern comparison expects the ordinary
resolved/raw source conflict while preserving the callback guard. The dated
fixture labels this retrospective source capture explicitly; fresh execution
is recorded separately, not claimed as a pre-removal observation.


26 October addendum: positive owners and explicit BC selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Review 5649950083
<https://github.com/openghg/openghg_inversions/pull/974#pullrequestreview-5649950083>`_
approves the bounded follow-up. Before code edits,
``retirements/review_974_2026_10_26.json`` captures revision
``1c15954cf191899462cf0fe2faeca5079873b842`` against pinned ``0a32d73a`` and
base ``b288ac25``: source hashes, exact original BC regression cases, the
prepared-schema constants and their values, and the previous contract ownership.
These are source observations, not a fresh scientific execution claim. The
original eighteen fixtures and earlier dated comparisons remain unchanged.

* **BC selection (I4/legacy-basis):** the previous configuration exposed exact
  files but no algorithm field; absent files passed ``None`` to the same NESW
  numerical calculation. Add canonical ``bc_basis_algorithm="NESW"`` as its
  default. Exact files win and make the algorithm inactive; disabled BCs resolve
  and read no BC basis resources. Preserve directional labels, temporal/grid
  alignment and lazy borrowed data. #719 delivers this selector; #951 retains
  independent basis decisions. Ordinary name/applicable-value validation remains
  separate from resource loading.
* **Python constants (prepared-input/O1/O2):** replace
  ``RHIME_PREPARED_INPUTS_SCHEMA`` and ``RHIME_PREPARED_INPUTS_SCHEMA_VERSION``
  with ``PREPARED_INPUTS_SCHEMA`` and ``PREPARED_INPUTS_SCHEMA_VERSION``.
  Persisted ``openghg_inversions.rhime_prepared_inputs``, version 1 and codec
  layout stay unchanged in this slice. This is not a permanent freeze:
  `#976 <https://github.com/openghg/openghg_inversions/issues/976>`_ owns the
  separate persisted-identifier migration and explicit version support.
  `OPE-233 <https://linear.app/openghg-inversions/issue/OPE-233>`_ tracks a selected
  VG/OGI baseline and migration; none is claimed delivered here.
* **Contract disposition (all slices):** replace negative deletion inventories
  in durable configuration/selection/execution requirements with positive
  supported-interface contracts. Preserve the exact namespace/option/command/
  alias/factory maps in :doc:`/usage/legacy_and_migration`, the earlier ledger
  addenda and dated source records. Canonical construction belongs to the
  scientific-input owner; codecs/replay remain with persistence/handoffs.
  Remove the already-delivered factory's duplicated active ``ADDED`` requirement.
  Keep the new BC delta and final archive pending; #719 owns reconciliation,
  #945/#663 broader Python naming and #876/#964 separate scientific amendments.

The 26 October approval supersedes the operative blanket unchanged-schema clause.
The 24 October unchanged-schema observations above remain true for that earlier
implementation. Neither this reconciliation nor Python naming imports #964's
pending numerical changes, promotes the unfinished handoff programme or implies
unsupported historical-format compatibility. Fresh acceptance is recorded in
revision-specific tasks/PR evidence, separately from these source comparisons.

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
