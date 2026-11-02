Migration resource evidence for issue 960
=========================================

This is the resource-selection part of the M1 migration evidence. It records
observed compatibility behavior and candidate replacements for reviewers of
`issue 960 <https://github.com/openghg/openghg_inversions/issues/960>`_.
Candidates below are not decisions about the future converter.

Revisions and evidence classes
------------------------------

**Executed current evidence (C):** compatibility-bearing devel commit
``0a32d73a5c4a72be0505a4ed682472e1613e49e9``. The 14 checks in
``tests/test_migration_960_resources.py`` execute resource selection and request
decoding only. They use temporary synthetic NetCDF files and no acquired data,
model construction or inference. The files carry a partition variable or the
retained-artifact schema marker; a schema marker alone is not a valid complete
retained scientific artifact.

**Historical source inspection (H):** published ``v0.7.4`` commit
``0e5d6ef76eb724da8cee1b780633578bdc05f2b4``. Resource findings were checked
directly against ``basis/_functions.py``, ``basis/_wrapper.py``,
``basis/_helpers.py`` and ``utils.py`` at that revision. Those sources were also
compared with published ``v0.7.3``
(``9c0d23c15000dc7a7e4c3306d3f1bfd4816bdb4f``); the relevant source files are
unchanged. H does not mean the historical resource workflow was executed.

The true historical route was traced explicitly: ``run_hbmcmc.py`` dispatches
``--legacy-fixedbasis`` to ``fixedbasisMCMC``; that calls
``hbmcmc/preparation.py:prepare_fixedbasis_inversion_data``, which imports
``basis_functions_wrapper`` from ``basis/__init__.py``. That public export
selects ``basis/_wrapper.py``; its saved-basis branch calls
``load_basis_functions`` and its BC branch calls ``basis/_helpers.py``'s
``bc_sensitivity``. Thus H in the ledger applies to the true legacy route,
even though those basis helpers were also shared with historical RHIME.
At the pinned current revision the true executor is already absent; C here
tests the resource adapter used by the ordinary wrapper and does not imply a
current ``--legacy-fixedbasis`` execution.

**Target decision (T):** an unresolved converter policy or scientific review.
Follow-up owners are the issue workstreams identified below, not an assertion
that a person has accepted an assignment.

Ledger
------

.. list-table:: Resource migration ledger
   :header-rows: 1
   :widths: 14 15 22 18 11 20 12

   * - Source / route
     - Old spelling / default
     - Effective meaning
     - Candidate modern replacement
     - Disposition
     - Evidence / distinction
     - Follow-up owner
   * - 0.7.4 true ``--legacy-fixedbasis``; current ordinary HBMCMC adapter
     - ``fp_basis_case`` + ``basis_directory``; default repository ``basis_functions``
     - Sorted ``<root>/<domain>/<case>_<domain>*.nc`` matches. Prefer a sole
       schema-tagged retained artifact; otherwise use all flat partitions.
     - Exact ``basis_path`` for a retained artifact, or
       ``basis_partition_paths`` for selected flat partitions.
     - Mechanical candidate; T: selection policy unresolved.
     - H: ``_wrapper.load_basis_functions`` and ``_basis_artifact_files``;
       C: ``sorted-partition-and-bc-files`` and
       ``single-retained-artifact-wins-over-partitions``.
     - #949 converter; #948 resource diagnostics; #951 basis/readers.
   * - Same routes
     - Multiple retained matches; missing saved case
     - Multiple retained artifacts raise ``ValueError``; no matching files
       raise ``FileNotFoundError``. Neither chooses an arbitrary file.
     - Explicit user selection; report ambiguity or missing resource.
     - Mechanical failure reporting; T: diagnostics unresolved.
     - H: ``_wrapper.load_basis_functions``; C:
       ``ambiguous-retained-artifacts``, ``missing-emissions-resource``.
       Exact error wording in JSON is current C, not historical H.
     - #949 converter; #948 resource diagnostics; #951 basis/readers.
   * - 0.7.4 true ``--legacy-fixedbasis``; current ordinary HBMCMC adapter
     - ``bc_basis_case="NESW"``; directory supplied or omitted
     - Case-insensitive NESW uses acquired directional contributions directly;
       no BC basis file discovery. ``use_bc=False`` also bypasses discovery.
     - Leave ``bc_basis_paths`` unset; retain ``use_bc=False`` when requested.
     - Mechanical candidate: retain shared no-file behavior.
     - H: ``_helpers.bc_sensitivity`` NESW early return and
       ``_wrapper.basis_functions_wrapper`` use_bc gate;
       C: ``nesw-does-not-discover``, ``disabled-bc-does-not-discover``.
     - #949 converter; #948 resource diagnostics.
   * - Same routes
     - Non-NESW ``bc_basis_case`` + ``bc_basis_directory``; default repository
       ``bc_basis_functions``
     - Sorted matching files under the domain directory. H filters unreadable
       files before loading. C selects paths; modern runner loading owns
       missing/unreadable-file errors before acquisition.
     - Exact ``bc_basis_paths`` collection.
     - Mechanical selection candidate; T: unreadable-file policy unresolved.
     - H: ``_functions.basis_boundary_conditions``;
       C: ``sorted-partition-and-bc-files``, ``missing-bc-resource``.
       Unreadable-file behavior is inspection-only here.
     - #949 converter; #948 resource diagnostics.
   * - Current ordinary HBMCMC wrapper / dictionary adapter
     - Old saved cases replaced by override ``fp_basis_case=None`` and
       ``bc_basis_case="NESW"``
     - Winning overrides apply before discovery. Superseded missing paths
       do not cause errors. Historical/exact selections supplied together
       instead produce explicit conflicts; discovery needs ``domain``.
     - Resolve overrides, then supply only exact winning selections.
     - Mechanical candidate; T: override interface unresolved.
     - C: ``superseded-missing-selections``, both ``*-conflict`` cases and
       ``discovery-requires-domain``. No historical execution claim.
     - #949 converter; #948 resource diagnostics.
   * - Current ``params_from_config`` and modern INI request loader
     - Relative ``country_file``, ``landsea_path``, ``outer_regions_path``,
       ``output_path`` in INI
     - Paths resolve against invocation working directory, even when the INI
       lives in a different directory. A config-file-relative base is not
       inferred from the filename.
     - Preserve explicitly chosen path meaning in emitted requests.
     - Mechanical path preservation; T: converter policy unresolved.
     - C: ``invocation.ini`` and ``invocation.expected.json``, exercised by
       ``test_characterize_entrypoint_paths_are_invocation_relative``.
     - #949 converter; #948 resource diagnostics.
   * - 0.7.4 Python/installed RHIME; ordinary and true fixedbasis routes
     - Relative explicit ``basis_directory``, ``bc_basis_directory`` and
       output/save filenames from INI or overrides
     - INI decoding does not prepend the INI's directory. Resource readers and
       output writers use those relative paths from the invocation directory.
       Packaged defaults have separate source-relative roots.
     - Emit explicit paths preserving the invocation context.
     - Mechanical candidate; source-inspected historical rule.
     - H: ``rhime.params.params_from_config`` reads values without rebasing;
       ``basis._functions.basis`` / ``basis_boundary_conditions`` use
       ``Path(directory)``; ``rhime.outputs._resolve_output_path`` uses
       ``Path(save_setting)`` or ``Path(output_path) / filename``.
     - #949 converter; #950 launch working directory.
   * - Current resource adapter helper only
     - Explicit ``base_dir`` plus invocation overrides
     - File values use explicit base; overriding values use working directory.
       Canonical output-path spelling wins over its deprecated alias. Current
       normal INI entrypoints do not pass their file's parent as this helper base.
     - No public workflow recommendation; keep origin evidence.
     - Observed helper distinction; T: converter design unresolved.
     - C: ``file-base-and-invocation-override-origins``; source
       ``hbmcmc/compatibility.py:adapt_resource_params``.
     - #949 converter; #948 resource diagnostics.
   * - 0.7.4 true ``--legacy-fixedbasis`` versus current run configuration
     - ``country_directory``; default packaged support maps
     - H: weighted basis uses ``country-land-sea_<domain>.nc`` from the
       directory; fixed outer regions use
       ``outer_region_definition_<domain>.nc``. Reporting separately uses
       ``country_file`` or historical domain-based reporting lookup.
       C rejects ``country_directory`` outright.
     - Select ``landsea_path`` and/or ``outer_regions_path`` for preparation;
       choose ``country_file`` separately for reporting.
     - Unresolved meaning: select each required map explicitly.
     - H: ``_functions.load_country_region_classes``,
       ``fixed_outer_regions_basis`` and ``utils.get_country_file_path``;
       C: ``country-directory-is-not-one-modern-file``.
     - #949 converter; #951 support maps; #719 removal.

Fixture format and rerun
------------------------

``tests/fixtures/migration_960/resources/cases.json`` contains original inputs,
small synthetic directory layouts, plain expected settings or exact current
errors, and a justification for each expectation. Expected path strings are
relative to the test invocation directory. ``files`` declares each synthetic
file as a partition or retained schema marker; the test driver writes those
files without deriving any expected result from the adapter.
The expectations were transcribed from source branches and checked against
historical implementations and existing resource-boundary tests where applicable;
they were not generated by serializing adapter output. The fixture data has no
Python/HBMCMC imports and can survive deletion of the current execution driver.

In the lean environment for the pinned current revision, first apply the cache
and thread environment setup in :doc:`migration_960`, preserving any existing
PyTensor flags, then run::

   .venv/bin/pytest -q tests/test_migration_960_resources.py

Observed result: **14 passed** on CPython 3.13.7. Ruff passed for this test module.
Interpreter/dependency and released-baseline provenance belong to the parent
migration evidence note. Historical directory defaults, read-permission filtering,
support-map content, relative packaged outer-map fallback, and complete retained
artifact loading remain inspection-only in this bounded resource characterization.
No inference equivalence is claimed.
