# Tasks

The current #719 branch implements namespace/adapters and the approved finite
API/command retirements, including the separately approved prepared factory.
Completed revision-specific checks are recorded below. The 26 October follow-up
adds explicit BC selection and reconciles positive owners; its fresh acceptance
remains pending. Delivered construction contracts are synchronized, while the
new BC delta and final archive remain open. Dated sections preserve prior
implementation, approval and validation history rather than current task status.

## Development-boundary amendment — 23 October 2026

#944/#961 supersede the future compatibility schedule. Checked 0.8 tasks below
preserve historical implementation/validation evidence, including their former
0.9 guidance; they do not renew that schedule. The durable
[evidence contract](../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence)
protects source observations and scientific expectations. The concrete #960
baseline, PR checklist and current owners are delivery instructions below,
not permanent API requirements. No deletion task is completed by this amendment.

### Migration evidence checklist and delivery order

The bounded removal scope is the `openghg_inversions.hbmcmc` namespace, including
`run_hbmcmc`, its dictionary-returning `params_from_config` adapter and legacy
data/basis utilities; plus temporary translation of `outputpath`, `outputname`,
`xprior`, `bcprior`, `sigprior`, `offsetprior`, `emissions_name`,
`outer_region_definition_file`, the `emissions_store`/`emissions_domain` and
`inner_emissions_store`/`inner_emissions_domain` flux selectors, and the output
names `hbmcmc`/`hbmcmc_postprocessing`. Wrapper-only `nit`, `nchain`, `verbose`
and `sampler_kwargs` translation retires with that wrapper. Existing explicitly
named API migrations retain their own contracts; there is no general exception
for historical signatures or result wrappers. The finite Python-alias amendment
below permits only its five named removals; other Python names retain their
existing contracts pending explicit per-interface decisions.

For each affected removal or deliberate behavior-change PR:

- Identify affected #960 ledger entries and plain cases; add bounded discoveries
  before removal with explicit source version and execution route.
- Compare the PR base and affected source with pinned development revision
  `0a32d73a5c4a72be0505a4ed682472e1613e49e9`, accounting for intervening changes
  and relevant merged PRs.
- Record disposition and follow-up owner: retained, mechanically migrated,
  deliberate scientific change, unsupported or unresolved. Preserve historical
  inputs/expectations; add dated modern comparisons rather than replacing them.
- Preserve or port relevant scientific coverage before retiring adapter/test
  plumbing. Test unchanged behavior on canonical inputs and deliberate scientific
  changes against separately agreed semantics. Distinguish inspected evidence
  from executed checks and identify the validated implementation revision.

#960 remains a pinned archive; refresh only affected evidence and live checks,
without requiring a whole-audit rerun. #719 owns runtime retirement and regression
migration. Complete #719, including the separately identified special-tracer
compatibility slice, before #964. #971 is a separate prerequisite for #964,
not for #719. #876/#964 own the agreed scientific-input/retained-population changes;
their historical witnesses remain distinct from new-semantics tests.

Finished converters (#949), launchers (#950), maintenance publication (#947),
integrated acceptance (#952) and complete migration guidance (#432) remain M2
release gates, not M1 development-removal prerequisites. #948 owns final source
coverage; #945 owns release numbering and unresolved public interfaces; #951
owns independent basis decisions. Completion of evidence does not resolve those
decisions or establish release readiness.

### CLI command-name amendment — 23 October 2026

The maintainer accepted this finite command map in
[#663](https://github.com/openghg/openghg_inversions/issues/663#issuecomment-6364272469)
and [#945](https://github.com/openghg/openghg_inversions/issues/945#issuecomment-6364278950),
then authorized the bounded contract amendment during #972 review. Implementation
is pending; this record does not remove any executable or parser alias.

| Retire | Canonical replacement |
| --- | --- |
| `openghg-inversions` executable | `rhime`, retaining its same subcommands |
| `run-rhime` under either executable | `rhime run --model standard` or `rhime run` |
| `run-rhime-multisector` under either executable | `rhime run --model multisector` |
| `run-rhime-nested` under either executable | `rhime run --model nested` |
| `run-co2` under either executable | `rhime run --model co2` |

The target has one installed executable, `rhime`, without forwarding commands
retained solely for naming compatibility. For this command-name migration,
preserve positional dates, `--kwargs`,
effective configuration, date/output override precedence, scientific execution,
result contents, outputs and artifact contracts. No replacement argument syntax,
Python package/distribution/install rename or persisted-identifier rename is agreed.
Positional dates and `--kwargs` remain open for discussion and possible revision;
preserving current argument meanings here does not finalize their design.
Broader Python names/imports require their own exact maps under #663/#945 and a
#776 revival; this finite CLI decision grants no general Python-name exception.

Apply the evidence checklist above to R1/R2 and CLI invocation cases. Source
inspection at this amendment found no `cli.py` changes since pinned
`0a32d73a5c4a72be0505a4ed682472e1613e49e9`; intervening `pyproject.toml` edits
affect dependencies/test markers, not console-script declarations. This is source
comparison evidence, not executed command/scientific validation. Preserve the
original #960 ledger, fixtures and expectations; add dated implementation
comparisons and port live tests to canonical routes when removal is implemented.

The implementation slice must update console scripts, parsers, tests, templates,
README/usage references and migration errors together. Preserve the distinct CO2
forwarding behavior used by canonical `run --model co2`; retain argument/override
assertions and verify retired commands and the second executable no longer ship.
#663/#945 own the command map and coordination, and #950 consumes it for launcher
guidance. This accepted CLI map can be delivered independently of unresolved
Python maps; it completes no runtime task or release gate.

### Named Python-alias amendment — 23 October 2026

The maintainer approved removal of these five currently implemented alias
assignments, with no continued aliases solely for naming compatibility:

| Retire | Canonical replacement |
| --- | --- |
| `run_rhime_multisector` | `run_multisector` |
| `run_rhime_nested` | `run_nested` |
| `RhimeMergedData` | `MergedData` |
| `RhimePreparedInputs` | `PreparedInputs` |
| `assemble_rhime_inputs` | `assemble_inputs` |

This amendment records permission and the target map, not runtime deletion.
The aliases remain direct assignments in the inspected source. Preserve the
canonical functions, class objects, constructors, argument/default meanings,
result contents, borrowed/lazy execution boundaries and scientific behavior.
Naming cleanup must not incidentally reinterpret scientific data. Supported
format/version acceptance, conversion or rejection remains the artifact owner's
decision, with no implied 0.7 format support or literal schema-identifier freeze.
Existing `run_rhime`, `RhimeRunSpec` and `RhimeResult`
migrations retain their separate explicit contracts; there is no general
signature/result-wrapper exemption.

Apply the migration evidence checklist to affected imports/callers and add
bounded source witnesses where the pinned #960 ledger does not cover an alias.
Preserve original historical inputs/expectations, compare affected source with
the pinned baseline and port relevant regression coverage before deletion.
The implementation slice updates exports, callers, tests, API documentation and
migration guidance together and checks the retired imports no longer resolve.
Broader symbols, types and module moves remain under #663/#945 and a #776 revival,
requiring exact replacement maps before their own removals. This finite decision
does not finalize CLI positional-date/`--kwargs` syntax or complete a delivery task.

## 1. Historical isolation delivery in 0.8

- [x] 1.1 Give retained RHIME aliases/validation runner/frontend owners and remove all
  production imports/re-exports of HBMCMC outside that package, including the
  RHIME dictionary adapter export and copied examples. Verify modern imports and
  representative configured/output-only execution with HBMCMC unavailable;
  preserve runner alias warnings, precedence and the independent flux-alias
  contract; verify canonical configuration knows no aliases or alias warnings.
- [x] 1.2 Wire historical selection into the actual HBMCMC CLI and normalized
  dictionary route after overrides and before canonical construction. Verify the
  reported NESW options through `run_hbmcmc.main`, file-backed selection,
  superseded missing inputs, dictionary normalization controls and documented
  canonical Path returns, self-contained discovery and clear missing-domain
  errors, early removed-option errors, unchanged caller mappings and established
  output policy.
- [x] 1.3 Add the small production import-boundary check and early modern errors
  for HBMCMC-only selectors. Verify absolute/relative import and re-export
  violations fail the check while legacy-entry tests remain allowed; verify
  errors provide modern replacements or the temporary legacy route.
- [x] 1.4 Update usage migration guidance and add a release fragment: document
  0.8 legacy-only routing and complete HBMCMC removal in 0.9 separately from
  other deprecations. Verify links, the full documentation build, and focused
  HBMCMC/configuration/resource tests, including corrected typed-path assertions.

## 2. Remove runtime compatibility at the development boundary

- [x] 2.1 Delete HBMCMC's wrapper, adapters and legacy utilities plus obsolete
  entry points and internal consumers, and remove temporary runner spelling/output-format/flux
  alias support after the affected evidence and regression gate. Retire special
  `use_tracer` handling in a separately identified #719 slice: every explicit
  occurrence, regardless of value, fails ordinary unsupported-option validation;
  omission remains valid.
  Complete #719 before #964; #971's separate filter retirement is not a
  prerequisite for #719. Leave unresolved
  public aliases and independent basis capabilities with #945/#951.
  Verify wheel/source distributions omit
  the package and no compatibility alias preserves its import path; run the
  import-boundary check and relevant modern workflow regressions.
- [x] 2.2 Update concise devel migration guidance and the removal fragment. Verify
  modern replacements and older-release guidance for legacy-only artifacts,
  with no accidental removal of independently supported output formats.
- [x] 2.3 Apply the approved #974 review clarification: remove legacy-specific
  request diagnostics and nested selector checks; validate supported names and
  applicable values generically while deferring known inactive values. Reconcile
  contracts and guidance, preserve affected evidence and run fresh checks on the
  rebased implementation before marking this follow-up complete.

- [x] 2.4 Deliver the 24 October approved prepared-input-factory retirement and
  review cleanup: preserve dated pre-removal cases, remove only the named factory,
  migrate its xarray caller without changing provenance/coordinates/laziness,
  localize request validation and basis helper ownership, remove migration-only
  import/duplicate-validation paths and stale template/doc text, reconcile the
  narrowly approved contract and run fresh implementation checks.

- [ ] 2.5 Deliver the 26 October approved explicit `bc_basis_algorithm="NESW"`
  selection with exact-file precedence and disabled-BC I/O bypass; reconcile
  positive contract owners, current status prose and Python-only prepared schema
  constants. Preserve pre-change evidence and obtain fresh focused/full/package/
  documentation checks before completing this follow-up. Persisted identifier
  migration remains #976/OPE-233; no release or VG integration is claimed here.

## 3. Promote only delivered contracts

- [x] 3.1 After 0.8 implementation and acceptance, verify the already-aligned
  grouped delta and active consumers still preserve the dependency boundary,
  then sync only delivered isolation/translation and
  configuration requirements. Rebase the remaining active delta to retain the
  pending 0.9 removal. Verify the durable alias contract and resource-I/O links
  express the accepted boundary without claiming deletion is delivered.
- [ ] 3.2 After fresh BC amendment acceptance, sync the remaining BC delta and
  archive this change. Verify durable owners, delivery status, links and absence
  of duplicate delivered `ADDED` requirements; preserve all dated evidence.
  The 26 October owner reconciliation does not complete archival.

## Issue 719 implementation evidence — 23 October 2026

- Base: `17e28ed081954b4155c9d85b66eaeef2ad81853c`, including merged #966/#972.
- Before-removal discoveries: dated `runtime_719_2026_10_23.json` in the #960
  retirement fixtures; original request/resource/released inputs and expectations
  remain unchanged. The appended migration ledger maps every removal slice to
  affected entries/cases, pinned/base comparisons, dispositions and follow-up owners.
- Source inspection: compatibility/runner/CLI/five-alias definitions are unchanged
  from pinned `0a32d73a`, except #956's explicit outer source concatenation in
  `legacy_data`; intervening modern merge/PyMC/fixed-OU/sampler checks remain in
  the base. This is inspection evidence, not historical scientific execution.
- Runtime delivery: namespace/adapters/executable surfaces, enumerated option and
  output aliases, every explicit tracer option, exact CLI map and five Python
  aliases are retired. Canonical callers and regression fixtures are ported;
  historical adapter-only drivers retire while plain source cases remain.
- Retained owners: canonical scientific/labeled/lazy behavior and modern codecs
  remain; #945 owns unresolved names/version, #951 independent basis decisions,
  #948 source coverage, #949/#950 tools, #946/#947 maintenance, #952 integrated
  acceptance and #432 complete migration guidance. #876/#964 retain imputation
  and nested alignment; #971 is a separate prerequisite for #964.
- Fresh implementation validation is recorded below only when executed. Earlier
  #972 documentation/contract checks and draft #964/#973 test counts are not
  runtime evidence for this revision. Task 3.2 remains pending; no archive or
  release-readiness claim is made.

### Initial complete acceptance and correction

At immutable implementation `01e34d0c32bc342863b24f3ec6d6858ac105aa6b`,
Slurm array `19667378` (callback `19667379`) executed the full suites:

- Locked Python 3.12 and 3.13 each: **4,124 passed, 14 failed, 8 skipped,
  4 deselected**. Thirteen failures asserted retired alias translation,
  command names or diagnostics. The remaining installed CO2 workflow completed
  preparation/prior/sampling before its test assumed a Git revision in the
  archive snapshot; installed provenance explicitly permits an unknown revision.
- Lint, all **4 borrowed-typing checks**, and **docs-full** passed. Documentation
  reported **10** existing duplicate/ambiguous API-reference warnings.
- Moving OpenGHG devel: **4,121 passed, 17 failed, 8 skipped, 4 deselected**;
  the same fourteen plus three independently baselined Zarr 3 serialization
  failures (prepared-input, source-covariance and linked-handoff artifacts).
- GitHub CI run `39983254785` used a Git checkout and therefore had thirteen
  stale-expectation failures (4,125 passes) on each locked interpreter; moving
  dependencies had sixteen failures (4,122 passes). Docs run `39983254841` passed.
  These failed initial runs are evidence, not final acceptance success.

The correction ports the remaining tests to canonical overrides/commands and
ordinary unsupported-option errors, retaining path precedence and scientific
assertions. CO2 provenance permits `None` and still checks forty hexadecimal
characters when known and equality between preparation/sample manifests. The
unknown-revision path was executed against the original no-Git snapshot: **2
checks passed**. No production code or scientific equations/defaults changed.
A rendered migration-guide check also adds the required explicit likelihood
choice to its Python/INI examples. Final revision-specific retry remains pending.

Correction verification: all **117** tests in the three affected standard
handoff/API/resource-boundary modules pass, preserving canonical flat/grouped
file and override path precedence while rejecting historical aliases. The
supplementary `legacy_codec_witnesses_2026_10_23.json` preserves ten explicit
codec inputs/expectations and exact source hashes without legacy imports;
it is source-inspected supplementary capture after the initial retirement.
All original #960 fixtures remain unchanged. Ruff, strict OpenSpec **10/10**
and whitespace checks pass. Final complete retry results are not yet available.

### Final runtime acceptance — 23 October 2026

Immutable revision `a4fd9ae4c170d3620ea7425120671394712662de` passed the
complete locked scientific suites in Slurm array `19668217`:

- Element 0: Python **3.12.11**, **4,142 passed**, 8 skipped, 4 deselected;
  lint, **4 borrowed-typing checks**, and **docs-full** passed.
- Element 1: Python **3.13.7**, **4,142 passed**, 8 skipped, 4 deselected.
- Element 2: moving OpenGHG devel, Python 3.12.11, **4,139 passed**, 3 failed,
  8 skipped, 4 deselected. Only the independently baselined Zarr 3 failures
  remain: linked handoff/source covariance string codecs and prepared-input
  Blosc encoding. This environment is not reported as passing; its unrelated
  provider/artifact compatibility remains outside #719 and with final integrated
  acceptance under #952.
- Callback `19668218` completed; the durable ticket reported all terminal states.
  Jobs used two CPUs, two Dask workers, one numerical thread and 8 GiB each.
- Full documentation built successfully with eleven Sphinx warnings. Ten concern
  existing duplicate/ambiguous API references; the introduced supplementary-ledger
  heading underline was corrected separately and checked by a bounded render.
- Fresh wheel/sdist and installed-wheel smoke checks at `a4fd9ae4` passed:
  no HBMCMC namespace or five retired aliases, canonical imports/model routes,
  and sole `rhime` executable. The corrected Python and INI migration examples
  resolve without acquisition/inference. Strict OpenSpec **10/10**, Ruff and
  `git diff --check` pass.
- GitHub Actions on the same head: test run `39987314862` and documentation
  run `39987314865`; supported locked suites and docs pass, while the moving
  dependency gate retains its known Zarr failures. PR #974 remains draft.

Tasks 2.1/2.2 are delivered with this evidence. The final follow-up fixes the heading, records this evidence and removes three
duplicate canonical `__all__` entries; scientific execution, serialization, tests,
package metadata and the lockfile remain unchanged from validated `a4fd9ae4`. Task 3.2 remains unchecked for subsequent specification
promotion/archive after merge; approved durable targets were not weakened or
rewritten. No merge, publication, release number or release-readiness claim.

Final documentation check: a bounded Sphinx warning-as-error render of six
current migration/CLI/configuration/evidence guides passed with zero warnings
in the existing documentation environment after the heading fix. References to
nonincluded pages were outside this bounded render. Rendered examples and
supplementary evidence wording were inspected. All eighteen original fixtures
remain byte-identical and all six supplemental source blob/SHA256 records match
their exact commits. GitHub's marked Numba checks also pass on both locked
interpreters. Independent final source, documentation and CI reviews found no
material retirement gaps; only three duplicate canonical export entries needed
removal, with focused import/command validation recorded below.

After the three export-list deduplications, **41 focused import/public-API/CLI
checks passed** (8.51 s). A source AST scan finds no duplicate literal export
lists. Ruff, strict OpenSpec **10/10** and whitespace checks pass. No full-suite
rerun is claimed for the final export-list/markup/evidence-only follow-up.


## Review 5634401507 follow-up — 23 October 2026

The maintainer clarified that modern validation rejects unknown names and invalid
applicable values generically, while known inactive groups defer value validation.
The earlier every-retired-output-value-occurrence wording contradicted phase-only
resolution; its approved correction preserves inactive values and introduces no
historical-token checks. Explicit `use_tracer` is still an unknown name.

- Remove the residual `_request_validation` diagnostic tables and dedicated
  nested historical-inner-selector checks/extraction names; use ordinary unknown
  option validation. Replacement guidance remains in documentation/#949 tooling.
- Reconcile the durable configuration/retirement wording and active retirement
  delta/design with generic name and phase-owned value validation. Preserve
  canonical science, independent codecs and #951 basis support.
- Compare the affected sources with pinned `0a32d73a` and rebased base `b288ac25`.
  `requests.py`, `_runner_compat.py` and `nested.py` are unchanged between those
  source revisions; the diagnostic helper was introduced by this retirement.
  #975 modern serialization changes remain in the rebased base. New plain
  `review_974_2026_10_23.json` cases capture source hashes, ZIP expectations,
  old diagnostics and known-inactive output-value semantics.
- Preserve all original archived cases and dated validation above. Those earlier
  results do not validate the rebased follow-up; its executed checks will be
  recorded separately. Task 3.2 remains pending. No new validation result,
  archival or release-readiness claim is made by this planning reconciliation.

Rebased review follow-up checks (working tree atop `8b7fe6fb`, base
`b288ac2530ad93ae8f67c5f1d47ee8ee7b666000`):

- **401** focused request/acquisition/resolved-run/migration boundary checks pass.
- **211** CLI/configuration/nested/phase-output checks and **2** direct generic
  invalid-output checks pass. Preparation defers known inactive output values
  across standard, multisector, CO2 and linked CO2; output consumers reject
  invalid effective values before saved scientific replay.
- **112** codec/source-covariance/linked-configuration regressions pass, verifying
  the retained #975 modern fixes. Counts overlap other suites and are not summed.
- Full Ruff, strict OpenSpec **10/10** and `git diff --check` pass. A bounded
  seven-guide warning-as-error Sphinx render passes with zero warnings; references
  to nonincluded pages remain outside that bounded check.
- Original eighteen historical fixtures remain byte-identical. New evidence
  preserves #975's retired ZIP codec source/test expectations as inspected
  observations, rather than claiming their historical execution or restoring
  the removed namespace. Modern #975 codec/source-covariance files and their
  three focused regression modules are unchanged from the rebased base.

Fresh complete tox and wheel/sdist checks are pending on the committed rebased
revision. Earlier acceptance results above do not validate this follow-up.
Task 2.3 remains unchecked until those results are inspected; 3.2 remains pending.

### Rebased review acceptance — 23 October 2026

Immutable revision `48057c5e630ce3f1fcb56dbad5ba7dbc675f84de`, based on
`b288ac2530ad93ae8f67c5f1d47ee8ee7b666000`, passes the fresh complete matrix
in Slurm array `19671842`:

| Environment | Executed result |
| --- | --- |
| Locked Python 3.12.11 | 4,162 passed; 8 skipped; 4 deselected |
| Locked Python 3.13.7 | 4,162 passed; 8 skipped; 4 deselected |
| Moving OpenGHG devel, Python 3.12.11 | 4,162 passed; 8 skipped; 4 deselected |
| Lint / borrowed types / docs-full | Pass / 4 passed / pass |

All three primary jobs completed with exit `0:0`. Each used 2 CPUs, 8 GiB,
two Dask workers and one numerical thread; maximum reported RSS was
6,308,812 KiB. The durable reconciler delivered the terminal wakeup under ticket
`sw-20261023T211050Z-738087746c24`; the independently registered `afterany`
callback `19671843` was still queued when inspected, so its completion is not
claimed. Local log files are `slurm-719-review-19671842_{0,1,2}.out`.

GitHub Actions independently pass on the same commit:
[tests 40024729738](https://github.com/openghg/openghg_inversions/actions/runs/40024729738)
and [docs 40024729782](https://github.com/openghg/openghg_inversions/actions/runs/40024729782).
The earlier three moving-dependency Zarr failures no longer occur after rebasing
onto #975. Full docs build successfully with ten existing duplicate/ambiguous
API-reference warnings; no warning-free full build is claimed. The seven-guide
bounded warning-as-error check remains separately recorded above.

Fresh wheel/sdist content and installed-wheel checks at this exact revision also
pass: the HBMCMC namespace and five retired Python aliases are unavailable,
canonical APIs/model routes import, and only `rhime` is installed as an executable.
Together with the focused boundary/phase tests, strict OpenSpec 10/10 and original
fixture/source-hash checks above, these results complete review task 2.3. The
following record-only commit changes this task file; production code, tests,
dependencies, scientific defaults and package metadata remain identical to the
validated revision. Prior dated failures and observations remain intact.

PR #974 remains draft and #719 remains open pending review/merge. Task 3.2 remains
unchecked for subsequent specification promotion/archive. No merge, publication,
release number or release-readiness claim is made.


## Approved scope addendum — 24 October 2026

[Decision 6380451664](https://github.com/openghg/openghg_inversions/pull/974#issuecomment-6380451664),
following [analysis 6380430675](https://github.com/openghg/openghg_inversions/pull/974#issuecomment-6380430675),
adds exactly `PreparedInputs.from_legacy_inputs` to the removal list. The
replacement is canonical construction with labeled `site_metadata`. This is an
additional compatibility-constructor disposition under #945/#663, not a sixth
alias or a #951 basis-reader/writer decision. No other constructor, persisted
schema, scientific variable, unit or artifact capability is removed.

- Before production edits, the plain
  `factory_and_cleanup_2026_10_24.json` and appended #960 ledger captured the
  exact `e1d020f0` signature/source hashes, two original factory cases, sole caller,
  provenance fallback/overrides, coordinate precedence/validation and owners.
  Factory source matches pinned `0a32d73a` and base `b288ac25`; the original
  eighteen #960 fixtures and earlier dated evidence/results remain unchanged.
- Port factory cases to canonical construction and retain xarray scalar/mobile
  coordinate, site/period-order and serialization coverage. Add provenance tests
  for absent/existing namespaced/plain metadata, input nonmutation and lazy arrays.
- Address [review 5637015484](https://github.com/openghg/openghg_inversions/pull/974#pullrequestreview-5637015484):
  remove the migration import hook and raw parsing before source-conflict errors;
  keep only resolved nested likelihood checking; give outer CO2 requests one
  generic name-validation owner and let consuming parsers own applicable values.
  Preserve inactive-name rejection and inactive-value deferral.
- Remove the root flux-alias helper and keep its private sentinel/alias logic
  inline in the sole basis consumer, retaining independently supported
  alias/conflict behavior and removing an unapproved warning date. Remove dead HBMCMC template branches and stale translation prose.
- Fresh runtime/package/documentation checks are pending for this scope addition.
  Task 2.4 stays unchecked until implementation and relevant checks finish;
  task 3.2 remains pending subsequent closeout. Prior complete matrices and the
  review's separate probes do not establish validation of this new revision.

Focused checks on the completed 24 October working tree, before committing the
related slices atop `e1d020f0`:

- **149 passed** across standard API, migration requests, resolved runners,
  nested core/runner and independent basis input coverage. The remaining
  prepared `model_builder` source-conflict assertion was then ported and its
  separate **1 test passed**, preserving the callback sentinel.
- **133 passed** across xarray preparation and prepared-input serialization,
  including two real-data cache round-trip/run cases. Four new provenance,
  coordinate/nonmutation and shared-lazy-array cases also passed against the
  original factory before deletion and against canonical construction afterward.
- CO2 configuration/CLI/phase-output coverage: **108 passed**; request/tracer
  ownership coverage: **325 passed**; broader ordinary/linked handoff, parity,
  model inference and installed full/prepared/saved/output coverage: **106 passed**.
  The review's no-acquisition probe now observes one whole-request name traversal
  rather than six. These suite counts overlap and are not summed.
- Full Ruff, strict OpenSpec **10/10**, original-fixture/source-hash verification
  and `git diff --check` pass. A bounded seven-guide Sphinx warning-as-error render
  passes with zero warnings; nonincluded references are outside that check.
  Fresh API-page generation with the simplified Jinja templates also passes,
  without HBMCMC or root flux-helper pages.

These focused checks do not replace committed-revision package checks or the
complete Slurm matrix. Task 2.4 remains unchecked pending fresh acceptance;
previous complete matrices retain their original revisions and dates.


### Factory and cleanup acceptance — 24 October 2026

Immutable revision `bbc9f64264b46248f1bd4d5226d4560d99485758`, based on
`b288ac2530ad93ae8f67c5f1d47ee8ee7b666000`, passes fresh complete acceptance
for the approved factory addition and review cleanup in Slurm array `19679134`:

| Environment | Executed result |
| --- | --- |
| Locked Python 3.12.11 | 4,167 passed; 8 skipped; 4 deselected |
| Locked Python 3.13.7 | 4,167 passed; 8 skipped; 4 deselected |
| Moving OpenGHG devel, Python 3.12.11 | 4,167 passed; 8 skipped; 4 deselected |
| Lint / borrowed types / docs-full | Pass / 4 passed / pass |

All three primary jobs and registered `afterany` callback `19679135` completed
with exit `0:0`; ticket `sw-20261024T055258Z-15b3adfbfe2f` delivered the terminal
wakeup. Each primary job used 2 CPUs, 8 GiB, two Dask workers and one numerical
thread, with maximum reported RSS 5,570,960 KiB. Local logs are
`slurm-719-oct24-19679134_{0,1,2}.out`.

GitHub Actions independently pass on the same revision:
[tests 40067887464](https://github.com/openghg/openghg_inversions/actions/runs/40067887464)
and [docs 40067887409](https://github.com/openghg/openghg_inversions/actions/runs/40067887409).
Full documentation retains ten existing duplicate/ambiguous API-reference
warnings; no warning-free full build is claimed. The separately recorded bounded
seven-guide warning-as-error render passed without warnings.

Fresh wheel/sdist content and installed-wheel smoke checks at `bbc9f642` pass:
no HBMCMC namespace, root flux helper, five retired aliases, compatibility factory
or migration-only import hook; canonical APIs/model routes, the existing prepared
schema and sole `rhime` executable remain. Focused provenance/coordinate/laziness,
source-conflict and CO2 ownership regressions are recorded above, including the
pre-removal comparison cases. Full Ruff, strict OpenSpec 10/10, whitespace checks
and original-fixture/source-hash verification pass. Original historical fixtures,
prior dated expectations and failed/successful validation records remain intact.

An independent subagent audit confirms the counts and passing canonical workflow
modules. All tracked production, test, specification, usage, configuration and
lock files in the executed snapshot match this revision; four generated API pages
have only cosmetic heading/newline differences.

These results complete task 2.4. The acceptance-record commit changes only this
task file; production code, tests, templates, dependencies, scientific defaults
and package metadata remain identical to validated `bbc9f642`. All seven inline
review threads have replies and are resolved. PR #974 remains draft and #719 open
pending review/merge. Task 3.2 remains unchecked for subsequent specification
promotion/archive. No merge, publication, release number or release-readiness
claim is made.


## Approved owner and BC amendment — 26 October 2026

- Approval: review 5649950083 and the maintainer's bounded implementation request.
- Before edits, `retirements/review_974_2026_10_26.json` captures `1c15954c`
  sources against pinned `0a32d73a` and base `b288ac25`, original BC regression
  inputs/expectations, schema constant values and requirement dispositions.
- Delivered canonical prepared construction and existing preparation semantics
  move to `rhime-scientific-inputs`; no #964 science is imported. Configuration
  owns options/applicable values, resource I/O owns BC loading, existing handoff
  and persistence contracts own stored formats/version/replay. Scientific model
  semantics and all-retained-chain products keep scientific owners.
- Durable requirements describe supported interfaces and generic validation.
  Historical deletion inventories/maps remain migration evidence/guidance.
  The duplicated delivered factory/isolation `ADDED` requirements are removed;
  only the new BC amendment remains in the active delta.
- Oct26 supersedes the operative schema-freeze restriction. Dated Oct24 approval
  and unchanged-schema observations stay intact. Python constants lose the
  redundant RHIME prefix; stored identifiers/versions/layout do not change here.
  #976 owns that separate migration; OPE-233 owns selected VG baseline integration.
- Fresh validation is pending for this amendment. Prior green matrices remain
  revision-specific evidence and do not complete task 2.5. Task 3.2 remains open;
  no archival, merge, publication or release-readiness claim is made.

Focused checks on the completed 26 October working tree atop `1c15954c`:

- BC resource/route checks: **77 passed**; the final explicit BC-selection
  module, including nested inner policy: **16 passed**. Broader configuration,
  ordinary/multisector handoff, nested and resolved-run coverage: **345 passed**.
  Directional labels/default sensitivities, lazy borrowed inputs, file time/grid
  alignment, disabled-BC bypass and selected-resource early failures are covered.
- Prepared serialization/public API/CO2 preparation: **117 passed**, including
  real-data NetCDF/Zarr round trips and replay. Python constants use their new
  names; assertions retain the exact stored schema identifier and version.
- Documentation contracts: **50 passed**; both modern INI templates and the
  satellite schema example: **3 passed**. Counts overlap and are not summed.
- Ruff, strict OpenSpec **11/11**, `git diff --check`, 37 specification
  links/anchors and 54 source revision hash records pass. All eighteen original
  fixtures remain byte-identical to the base. A bounded nine-page Sphinx
  warning-as-error render passes; references outside that set are excluded, so
  it does not replace the complete documentation check.
- Separate follow-ups are created: GitHub **#976** owns exact persisted schema
  identifiers/versions and one selected campaign migration; Linear **OPE-233**,
  under **OPE-145**, owns the supported future VG/OGI baseline and its integration.
  No VG artifacts, environments or campaign references have been modified.

Fresh committed-revision wheel/sdist and full tox results remain pending;
task 2.5 is intentionally unchecked. The existing green GitHub runs
`40069509661` (tests) and `40069509766` (docs) are the `1c15954c` baseline,
not validation of this amendment. Task 3.2 remains pending acceptance and
subsequent BC contract promotion/archive.
