# Tasks

The maintainer approved this plan for implementation on 11 October 2026. These
are implementation tasks, all unchecked until their acceptance evidence exists.
Start with a bounded implementation PR stacked on #841. This planning change
itself changes no runtime behavior. Follow [design.md](design.md) and the three
new capability specs and the
[configuration capability](../../specs/rhime-configuration/spec.md) and
[resource-I/O prerequisite](../../specs/rhime-resource-io/spec.md).
Their durable capability specs describe the delivered baselines.
The four delivery sections below define proposed bounded
implementation-child scopes for OPE-207; they do not create Linear issues or
represent one assignment spanning every family.

Keep each saved-handoff writer, reader and caller in the same implementation PR.
Deliver applicable commands, examples, documentation, focused checks and release
notes with each family. The standard route must pass its scientific acceptance
gate before multisector or ordinary CO2 adoption; those two adoptions can then
proceed independently. Cached CO2 adoption follows the ordinary CO2 handoff and
result conventions. Preparation and ordinary-model work can proceed in parallel
with that proof. OPE-165 needs the actual shared interfaces it adopts, not
completion of every OPE-207 delivery. A passing mock-forwarding test alone clears
none of the scientific gates. Complete #769's production predictive integration,
following #795, before adding new cached predictive-stage plumbing.

The reviewed baseline is devel `7269f50f`: reuse the merged acquisition/inference
ownership from #773/#774 and configuration/preparation work from #822–#825.
Cleanup #826/#830, legacy isolation #832 and boundary work #834–#837 are merged.
#854 records existing model selection and composition in current capability specs;
consume those contracts without reopening their implementation or counting them
as completed handoff tasks. The #856 coordinate-registry limitation stays separate.
#840 addresses #838's explicit basis paths; #841 addresses #839's runtime-flux/
operator-loading work and remains under review. Consume resulting APIs and record merged evidence rather than duplicating
these owners or treating open PRs as landed. Coordinate #808's removal of redundant
manifest/hash gates and its consumer fixes within the implementation that needs it, without
making it a separate delivery prerequisite. Acquisition persistence/replay belongs to
deferred #829 and is not required for prepared/sample handoffs. OPE-164 records
the initial ordinary/cached CO2 staged delivery; the CO2 sections below adopt the
replacement handoffs rather than reopening that completed scope. Linked CO2/O2
adoption remains in OPE-165, retaining its completed numerical persistence.

The 13 October review adds one bounded prerequisite before #866:
[#868](https://github.com/openghg/openghg_inversions/issues/868) resolves resource
paths at the runner/frontend boundary and moves optional basis saving from
`prepare_basis` into runners. Consume that delivery, including its caller/API
migration, without reimplementing it here. #840's exact basis destinations are
already an acceptance baseline; the rebasing in #866 is a regression against it.

## 1. Complete standard preparation-to-products route

Implementation started in [#843](https://github.com/openghg/openghg_inversions/pull/843),
stacked on #841 at `7f84645a`. Its first slice covers live sampler selection and
sampler-free result/replay foundations from 1.2 and 1.8; it does not complete either
task's acceptance across the new handoff APIs or the standard delivery gate.
Finish the local validation and descriptive-metadata simplifications in #843;
they do not require preliminary cleanup or new configuration infrastructure.
At implementation head `61d0036c`, 39 focused sampler/prepared tests, 67 replay/stage
tests and 330 configured-run/compatibility tests passed. Broader family coverage
had 252 passes and one direct prior-predictive Numba precision failure reproduced
with the same selection on #841. Independent review, changed-path Ruff and diff
checks passed. Keep the task checkboxes open pending their complete acceptance;
the PR records the bounded implementation and validation details.

### Standard-route integration evidence, 14 October 2026

PR #866 integrates devel `52a6b12c`, including merged #869's resource I/O,
canonical-only configuration and HBMCMC isolation. #863 delivered #840/#841's
operator/current-flux contracts, #843 supplied invocation-local samplers, #855
supplied stage-I/O cleanup, and #795 supplied predictive integration. The separate
configuration change owns the implemented `DataConfig` / `RhimeConfig.data` /
`data` grouping; its sync/archive is complete in the 16 October 2026 configuration closeout.

The runnable `examples/standard_handoffs.py` prepares once, samples two Normal
prior choices, and makes basic and inversion outputs from each saved posterior.
`tests/test_standard_handoff_parity.py` compares those results with two full
in-memory runs and checks Normal density against SciPy (`rtol=1e-6`, `atol=1e-5`
for float32 graph arithmetic). The generated recipe supplies an independent
Student-t density check and saved replay. Default log-normal sampling retains
both `x` and `x_latent`; explicitly selecting only `x_latent` and `mu` fails
before a completed sample manifest.

The unequal-source acceptance supplies inventory fields `[[1,2],[3,4]]` and
`[[.3,1.2],[2.1,.4]]`. It independently reconstructs flux from scaling draws and
their sum, checks first-source basis weighting separately, reverses source order,
and verifies one native-flux resolution per preparation. The saved operator has
a different historical prior. Deleting that operator and forbidding acquisition
and basis export during direct scientific preparation/replay still passes.
This exercises supplied standard merged data; it does not expand configured
source-count support or claim multisector handoff adoption.

Validation after integration:

- Configuration, HBMCMC shim and full-run tests: 480 passed.
- Scientific parity, basis/resource paths, resolved runs, generated/custom recipe
  consumers: 159 passed.
- Handoffs, deferred stages and CO2 staged outputs: 98 passed; output contracts:
  9 passed.
- CLI and nested consumers: 44 passed initially; amended CLI regressions also
  cover flat saved-model claims and rejected unused prior-readiness sampler overrides.
- Additional resource, sampler, replay, HBMCMC boundary and builder coverage:
  142 checks passed (three focused reruns after migrating one stale test and
  selecting a writable compiler cache).
- Grouped full-run alias/canonical-boundary and phase compatibility coverage:
  172 passed. Prepared serialization, tutorials and xarray adapters: 134 passed
  after updating their mocked posteriors to satisfy the retained-state contract.
- Changed-path Ruff and both active changes' strict OpenSpec validation pass.

Review follow-up on 14 October adds a real fixed-error/offset regression:
a declared scientific offset must be retained even when latent offsets exist;
predictive sampling may remain disabled. Empty, missing and nonfinite returned
prior evidence is saved alongside a failed readiness report, and serialization
errors propagate. Phase errors reuse the modern owner's resource migration
guidance while collecting all unsuitable keys. Obsolete complete-request
validators, duplicate replay checks and the retired standard envelope branch
are removed; deferred-family behavior remains covered. Combined phase, handoff,
scientific parity, RHIME, contract and compatibility checks passed 408 tests;
the remaining custom-likelihood mock needed its declared offset added and passed
on rerun. The separate handoff/deferred-family selection passed 94 tests.

Review restructure, 14 October 2026 (design decision 8): `run_standard`,
`prepare`/`sample`/`postprocess`/`prior_predictive` and the standard branch of
`run_rhime_from_prepared_inputs` now call one step sequence (`acquire_merged_data`,
`prepare_standard`, `export_basis`, `build_standard`, `sample_rhime_model`,
`make_standard_rhime_result`, `make_standard_rhime_outputs`). Preparation reloads
the recorded `DataConfig`; the model record is owned by `RhimeModelSpec` and no
longer computes lazy priors (execution-count regression); provenance comes from
the installed-package owner on both routes; outputs are returned rather than
patched. One loader (`load_rhime_params`) serves full runs and phases. Preparation
steps pass site datasets and `SiteOptions`, keeping `MergedData` acquired-only.
The multisector envelope moved to `_multisector_stages.py` with restored tests.
New acceptance covers custom-likelihood saved sampling against the full route and
retrying a failed preparation into the same destination.

Settled review decisions, 14 October 2026: standard construction now receives
`RhimeModelSpec` without output choices. Standard prepared execution rejects the
complete-model-builder hook while preserving custom likelihoods and alternative
preparation. Posterior `var_names` overrides fail at the sampler boundary; result
and inversion-output construction trust the produced trace. Full and saved
configuration paths resolve from the invocation working directory; final products
share the full runner overwrite behaviour. Callers own checkpoint allocation,
collision avoidance and isolation; manifest-last completion remains required.
These decisions supersede the earlier standard complete-builder acceptance and
trace-retention test details above.

Revised delivery plan, 15 October 2026: standard full/prepared/replay execution
returns `InversionOutput`, including for no-output runs. The Python `run_rhime`
name is removed in 0.8 with helpful import guidance; CLI files/logs remain
unchanged. Consolidate the builders into one eager operation and explicitly
materialize related selected inputs and enabled priors in callers. One sample
reader/writer owns reference reconstruction and preparation generation binding.
Runners own optional basis, trace and inversion-output exports without rewriting
scientific inputs for receipts. The earlier recorded builder/result shapes are
superseded; their historical test counts do not establish the revised acceptance.
Further cleanup review at f4fb9542: standard prepared input now enters through
`run_standard(prepared=..., config=...)`; the old standard prepared runner is
removed, leaving its multisector compatibility interface. One public operational
artifact saver replaces repeated write/export blocks. Public preparation loads
return the generation, and external sample saves must provide it. Relative
preparation references and required/used manifest fields replace the earlier
absolute-reference and unused-timing records. This supersedes the intermediate
shapes described in the dated validation history above.
At that revision, task 1.11 remained pending actual maintainer review.
The acceptance recorded below supersedes that status.

No multisector/CO2 adoption, final integrated-family gate, or spec promotion is
claimed by this standard delivery. CI verifies the pushed integration's full
suite, supported Python versions, typing and documentation.

- [x] 1.1 Consume the separate [configuration delivery](../archive/2026-10-16-group-rhime-configuration-by-owner/tasks.md)
  delivering `RhimeConfig.data: DataConfig` and the grouped `data` key. Record
  its implementation evidence and verify full and phase-only inputs resolve the same applicable choices
  without demanding unrelated sections, touching arrays or mutating caller values.
  Cover omitted/explicit model `use_bc` against enabled/disabled preparation,
  preserved legacy flat routing, and omitted/equivalent/conflicting `bc_freq`
  claims against the baked temporal policy.
- [x] 1.2 Use live `sampler="default"` consistently in new and existing
  default-selecting entry points as a request for a fresh family-default
  sampler on each invocation, without selecting execution defaults from saved
  metadata.
  Give custom `RhimeSampler` invocations owned copies of nested keyword values;
  verify list/tuple/array fidelity, isolated overrides and unchanged caller values.
  Reject `idata_kwargs.coords`/`dims` once at configuration parsing or sampler
  construction; trust valid resolved choices in resolution and sampling. Do not
  duplicate PyMC's model-dependent validation or add a sampler options class.
  Keep ownership copies where backend side effects require them.
  Verify configured full runs preserve `config.sampler`, omitted/explicit default
  calls select fresh defaults, and invalid strings/explicit `None` fail at all live
  sampling boundaries. Remove `None`-means-default compatibility adapters and
  provide migration errors; update annotations, docstrings, examples and caller
  tests together. Keep `None` as absence on results and object-only arguments on
  lower-level operations that already require a resolved sampler.
- [x] 1.3 Consolidate standard scientific calls at their existing owners in the
  visible order observation errors, filtering, basis, sensitivities and assembly.
  Keep acquisition/supplied-data selection in the runner, explicitly selecting
  each operation's named arguments from `config.data`; verify scientific
  preparation accepts acquired inputs with retrieval disabled and no export writes.
  Consume #837's explicit assembly/execution boundaries and the named joint PyMC
  materialization boundary. Verify retained-site/averaging alignment, empty-set
  rejection, borrowed lazy arrays and custom-likelihood contracts across
  full, supplied-merged/prepared and saved routes; remove forwarding-only wrappers.
  Standard construction receives model choices without output configuration;
  complete-model builders remain only at the retained multisector boundary.
  Preserve the current model-composition seam and registered-coordinate owner;
  do not broaden the canonical custom-likelihood contract through role handling.
- [x] 1.4 Consume #838/#839's explicit basis and runtime-flux contracts. Verify a
  selected operator A binds current acquired native flux B, saving retains B and
  replay restores B without acquisition or the original inventory paths. Cover
  first-source weighting separately from all-source retention, source order and
  once-per-domain/grid native-flux resolution; use unequal source fields and an
  independent reconstruction calculation. Record which prerequisite PRs supply
  these behaviors rather than implementing them again here.
  Consume #868's runner-owned optional basis save and resource-path resolution;
  verify the new routes preserve the existing exact-export behavior rather than
  introducing checkpoint-directory rebasing.
- [x] 1.5 Validate declared coordinate/output roles even when immediate products
  are disabled. Share the existing variable-role/output-contract owner between
  predictive sampling and postprocessing; remove independent concentration-name
  inference in the sampler. Keep species-dependent mass conversion in
  postprocessing. Verify custom-likelihood construction with model-only choices,
  posterior variable-selection rejection at the sampler boundary, and necessary
  role validation at external save/load boundaries. Standard complete-model
  builders are explicitly rejected; direct scientific arguments, conflict checks
  and established returns remain usable.
- [x] 1.6 Implement standard `prepare` returning one absolute manifest path,
  using existing numerical codecs and #808's bounded contract. Omit the obsolete
  filtered checkpoint without adding acquisition persistence. Verify acquisition
  and retained subsets with unequal site options, original requested dates,
  ordinary missing/malformed inputs, save/reopen and relative/shared-cache paths.
  Use JSON records and the numerical formats supported by existing codecs.
  Keep the writer, loader and callers together.
- [x] 1.7 Implement standard `sample` from that manifest and current model/sampler
  choices. Record bounded actual inference provenance and embedded `OutputContract`.
  Verify two compatible priors/likelihoods reuse unchanged preparation, while
  missing required arrays or changed baked policy fail before inference. Distinguish
  scientific compatibility from descriptive provenance. Record a Git revision
  when available without making it a requirement for execution.
  Retain scientific output state `x` by default. Check that explicit omission
  cannot complete an incompatible handoff without supported reconstruction;
  `_latent` values alone do not satisfy the scientific output role.
- [x] 1.8 Implement standard `postprocess` from one sample manifest and current
  output choices. Replayed results use `sampler=None` and plain saved metadata;
  configuration construction is permitted, but replay does not build a model or
  run inference. Use retained, referenced or supplied run context without filling
  missing provenance from current defaults. Verify save/reopen and two output
  choices from the same samples, preserving requested window/retained metadata
  and matching live products. Let ordinary loaders report missing/malformed
  files and scientific boundaries reject incompatible roles, labels or units.
- [x] 1.9 Preserve standard prior-predictive/readiness catch boundaries and
  independent diagnosis with its supported inputs. Verify
  fail/pass/unknown, strict exit status, invalid predictive evidence and propagation
  of loading/serialization errors; readiness does not silently perform posterior
  inference or repair missing evidence during replay.
- [x] 1.10 Deliver standard one-manifest CLI forms, full/phase-only examples and
  cookiecutter consumer acceptance with this route. Verify family
  selection, explicit mismatch rejection, once-only override resolution, help and
  destination behavior, and an executable generated scientific recipe. Preserve
  the model-selection baseline's full-run choices and aliases; phase commands
  do not implicitly extend the full-run CO2/config-file interface. Update
  stage guidance/docstrings with arguments, returns, file lifetime and reuse limits;
  remove premature `openghg-run` references in affected guidance. Coordinate the
  0.8 staged-contract release note with #808, distinguishing preserved numerical,
  product and direct scientific APIs from retired staged envelopes/aliases;
  add no 0.7 staged compatibility adapters.
  Cover canonical file options versus deprecated overrides, one error naming all
  unsuitable keys, and the shared invocation-relative configuration path base and manifest-relative references.
- [x] 1.11 Clear the standard gate with a runnable preparation/two-model/two-product
  example, distinct sample destinations and real prior or likelihood
  calculations checked against an independent calculation or existing regression
  reference. Demonstrate equivalent full execution with intermediate saves
  disabled. Record focused/broader relevant test results and
  review scientific parity, borrowed-array behavior and recipe readability before
  family expansion.

## 2. Multisector reusable-handoff adoption

- [x] 2.1 Give multisector concrete preparation, inference and products calls using
  the standard delivery's established persistence mechanics through ordinary
  functions. Verify full/staged/replay preserve sector/source-labelled inputs,
  scientific roles, retained-site subsets, all-source flux/source order and borrowed
  arrays without introducing shared scientific dispatch. Follow design decision 8:
  the full multisector runner and its saved phases call the same step functions
  (including a `prepare_multisector(merged, ...)` preparation owner), then delete
  the `_multisector_stages.py` envelope, its `stages.py` dispatch and the CLI
  `--prepared-inputs`/`--posterior` arguments for multisector.
- [x] 2.2 Verify compatible sector-prior/likelihood changes reuse preparation and
  genuine preparation changes fail. Exercise saved-result products with unchanged
  sample meaning and current reporting choices, including graph-free
  replay and rejection of scientifically incompatible source layouts.
- [x] 2.3 Deliver the applicable multisector CLI/example, guidance/docstrings,
  readiness/diagnosis checks and release notes with its writer/reader/callers.
  Verify full/phase-only choices, requested-window metadata and declared role
  validation independently of immediate product requests.
- [ ] 2.4 Clear the multisector gate with existing scientific/product references
  across full/staged/replay plus focused and relevant broader checks. Record the
  merged implementation and evidence before marking this delivery complete.


### Multisector implementation evidence, 16 October 2026

The multisector slice starts from current `devel` `208a21d9`, consuming merged
standard #866 and configuration closeout #883. It owns tasks 2.1–2.3 and the
implementation/validation portion of 2.4. **Task 2.4 remains unchecked**: merged
implementation evidence and maintainer review are outstanding. No programme
completion, maintainer readability acceptance, spec promotion or merge is claimed.

The maintainer explicitly directed this implementation to follow the standard
pattern, including its result migration, on 16 October. Accordingly,
`run_multisector`, `run_multisector(prepared=..., config=...)` and saved replay
return `InversionOutput`; this extends the earlier standard-only result amendment
and is flagged for PR review. Full and saved routes compose `prepare_multisector`,
`materialize_multisector_inputs`, `build_multisector_model`, sampling and trace
annotation, `make_model_output`, `postprocess_multisector_outputs` and explicit
artifact exports. Acquisition/validation and optional basis exports remain visible
in the runners; construction consumes the supplied eager Dataset and prior values.

`rhime.multisector_stages` saves/loads preparation and samples through the existing
codecs and checkpoint functions, with manifest-relative references, generation
binding and embedded `OutputContract`. The obsolete `_multisector_stages` envelope,
configuration identity/equality gate, output-binding sidecar reader, legacy graph
recovery and multisector envelope dispatch are removed together with their callers.
Multisector CLI saved phases consume one manifest and phase-owned choices.

The runnable `examples/multisector_handoffs.py` prepares once, samples Normal
scaling priors with two widths and fixed-error/Student-t likelihoods into distinct
locations, and compares full/prepared/staged/replay traces and flux products.
Independent SciPy density checks cover three two-sector states per likelihood
(`rtol=1e-6`, `atol=1e-5` for float32 arithmetic). Independent native-flux
reconstruction uses unequal inventory fields `[[1,2],[3,4]]` and
`[[.3,1.2],[2.1,.4]]`, distinct from the historical operator's prior.
Native source ordering and model-sector declaration ordering are preserved
independently by labels and explicit recorded associations.

`tests/test_multisector_handoff_parity.py`: **14 passed** after the source and
eager-builder review fixes, plus **1 passed** in the added signed-source replay
case (negative source flux/sensitivity and independently reconstructed native totals). This includes default lognormal scientific/latent
state retention; joint lazy prior/input materialization and zero-state pruning;
retained-site/requested-window metadata; omitted, extra and incompatible source
layouts; borrowed/unselected lazy arrays; unchanged prepared numerical files;
equivalent default/shorthand preparation records; and fresh-process replay after
removing the original operator with acquisition, construction, materialization and
inference forbidden. Full and prepared runs also pass with checkpoint serialization
and manifest operations forbidden.

Final consolidated local evidence: **1,026 passed** across 22 focused/relevant
broader test files covering RHIME science and products, multisector and standard
handoffs/parity, CLI/readiness/diagnosis, CO2 facade preservation, tutorial and
xarray consumers, configuration/site resolution, nested consumers, resource paths,
basis inputs, merged/prepared codecs, output contracts and prior registration.
The malformed-schema CLI regressions pass in this consolidated run. Changed-path
Ruff, `git diff --check` and strict validation of this active OpenSpec change pass.
Full compatibility/type/documentation Slurm results and comparison against
`208a21d9` remain pending and will be recorded with the PR evidence.

CI run `38878858180` exposed two unmigrated test consumers: six multisector
basis-export cases stopped inside sensitivities, before scientific preparation
now completes; the import-boundary test still called the retired multisector
stage resolver. The export checks now stop at materialization, as standard does,
and assert preserved scientific payloads/metadata without a destination receipt.
The import-boundary check exercises configured/raw multisector execution and
current staged preparation with HBMCMC imports blocked. Local `.venv` execution
of both complete test modules passes **86 tests**, with changed-path Ruff and
`git diff --check` clean. Production code is unchanged by this CI correction.
The unchanged `208a21d9` Slurm baseline (`19448566`) reports **164 Mypy errors in
52 files** and **318 strict-docs warnings**; those diagnostics are retained for
candidate comparison, without claiming the branch introduces no regressions.

Scientific checks retained for #808 have concrete owners: selected arrays and
source/basis coverage before materialization, baked BC policy at model resolution,
prepared labels/units at numerical loading, and scientific roles at construction
and external save/load. A model omitting a retained source now fails rather than
silently excluding its flux from totals. Replay uses original requested dates and
actual inference descriptions with current output choices, without fresh sampler
settings or graph recovery.

The maintainer's subsequent 16 October correction retires the remaining
`run_rhime_from_prepared_inputs` route and its multisector build/result/output
adapters rather than preserving them for test/example consumers. All affected
ordinary prepared-input callers migrate to `run_multisector(prepared=..., config=...)`
and explicit scientific composition; the `RhimeRunSpec` complete-builder hook and
unused callable adapter/protocol are removed. Neither standard nor multisector
execution constructs `RhimeResult` or `RhimeRunSpec`. Shared envelope types and
the nested graph-validation context remain because nested and CO2 still
consume them; their owning slices must migrate those real consumers before
global deletion. Ordinary CO2 #886 is coordinated separately and already uses
its concrete scientific output. Nested/cached CO2, model-input ownership,
artifact/storage policy and integrated acceptance remain outside this delivery.

The maintainer also directed shared output ownership rather than duplicated
standard/multisector functions. Both recipes and their saved routes now call
`make_model_output(..., split_by_sectors=...)` and `save_inversion_output`;
family-specific product computation remains explicit. The identical multisector
constructor and forwarding exporter are deleted, with standard/multisector
callers, custom/cookiecutter examples, API docs and migration guidance updated
together. The model-composition delta records removal of the former retained
multisector complete-builder requirement for review.

Retirement/shared-output local evidence: **313 passed** in `test_rhime.py`;
**225 passed** across serialization, xarray adapter, reporting preflight,
multisector scientific parity, standard API and three custom-runner integration
modules; **193 passed** across standard/multisector handoffs, phase CLI,
readiness, resource boundaries, tutorials and model-family documentation.
Shared nested/CO2 preservation adds **72 passed**; public API/configuration
preservation adds **155 passed**. These are bounded selections, not a claim of
full-suite completion. All changed Python paths pass Ruff; strict active-change
validation and diff checks pass. Documentation checks resolve 164 API references
and parse 28 Python blocks; the revised complete-composition snippet runs against
real synthetic preparation/sampling (two draws, execution smoke only).

The further readability review follows the pre-#866 standard history:
`6822bd3b` removed the whole-model hook, `b0251e7e` consolidated construction,
`189569f9` removed the result/run-spec route, and `638067db` exposed acquisition
and output operations. The final function comparison is:

| Standard | Multisector | Difference that remains |
| --- | --- | --- |
| `_require_component_inputs` | same name | Selected component diagnostics. |
| `standard_model_input_names` | `multisector_model_input_names` | Source-resolved model inputs. |
| `_standard_input_names` | `_multisector_input_names` | Sector identities and exact source coverage. |
| `_standard_prior_parameters` | `_multisector_prior_parameters` | Separate eager priors for each sector. |
| `materialize_standard_inputs` | `materialize_multisector_inputs` | Retained source-specific basis compatibility. |
| `build_standard_model` | `build_multisector_model` | Source loop, state/prior alignment and coordinate namespaces. |
| `prepare_standard` | `prepare_multisector` | Source-resolved sensitivity preparation. |
| `run_standard` | `run_multisector` | Sector-specific products and preflight. |

The only extra multisector function is `_validate_multisector_basis_layout`:
source-specific operators may have different state coordinates, so preparation
reuse and replay must protect each source-to-state binding. The raw
`build_multisector_rhime_model`, eager forwarding layer and tuple-producing sector
preparation helper are replaced by one visible eager builder; optional
`sigma_alignment` remains available. The builder now rejects extra independently
supplied H sources as well as incompatible retained preparation. Shared output
construction is named `make_model_output`; new names avoid a redundant RHIME
prefix while existing delivered specification/result types remain unchanged.
Multiline signatures/calls use one argument per line. Repository Ruff runs lint
only and did not require the earlier packed formatting.

Readability/construction validation: **215 passed** across full/staged/replay
scientific parity, standard/multisector checkpoints, phase CLI, output API,
tutorials, model-family docs and custom/cookiecutter consumers. The four-module
scientific builder/state-policy/serialization run initially passed **423 cases**
with five obsolete one-source fixtures failing the multisector input boundary;
those fixtures were corrected without weakening validation, and their complete
**11-case containing subset passed**. The **16 integration cases** also pass.
The independent eager-builder extra-source rejection is exercised in the parity
suite. Ruff on all changed Python paths, diff checks and strict OpenSpec
validation pass. Fifteen changed documentation Python blocks parse and their
direct document targets resolve. Full compatibility/type/docs checks must run
on the new fixed revision; earlier green GitHub evidence at `f900d539` remains
labelled with that revision rather than claimed for the consolidation.

Slurm `19451784` completed all seven environments at `9348c610`. Python 3.12
and 3.13 locked runs each report **3,019 passed / 8 skipped**, and OpenGHG devel
reports **3,015 passed / 12 skipped**, with exactly two failures in each: the
preserved CO2 installed-stage tests require a Git revision, while the validation
archive deliberately contained no `.git` and returned `revision=None`. This is
a validation-context failure, not evidence of a scientific change; compatibility
must be rerun from a detached Git checkout with the fixed revision. Lower-bound
PyMC and borrowed-reference typing environments pass.

Mypy reports **162 errors in 50 files**, against baseline **164 in 52**, with
zero introduced normalized diagnostics. Strict docs report **309 warnings**,
against baseline **318**: one new heading underline warning is fixed; two
existing external TensorVariable-reference warnings relocate to the new stage
page while six disappear elsewhere, and six old RhimeConfig ambiguity warnings
disappear. No baseline warning suppression was added. A fresh strict-docs run
must confirm the fix; comparison artifacts retain location-aware and diagnostic
content counts. The retry uses a detached local clone, with generated API docs
outside that checkout to keep scientific provenance stable during tests.

## 3. Ordinary CO2 reusable-handoff adoption

- [ ] 3.1 Resolve owned CO2 phase choices consistently for configured full and
  staged execution, reusing its readable preparation/building/inference owners.
  Verify the actual configured full-run entry point consumes model/sampler/output
  choices and writes the selected products/destination, while raw runner trace
  returns remain intact. Remove redundant stage-only configuration wrappers;
  keep the full route independent of manifest machinery while sharing its step
  functions with the saved phases (design decision 8). Delete the CO2 branches of
  the staged envelope facade once the handoffs replace them.
- [ ] 3.2 Implement ordinary CO2 prepared/sample handoffs using existing numerical
  persistence and trace/affine output meaning. Verify independent BC/offset priors
  or supported mismatch changes reuse coherent preparation; changed defining
  native prior/covariance/operator/projection requires new preparation. Check that
  the affine reconstruction uses the prepared sources and grid.
- [ ] 3.3 Preserve authoritative requested dates separately from retained
  observation extrema for imported and newly produced CO2 preparations. Verify
  filtering both period edges does not shorten output metadata; make missing
  required imported facts explicit rather than inferring them from observations.
  For new affine outputs, save prepared inputs and their companion with existing
  codecs, check their scientific compatibility, then save the manifest last.
  Verify imported bound pairs remain valid and native/country replay works for a newly
  produced pair as well as an imported pair.
- [ ] 3.4 Validate per-site model mappings at inference against prepared retained
  labels: permit valid extra requested sites, validate every supplied value even
  for dropped sites, require retained coverage, then select retained labels.
  Cover nonnegative ordinary fixed sigma and positive OU time scales, including
  zero/negative/nonfinite values and unequal per-site options.
- [ ] 3.5 Deliver ordinary CO2 CLI/examples, guidance/docstrings, readiness,
  independent diagnosis and applicable release notes with its writer/reader/callers.
  Verify unsupported science/output requests fail before expensive work or report
  destination creation, predictive evidence retains its scientific meaning and
  replay uses saved metadata with `sampler=None` without rebuilding a graph or
  running inference.
- [ ] 3.6 Clear the ordinary CO2 gate against existing coherent, likelihood and
  affine reconstruction references, plus configured-full/staged/replay products
  and focused/relevant broader checks. Record the merged evidence and the public
  product entry point before cached adoption; leave linked CO2/O2 work in OPE-165.

## 4. Cached CO2 reusable-handoff adoption

- [ ] 4.1 Connect the ordinary CO2 handoff/result conventions to the matched cached
  construction/inference operation. Verify required backend, sigma-then-state step
  order and accepted-cache lifecycle; reject unsupported generic sampler/tuning
  substitutions before expensive work. Preserve the existing direct cached runner.
- [ ] 4.2 Validate cached per-site initialization and OU time scales as positive,
  requiring retained coverage after checking all supplied values. Verify compatible
  mismatch changes reuse numerical preparation while dependent runtime caches are
  rebuilt for the selected model; reject a scientifically incompatible covariance
  cache/eigenbasis at its owning boundary.
- [ ] 4.3 Preserve exact prior-predictive support, conditional prediction, complete
  component contributions and cached annotations across full/staged/replay. Verify
  readiness/independent diagnosis, scientific roles and graph-free output
  replay using existing cached references; check the likelihood at varied fixed
  states/sigma against a dense oracle and posterior moments against the existing
  independent reference, rather than requiring identical seeded draws from
  different inference algorithms.
- [ ] 4.4 Deliver cached CLI/examples, guidance/docstrings and applicable release
  notes with its writer/reader/callers. Clear the cached gate with focused and
  relevant broader scientific, predictive and output checks; record merged evidence
  and supported/unsupported combinations without adding a second sampling harness.

## Review follow-up tracking — 15 October 2026

GitHub [#880](https://github.com/openghg/openghg_inversions/issues/880) mirrors
OPE-207 and indexes the remaining family slices. The standard #866 readability
corrections received maintainer readability acceptance on 16 October 2026. Deferred alignment
ownership is [#876](https://github.com/openghg/openghg_inversions/issues/876)
(OPE-105); durable input decisions and output consumer migration are
[#877](https://github.com/openghg/openghg_inversions/issues/877) and
[#878](https://github.com/openghg/openghg_inversions/issues/878) (OPE-106/107).
Compact/recoverable output implementation is
[#879](https://github.com/openghg/openghg_inversions/issues/879) (OPE-125),
under existing storage policy #465 and output umbrella #850. Nested remains
outside this change, with runner follow-up assigned to existing #408.

## 5. Integrated acceptance and completion

- [ ] 5.1 Reconcile all four family deliveries against the three new capability specs and the configuration/resource-I/O capabilities
  while preserving the current model-selection and model-composition baselines;
  record merged PRs, commands/results and justified numerical tolerances. Run
  relevant broader scientific, configuration, CLI, staged/replay, builder and output
  coverage; run Ruff on changed Python paths and `git diff --check`. Submit required
  full-suite, compatibility and type environments through `scripts/slurm_tox.sh`,
  not local tox. Reconcile pending-owner status without claiming #839 or another
  unmerged delivery complete, and record the coordinated #808 work where applicable.
- [ ] 5.2 Verify active routes retain no obsolete filtered checkpoint, execution of
  recorded callables, inference during product replay or full-config equality gate.
  Confirm every scenario has implementation and validation evidence before syncing
  these capabilities and archiving this completed change. Planning merge alone
  does not complete OPE-207; do not archive the superseded proposal as successfully
  implemented or count linked OPE-165 work as delivered by this change.
  Update remaining consumers and the developer contract map to the delivered
  capabilities when archiving; use the actual archive location for historical
  references. Verify links in promoted specs and remaining active changes.

## Standard gate accepted — 16 October 2026

The maintainer explicitly confirmed that readability is good and authorized an
admin merge of #866. Task 1.11 is complete. Runtime head `da832a5a` passed both
full locked Python 3.12/3.13 suites in Slurm 19436596. Strict documentation has
318 warnings versus 321 at pre-review `460d993a`; Mypy has the identical 164
diagnostics on both revisions. No introduced diagnostics; existing debt remains
#679/#881. Earlier scientific parity evidence remains pinned to its recorded
revision in #866. This acceptance record changes no runtime code.

The standard gate permits the separately reviewed multisector and ordinary CO2
slices to proceed. Remaining family tasks, integrated acceptance and capability
promotion/archive remain open under OPE-207/#880.
