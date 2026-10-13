# Tasks

The maintainer approved this plan for implementation on 11 October 2026. These
are implementation tasks, all unchecked until their acceptance evidence exists.
Start with a bounded implementation PR stacked on #841. This planning change
itself changes no runtime behavior. Follow [design.md](design.md) and the three
new capability specs and the
[configuration amendments](../group-rhime-configuration-by-owner/proposal.md) and
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

- [ ] 1.1 Consume the separate [configuration change](../group-rhime-configuration-by-owner/tasks.md)
  delivering `RhimeConfig.data: DataConfig` and the grouped `data` key. Record
  its implementation evidence and verify full and phase-only inputs resolve the same applicable choices
  without demanding unrelated sections, touching arrays or mutating caller values.
  Cover omitted/explicit model `use_bc` against enabled/disabled preparation,
  preserved legacy flat routing, and omitted/equivalent/conflicting `bc_freq`
  claims against the baked temporal policy.
- [ ] 1.2 Use live `sampler="default"` consistently in new and existing
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
- [ ] 1.3 Consolidate standard scientific calls at their existing owners in the
  visible order observation errors, filtering, basis, sensitivities and assembly.
  Keep acquisition/supplied-data selection in the runner, explicitly selecting
  each operation's named arguments from `config.data`; verify scientific
  preparation accepts acquired inputs with retrieval disabled and no export writes.
  Consume #837's explicit assembly/execution boundaries and the named joint PyMC
  materialization boundary. Verify retained-site/averaging alignment, empty-set
  rejection, borrowed lazy arrays and custom builder/likelihood contracts across
  full, supplied-merged/prepared and saved routes; remove forwarding-only wrappers.
  Preserve the current model-composition seam and registered-coordinate owner;
  do not broaden the canonical custom-likelihood contract through role handling.
- [ ] 1.4 Consume #838/#839's explicit basis and runtime-flux contracts. Verify a
  selected operator A binds current acquired native flux B, saving retains B and
  replay restores B without acquisition or the original inventory paths. Cover
  first-source weighting separately from all-source retention, source order and
  once-per-domain/grid native-flux resolution; use unequal source fields and an
  independent reconstruction calculation. Record which prerequisite PRs supply
  these behaviors rather than implementing them again here.
  Consume #868's runner-owned optional basis save and resource-path resolution;
  verify the new routes preserve the existing exact-export behavior rather than
  introducing checkpoint-directory rebasing.
- [ ] 1.5 Validate declared coordinate/output roles even when immediate products
  are disabled. Share the existing variable-role/output-contract owner between
  predictive sampling and postprocessing; remove independent concentration-name
  inference in the sampler. Keep species-dependent mass conversion in
  postprocessing. Verify a valid custom no-product model is accepted, an invalid
  declared role fails with `output_format="none"`, and direct scientific arguments,
  conflict checks and established returns remain usable.
- [ ] 1.6 Implement standard `prepare` returning one absolute manifest path,
  using existing numerical codecs and #808's bounded contract. Omit the obsolete
  filtered checkpoint without adding acquisition persistence. Verify acquisition
  and retained subsets with unequal site options, original requested dates,
  ordinary missing/malformed inputs, save/reopen and relative/shared-cache paths.
  Use JSON records and the numerical formats supported by existing codecs.
  Keep the writer, loader and callers together.
- [ ] 1.7 Implement standard `sample` from that manifest and current model/sampler
  choices. Record bounded actual inference provenance and embedded `OutputContract`.
  Verify two compatible priors/likelihoods reuse unchanged preparation, while
  missing required arrays or changed baked policy fail before inference. Distinguish
  scientific compatibility from descriptive provenance. Record a Git revision
  when available without making it a requirement for execution.
  Retain scientific output state `x` by default. Check that explicit omission
  cannot complete an incompatible handoff without supported reconstruction;
  `_latent` values alone do not satisfy the scientific output role.
- [ ] 1.8 Implement standard `postprocess` from one sample manifest and current
  output choices. Replayed results use `sampler=None` and plain saved metadata;
  configuration construction is permitted, but replay does not build a model or
  run inference. Use retained, referenced or supplied run context without filling
  missing provenance from current defaults. Verify save/reopen and two output
  choices from the same samples, preserving requested window/retained metadata
  and matching live products. Let ordinary loaders report missing/malformed
  files and scientific boundaries reject incompatible roles, labels or units.
- [ ] 1.9 Preserve standard prior-predictive/readiness catch boundaries and
  independent diagnosis with its supported inputs. Verify
  fail/pass/unknown, strict exit status, invalid predictive evidence and propagation
  of loading/serialization errors; readiness does not silently perform posterior
  inference or repair missing evidence during replay.
- [ ] 1.10 Deliver standard one-manifest CLI forms, full/phase-only examples and
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
  unsuitable keys, and the distinct file-relative and invocation-relative path bases.
- [ ] 1.11 Clear the standard gate with a runnable preparation/two-model/two-product
  example, distinct sample destinations and real prior or likelihood
  calculations checked against an independent calculation or existing regression
  reference. Demonstrate equivalent full execution with intermediate saves
  disabled. Record focused/broader relevant test results and
  review scientific parity, borrowed-array behavior and recipe readability before
  family expansion.

## 2. Multisector reusable-handoff adoption

- [ ] 2.1 Give multisector concrete preparation, inference and products calls using
  the standard delivery's established persistence mechanics through ordinary
  functions. Verify full/staged/replay preserve sector/source-labelled inputs,
  scientific roles, retained-site subsets, all-source flux/source order and borrowed
  arrays without introducing shared scientific dispatch.
- [ ] 2.2 Verify compatible sector-prior/likelihood changes reuse preparation and
  genuine preparation changes fail. Exercise saved-result products with unchanged
  sample meaning and current reporting choices, including graph-free
  replay and rejection of scientifically incompatible source layouts.
- [ ] 2.3 Deliver the applicable multisector CLI/example, guidance/docstrings,
  readiness/diagnosis checks and release notes with its writer/reader/callers.
  Verify full/phase-only choices, requested-window metadata and declared role
  validation independently of immediate product requests.
- [ ] 2.4 Clear the multisector gate with existing scientific/product references
  across full/staged/replay plus focused and relevant broader checks. Record the
  merged implementation and evidence before marking this delivery complete.

## 3. Ordinary CO2 reusable-handoff adoption

- [ ] 3.1 Resolve owned CO2 phase choices consistently for configured full and
  staged execution, reusing its readable preparation/building/inference owners.
  Verify the actual configured full-run entry point consumes model/sampler/output
  choices and writes the selected products/destination, while raw runner trace
  returns remain intact. Remove redundant stage-only configuration wrappers;
  keep the full route independent of manifest machinery.
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
