# Tasks

These are future implementation tasks, all unchecked. This planning change
changes no runtime behavior. Follow [design.md](design.md) and the three
new capability specs and configuration amendment. The four delivery sections below define proposed bounded
implementation-child scopes for OPE-207; they do not create Linear issues or
represent one assignment spanning every family.

Keep each saved-handoff writer, reader and caller in the same implementation PR.
Deliver applicable commands, examples, documentation, focused checks and release
notes with each family. The standard route must pass its scientific acceptance
gate before multisector or ordinary CO2 adoption; those two adoptions can then
proceed independently. Cached CO2 adoption follows the ordinary CO2 handoff and
result conventions. A passing mock-forwarding test alone clears none of these
gates.

The reviewed baseline is devel `80b0ebf0`: reuse the merged acquisition/inference
ownership from #773/#774 and configuration/preparation work from #822–#825.
Cleanup #826/#830 is merged; #832 and boundary work #834–#837 remain pending.
#840 addresses #838's explicit basis paths; #841 addresses #839's runtime-flux/
operator-loading work and remains under review. Consume resulting APIs and record merged evidence rather than duplicating
these owners or treating open PRs as landed. #808 retains the bounded manifest,
hash and consumer-contract work. Acquisition persistence/replay belongs to
deferred #829 and is not required for prepared/sample handoffs. OPE-164 records
the initial ordinary/cached CO2 staged delivery; the CO2 sections below adopt the
replacement handoffs rather than reopening that completed scope. Linked CO2/O2
adoption remains in OPE-165, retaining its completed numerical persistence.

## 1. Complete standard preparation-to-products route

- [ ] 1.1 Group the existing resolved `RhimeConfig` preparation fields in one
  `PreparationConfig` at their current owner, moving fields/defaults rather than copying
  them into a second schema. Keep model, sampler and output choices with their
  owners. Verify full and phase-only inputs resolve the same applicable choices
  without demanding unrelated sections, touching arrays or mutating caller values.
- [ ] 1.2 Add live `sampler="default"` as a request for a fresh family-default
  sampler on each invocation, without importing a sampler from recorded metadata.
  Give custom `RhimeSampler` invocations owned copies of nested keyword values;
  verify list/tuple/array fidelity, isolated overrides and unchanged caller values.
  Reject `idata_kwargs.coords`/`dims` in resolved and direct sampling paths.
  Verify configured full runs preserve `config.sampler`, omitted/explicit default
  calls select fresh defaults, invalid strings/`None` fail at new live boundaries,
  and established direct `None` adapters retain their documented behavior.
- [ ] 1.3 Consolidate standard scientific calls at their existing owners in the
  visible order observation errors, filtering, basis, sensitivities and assembly.
  Consume #837's explicit assembly/execution boundaries and the named joint PyMC
  materialization boundary. Verify retained-site/averaging alignment, empty-set
  rejection, borrowed lazy arrays and custom builder/likelihood contracts across
  full, supplied-merged/prepared and saved routes; remove forwarding-only wrappers.
- [ ] 1.4 Consume #838/#839's explicit basis and runtime-flux contracts. Verify a
  selected operator A binds current acquired native flux B, saving retains B and
  replay restores B without acquisition or the original inventory paths. Cover
  first-source weighting separately from all-source retention, source order and
  once-per-domain/grid native-flux resolution; use unequal source fields and an
  independent reconstruction calculation. Record which prerequisite PRs supply
  these behaviors rather than implementing them again here.
- [ ] 1.5 Validate declared coordinate/output roles even when immediate products
  are disabled. Verify a valid custom no-product model is accepted, an invalid
  declared role fails with `output_format="none"`, and direct scientific arguments,
  conflict checks and established returns remain usable.
- [ ] 1.6 Implement standard `prepare` returning one absolute manifest path,
  using existing numerical codecs and #808's bounded contract. Omit the obsolete
  filtered checkpoint without adding acquisition persistence. Verify acquisition
  and retained subsets with unequal site options, original requested dates,
  malformed inputs, parent-relative relocation, invalid references and digests,
  protected saved dependencies and interrupted saving without a completed
  manifest. Keep the writer, loader and callers together.
- [ ] 1.7 Implement standard `sample` from that manifest and current model/sampler
  choices. Record bounded actual inference provenance and embedded `OutputContract`.
  Verify two compatible priors/likelihoods reuse unchanged preparation, while
  missing required arrays or changed baked policy fail before inference. Distinguish
  scientific compatibility from content integrity and descriptive provenance.
- [ ] 1.8 Implement standard `postprocess` from one sample manifest and current
  output choices. Replayed results use `sampler=None` and plain saved metadata;
  no sampler or callable is reconstructed. Verify fresh-process relocated replay
  with original configuration, acquisition and graph constructors unavailable,
  preserving requested window/retained metadata and matching live products from
  the same samples. Reject missing, changed, swapped, malformed, unsupported or
  down-labelled records and invalid roles/labels/units before product writes;
  structural/digest failures precede posterior loading.
- [ ] 1.9 Preserve standard prior-predictive/readiness catch boundaries and
  independent diagnosis, including its declared historical sample support. Verify
  fail/pass/unknown, strict exit status, invalid predictive evidence and propagation
  of loading/serialization errors; readiness does not silently perform posterior
  inference or repair missing evidence during replay.
- [ ] 1.10 Deliver standard one-manifest CLI forms, full/phase-only examples and
  cookiecutter consumer acceptance with this route. Verify authenticated family
  selection, explicit mismatch rejection, once-only override resolution, help and
  destination behavior, and an executable generated scientific recipe. Update
  stage guidance/docstrings with arguments, returns, file lifetime and reuse limits;
  remove premature `openghg-run` references in affected guidance. Coordinate the
  0.8 staged-contract release note with #808, distinguishing preserved numerical,
  product and direct scientific APIs from retired staged envelopes/aliases.
- [ ] 1.11 Clear the standard gate with a runnable preparation/two-model/relocated-
  replay example, distinct sample destinations and real prior or likelihood
  calculations checked against an independent calculation or existing regression
  reference. Demonstrate equivalent full execution with saved-handoff writers,
  digests and encoding disabled. Record focused/broader relevant test results and
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
  sample meaning and current reporting choices, including relocated graph-free
  replay and rejection of swapped sector/source output associations.
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
  native prior/covariance/operator/projection or an unrelated affine companion
  cannot masquerade as compatible preparation.
- [ ] 3.3 Preserve authoritative requested dates separately from retained
  observation extrema for imported and newly produced CO2 preparations. Verify
  filtering both period edges does not shorten output metadata; make missing
  required imported facts explicit rather than inferring them from observations.
  For new affine outputs, save prepared inputs, bind the companion to their saved
  identity, validate the pair, then save the manifest last. Verify imported bound
  pairs remain valid and relocated native/country replay works for a newly
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
  replay uses saved metadata with `sampler=None` without graph/sampler construction.
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
  rebuilt or proven matching; a valid preparation does not authenticate an unrelated
  covariance cache/eigenbasis.
- [ ] 4.3 Preserve exact prior-predictive support, conditional prediction, complete
  component contributions and cached annotations across full/staged/replay. Verify
  readiness/independent diagnosis, scientific roles and graph/sampler-free output
  replay using existing cached references; check the likelihood at varied fixed
  states/sigma against a dense oracle and posterior moments against the existing
  independent reference, rather than requiring identical seeded draws from
  different inference algorithms.
- [ ] 4.4 Deliver cached CLI/examples, guidance/docstrings and applicable release
  notes with its writer/reader/callers. Clear the cached gate with focused and
  relevant broader scientific, predictive and output checks; record merged evidence
  and supported/unsupported combinations without adding a second sampling harness.

## 5. Integrated acceptance and completion

- [ ] 5.1 Reconcile all four family deliveries against the three new capability specs and configuration amendment
  and record merged PRs, commands/results and justified numerical tolerances. Run
  relevant broader scientific, configuration, CLI, staged/replay, builder and output
  coverage; run Ruff on changed Python paths and `git diff --check`. Submit required
  full-suite, compatibility and type environments through `scripts/slurm_tox.sh`,
  not local tox. Reconcile pending-owner status without claiming #839 or another
  unmerged delivery complete, and keep #808's independent acceptance visible.
- [ ] 5.2 Verify active routes retain no obsolete filtered checkpoint, executable
  replay configuration, sampler reconstruction or full-config equality gate.
  Confirm every scenario has implementation and validation evidence before syncing
  these capabilities and archiving this completed change. Planning merge alone
  does not complete OPE-207; do not archive the superseded proposal as successfully
  implemented or count linked OPE-165 work as delivered by this change.
