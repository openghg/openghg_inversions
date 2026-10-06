# Tasks

These are future implementation tasks, all unchecked. The planning PR changes no
runtime behavior. Follow [design.md](design.md) and the three capability specs.
Keep each saved-handoff writer, reader and caller in the same implementation PR.
Deliver documentation and focused checks with each slice; the grouping below is
dependency order, not a requirement for five PRs.

The first acceptance gate is a working standard route demonstrating two models
from one preparation and graph-free replay. Review that before porting the design
across families. A passing mock-forwarding test alone does not clear this gate.

## 1. Scientific operations and phase choices

- [ ] 1.1 Separate preparation, model, sampler and output resolution at existing
  configuration owners; remove unused stage/setup aliases. Verify full config
  and phase-only inputs resolve the same applicable choices without demanding
  unrelated sections, touching arrays or mutating caller values.
- [ ] 1.2 Implement immutable sampler choices and explicit runtime creation;
  reject `idata_kwargs.coords`/`dims` in resolved and direct sampling paths.
  Verify nested list/tuple/array values round-trip, mutation stays isolated and
  two calls with an override do not alter each other's choices.
- [ ] 1.3 Consolidate standard preparation and construction at existing science
  owners, including retained-site alignment and named joint materialization.
  Verify full, merged/prepared-input and stage routes preserve borrowed arrays,
  custom builder/likelihood contracts and scientific variable roles; remove
  forwarding-only construction wrappers.
- [ ] 1.4 Make declared coordinate/output-role validation independent of immediate
  product requests. Verify a valid custom no-product model is accepted and an
  invalid declared role fails even when `output_format="none"`; preserve direct
  scientific arguments, conflict checks and returns.

## 2. Standard preparation-to-products proof

- [ ] 2.1 Implement the prepared record/codec caller and common version, relative
  path, digest and publication rules in the existing artifact owners. Verify
  parent-relative shared dependencies relocate correctly, malformed references
  fail, destinations cannot overwrite published dependencies, and an interrupted
  write cannot publish a completed manifest.
- [ ] 2.2 Implement standard `prepare` returning one absolute manifest path;
  preserve pre-filter merged-cache behavior and omit the filtered snapshot.
  Verify acquisition, compatible reload and filtering retained subsets (with
  unequal site options), empty-set rejection and malformed-cache/input behavior.
- [ ] 2.3 Implement standard `sample` from that manifest plus current model/sampler
  choices, recording bounded inference provenance and embedded `OutputContract`.
  Verify different compatible priors/likelihoods use unchanged preparation;
  absent required arrays or changed baked policy fail before inference.
- [ ] 2.4 Adapt result/output metadata construction to recorded sampler provenance
  and implement standard `postprocess` from one sample manifest and output policy.
  Verify replay works in a new process with original config and graph constructors
  unavailable, preserves original window/retained metadata, and matches live
  products from the same samples.
- [ ] 2.5 Exercise authentication and output semantics together: reject missing,
  changed, swapped, malformed, unsupported and down-labelled records; check
  scientific roles/labels/units after loading. Verify failures precede product
  writes, structural/digest failures precede posterior loading, and replay never
  executes recorded callables or creates a sampler.
- [ ] 2.6 Present one runnable example using the public calls in the design, with
  two compatible models, distinct sample destinations and relocated replay.
  Verify real model/prior or likelihood calculations against an independent
  calculation or existing regression reference, plus equivalent full execution
  with intermediate writers disabled. Use this evidence for the first design gate.

## 3. Multisector and CO2 family adoption

- [ ] 3.1 Give multisector concrete stages using its canonical preparation,
  construction and products. Verify sector-labelled inputs/roles and existing
  products across full/staged/replay routes, including retained-site subsets,
  prior reuse and array ownership; do not reintroduce shared scientific dispatch.
- [ ] 3.2 Resolve owned CO2 configuration consistently for configured full and
  staged execution, relocating stage-only run/output setup to its actual owner.
  Verify output settings govern products/destinations and raw runner trace returns
  remain intact; remove redundant configuration aliases/wrappers.
- [ ] 3.3 Implement ordinary CO2 handoffs using existing numerical persistence and
  trace/affine output meaning. Verify independent BC/offset or supported mismatch
  changes reuse coherent preparation, while changed defining native prior/operator
  or mismatched affine companions cannot masquerade as compatible preparation.
- [ ] 3.4 Connect cached CO2 stages to the matched construction/inference operation.
  Verify backend/step order, cache updates, conditional prediction, annotations
  and covariance-specific cache validity against existing cached CO2 references;
  do not accept a generic sampler substitution as parity.
- [ ] 3.5 Verify per-site model mappings at inference, using prepared retained
  labels: valid extra requested sites are allowed; all supplied values retain
  numeric/finiteness/sign checks; missing retained values fail. Cover ordinary
  fixed sigma (nonnegative), cached initialization and time scales (positive).

## 4. Commands, checks and explanation delivered with each slice

- [ ] 4.1 Update the existing CLI commands to the design's one-manifest forms;
  infer family from the authenticated handoff, reject an explicit mismatch and
  resolve applicable config/overrides once. Verify executable examples for full
  and phase-only files, help text and explicit destination behavior.
- [ ] 4.2 Preserve family-specific prior-predictive/readiness catch boundaries and
  independent diagnosis, including declared historical sample support. Verify
  fail/pass/unknown, strict exit status, invalid predictive evidence, propagation
  of load/serialization errors and CO2 errors before report destination creation.
- [ ] 4.3 Update the stage introduction, family module/function docstrings and
  workflow examples with the diagram, exact arguments/returns, file lifetime,
  compatible reuse limits and extension examples. Verify a newcomer can follow
  preparation through products without reading OpenSpec; remove premature
  `openghg-run` references in affected guidance.
- [ ] 4.4 Update cookiecutter and its consumer acceptance test with the supported
  customization and staged argument contract. Verify a generated recipe executes
  the documented scientific sequence, rather than asserting an obsolete keyword set.
- [ ] 4.5 Add next-minor Towncrier fragments for the staged API/envelope reset,
  filtered-checkpoint removal and retained-site correction alongside implementation.
  Verify preserved numerical/product formats and direct scientific APIs are
  distinguished from intentionally retired aliases and staged metadata.

## 5. Integrated acceptance and completion

- [ ] 5.1 Run the relevant broader scientific, configuration, CLI, staged/replay,
  builder and output tests after the slices pass focused checks. Run Ruff only
  on changed Python paths and `git diff --check`; submit required full-suite,
  compatibility and type environments through `scripts/slurm_tox.sh`, not local
  tox. Record commands/results and justify scientific comparison tolerances.
- [ ] 5.2 Reconcile delivered behavior with all scenarios in the three capability
  specs and the design's API/record/reuse examples. Verify no retired wrapper,
  filtered checkpoint or full-config equality requirement remains on active paths;
  only then sync these new capabilities and archive this completed change.
  Do not archive the superseded proposal as a successfully implemented change.
