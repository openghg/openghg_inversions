# Tasks

This checklist records the configuration slice, PR #824, and subsequent preparation
contract corrections in the
[#815 stack](https://github.com/openghg/openghg_inversions/issues/815).
It replaces the accumulated #813 implementation amendments. Checked items
record the implemented contracts in this slice; historical validation below
applies only to the named revisions and does not certify the current stack.

## 1. One requested configuration

- [x] 1.1 Put acquisition and preparation fields directly on `RhimeConfig`,
  reusing `SiteOptions`, model/output specifications and the existing sampler.
  Remove `RhimePreparationConfig`, its projections, `RhimeRunnerSetup`,
  `make_rhime_runner_setup` and `resolve_rhime_options`.
- [x] 1.2 Make `RhimeConfig.from_params` the single construction entry point;
  remove `resolve_rhime_config`. Derive supported and required names from the
  owning records and advertised consumer subsets without signature inspection.
- [x] 1.3 Resolve configuration-only defaults and reject malformed effective
  choices before acquisition, including scalar/per-site minimum errors,
  non-finite-flux and aggregation-error modes, active basis algorithms and
  unsupported tracer requests. Preserve saved-basis precedence.
- [x] 1.4 Own ordinary caller containers, including arbitrary `Mapping` inputs,
  while borrowing scientific arrays and opaque values. Configuration access
  performs no scientific copying, computation, model construction or sampling.

## 2. Decode, edit and resolve once

- [x] 2.1 Keep `read_rhime_ini(path)` in `rhime.ini`, returning a flat dictionary
  with the existing section flattening and first-occurrence rules. The reader
  does not return a configuration or accept resolution/override arguments.
- [x] 2.2 Apply winning overrides and consume recipe-owned options before
  `RhimeConfig.from_params`. Expand site shorthand and resolve date-dependent
  defaults only after those edits; preserve file/Python equivalence.
- [x] 2.3 Invoke historical alias translation once inside `from_params`, using
  definitions owned by `hbmcmc.compatibility`. Warn only for translated or
  removed deprecated spellings/values; canonical options win and remain quiet.
- [x] 2.4 Preserve the deprecated `params_from_config` dictionary adapter and
  fixedbasis translation at their compatibility owner. Remove transitional
  old-location imports rather than adding forwarding aliases; retain intentional
  package-level public exports and document the breaking imports.
- [x] 2.5 Reuse complete `config=` requests in ordinary runners and the HBMCMC
  shim without another resolution pass. Reject combining resolved configuration
  with raw/file input, and preserve shim validation before copy side effects.

## 3. Explicit scientific consumers

- [x] 3.1 Consume the public, complete `SiteOptions` record and resolved-selector
  retrieval body supplied by the acquisition slice. Preserve scalar/aligned
  input equivalence, retained-label selection and no repeated expansion.
- [x] 3.2 Acquire in memory with `RhimeMergedData.from_options` or reuse
  compatible supplied handoffs. #826 supersedes the earlier modern cache
  loading/saving requirement; historical codecs remain isolated under HBMCMC.
- [x] 3.3 Forward named scientific inputs directly or through shallow
  `RhimeConfig.select(*names)`. Preserve borrowed selected values and errors for
  unknown attributes; remove named stages' positional `data_args` adapters.
- [x] 3.4 Keep requested options unchanged when preparation retains fewer sites.
  Build `RhimeRunSpec` after preparation from retained sites/periods, requested
  dates, prepared layout and resolved model/output choices.
- [x] 3.5 Migrate nested, staged, shim and custom-runner consumers. Keep nested
  model/output choices coherent, avoid traversing output options for preparation
  identity, and exercise custom-basis resolution without an unused built-in
  basis selector masking the test.

## 4. Documentation and current delivery

- [x] 4.1 Describe the current constructor, decoding reader, public owners,
  selected forwarding and breaking imports in usage/development guidance,
  examples, API documentation and release fragments. Separate advanced
  configuration material from the ordinary run guide.
- [x] 4.2 Consolidate this checklist so superseded resolved-reader and
  compatibility-wrapper designs are no longer presented as current contracts.
- [x] 4.3 Record focused and relevant broader validation for the final #824
  revision, including changed-path lint, strict OpenSpec, whitespace and
  affected documentation checks. Identify any stack-tip results separately.
- [x] 4.4 Push the final #824 revision and update its review evidence with the
  exact revision and validation scope. Historical #813 delivery is not current
  delivery evidence.

## Separate slices and deferred work

The #823–#832 stack supplies in-memory acquisition, retained selectors and
provenance, dataset-based basis construction, and isolated legacy adapters.
#826 removed `prepare_rhime_inputs`, its dispatcher, modern cache options and
the filtered merged-data checkpoint. Acquisition replay remains deferred to #829.

The preparation follow-up removes forwarding wrappers and derives observation
errors before filtering/aggregation, including both nested domains. It retains
custom errors, zero-error fallback, borrowed data and supported minimum-error
normalization. Modern flux terminology/deprecation and public API/CLI naming
are separate reviewable changes; #817 owns coordinated materialization and
#818 retains filter retry policy.

Configuration export/logging and an INI writer remain #814. Full route-wide
retained-site parity and replacement handoff contracts remain #802/#808; this
preparation change does not claim completion of those contracts or new family
checkpoint support. HBMCMC retirement remains #719.

## Earlier configuration/stack validation

At configuration revision `293624fd`, strict OpenSpec and whitespace checks
pass and `docs-full` passed in Slurm `19291950_0`. This revision changes planning
and reference documentation only; the configuration Python validation remains
attached to the owning earlier revisions recorded on PR #824 and issue #815.

At stack-tip revision `783517be`, 75 artifact/runner and 30 nested/documentation
tests passed locally on Python 3.13. Locked Python 3.12 artifact/runner tests,
CI borrowed-reference typing, full Ruff lint and `docs-full` passed in
`19291950_1`. Built API navigation, both new public method anchors and the
nested rejection contract were checked. These stack-tip checks validate #826's
UTC correction; they are not attributed to the earlier configuration slice.

Both revisions were pushed through the stack. This checklist completion record
changes no Python code, tests or Sphinx source. Issue #815 and the PR descriptions
record current heads, immutable validation evidence and remaining upstream
limitations; no fresh full compatibility matrix is claimed here.

## Historical validation ledger — not current acceptance

These records preserve the validation history of #813 and its amendments.
Passing an earlier implementation does not validate a later contract, the split
PRs or their current heads. Earlier "ready for review" and delivery statements
applied to #813 at those revisions only.

| Revision or round | Recorded validation | Scope and limits |
| --- | --- | --- |
| Original preparation-config implementation | Locked Python 3.12/3.13 job `19270831`; docs-full job `19270662` passed. | Predates the direct-field redesign. |
| `436379ca`, synchronized with `devel` at `e49632fd` | Relevant Python 3.12/3.13 suites in `19278392`; docs-full in `19278394`, both exit zero. Eight rendered pages and local links inspected; focused tests, Ruff, OpenSpec, whitespace and executable examples passed. | Predates later ownership, decoding and constructor amendments. |
| Configuration finite-choice follow-ups | Initially 36 configuration and 60 selected consumer tests; then 56 configuration and 67 selected consumer/basis tests passed on Python 3.13, with Ruff, OpenSpec and whitespace checks. | Recorded local selections cover mode validation, active basis choices and minimum-error methods; they overlap and have no separate revision recorded here. |
| `f0f47030` | Locked Python 3.12/3.13 suites in `19280326`; docs-full in `19280272` passed. Nine rendered pages inspected; Ruff, OpenSpec and whitespace passed. | A local PyMC 5.26.1 staged-result mismatch reproduced on both revisions. The passing cluster retry used sequential environments and one pytest worker after an earlier memory/dependency failure; repository tox configuration was unchanged. |
| `2b80e490` | Python 3.12/3.13 suites and docs-full in `19280761`, exit zero. Eight rendered pages and 1,316 local anchors checked; generated references matched. | Selected-forwarding round; earlier focused groups of 69, 55, 69 and 10 tests overlapped. Rendered inspection covered HTML structure/text, not browser screenshots. |
| `7c58af3b` | Relevant Python 3.12/3.13 suites and docs-full in `19281228`, exit zero. Eight pages and 940 local links/anchors checked. | Decoder/compatibility separation before the final constructor amendment. |
| `81405294` | Relevant Python 3.12/3.13 suites and docs-full in `19281458`, exit zero. Focused amendment checks covered 173 consumer and 35 compatibility/shim tests; Ruff, OpenSpec and whitespace passed. Eight pages and 936 local links/anchors checked; generated references matched. | One-constructor amendment, pushed to #813 before splitting into the #815 stack. Rendered inspection covered HTML structure/text, not browser screenshots. |
