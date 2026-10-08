# Tasks

This checklist supersedes the previous completed implementation checklist.
PR #813 originally implemented the former preparation-config design. Prior
locked Python 3.12/3.13 coverage (job 19270831) and docs-full (job 19270662) passed,
but do not verify the revised plan. The planning artifacts are approved and
finalized; pending boxes track implementation against those contracts.

## 1. Revised planning

- [x] 1.1 Reconcile proposal, design and behavioral requirements around one requested configuration, direct preparation fields and explicit roles/contracts; verify strict OpenSpec validation and the key-name table.

## 2. Configuration and compatibility

- [x] 2.1 Remove RhimePreparationConfig and its projection plumbing; put its explicit fields directly on RhimeConfig while reusing existing site/model/output/sampler values; verify direct option access, file/Python equivalence, winning overrides (including date-dependent defaults), early errors and caller ownership; consolidate checks at their owning input boundaries.
- [x] 2.2 Remove RhimeRunnerSetup, make_rhime_runner_setup and resolve_rhime_options and migrate ordinary/nested/staged/shim/example consumers to RhimeConfig; remove encoding-only restoration of raw site forms and sparse priors; verify scientific behavior and entry-point results, absence of removed-type imports and explicit supported staged-contract handling.
- [x] 2.3 Promote the existing acquisition-owned _SiteOptions to SiteOptions and export it from inversion_data; update annotations/imports/consumers while keeping one record; reuse its from_inputs and applicable small alignment helpers independently of file syntax; verify public construction, scalar/list equivalence, resolved-constructor invariants and selection without request mutation or repeated expansion.

## 3. Scientific boundaries and names

- [x] 3.1 Replace load_rhime_config with read_rhime_ini(path, *, overrides=None, multisector=False) returning complete RhimeConfig; retain INI interpretation inside the frontend and apply overrides before shared semantic/site resolution; preserve params_from_config dictionary/normalization behavior through shared internal decoding; migrate configured consumers to use the result without resolving twice and verify reader results, override precedence and existing INI section behavior.
- [x] 3.2 Collapse retrieval/reload forwarding layers into load_rhime_data; verify supplied-data no-I/O behavior, cache loading/fallback, layout/selector checks and retained alignment across ordinary/nested/staged consumers.
- [x] 3.3 Forward needed resolved values explicitly to named stages, retain aligned selectors by label and derive run descriptions after preparation; verify retrieval/reload/filter/drop/supplied-data authority, tracer rejection and requested-versus-retained metadata without request mutation.
- [x] 3.4 Rename fresh acquisition to retrieve_inversion_data, sharing the resolved-selector body; preserve data_processing_surface_notracer as a deprecated same-signature/six-tuple wrapper; verify warning, forwarding, imports and equivalent surface/column results without repeated expansion.

## 4. Documentation and delivery

- [ ] 4.1 Reconcile user/developer/API guidance, examples and the existing Issue 804 fragment with the new roles and names; verify examples and affected rendered documentation.
- [ ] 4.2 Run focused and relevant broader tests, changed-path Ruff, strict OpenSpec and whitespace checks for the reconciled implementation; inspect ownership and public compatibility before marking complete.
- [ ] 4.3 SSH-push the reconciled implementation and update #813 using the repository template; verify remote head, target branch, review readiness and app attachment confirmation.

Configuration serialization, resolved-settings logging and an INI writer are
deferred to [#814](https://github.com/openghg/openghg_inversions/issues/814).
No serializer or writer implementation is part of this checklist.
