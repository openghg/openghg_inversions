# Design

## Context

See [proposal.md](proposal.md). At PR #869 revision `e41e2838`, RHIME runners,
configuration and public exports import `hbmcmc.compatibility`. Conversely,
`run_hbmcmc.main` calls `fixedbasis_params_to_rhime` and then the canonical
constructor without resource translation; historical BC selectors can fail as
unsupported parameters. The delivered configuration spec and the active grouped
amendment still explicitly assign shared alias translation to HBMCMC.

## Goals / Non-Goals

**Goals:** make modern execution independent of legacy package availability,
retain a working 0.8 migration entry point, and make complete 0.9 deletion a
bounded release operation.

**Non-goals:** no model, sampler, output-codec, checkpoint or configuration-grouping
redesign; no immediate deletion of HBMCMC from the 0.8 implementation.

## Decisions

1. Keep fixedbasis policy and historical resource discovery in HBMCMC. Wire the
   CLI and normalized dictionary route after overrides, before construction.
   Keeping calls in modern runners or placing discovery in `from_params` would
   retain the reverse dependency or violate data-free construction.
2. Move genuinely modern aliases and validation to modern owners. The existing
   RHIME alias vocabulary and output-format aliases remain supported under their
   current contracts; fixedbasis switches and case/directory discovery do not
   become modern aliases. Remove the RHIME `params_from_config` re-export. Use
   small duplication in HBMCMC where extraction would couple modern behavior
   back to legacy policy; do not introduce a shared compatibility framework.
3. Leave path conversion, BC loading and packaged-map selection with the
   [resource-I/O change](../move-rhime-resource-io-to-runners/design.md).
   Its independent converter accepts owner declarations and a base; it does not
   import configuration owners. This change consumes that behavior rather than
   defining a second converter or resource policy.
4. Enforce the package boundary with a small production import check, covering
   absolute/relative imports and re-exports, plus a representative modern import
   and execution check that rejects attempts to load HBMCMC. Test modules may
   still import legacy APIs. Avoid a general dependency-analysis framework.
5. Separate 0.8 isolation acceptance from 0.9 removal. Keep retirement tasks open
   after 0.8 work; do not declare the whole change delivered or promote pending
   removal as implemented. The resource-I/O change can complete independently
   once its own behavior and the required 0.8 boundary are accepted.

## Risks / Trade-offs

- Mistaking supported modern aliases for fixedbasis policy → retain alias and
  conflict-precedence tests while adding early errors for legacy-only selectors.
- Tests exercise helpers but miss the CLI → reproduce the reported NESW request
  through `run_hbmcmc.main` with acquisition mocked at its owning boundary.
- A later configuration delta restores HBMCMC ownership → reconcile the grouped
  delta before sync and verify promoted requirements and active consumers.
- Legacy artifact users lose utilities in 0.9 → document supported modern
  replacements and older-release use; do not promise new artifact compatibility.

## Migration Plan

Review these artifacts on PR #869 before further code changes. Implement and
validate 0.8 isolation together with the resource-I/O correction, including
migration guidance and a release fragment. Keep complete 0.9 removal as an
explicit outstanding phase. After 0.8 implementation and acceptance, sync only
the delivered isolation/translation requirements and configuration amendment;
leave the 0.9 removal requirement in this active change and rebase its remaining
delta against the promoted capability. This permits resource-I/O to reference a
durable isolation contract without keeping the obsolete alias-ownership rule.
Verify distribution contents and modern execution when 0.9 deletion lands; only
then sync the remaining requirement and archive this change. Durable specs
remain untouched during this proposal.

The pending configuration delta replaces the old HBMCMC-owned alias requirement.
The resource-I/O compatibility amendment references this capability for legacy
routing exceptions; its requirements retain path and resource semantics. Before
sync, reconcile overlapping grouped-configuration requirements so none restores
the superseded dependency rule. Update active-change links to the delivered
capability or actual archive location during promotion.
