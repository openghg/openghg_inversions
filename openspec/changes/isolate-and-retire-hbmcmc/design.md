# Design

## Context

See [proposal.md](proposal.md). At PR #869 revision `e41e2838`, RHIME runners,
configuration and public exports import `hbmcmc.compatibility`. Conversely,
`run_hbmcmc.main` calls `fixedbasis_params_to_rhime` and then the canonical
constructor without resource translation; historical BC selectors can fail as
unsupported parameters. The delivered configuration spec still assigns shared
alias translation to HBMCMC; the active grouped amendment is now aligned with
this pending correction while retaining its phase-only resolution requirements.

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
2. Preserve supported spelling and flux aliases at runner/frontends through 0.8,
   warning on devel and 0.8 and removing them in 0.9. Canonical configuration
   knows only canonical names, with ordinary unsupported-name errors and no
   alias warnings. Keep the compatibility owner outside HBMCMC and remove
   `rhime._aliases`; do not route translation through `RhimeConfig.from_params`.
   HBMCMC discovery is self-contained; separately deprecated basis adapters
   retain their own schedules. Small duplication avoids reverse dependencies.
3. Leave path conversion, BC loading and packaged-map selection with the
   [resource-I/O change](../../specs/rhime-resource-io/spec.md).
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
- A later configuration delta restores HBMCMC ownership → keep the already-aligned
  grouped delta consistent during implementation and verify it again before sync.
- Legacy artifact users lose utilities in 0.9 → document supported modern
  replacements and older-release use; do not promise new artifact compatibility.

## Delivery and remaining migration

The 0.8 decisions above are implemented and accepted for merge on PR #869 after
CI passed at `7260b43d`. Isolation, canonical configuration and resource-I/O
requirements are promoted to durable capabilities. The resource-I/O change is
[archived](../archive/2026-10-13-move-rhime-resource-io-to-runners/tasks.md).
The grouped-configuration change retains its unimplemented phase-only selection
requirements and consumes the delivered boundary.

Only the 0.9 retirement delta remains here. Delete the HBMCMC package and its
entry points, remove temporary runner spelling/flux aliases, update migration
notes, and verify distribution contents and modern execution. After that
implementation is accepted, sync the remaining removal requirement and archive
this change. Do not mark deletion complete because isolation has shipped.
