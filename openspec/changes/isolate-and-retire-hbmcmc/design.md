# Design

## Development-boundary amendment — 23 October 2026

The accepted #944/#961 treatment supersedes the future version schedule below.
Delivered 0.8 isolation remains historical evidence. Pending #719 retirement may
proceed on devel while preserving the durable
[evidence guarantees](../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence)
and following the [delivery checklist](tasks.md#migration-evidence-checklist-and-delivery-order).
That checklist owns the finite namespace/translation scope, pinned #960 baseline,
per-PR checks and current delivery order. It grants no general public
signature/result-wrapper exception. The five-name Python amendment below is a
finite exception; other names retain their contracts.
Converters, launchers, maintenance publishing and integrated acceptance remain M2
release work. Final release/public-alias choices stay with #945; independent
basis readers, writers and discovery stay with #951. This amendment implements
no runtime removal.

The [finite CLI command map](tasks.md#cli-command-name-amendment--23-october-2026)
adds separately accepted command-name retirement. Target one `rhime` executable
and canonical `run --model` routes, preserving positional dates, `--kwargs`,
configuration/override meanings, scientific execution and artifacts. Current
aliases remain implemented until the removal slice lands; no argument redesign,
package rename or blanket Python-name migration follows from this amendment.

The [five-name Python alias map](tasks.md#named-python-alias-amendment--23-october-2026)
permits removing only those alias assignments without continued naming-only
forwarders. Recommend canonical `run_multisector`, `run_nested`, `MergedData`,
`PreparedInputs` and `assemble_inputs`, preserving the same callable/class objects,
constructors, result contents, scientific behavior, borrowed execution boundaries
and persisted schema identifiers/artifacts. Current source still implements the
aliases; broader types, symbols and module moves require separately agreed maps.
Positional dates and `--kwargs` remain open design questions; preserving their
current meanings for command renaming does not decide their final syntax.

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
preserve delivered isolation, and make complete development retirement a
bounded operation after affected evidence and scientific regressions exist.

**Non-goals:** no model, sampler, output-codec, checkpoint or configuration-grouping
redesign; no runtime deletion or release-readiness claim in this amendment.

## Decisions

1. Keep fixedbasis policy and historical resource discovery in HBMCMC. Wire the
   CLI and normalized dictionary route after overrides, before construction.
   Keeping calls in modern runners or placing discovery in `from_params` would
   retain the reverse dependency or violate data-free construction.
2. Retire temporary spelling, output-format and `emissions_*` option translation
   through #719 after its affected evidence and scientific regressions exist.
   Until each removal lands, existing frontend precedence and warnings apply
   outside HBMCMC. Canonical configuration knows only canonical names; normal
   execution must not invoke migration tooling. Outside the accepted finite
   CLI/five-name Python maps above, public names retain their contracts pending
   exact #945 decisions. Independent basis adapters remain
   with #951, outside blanket retirement. HBMCMC discovery stays self-contained
   until removal.
3. Leave path conversion, BC loading and packaged-map selection with the
   [resource-I/O change](../../specs/rhime-resource-io/spec.md).
   Its independent converter accepts owner declarations and a base; it does not
   import configuration owners. This change consumes that behavior rather than
   defining a second converter or resource policy.
4. Enforce the package boundary with a small production import check, covering
   absolute/relative imports and re-exports, plus a representative modern import
   and execution check that rejects attempts to load HBMCMC. Test modules may
   still import legacy APIs. Avoid a general dependency-analysis framework.
5. Separate delivered isolation from pending development retirement. Keep
   deletion tasks open until implementation and revision-specific validation
   exist. Check intervening changes against #960 for each removal, preserve
   historical expectations and port affected scientific tests. Unresolved
   dispositions do not authorize deletion. Resource I/O and independently
   retained codecs keep their own contracts.

## Risks / Trade-offs

- Mistaking supported modern aliases for fixedbasis policy → retain alias and
  conflict-precedence tests while adding early errors for legacy-only selectors.
- Tests exercise helpers but miss the CLI → reproduce the reported NESW request
  through `run_hbmcmc.main` with acquisition mocked at its owning boundary.
- A later configuration delta restores HBMCMC ownership → keep the already-aligned
  grouped delta consistent during implementation and verify it again before sync.
- Legacy artifact users lose utilities at retirement → document supported modern
  replacements and older-release use; do not promise new artifact compatibility.

## Delivery and remaining migration

The 0.8 decisions above are implemented and accepted for merge on PR #869 after
CI passed at `7260b43d`. Isolation, canonical configuration and resource-I/O
requirements are promoted to durable capabilities. The resource-I/O change is
[archived](../archive/2026-10-13-move-rhime-resource-io-to-runners/tasks.md).
The grouped-configuration change retains its unimplemented phase-only selection
requirements and consumes the delivered boundary.

Development retirement remains pending here. After the affected evidence and
scientific regression gate, delete the HBMCMC package and its entry points,
remove temporary runner spelling/output-format/flux aliases, update migration
notes, and verify distribution contents and modern execution. After that
implementation is accepted, sync the remaining removal requirement and archive
this change. Finished converter/launcher interfaces are not a development
prerequisite. Do not mark deletion complete because isolation or this amendment
has shipped.
