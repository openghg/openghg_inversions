# Design

## Context

See [proposal.md](proposal.md). The delivered `RhimeConfig` keeps acquisition and
preparation fields flat. The reviewed amendment groups those choices while the
existing `SiteOptions`, model, sampler and output owners retain their roles.

## Goals / Non-Goals

Keep one owner for each choice and default, with equivalent full/saved resolution.
This change does not introduce a configuration registry, resource manager or
handoff codec. Acquisition and preparation remain distinct operations.

## Decisions

Move fields/defaults into `DataConfig` and expose `config.data`; do not keep a
second flat copy or an alias for the unshipped `preparation` group. Runners select
named acquisition arguments and preparation arguments explicitly. Apply winning
raw overrides before the existing alias owner, then resolve only applicable
groups. Bind inference choices against requested or retained preparation facts
as specified in the delta. Reuse the established sampler and model owners.

Keep resource conventions with the prerequisite resource-I/O change. Its small
compatibility amendment lands first; this delta retains that exception when it
updates the same public-compatibility requirement. This order avoids undoing
the basis-export migration while grouping configuration.

## Risks / Trade-offs

- Duplicate defaults or divergent aliases: cover equivalent full/saved inputs.
- Shared BC choices drifting: cover omitted, explicit and conflicting values
  against saved preparation facts, without recomputing scientific inputs.
- Premature completion: keep these requirements under `changes/` until the
  corresponding #866 acceptance evidence exists.

## Migration Plan

Migrate configured runners, CLI and examples together with #866. Reuse the
handoff change's scientific acceptance rather than adding another test harness.
Sync this delta and archive only after its tasks are complete; planning approval
or merging #867 does not complete implementation.
