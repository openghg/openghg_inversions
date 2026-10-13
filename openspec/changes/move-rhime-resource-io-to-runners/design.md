# Design

## Context

See [proposal.md](proposal.md). `resolve_basis_inputs` already resolves selected
maps before acquisition, and basis codecs already accept exact paths. The current
staged loader merges overrides before applying its config-directory path base;
`prepare_basis` still optionally saves its result. Those are the bounded #868
changes. The exact-export guarantee from #840 is already a durable baseline.

## Goals / Non-Goals

Make path interpretation and optional export visible at invocation boundaries.
Preserve science, exact destinations, numerical codecs and borrowed arrays.
No configuration grouping, new checkpoints or generic resource framework.

## Decisions

Resolve file-origin resource paths before combining explicit invocation overrides;
resolve the latter against the invocation directory. Forward exact absolute paths
and reuse `resolve_basis_inputs` for selected-resource checks and map loading.
Preserve other documented frontend conventions.

Remove `output_path` from canonical `prepare_basis`. Let the runner save the
returned basis or the basis in its prepared handoff through `BasisFunctions.save`,
preserving export metadata where existing consumers require it. Update all
standard, multisector, nested, staged and copied/example callers together.
Checkpoint writers retain only their checkpoint-owned destinations.

## Risks / Trade-offs

- Different path origins selecting different files: exercise file values and
  invocation overrides from another working directory.
- Export migration changing metadata or science: preserve codec behavior and
  run focused caller/parity tests, including loaded operators and partitions.
- Overlap with the grouped-configuration compatibility delta: land this change
  first; the later delta preserves its migration exception.

## Migration Plan

Deliver #868 before #866 consumes these boundaries, with API documentation and a
release note. Sync the resource and compatibility deltas only after acceptance;
the durable spec keeps the existing optional preparation-save API until then.
