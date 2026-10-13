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

Apply winning overrides before semantic resolution while retaining each winning
value's origin. At each external configuration boundary, resolve file-origin
resources against their documented file base and invocation values against the
invocation directory. Forward exact absolute paths and reuse `resolve_basis_inputs`
for selected-resource checks and map loading. Preserve other documented frontend
conventions; superseded inputs must not trigger discovery or validation.

Keep syntax decoding separate from compatibility interpretation. After applying
winning overrides, the legacy HBMCMC input adapter resolves supported historical
case/directory selections once. Legacy routing and modern dependency direction
are owned by the separate [HBMCMC isolation/retirement change](../../isolate-and-retire-hbmcmc/proposal.md).
Modern entry adapters receive explicit selections; modern resolved requests and
numerical functions do not carry discovery rules. Remove `country_directory` in 0.8 and reject it
early with named replacements, rather than retaining a path-resolution exception.

Use declared filesystem types to drive a small standard-library converter
with the owning record's declarations supplied by that owner (for example,
`typing.get_type_hints`). The converter imports no concrete configuration classes
and does not combine independently owned schemas. Canonical paths use `Path`;
annotations identify the values, while the frontend provides their origin/base.
Each external value is converted once; factories, configured runners and identity
encoding forward already canonical values without another normalization pass.
Standalone output configuration normalizes only its own external inputs and does
not require a full run request. Direct canonical dataclass construction expects
resolved values; external-string factories remain available. Do not introduce
Pydantic, recursive object traversal or another registry of path option names.
Configuration construction stays data-free, including no globbing or candidate
artifact inspection; do not move the legacy adapter into `from_params`. Runners
load selected resources before acquisition.

For file-backed BCs, replace case/directory discovery inside sensitivities with
explicit file selection at the boundary and resolved labelled data as the
scientific input. Preserve existing time alignment for an explicitly selected
file or collection of files; do not replace the legacy glob with another hidden
search. NESW uses the directional BC contributions already in acquired data,
without a basis file. A legacy directory supplied with NESW is unused: warn
about that at the compatibility boundary and perform no basis-file lookup.

Keep `country_file` an explicit reporting choice. For preparation, the runner
resolves omitted `landsea_path` and `outer_regions_path` to the exact domain's
vendored defaults only when the selected algorithm needs them. Explicit paths
win; missing files/defaults fail without fallback to another domain or creating
directories. Reuse existing support-map loaders and pass labelled arrays onward.

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
Add a 0.8 migration section covering the removed arguments, explicit BC inputs,
NESW's file-free behavior, packaged preparation defaults and path origins. List
planned 0.9 deprecations separately from removals. The separate HBMCMC change
owns its complete removal in 0.9; preserve other schedules rather than treating
every deprecation version as a removal commitment.
