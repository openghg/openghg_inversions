# Proposal

## Why

Resource paths need one interpretation at the invocation boundary, and optional
basis exports should be visible runner operations. #840 already guarantees exact
destinations; #868 completes ownership without changing that delivered guarantee.

The maintainer's #869 review extends this prerequisite to remove legacy resource
selection from scientific preparation, including BC basis discovery.

## What Changes

- Resolve direct Python/CLI paths and explicit overrides from the invocation
  directory; retain configuration-relative saved-phase file resources and other
  documented frontend conventions. Pass exact absolute paths onward.
- **BREAKING:** Remove `prepare_basis(output_path=...)`; runners optionally save
  the returned basis using its existing codec and the selected export destination.
- Preserve exact exports, early selected-resource checks, scientific results,
  borrowed inputs and existing numerical codecs.
- Translate supported historical case/directory selections at the input
  HBMCMC compatibility boundary under the separate isolation/retirement contract;
  pass explicit BC basis selections to the runner and
  resolved basis data to sensitivities. NESW requires no basis file.
- **BREAKING (0.8):** Remove `country_directory`; use `country_file` for reporting
  and explicit or runner-selected packaged support maps for preparation.
- Identify filesystem fields through type annotations and normalize them with
  a small independent standard-library converter supplied with the owning
  record's annotations and frontend base. Convert each external value once and
  forward canonical paths, including independently usable output configuration.

## Capabilities

### Modified Capabilities

- `rhime-resource-io`: Runner-owned external path resolution and optional basis saves.
- `rhime-configuration`: Explicit compatibility exception for the basis-export
  and legacy-resource migrations, with typed canonical path fields.

## Impact

Implement under [#868](https://github.com/openghg/openghg_inversions/issues/868)
before #866 consumes these boundaries. Standard, multisector, nested, staged and
copied/example callers migrate together. Grouped configuration, new handoff
formats and acquisition persistence remain outside this change. Update migration
guidance with a 0.8 section and a separate list of planned 0.9 deprecations and
removals. No resource manager or replacement serialization layer is introduced.
Sync only after implementation and acceptance; the durable resource spec retains
the existing optional preparation-save contract until then.

Dependency direction, legacy-only routing and complete HBMCMC removal in 0.9
belong to the separate [isolation/retirement change](../../isolate-and-retire-hbmcmc/proposal.md).
This change consumes that boundary without owning or duplicating its prohibition.
The 0.8 isolation correction is required before acceptance here; 0.9 deletion is
tracked separately and does not block the resource-I/O delivery.
