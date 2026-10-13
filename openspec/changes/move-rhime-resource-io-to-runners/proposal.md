# Proposal

## Why

Resource paths need one interpretation at the invocation boundary, and optional
basis exports should be visible runner operations. #840 already guarantees exact
destinations; #868 completes ownership without changing that delivered guarantee.

## What Changes

- Resolve direct Python/CLI paths and explicit overrides from the invocation
  directory; retain configuration-relative saved-phase file resources and other
  documented frontend conventions. Pass exact absolute paths onward.
- **BREAKING:** Remove `prepare_basis(output_path=...)`; runners optionally save
  the returned basis using its existing codec and the selected export destination.
- Preserve exact exports, early selected-resource checks, scientific results,
  borrowed inputs and existing numerical codecs.

## Capabilities

### Modified Capabilities

- `rhime-resource-io`: Runner-owned external path resolution and optional basis saves.
- `rhime-configuration`: Explicit compatibility exception for the basis-export
  argument migration and changed saved-phase override path convention.

## Impact

Implement under [#868](https://github.com/openghg/openghg_inversions/issues/868)
before #866 consumes these boundaries. Standard, multisector, nested, staged and
copied/example callers migrate together. Grouped configuration, new handoff
formats, BC case/directory migration and acquisition persistence are outside this
change. No resource manager or replacement serialization layer is introduced.
Sync only after implementation and acceptance; the durable resource spec retains
the existing optional preparation-save contract until then.
