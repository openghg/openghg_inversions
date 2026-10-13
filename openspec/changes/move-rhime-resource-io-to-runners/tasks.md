# Tasks

Deliver under #868 before #866 consumes these boundaries. Exact destinations and
selected-resource checks already exist; preserve their tests and scientific behavior.

## 1. Runner-owned resources and exports

- [ ] 1.1 Resolve file resource values and explicit invocation overrides at their
  owning boundary; verify selected paths from a different working directory,
  absolute paths and equivalent full/staged requests.
- [ ] 1.2 Remove optional saving from `prepare_basis` and migrate every runner,
  staged and copied/example caller to the existing save codec; verify generated,
  reused and partition bases, export metadata, no unrequested writes and exact
  destinations inside/outside checkpoint directories without a repeated prefix.
- [ ] 1.3 Update affected API documentation and add the #868 release note; pass
  focused caller/scientific parity checks, including early missing-resource errors
  without fallback or read-side directory creation.

## 2. Completion

- [ ] 2.1 Record merged implementation and acceptance, then sync both deltas and
  archive this change before the grouped-configuration delta is synced. Confirm
  the durable resource spec describes the resulting implemented save boundary.
