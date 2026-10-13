# Tasks

The 0.8 implementation was accepted for merge on PR #869 after green CI at
`7260b43d`. Delivered contracts are promoted; 0.9 deletion and final archive
remain open. Spec promotion precedes the authorized merge by maintainer request.

## 1. Isolate legacy behavior in 0.8

- [x] 1.1 Give retained RHIME aliases/validation runner/frontend owners and remove all
  production imports/re-exports of HBMCMC outside that package, including the
  RHIME dictionary adapter export and copied examples. Verify modern imports and
  representative configured/output-only execution with HBMCMC unavailable;
  preserve runner alias warnings, precedence and the independent flux-alias
  contract; verify canonical configuration knows no aliases or alias warnings.
- [x] 1.2 Wire historical selection into the actual HBMCMC CLI and normalized
  dictionary route after overrides and before canonical construction. Verify the
  reported NESW options through `run_hbmcmc.main`, file-backed selection,
  superseded missing inputs, dictionary normalization controls and documented
  canonical Path returns, self-contained discovery and clear missing-domain
  errors, early removed-option errors, unchanged caller mappings and established
  output policy.
- [x] 1.3 Add the small production import-boundary check and early modern errors
  for HBMCMC-only selectors. Verify absolute/relative import and re-export
  violations fail the check while legacy-entry tests remain allowed; verify
  errors provide modern replacements or the temporary legacy route.
- [x] 1.4 Update usage migration guidance and add a release fragment: document
  0.8 legacy-only routing and complete HBMCMC removal in 0.9 separately from
  other deprecations. Verify links, the full documentation build, and focused
  HBMCMC/configuration/resource tests, including corrected typed-path assertions.

## 2. Remove the package for 0.9

- [ ] 2.1 Delete HBMCMC's wrapper, adapters and legacy utilities plus obsolete
  entry points and internal consumers, and remove temporary runner spelling/flux
  alias support. Verify wheel/source distributions omit
  the package and no compatibility alias preserves its import path; run the
  import-boundary check and relevant modern workflow regressions.
- [ ] 2.2 Finalize migration documentation and the 0.9 removal fragment. Verify
  modern replacements and older-release guidance for legacy-only artifacts,
  with no accidental removal of independently supported output formats.

## 3. Promote only delivered contracts

- [x] 3.1 After 0.8 implementation and acceptance, verify the already-aligned
  grouped delta and active consumers still preserve the dependency boundary,
  then sync only delivered isolation/translation and
  configuration requirements. Rebase the remaining active delta to retain the
  pending 0.9 removal. Verify the durable alias contract and resource-I/O links
  express the accepted boundary without claiming deletion is delivered.
- [ ] 3.2 After 0.9 deletion and acceptance, sync the remaining removal requirement
  and archive this change. Validate both capabilities, delivery status and links;
  verify no active consumer restores the superseded dependency rule.
