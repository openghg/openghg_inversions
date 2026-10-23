# Tasks

The 0.8 implementation was accepted for merge on PR #869 after green CI at
`7260b43d`. Delivered contracts are promoted; development deletion and final archive
remain open. Spec promotion precedes the authorized merge by maintainer request.

## Development-boundary amendment — 23 October 2026

#944/#961 supersede the future compatibility schedule. Checked 0.8 tasks below
preserve historical implementation/validation evidence, including their former
0.9 guidance; they do not renew that schedule. The durable
[evidence contract](../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence)
protects source observations and scientific expectations. The concrete #960
baseline, PR checklist and current owners are delivery instructions below,
not permanent API requirements. No deletion task is completed by this amendment.

### Migration evidence checklist and delivery order

The bounded removal scope is the `openghg_inversions.hbmcmc` namespace, including
`run_hbmcmc`, its dictionary-returning `params_from_config` adapter and legacy
data/basis utilities; plus temporary translation of `outputpath`, `outputname`,
`xprior`, `bcprior`, `sigprior`, `offsetprior`, `emissions_name`,
`outer_region_definition_file`, the `emissions_store`/`emissions_domain` and
`inner_emissions_store`/`inner_emissions_domain` flux selectors, and the output
names `hbmcmc`/`hbmcmc_postprocessing`. Wrapper-only `nit`, `nchain`, `verbose`
and `sampler_kwargs` translation retires with that wrapper. Existing explicitly
named API migrations retain their own contracts; there is no general exception
for historical signatures or result wrappers. The finite Python-alias amendment
below permits only its five named removals; other Python names retain their
existing contracts pending explicit per-interface decisions.

For each affected removal or deliberate behavior-change PR:

- Identify affected #960 ledger entries and plain cases; add bounded discoveries
  before removal with explicit source version and execution route.
- Compare the PR base and affected source with pinned development revision
  `0a32d73a5c4a72be0505a4ed682472e1613e49e9`, accounting for intervening changes
  and relevant merged PRs.
- Record disposition and follow-up owner: retained, mechanically migrated,
  deliberate scientific change, unsupported or unresolved. Preserve historical
  inputs/expectations; add dated modern comparisons rather than replacing them.
- Preserve or port relevant scientific coverage before retiring adapter/test
  plumbing. Test unchanged behavior on canonical inputs and deliberate scientific
  changes against separately agreed semantics. Distinguish inspected evidence
  from executed checks and identify the validated implementation revision.

#960 remains a pinned archive; refresh only affected evidence and live checks,
without requiring a whole-audit rerun. #719 owns runtime retirement and regression
migration. Complete #719, including the separately identified special-tracer
compatibility slice, before #964. #971 is a separate prerequisite for #964,
not for #719. #876/#964 own the agreed scientific-input/retained-population changes;
their historical witnesses remain distinct from new-semantics tests.

Finished converters (#949), launchers (#950), maintenance publication (#947),
integrated acceptance (#952) and complete migration guidance (#432) remain M2
release gates, not M1 development-removal prerequisites. #948 owns final source
coverage; #945 owns release numbering and unresolved public interfaces; #951
owns independent basis decisions. Completion of evidence does not resolve those
decisions or establish release readiness.

### CLI command-name amendment — 23 October 2026

The maintainer accepted this finite command map in
[#663](https://github.com/openghg/openghg_inversions/issues/663#issuecomment-6364272469)
and [#945](https://github.com/openghg/openghg_inversions/issues/945#issuecomment-6364278950),
then authorized the bounded contract amendment during #972 review. Implementation
is pending; this record does not remove any executable or parser alias.

| Retire | Canonical replacement |
| --- | --- |
| `openghg-inversions` executable | `rhime`, retaining its same subcommands |
| `run-rhime` under either executable | `rhime run --model standard` or `rhime run` |
| `run-rhime-multisector` under either executable | `rhime run --model multisector` |
| `run-rhime-nested` under either executable | `rhime run --model nested` |
| `run-co2` under either executable | `rhime run --model co2` |

The target has one installed executable, `rhime`, without forwarding commands
retained solely for naming compatibility. For this command-name migration,
preserve positional dates, `--kwargs`,
effective configuration, date/output override precedence, scientific execution,
result contents, outputs and artifact contracts. No replacement argument syntax,
Python package/distribution/install rename or persisted-identifier rename is agreed.
Positional dates and `--kwargs` remain open for discussion and possible revision;
preserving current argument meanings here does not finalize their design.
Broader Python names/imports require their own exact maps under #663/#945 and a
#776 revival; this finite CLI decision grants no general Python-name exception.

Apply the evidence checklist above to R1/R2 and CLI invocation cases. Source
inspection at this amendment found no `cli.py` changes since pinned
`0a32d73a5c4a72be0505a4ed682472e1613e49e9`; intervening `pyproject.toml` edits
affect dependencies/test markers, not console-script declarations. This is source
comparison evidence, not executed command/scientific validation. Preserve the
original #960 ledger, fixtures and expectations; add dated implementation
comparisons and port live tests to canonical routes when removal is implemented.

The implementation slice must update console scripts, parsers, tests, templates,
README/usage references and migration errors together. Preserve the distinct CO2
forwarding behavior used by canonical `run --model co2`; retain argument/override
assertions and verify retired commands and the second executable no longer ship.
#663/#945 own the command map and coordination, and #950 consumes it for launcher
guidance. This accepted CLI map can be delivered independently of unresolved
Python maps; it completes no runtime task or release gate.

### Named Python-alias amendment — 23 October 2026

The maintainer approved removal of these five currently implemented alias
assignments, with no continued aliases solely for naming compatibility:

| Retire | Canonical replacement |
| --- | --- |
| `run_rhime_multisector` | `run_multisector` |
| `run_rhime_nested` | `run_nested` |
| `RhimeMergedData` | `MergedData` |
| `RhimePreparedInputs` | `PreparedInputs` |
| `assemble_rhime_inputs` | `assemble_inputs` |

This amendment records permission and the target map, not runtime deletion.
The aliases remain direct assignments in the inspected source. Preserve the
canonical functions, class objects, constructors, argument/default meanings,
result contents, borrowed/lazy execution boundaries and scientific behavior.
Naming cleanup must not incidentally reinterpret scientific data. Supported
format/version acceptance, conversion or rejection remains the artifact owner's
decision, with no implied 0.7 format support or literal schema-identifier freeze.
Existing `run_rhime`, `RhimeRunSpec` and `RhimeResult`
migrations retain their separate explicit contracts; there is no general
signature/result-wrapper exemption.

Apply the migration evidence checklist to affected imports/callers and add
bounded source witnesses where the pinned #960 ledger does not cover an alias.
Preserve original historical inputs/expectations, compare affected source with
the pinned baseline and port relevant regression coverage before deletion.
The implementation slice updates exports, callers, tests, API documentation and
migration guidance together and checks the retired imports no longer resolve.
Broader symbols, types and module moves remain under #663/#945 and a #776 revival,
requiring exact replacement maps before their own removals. This finite decision
does not finalize CLI positional-date/`--kwargs` syntax or complete a delivery task.

## 1. Historical isolation delivery in 0.8

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

## 2. Remove runtime compatibility at the development boundary

- [ ] 2.1 Delete HBMCMC's wrapper, adapters and legacy utilities plus obsolete
  entry points and internal consumers, and remove temporary runner spelling/output-format/flux
  alias support after the affected evidence and regression gate. Retire special
  `use_tracer` handling in a separately identified #719 slice: every explicit
  occurrence, regardless of value, fails ordinary unsupported-option validation;
  omission remains valid.
  Complete #719 before #964; #971's separate filter retirement is not a
  prerequisite for #719. Leave unresolved
  public aliases and independent basis capabilities with #945/#951.
  Verify wheel/source distributions omit
  the package and no compatibility alias preserves its import path; run the
  import-boundary check and relevant modern workflow regressions.
- [ ] 2.2 Update concise devel migration guidance and the removal fragment. Verify
  modern replacements and older-release guidance for legacy-only artifacts,
  with no accidental removal of independently supported output formats.

## 3. Promote only delivered contracts

- [x] 3.1 After 0.8 implementation and acceptance, verify the already-aligned
  grouped delta and active consumers still preserve the dependency boundary,
  then sync only delivered isolation/translation and
  configuration requirements. Rebase the remaining active delta to retain the
  pending 0.9 removal. Verify the durable alias contract and resource-I/O links
  express the accepted boundary without claiming deletion is delivered.
- [ ] 3.2 After development deletion and acceptance, sync the remaining removal requirement
  and archive this change. Validate both capabilities, delivery status and links;
  verify no active consumer restores the superseded dependency rule.
