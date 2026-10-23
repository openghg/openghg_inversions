# Pending HBMCMC retirement

## ADDED Requirements

### Requirement: HBMCMC retirement follows the development evidence gate

The agreed development retirement SHALL remove the entire
`openghg_inversions.hbmcmc` subpackage, including the executable wrapper,
compatibility adapters and legacy data/basis utilities within that subpackage.
Retirement SHALL preserve the durable
[Development removals require affected migration evidence](../../../../specs/hbmcmc-isolation-retirement/spec.md#requirement-development-removals-require-affected-migration-evidence)
guarantees. Temporary runtime translation SHALL retire only for the spellings
enumerated in
[Compatible public adapters and scientific choices](../../../../specs/rhime-configuration/spec.md#requirement-compatible-public-adapters-and-scientific-choices);
canonical names and modern returns SHALL retain their contracts. Modern execution SHALL NOT invoke
migration tooling. No import aliases, forwarding shims or entry points SHALL
preserve the HBMCMC namespace after its removal.

Supported modern workflows, modern `output_format="legacy"`, saved
`InversionOutput` exports and independently supported basis/scientific codecs
SHALL remain available. Python names SHALL follow only the finite
[five-alias retirement](../../../../specs/rhime-model-selection/spec.md#requirement-retire-named-python-aliases),
preserving canonical callables/types and artifact identities. Other Python names
and independent basis capabilities SHALL retain their separately specified support. CLI names SHALL
follow only the finite
[installed-command exceptions](../../../../specs/rhime-model-selection/spec.md#requirement-bounded-installed-command-retirement),
preserving canonical operations and argument behavior without a broader Python
import/signature/result-wrapper exemption.

Migration guidance SHALL distinguish delivered isolation, permission for gated
development retirement and removal actually implemented. It SHALL identify
modern replacements for retained workflows and matching older software for
historical execution or unsupported artifacts.

#### Scenario: Install a distribution after retirement is implemented

- **WHEN** a user installs a wheel or source distribution after gated retirement is implemented
- **THEN** it contains no HBMCMC subpackage or executable entry point
- **AND** the HBMCMC module path cannot be imported through a compatibility shim
- **AND** modern workflows and separately supported output formats remain usable

#### Scenario: Follow development migration guidance

- **WHEN** a user reviews the legacy migration documentation before upgrading
- **THEN** delivered removals and pending development retirement are identified separately
- **AND** RHIME execution and retained prepared-input/basis artifacts are identified where applicable
- **AND** unsupported legacy-only operations are identified without promising new codecs
