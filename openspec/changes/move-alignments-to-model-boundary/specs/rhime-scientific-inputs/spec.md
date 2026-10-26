# Spec Delta

## ADDED Requirements

### Requirement: Retained backend-neutral scientific preparation

Standard, multisector and shared nested scientific preparation SHALL retain
backend-neutral observations and spatial sensitivities with their observation
labels and units. Model-specific BC-period expansion and minimum-error floors
SHALL NOT be baked into these reusable raw inputs. Borrowed datasets SHALL NOT
be mutated. Selection and labelled derivation SHALL preserve lazy array backing
except at explicit scientific operations that require execution.
Access to borrowed scientific inputs SHALL NOT hide copying, computation,
persistence, densification or rechunking. Related consumed arrays SHALL preserve
shared lazy graphs and materialize together at the explicit boundary which needs
their concrete values. These rules SHALL NOT require an eager scientific kernel
to retain lazy execution.

Observation-selection filters SHALL run before missing `mf_error` values are
derived or zero errors are imputed. Rejected observations SHALL NOT contribute
to the fallback concentration spread or nonzero-error median. Supplied
`mf_error` SHALL remain authoritative, including zeros, NaNs, labels and metadata.
The established component/quadrature rules SHALL remain unchanged: missing both
components SHALL fail when an error must be derived; zero replacement SHALL use
the larger of the retained nonzero-error median and concentration population
standard deviation. Existing NaN behavior SHALL remain unchanged, including
unfilled repeatability NaNs when `averaging_error=False`. Selection and
imputation SHALL NOT silently define a new resampling or uncertainty-propagation
algorithm. Accepted filter names belong to `rhime-configuration`.

Nested preparation SHALL retain only sites present in both domains, then apply
observation selection once to the canonical outer data. It SHALL align inner
data to the selected outer observations before deriving errors independently
from each retained domain population. Missing required inner coverage SHALL
fail rather than silently removing selected outer observations. Common-site
selection and error derivation SHALL NOT require errors from excluded sites.

#### Scenario: Error imputation uses retained observations

- **WHEN** an observation-selection filter rejects rows that would change the
  fallback concentration spread or nonzero-error median
- **THEN** derived errors use only retained observations
- **AND** supplied custom errors, established missing-component/NaN behavior,
  labels and metadata are preserved without mutating borrowed datasets

#### Scenario: Common nested sites precede error derivation

- **WHEN** acquired or supplied nested domains contain different site sets
- **THEN** only sites present in both domains can enter nested preparation
- **AND** uncertainty components from excluded sites are not required or derived
- **AND** an empty common-site set fails before scientific preparation

#### Scenario: Nested imputation uses selected and aligned populations

- **WHEN** outer observation selection rejects influential rows and inner data
  are aligned to the retained outer observation timestamps
- **THEN** error derivation occurs after that selection and alignment in both
  domain populations
- **AND** rejected outer rows and unused inner rows cannot influence their
  retained domain's fallback estimates
- **AND** missing required inner coverage fails without silently dropping a
  selected outer observation

#### Scenario: Shared nested spatial support

- **WHEN** retained nested sites share one outer domain and one inner domain
- **THEN** preparation normalizes coordinate labels against their domain
  definitions, falling back to the first retained site with a warning for an
  unregistered domain
- **AND** incompatible grids are rejected before overlap masking or basis projection
- **AND** compatible sites share one inner rectangular extent, applied on each
  actual outer footprint and flux grid without mutating or eagerly computing
  borrowed arrays
- **AND** output metadata retains the applied footprint rectangle; externally
  pre-masked inputs with no supplied extent retain an unknown mask extent

### Requirement: Model-owned boundary temporal operands

Compatible raw spatial BC sensitivities SHALL support separate valid constant,
calendar-month and fixed-duration model policies without repeating acquisition,
selection or spatial sensitivity work. Resolution SHALL align each temporal
operand with the retained observation labels and preserve its units and
curtain/period association. Related raw variables, auxiliary coordinates and
borrowed preparations SHALL remain unchanged; replacing the fitted boundary
state axis SHALL NOT discard independently retained variables sharing the raw
spatial axis.

For fixed-duration numerical resolution, an explicit no-anchor choice SHALL
use the earliest retained observation as the effective anchor. Configured
omitted-anchor resolution belongs to `rhime-configuration`. Resolution SHALL
remain independent of a probabilistic-programming backend. Lazy payloads and
required lazy auxiliary coordinates SHALL execute only at the explicit boundary
which needs concrete selected operands, with related selected arrays eligible
for joint materialization.

Consumers requiring final operands SHALL reject unresolved marked raw BC input,
even when eager. If producer-recorded period or anchor policy conflicts with
the requested effective policy, consumers SHALL reject it rather than silently
reinterpret the values. Equivalent effective policies SHALL remain compatible.
Independently supplied unmarked final constant BC matrices SHALL remain supported
for the constant-policy direct builder route; this allowance SHALL NOT imply
acceptance of raw input for a temporal policy requiring expansion. Disabled BC
terms SHALL NOT require alignment or execution of unused BC operands.

#### Scenario: Reuse one spatial preparation for two period choices

- **GIVEN** one raw preparation with labelled spatial BC sensitivities
- **WHEN** two models select different supported temporal policies
- **THEN** each receives its own observation-aligned curtain/period operand
- **AND** the raw preparation and unrelated BC-related variables remain intact

#### Scenario: Resolve an explicit no-anchor numerical policy

- **WHEN** fixed-duration boundary resolution receives an explicit no-anchor
  policy with retained observation timestamps
- **THEN** the earliest retained timestamp supplies the effective numerical anchor
- **AND** observation order and labels are preserved

#### Scenario: Reject a conflicting resolved boundary policy

- **WHEN** final BC values carry producer-recorded period or anchor choices
  incompatible with the requested effective policy
- **THEN** consumption fails before model construction with guidance to resolve
  the raw inputs for the requested policy
- **AND** equivalent duration or timestamp spellings are accepted

#### Scenario: Accept independently supplied final constant BC values

- **WHEN** a direct model builder receives an unmarked final constant BC matrix
  with a compatible constant-policy request
- **THEN** it can consume those values without raw-input temporal resolution
- **AND** marked raw inputs still require model-owned resolution before building

### Requirement: Likelihood-owned minimum-error operands

A named minimum-error method SHALL use retained observations after unusable
fitted rows have been removed. Residual and percentile methods SHALL preserve
their established definitions and unit conversion on that retained population;
scalar and retained-site mapping inputs SHALL retain their validated numerical
meaning and observation/site association. These floors SHALL remain distinct
from supplied observation-error standard deviations.

Only a likelihood that consumes a floor SHALL resolve or validate its required
floor operands. Fixed-error or custom likelihoods with no floor, disabled BC
terms and additive-sigma likelihoods without explicit floor enablement SHALL
NOT cause unused estimator operands or calculations to execute. Required
selected numerical inputs SHALL materialize at an explicit boundary; unrelated
lazy scientific extensions SHALL remain unexecuted.

When supplied final floors carry producer-recorded source choices, incompatible
requested effective choices SHALL fail before model construction rather than
silently reusing stale values. Site-mapping comparison SHALL concern the
retained producer sites, and irrelevant estimator options SHALL NOT create a
false conflict. Independently supplied unmarked final floors SHALL remain
authoritative. Durable fitted operands and replay follow
`sampled-recipe-handoff`; this requirement does not prescribe a storage format.

#### Scenario: Named floor uses retained fitted observations

- **WHEN** invalid sensitivities or observation errors remove otherwise finite
  observations before a named floor is calculated
- **THEN** the floor uses the retained observation population and preserves its
  observation labels, site association and units
- **AND** borrowed prepared observations remain unchanged

#### Scenario: Unused estimator operands stay lazy

- **WHEN** a selected likelihood does not consume a minimum-error floor
- **THEN** missing or lazy estimator-only operands are neither required nor executed
- **AND** unused BC operands and unrelated scientific extensions stay lazy

#### Scenario: Validate a supplied resolved floor policy

- **WHEN** final floor values carry producer-recorded source choices
- **THEN** a conflicting requested effective estimator or scalar/site policy fails
  before model construction
- **AND** equivalent effective choices are accepted without recomputing the values
- **AND** an independently supplied final floor without such producer policy
  remains authoritative
