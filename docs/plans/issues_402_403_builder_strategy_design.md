# Issues 402/403 Builder Strategy Design Notes

Date: 2026-07-24

Status: second-pass architecture notes, not a formal implementation plan.

## Scope And Evidence

These notes revisit:

- [#402](https://github.com/openghg/openghg_inversions/issues/402):
  multi-sector RHIME with shared-basis inputs and a builder-strategy seam.
- [#403](https://github.com/openghg/openghg_inversions/issues/403):
  route multiple flux sources and per-sector priors into RHIME inputs.

They treat the following work as additional constraints:

- [#414](https://github.com/openghg/openghg_inversions/issues/414):
  backend-neutral component preparation and deterministic reconstruction.
- [#456](https://github.com/openghg/openghg_inversions/issues/456):
  grouped state-vector layouts for layered and inner/outer bases.
- [#509](https://github.com/openghg/openghg_inversions/issues/509):
  source-neutral prepared inputs and grouped outputs.
- [#407](https://github.com/openghg/openghg_inversions/issues/407) through
  [#410](https://github.com/openghg/openghg_inversions/issues/410):
  high-resolution inner/outer inputs, models, outputs, and extension guidance.
- [#411](https://github.com/openghg/openghg_inversions/issues/411) through
  [#413](https://github.com/openghg/openghg_inversions/issues/413):
  linked primary-species/tracer specs, models, and outputs.

Relevant merged work since the first version of this note:

- [PR #472](https://github.com/openghg/openghg_inversions/pull/472)
  merged sector-aware latest-PARIS flux outputs on 2026-07-20.
- [PR #510](https://github.com/openghg/openghg_inversions/pull/510)
  merged `run_rhime_from_prepared_inputs(...)` on 2026-07-22.
- [PR #526](https://github.com/openghg/openghg_inversions/pull/526)
  merged backend-neutral `SigmaAlignment` preparation and reconstruction on
  2026-07-24.

Draft PRs #513 through #516 are useful design evidence under #509, but they are
not treated as merged behavior here.

The OpenGHG `ModelScenario` source used for source-combination behavior is:

```text
~/Documents/openghg/openghg/analyse/_scenario.py
```

The local worktree snapshot predates some of the July merges. Statements about
current `devel` after those merges were checked through GitHub; statements about
the existing RHIME implementation were also checked against the local source.

## Recommendation

Revise and split the earlier design.

The first note correctly separated input sources, model grouping, builder
strategy, virtual totals, and product names. Its weak point was the term
"model contribution": it was asked to mean both a sampled state and its
contribution to modelled observations.

That works for current multisector RHIME because every sector has exactly one
state and exactly one `mu_<sector>`. It fails for linked CO2/O2, where one state
must feed several forward terms, and it obscures 6 km models, where inner and
outer are usually state-space groups rather than emission sectors.

The extensible semantic core is a small directed linear model:

```text
input source provenance
-> physical flux components
-> latent state blocks
-> linear forward terms
-> observation channels and likelihoods
-> virtual aggregates and output views
```

Compilation decides how that model becomes a PyMC graph. It must not determine
the scientific identities in the model.

For the narrow closure of #402 and #403, this does not require a public
`SemanticModel`, a strategy registry, or a component-class framework. The
smallest useful seam is:

```text
RhimeModelSpec + RhimePreparedInputs
-> ordered backend-neutral linear model data
-> private loop-sum PyMC compiler
```

That seam should align with #414 rather than introduce a second RHIME-specific
component framework.

## Current Status

### Issue 402

The current code already has most of the mathematical behavior:

- `SectorSpec` and `RhimeModelSpec` represent one or more optimized sectors.
- The multisector builder creates one `x_<suffix>` and `mu_<suffix>` per
  sector.
- Total `mu` is the sum of sector contributions.
- Per-sector priors produce separate latent variables.
- Shared-basis preparation retains source-resolved `H`.
- The basis layer can represent source-specific ragged states through
  `MultiSourceBucketBasisOperator`.
- Generic and PARIS postprocessing reconstruct sector fluxes and calculate
  physical totals from reconstructed fluxes rather than summed scale factors.

The central acceptance item is still missing:

- The loop implementation is hard-coded in
  `build_rhime_multisector_model(...)`.
- Standard and multisector runners still select different builders and output
  paths through a `multisector` boolean.
- `run_rhime_from_prepared_inputs(...)` is the correct new execution boundary,
  but it still requires model sector count, `split_by_sectors`, and the
  presence of `H.source` to encode the same binary mode.
- The current builder consumes padded `H(region, nmeasure, source)` even though
  the basis layer can represent ragged state spaces without padding.

The conclusion is stronger than in the first note: #402 should remain open
until a real internal compilation seam exists. A public strategy option is not
required, but a private normalized linear-term boundary is.

### Issue 403

The current code also implements most of the narrow routing behavior:

- New config and API surfaces use `flux_sources`.
- Multiple OpenGHG sources can be retrieved for multisector runs.
- `sector_priors` supplies separate prior dictionaries.
- `sector_sources` separates user-facing sector labels from OpenGHG source
  names.
- Source-resolved sensitivity input is available as `H(..., source)`.
- Validation covers missing sources, malformed prior structures, and duplicate
  sanitized PyMC names.

Important limitations and one correction:

- `run_rhime` still requires exactly one source, despite the lower-level
  `ModelScenario` ability to combine several sources into one total prior flux.
- Each `SectorSpec` contains exactly one `flux_source`. One optimized component
  cannot be backed by several sources.
- `sector_sources` is not a strict one-to-one mapping. Multiple sectors may
  point to the same source because validation compares sets of source values.
  That can create separate, potentially non-identifiable states over the same
  sensitivity. The actual invariant is "one source reference per sector", not
  a bijection.
- The canonical prepared coordinate is `source`, not `sector`. Renaming that
  coordinate to `sector` merely to satisfy the issue wording would make the
  architecture less clear.
- Unknown or unused sector-prior keys should be treated deliberately rather
  than silently becoming configuration debris.

The original #403 acceptance criteria can be closed narrowly around the current
one-source-per-sector case once its prepared-data contract and validation are
explicit. Grouped sources controlling one state are a distinct semantic
feature and are better tracked explicitly than hidden inside a renaming map.

## Why "Contribution" Is Too Broad

The relationships required by milestones 9 through 11 are not one-to-one.

| Relationship | Current case | Future cardinality |
|---|---|---|
| Input source to physical component | Usually one source per sector | Several sources may form one component; one source may feed several diagnostic views |
| Physical component to state block | One sector owns one state | A component may have inner and outer states or layered states |
| State block to forward term | One state creates one `mu` | One state may feed CO2 and O2 terms or several source-resolved terms |
| Observation channel to forward term | One channel receives all sectors | Each channel receives several terms, potentially from shared states |
| Semantic identity to backend name | Same suffix used everywhere | Stable IDs, trace names, and product names must be separate |
| Component to output view | One sector output | Sector, basis group, species, domain, and virtual totals may be different views |

The word `contribution` is still useful for an observation-space term such as
`H_ff @ x_ff`. It should not imply ownership of the state `x_ff`.

## Proposed Vocabulary

These are conceptual roles, not committed public class names.

### Input source

An acquisition and provenance identity, normally an OpenGHG `source` value.
It says where prior flux data came from. It does not say how many states are
sampled.

### Source group

One or more input sources plus an explicit combination policy.

The policy matters. A tuple of names is insufficient because the model may
mean:

- combine aligned prior fluxes before applying one scale state;
- preserve source-resolved sensitivities but apply the same state to every
  source term;
- optimize every source independently;
- use one source only for provenance or an output diagnostic.

### Flux component

A stable scientific or reporting identity such as `ff`, `ocean`, `biosphere`,
or `anthro`. A sector is a flux component whose scientific meaning is an
emission or uptake sector.

A flux component need not own exactly one state.

### Basis group

A state-space partition within a component, such as `default`, `outer`,
`inner_6km`, a layer, a mask class, or another grouped basis partition.

This follows the direction of #456 and #509: sector/source and basis group are
orthogonal. Inner and outer should not be disguised as sectors simply because
both produce additive terms in observation space.

### State block

One set of sampled or fixed degrees of freedom with:

- a stable semantic ID;
- a prior or fixed-state policy;
- one state-space/basis reference;
- optional component and basis-group labels;
- activity and full-state reconstruction metadata.

The gathered, padded, or separately named backend layout of a state block is a
compilation choice, not its semantic identity.

### Linear forward term

A labelled map from one state block to one observation channel:

```text
mu_term = coefficient_term * (H_term @ x_state)
```

The coefficient can carry a sign, oxidation ratio, unit conversion, or other
declared transform. It may be fixed or have a prior, and it may be scalar or
labelled over coordinates such as time, domain, component, or basis group.
Non-scalar coefficients need an explicit preparation/alignment step analogous
to `SigmaAlignment`; backend broadcasting is not a semantic alignment policy.

If both the coefficient and `x_state` are sampled, the term is bilinear in the
joint latent variables even though it remains linear in either one
conditionally. The semantic representation should therefore preserve the
coefficient as a separate relation rather than folding it irreversibly into
`H_term`. The first #402 compiler only needs the current fixed-scalar case, but
its internal term representation should use a neutral `coefficient` name and
leave room for a fixed value, a prepared labelled value, or a prior-backed
parameter. The term references prepared labelled data; it does not own or
recreate the latent state.

### Observation channel

One observation and likelihood stream, for example CO2 or O2, with:

- its own observation coordinate;
- terms that sum into its modelled mean;
- boundary-condition and fixed-baseline terms;
- offset and error components;
- likelihood configuration.

CO2 and O2 observations must not be required to share one `nmeasure` axis.

### Virtual aggregate

A derived total or grouping over semantic objects. A virtual aggregate is not
automatically an input source or sampled state.

Examples:

- `mu_total` is a sum of compatible observation-space terms.
- `flux_total` is a sum, mosaic, or projection of reconstructed physical flux
  fields.
- a total uncertainty must include covariance cross terms.
- `x_total` is generally undefined unless an explicit, scientifically valid
  scale-factor reduction is declared.

### Output view

A product-neutral statement of what should be reconstructed or grouped, using
stable semantic IDs and coordinates. Product adapters then map those views to
generic, PARIS, or legacy variable names.

### Compilation plan

A record of backend-only decisions:

- loop-sum versus fused or stacked linear algebra;
- separate, gathered, active/fixed, or concatenated sampler layout;
- concrete PyMC and model-data names;
- retained versus reconstructable deterministics.

It should record a decision, not redefine the semantic model.

## Speculative Semantic Kernel

A minimal future model can be described with four relations plus output views.
Illustrative names:

```text
FluxComponentSpec:
    id
    source_group
    source_combination_policy
    label

StateBlockSpec:
    id
    component_id
    basis_group
    state_space_ref
    prior_or_fixed_policy

ObservationSpec:
    id
    species
    likelihood_policy
    nuisance_component_refs

LinearTermSpec:
    id
    state_block_id
    observation_id
    prepared_design_ref
    coefficient_ref

CoefficientSpec:
    id
    value_or_prior_policy
    coordinate_scope
    alignment_policy
    units
    sign_and_direction

OutputViewSpec:
    id
    selectors
    aggregation_policy
    product_neutral_label
```

These relations contain no PyMC suffixes, state-vector slices, loop/stack
choice, or PARIS codes.

This is deliberately smaller than an arbitrary component DAG. It represents
all current and anticipated linear RHIME variants without committing #402 or
#403 to a public semantic IR.

## Two Preparation Levels

PR #510 and issue #509 establish the first source-neutral execution boundary:

```text
source adapter
-> RhimePreparedInputs
-> model builder
-> sampler
-> output bundle
```

Issue #414 introduces a second, component-local preparation boundary:

```text
canonical inversion inputs + component specification
-> prepared component data
-> backend adapter
-> reconstruction
```

These fit together:

```text
source adapter
-> RhimePreparedInputs
-> semantic model normalization
-> backend-neutral prepared linear model
-> compilation plan
-> PyMC graph
```

Conceptually, a prepared linear term needs:

```text
term_id
state_block_id
observation_id
design: H_term(nmeasure_for_channel, state_for_block)
reconstruction and provenance metadata
```

A mapping of separately labelled terms is preferable to one universal xarray
cube over species, sector, basis group, source, and state. A universal cube
would create invalid Cartesian combinations and repeat the ragged-padding
problem already visible in multisource bases.

`RhimePreparedInputs` remains the reusable run boundary. It should not have to
encode the full semantic model solely through `H.source` and
`split_by_sectors`.

## Builder Strategy

The builder seam should compile state blocks and forward terms, not sectors as
indivisible objects.

A compiler should:

1. Create each semantic state block exactly once.
2. Apply every prepared forward term that references that state.
3. Sum terms by observation-channel ID.
4. Attach channel-specific BC, fixed baseline, offset, error, and likelihood
   components.
5. Record state, term, data, and output-role mappings for reconstruction.

The current `add_linear_component(...)` creates a prior/state and applies one
design matrix in one operation. That is convenient for the current one-state to
one-term cardinality, but linked tracers require state creation and linear
application to be separable. The proposed linear seam in #414 is the right
place to establish that separation.

### Loop-sum

The first compiler should preserve the current transparent behavior:

```text
for each state block:
    create state once

for each forward term:
    apply H_term to referenced state

for each observation channel:
    mu_channel = sum(channel terms)
```

This is sufficient for #402. It is easy to validate against current
deterministics and preserves per-term diagnostics.

### Fuse terms sharing a state

If several compatible terms reference the same state and observation channel:

```text
H_a @ x + H_b @ x
-> (H_a + H_b) @ x
```

This is a valid optimization only after checking coordinates, units, state
layout, and coefficient policies. Terms with independently sampled
coefficients cannot in general be fused this way. The optimization is
especially relevant to one-state
multiple-source models.

### Stack independent state blocks

If terms target one observation channel but reference independent state
blocks:

```text
H_a @ x_a + H_b @ x_b
-> concat_state(H_a, H_b) @ concat(x_a, x_b)
```

The compilation plan must retain the block-to-state lookup needed for priors,
posterior reconstruction, and outputs.

### Multiple observation channels

CO2/O2 can be lowered as separate loops, a block matrix, or another backend
representation. The key invariant is that a shared state is created once.
Stacking must not duplicate that state or accidentally apply its prior twice.

No loop/stack choice needs to be public configuration today. An internal
callable or compiler function, with the selected decision recorded in output
metadata, is enough to satisfy the strategy seam.

## Current Runners As Special Cases

### `run_rhime`

Current meaning:

```text
one input source
-> one flux component
-> one state block
-> one forward term
-> one observation channel
```

Future grouped-source meaning:

```text
several input sources
-> one flux component
-> one state block
-> one or several source-resolved forward terms
-> one observation channel
```

The semantic assumption is "one scale state controls the grouped prior flux",
not "there happen to be several sectors with equal values".

### `run_rhime_multisector`

Current meaning:

```text
one source reference per sector
-> one state block per sector
-> one forward term per sector
-> terms summed into one observation channel
```

The current loop-sum builder is one compiler for that semantic shape.

### `run_rhime_from_prepared_inputs`

PR #510 supplies the correct shared post-preparation executor. Its present
validation still derives model mode from:

- sector count;
- `split_by_sectors`;
- whether `H` has a `source` dimension.

That is reasonable compatibility validation for current builders, but it
cannot be the durable semantic rule. A one-state grouped-source model may keep
source-resolved `H`; a 6 km model may have several state blocks without a
source dimension; a tracer model has several channels while reusing states.

The future executor should dispatch from the normalized model relation, while
the current boolean remains a compatibility property of the two M9 wrappers.

## Source Group Semantics And ModelScenario

OpenGHG `ModelScenario` already provides one useful grouped-source behavior.

With `split_by_sectors=False`, `combine_flux_sources(...)`:

- aligns source flux datasets to the highest-frequency time coordinate;
- forward-fills lower-frequency datasets;
- sums the quantified datasets;
- produces one total `fp_x_flux` without a `source` dimension.

This is "combine before scaling": one state acts on the summed prior flux.

With `split_by_sectors=True`, `calc_modelled_obs(...)` also computes
source-resolved `fp_x_flux_sectoral(..., source)` while retaining total
`fp_x_flux`.

That gives two possible prepared forms for one-state multiple-source semantics:

```text
materialized grouped term:
    H_group @ x

source-resolved terms sharing one state:
    H_source_a @ x + H_source_b @ x + ...
```

They are mathematically equivalent only when alignment, units, basis/state
layout, masks, and fixed transforms are compatible. The source-resolved form
retains better provenance and can support source-level diagnostic flux output;
the compiler may still fuse it when valid.

A future `source_group` therefore needs a declared combination policy and
validation invariants. It must not be just `tuple[str, ...]`.

## Speculative Model Shapes

### Standard RHIME

```text
component[total]
    sources = [inventory]

state[total.default]
    component = total
    basis_group = default

term[total_to_ch4]
    state = total.default
    observation = ch4
    mu = H_total @ x_total

observation[ch4]
    mu = term[total_to_ch4]
    likelihood = current RHIME likelihood
```

### One State Backed By Several Sources

```text
component[anthro]
    sources = [ff, industry, waste]
    combine = shared_state_source_terms

state[anthro.default]

term[ff_to_co2]       = H_ff @ x_anthro
term[industry_to_co2] = H_industry @ x_anthro
term[waste_to_co2]    = H_waste @ x_anthro

observation[co2].mu = sum(anthro terms)
```

This preserves source-level reconstruction while sampling one state. A
`combine_before_scaling` adapter may instead materialize `H_anthro`.

### Shared-Basis Multisector

```text
state[ff.default]    -> term[ff_to_ch4]    --\
state[ocean.default] -> term[ocean_to_ch4] ---+-> observation[ch4].mu
state[bio.default]   -> term[bio_to_ch4]   --/
```

The states share a basis definition but remain independently sampled.

### Non-shared Or Ragged Multisector

```text
state[ff.default](state_ff)
    -> H_ff(nmeasure, state_ff)

state[ocean.default](state_ocean)
    -> H_ocean(nmeasure, state_ocean)
```

The builder does not need a padded
`H(region, nmeasure, source)`. Each prepared term carries its own state
dimension. A stacked compiler can create an explicit state layout later.

The current gathered `(source, region_in_source)` basis representation is a
valid storage or compiler layout, not the scientific meaning of the model.

### 6 km Inner/Outer

For one physical component:

```text
component[ff]

state[ff.outer]
    basis_group = outer
    basis = B_outer

state[ff.inner_6km]
    basis_group = inner_6km
    basis = B_inner

term[ff_outer_to_co2] = H_outer @ x_ff_outer
term[ff_inner_to_co2] = H_inner @ x_ff_inner

observation[co2].mu = sum(outer, inner)
```

For multisector 6 km, state blocks may exist for selected
`(flux_component, basis_group)` pairs. The representation should not force a
dense sector x basis-group Cartesian product when some combinations do not
exist.

Observation-space addition does not define physical flux-output composition.
Inner and outer outputs may require:

- a complementary-mask sum;
- an inner-over-outer mosaic;
- projection to a common grid before summing;
- separate native-grid products plus a coarser total.

The chosen support and composition policy must be explicit to avoid overlap or
double counting.

### Linked CO2/O2

```text
state[ff.default] -------------------------------\
    |                                             \
    +-> term[ff_to_co2] = H_co2_ff @ x_ff          +-> observation[co2].mu
    |
    +-> term[ff_to_o2] = r_ff * H_o2_ff @ x_ff ----+-> observation[o2].mu

state[bio.default]
    +-> term[bio_to_co2] = H_co2_bio @ x_bio
    +-> term[bio_to_o2] = r_bio * H_o2_bio @ x_bio
```

Each state exists once. Each observation channel has its own observations,
boundary/fixed baseline, offset, error model, and likelihood.

The conversion-factor relation must record:

- fixed value or prior policy;
- units;
- direction of conversion;
- sign convention;
- component/sector scope;
- coordinate scope, including time or domain dependence;
- alignment and broadcasting policy.

A sampled conversion coefficient that multiplies a sampled flux state makes
that forward relation bilinear. This is representable without copying the flux
state, but it is not part of the strictly linear compiler proposed for the
narrow #402 closure.

The state itself needs an explicit scientific meaning. It might be a
dimensionless scaling of primary-species prior flux, a physical carbon flux, or
another shared latent. Ratio placement and output reconstruction depend on that
choice.

### Combined Multisector, 6 km, CO2/O2

The pressure-test model is:

```text
state[ff.outer]     -> CO2 and O2 terms
state[ff.inner]     -> CO2 and O2 terms
state[bio.outer]    -> CO2 and O2 terms
state[bio.inner]    -> CO2 and O2 terms
state[ocean.outer]  -> CO2 term, and O2 only if scientifically declared
```

This is still the same relation:

- sectors are flux components;
- inner/outer are basis groups;
- states are sampled blocks;
- CO2/O2 are observation channels;
- edges are linear terms;
- totals are output views.

No new runner family is required by the semantic model.

## Virtual Totals

`total` should not be a magic sector.

### Observation totals

For each channel:

```text
mu_channel = sum(forward terms targeting channel)
```

This is always valid once dimensions and units are compatible.

### Flux totals

Reconstruct every physical flux component first, then apply an explicit output
aggregation:

```text
flux_total = aggregate(reconstructed component/group fluxes)
```

For ordinary sectors on one grid, the aggregate is a sum. For inner/outer
domains, it may be a mosaic or projection. For linked tracers, derived
tracer-flux fields need their own unit and sign metadata.

### Uncertainty totals

Totals must retain covariance cross terms. PR #472 established this for
between-sector country totals, and #509 correctly extends the requirement to
sector x basis-group outputs.

### State totals

Scale factors do not have an unconditional total. If all states share an
aligned basis and a product needs a total scale diagnostic, it must define a
scientifically valid reduction such as a prior-flux-weighted scale. That is an
output view, not a sampled `x_total`.

## Output Identity And PR 472

PR #472 is now merged and demonstrates the value of sector-aware
reconstruction:

- sector and total flux variables are emitted;
- totals are reconstructed from physical flux fields;
- sector country totals and covariance diagnostics are retained;
- the PARIS `sector` coordinate is supported by the latest template.

The naming concern from the first note remains real in current `devel`:

- generic output resolves traces and variables through `variable_suffix`;
- `_paris_sector_name_by_suffix(...)` derives PARIS sector codes from that
  PyMC-safe suffix;
- generic and PARIS modules each parse sector metadata into a local
  `OutputSector` shape.

This creates the chain:

```text
OpenGHG source
-> SectorSpec.name
-> variable_suffix
-> generic output variable
-> PARIS sector code
```

Issue #414 now explicitly states the stronger rule: dynamic sector names and
variable roles should be carried explicitly and must not be inferred from
sanitized variable suffixes.

A durable output boundary needs a serializable compilation/reconstruction
manifest:

```text
semantic component/state/term IDs
-> concrete trace and model-data names
-> product-neutral output coordinates
-> product-specific names
```

At minimum it should record:

- state block ID to sampled variable name;
- linear term ID to model-data and optional deterministic names;
- flux component, basis group, observation/species, and source provenance;
- virtual aggregate definitions;
- reconstruction references and conversion factors;
- stable product-neutral labels.

`variable_suffix` can remain a backward-compatible backend naming device. Old
artifacts can retain suffix-based fallback. New output identities should not
depend on changing a PyMC variable name.

PARIS should have one central mapping from product-neutral component identity
to a recorded PARIS sector code. Sanitization can remain a fallback at the
adapter boundary, but the resolved code should be stored and reused.

## Alignment With Current Issues

| Issue | Architectural ownership |
|---|---|
| #402 | Normalize current state/term relations and add the private linear compiler seam; preserve loop-sum behavior |
| #403 | Route current source and prior shorthands into the normalized relation; document one-source-per-sector as a current limitation |
| #414 | Prepare, apply, and reconstruct backend-neutral linear component data; define deterministic retention |
| #456 | Define explicit grouped state layout and metadata without positional slice inference |
| #509 | Own source-neutral adapters, prepared-input execution, state-group outputs, and separation of sampler layout from output accounting |
| #407-#410 | Define inner/outer acquisition, supports, basis groups, composition, and extension guidance |
| #411-#413 | Define shared-state multi-channel relations, conversion metadata, channel likelihoods, and tracer-aware outputs |
| #415/#442 | Persist prepared data, reconstruction manifests, MultiIndexes, and run-bundle provenance |
| #444 | Decide how generic variable roles are selected; this should consume explicit semantic roles rather than recreate suffix inference |

This division avoids having #402 implement a full semantic IR while still
making it a useful precursor.

## Narrow Closure Versus Future Design

### Reasonable closure for #402

The issue can close when:

- standard and multisector model specs normalize into backend-neutral prepared
  linear terms;
- one private compiler constructs the existing loop-sum graph;
- the compiler creates each state once, sums terms by observation channel, and
  records explicit roles;
- `RhimeModelSpec` remains independent of compiler layout;
- current trace names, priors, numerical behavior, and outputs remain
  compatible;
- the non-shared case is documented as separate state blocks/design matrices,
  without requiring padded source cubes.

A public strategy parameter or stacked compiler is not required for closure.
The seam must be real and testable, not only described.

### Reasonable closure for #403

The narrow original issue can close around the one-source-per-sector case when:

- `H.source` is documented as input provenance rather than semantic sector
  identity;
- the sector-to-source relation and duplicate-source behavior are explicit;
- sector-prior keys and missing/unused mappings are validated deliberately;
- error messages identify both semantic sector and source;
- source-resolved prepared inputs feed the normalized linear-term path.

One-state grouped-source support should be tracked as an explicit follow-up
unless #403 is deliberately broadened. Its semantic design is recorded here so
the M9 implementation does not block it.

## Invariants

A future semantic or normalized model should enforce:

- Stable semantic IDs are unique and serializable.
- Every state, term, channel, source group, and output reference resolves.
- Every state block is created at most once per model.
- Every linear term references exactly one state and one observation channel.
- A state may feed several terms and channels.
- Terms summed in one channel have compatible observation coordinates and
  units.
- Source-group combination declares alignment, units, and combination policy.
- Basis groups and state spaces are explicit; slices are not inferred from
  position or numeric labels.
- `total` is never an implicit source or latent state.
- Physical flux aggregation declares sum, mosaic, masking, or projection
  semantics.
- Conversion coefficients declare fixed value or prior, units, sign,
  direction, coordinate scope, and alignment policy.
- Output identities do not depend on backend name sanitization.
- Compilation choices are recorded but do not alter semantic meaning.

## Pressure Tests

The design is extensible only if these can be represented without adding a new
runner family:

- one OpenGHG source, one state, one channel;
- several OpenGHG sources, one shared state, source-resolved diagnostics;
- several sectors with independent priors on a shared basis;
- sectors with different and ragged state spaces;
- one sector with independent outer and inner state blocks;
- selected sector x basis-group combinations without a dense Cartesian cube;
- one state feeding CO2 and O2 channels with different observation indexes;
- channel-specific BC, fixed baseline, offset, error, and likelihood terms;
- fixed or inactive states that remain in full ordered output;
- output grouping by sector, basis group, source provenance, and species;
- total flux uncertainty including sector/group covariance cross terms;
- renaming PyMC variables without changing generic or PARIS product identity.

If one of these requires redefining `source`, inventing a fake sector, copying a
latent state, or parsing a suffix, the design is still encoding current
implementation details as model semantics.

## Open Questions

- Which grouped-source combination policies should be public, and which should
  remain adapter-specific?
- When sources differ in temporal coverage, grid, units, or basis, what
  equivalence is required before their forward terms may share one state?
- Is a flux component always a reporting identity, or can several components
  intentionally share one state?
- Are inner and outer supports complementary, overlapping, or nested, and how
  is double counting prevented?
- Which grid owns a physical total when inner and outer outputs have different
  native resolutions?
- What is the scientific meaning and unit of a shared CO2/O2 latent state?
- Are oxidation coefficients fixed, or prior-backed; and may they vary by
  sector, time, domain, or basis group?
- Can nuisance states be shared across observation channels, or must sharing
  always be explicit?
- Should compiler strategy be selected automatically, injected for tests, or
  only recorded after selection?
- What is the smallest additive schema change that can persist explicit
  state/term/output roles while preserving old `InversionOutput` artifacts?

## Summary

The durable model is not "single-sector versus multisector", and it is not one
generic state-bearing contribution object.

It is:

```text
sources identify provenance
flux components identify physical/reporting meaning
state blocks identify sampled degrees of freedom
forward terms map states into observation channels
channels own likelihood-local behavior
virtual aggregates and output views define totals
compilation plans choose backend graph shape
product adapters own product names
```

M9 can lower this relation with the current loop-sum behavior. M10 adds explicit
basis groups, supports, and multiple grids. M11 adds shared-state fan-out into
multiple observation channels. These are extensions of one relational linear
model, not separate RHIME model families.
