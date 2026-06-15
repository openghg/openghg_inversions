# Issues 402/403 Builder Strategy Design Notes

Date: 2026-06-09

Status: design notes, not an implementation plan.

## Scope

These notes revisit:

- [#402](https://github.com/openghg/openghg_inversions/issues/402):
  multi-sector RHIME with shared-basis inputs and a builder-strategy seam.
- [#403](https://github.com/openghg/openghg_inversions/issues/403):
  route multiple flux sources and per-sector priors into RHIME inputs.
- [#472](https://github.com/openghg/openghg_inversions/pull/472):
  sector-level latest-PARIS flux outputs.

They also check the design against milestone 10 high-resolution requirements
([#407](https://github.com/openghg/openghg_inversions/issues/407),
[#408](https://github.com/openghg/openghg_inversions/issues/408),
[#409](https://github.com/openghg/openghg_inversions/issues/409),
[#410](https://github.com/openghg/openghg_inversions/issues/410)) and
milestone 11 tracer requirements
([#411](https://github.com/openghg/openghg_inversions/issues/411),
[#412](https://github.com/openghg/openghg_inversions/issues/412),
[#413](https://github.com/openghg/openghg_inversions/issues/413)).

The source checkout used for OpenGHG `ModelScenario` behavior is:

```text
~/Documents/openghg/openghg/analyse/_scenario.py
```

## Position

The durable distinction is not "single-sector versus multi-sector". The
durable distinction is how a set of named flux contributions is compiled into
one or more modelled-observation vectors.

Current RHIME already has the bones of one pattern:

```text
OpenGHG flux sources
-> prepared sensitivities and prior fluxes
-> named model contributions
-> summed concentration mean
-> shared BC / offset / error / likelihood
-> product-specific outputs
```

On that framing:

- `run_rhime` is one model contribution, currently restricted to one
  `flux_source`.
- `run_rhime_multisector` is multiple model contributions, currently one
  contribution per `flux_source`, compiled with a loop-sum strategy.
- A future single-sector multi-source inversion is one model contribution backed
  by a group of input sources.
- A future high-resolution model is multiple domain contributions, for example
  outer plus inner, compiled into one total `mu`.
- A future linked CO2/O2 model has one or more latent sector states feeding
  multiple species or tracer forward models and likelihoods.

That suggests a small "contribution plan" as the local bridge toward the
semantic model representation. It can remain private and RHIME-specific for now,
but it should use the same concepts that a later `SemanticModel` would use.

## Current Coverage And Gaps

### Issue 402

Already mostly done:

- `SectorSpec` and `RhimeModelSpec` exist and keep model metadata out of the
  runner.
- `build_rhime_multisector_model(...)` builds one `x_<suffix>` and
  `mu_<suffix>` per sector and creates total deterministic `mu` as the sum of
  sector contributions.
- Per-sector priors are passed through to distinct latent variables.
- Shared-basis multisector preparation carries source-resolved sensitivity data
  in `H(..., source)`.
- Source-specific/ragged basis operators already exist in the basis layer.

Remaining design gap:

- The builder strategy seam is not real yet. The loop is hard-coded in the
  multisector builder, and the runner chooses separate builders with a
  `multisector` boolean.
- The current model metadata says "sector", but the prepared data coordinate is
  still OpenGHG `source`. That is acceptable as a transitional layout, but it is
  not a semantic sector model.
- Non-shared basis support is partly present below the builder, but the current
  RHIME builder still wants legacy padded source structure rather than a direct
  gathered-state contribution layout.

Minimum closure for #402 may not require a public strategy registry. It does
require a place where the model builder first normalizes the model into ordered
linear contributions, then applies a compilation shape such as loop-sum. The
spec should continue to describe the math and labels, not whether PyMC uses a
loop, a stacked design matrix, or some later PyTensor operation.

### Issue 403

Already mostly done:

- New API/config surfaces prefer `flux_sources`.
- `sector_sources` can map user-facing sector names to OpenGHG source names.
- `sector_priors` accepts per-sector prior dictionaries.
- Validation catches duplicate sanitized sector names, malformed sector priors,
  and mismatched `sector_sources`.
- `run_rhime_multisector` can retrieve multiple OpenGHG flux sources and prepare
  source-resolved sensitivities.

Remaining design gap:

- `run_rhime` currently rejects multiple `flux_sources`, even though OpenGHG
  `ModelScenario` has a coarse single-total multi-source behavior.
- `sector_sources` currently enforces a one-to-one mapping between model sectors
  and OpenGHG sources. That makes it mainly a renaming device when there is no
  real grouping.
- The canonical "sector" sensitivity is really source-resolved sensitivity. It
  does not yet express one model sector backed by multiple source inputs.

The important fresh constraint is that "multiple sources" and "multiple model
states" are not the same thing.

## ModelScenario Constraint

OpenGHG `ModelScenario` already distinguishes two useful modes.

With `split_by_sectors=False`, `combine_flux_sources(...)` takes one or more
OpenGHG flux sources and combines them into one flux dataset. The downstream
`fp_x_flux` has no `source` dimension. That is the old coarse behavior: many
input sources, one modelled flux field, one possible state vector.

With `split_by_sectors=True`, `calc_modelled_obs(...)` still computes the usual
total `mf_mod` and `fp_x_flux`, then loops over each source to add
`mf_mod_sectoral` and `fp_x_flux_sectoral` with a `source` dimension.

This is a strong argument against using `source` and `sector` interchangeably.
OpenGHG `source` is input provenance. A RHIME sector or contribution is a model
grouping decision.

It also means a single-sector multi-source RHIME path is not an exotic future
feature. It is the behavior that the current `run_rhime` wrapper blocks at the
parameter layer.

## Builder Strategy As General Pattern

The builder-strategy seam should sit behind the RHIME model-spec boundary, close
to `models/rhime.py`.

The runner should not grow a tree of special cases. Parameter parsing should not
know whether the PyMC graph is a loop, a stacked dot product, or a shared-state
projection. Data preparation should provide source-resolved or grouped
sensitivities, but the builder should own how model contributions are compiled.

A useful contribution record, conceptually, needs these roles:

- model contribution id: stable model/component name, for example `ff`,
  `ocean`, `outer`, or `inner`.
- input source group: one or more OpenGHG `source` values that back the
  contribution.
- state space: shared basis region, source-specific gathered state, inner-domain
  state, outer-domain state, or shared tracer-linked state.
- prior policy: one prior for the contribution state, or a future policy for
  source-specific subparts.
- output labels: product-neutral labels for generic diagnostics.
- links: optional typed dependencies, such as "this O2 forward map uses the CH4
  fossil-fuel state with a conversion factor".

This record is not necessarily a new public class today. It is the shape that
keeps `run_rhime` and `run_rhime_multisector` from remaining separate model
families.

### Current Special Cases

`run_rhime` can be viewed as:

```text
contribution total:
    input source group = requested flux_sources
    state = one optimized state
    builder strategy = single linear contribution
    aggregate mu = contribution mu
```

Today this is artificially limited to one source. Relaxing that limit would
recover the `ModelScenario` coarse total behavior: multiple input sources, one
state, one `mu`.

`run_rhime_multisector` can be viewed as:

```text
contribution ff:
    input source group = [ff-source]
    state = x_ff
    mu = H_ff @ x_ff

contribution ocean:
    input source group = [ocean-source]
    state = x_ocean
    mu = H_ocean @ x_ocean

aggregate total:
    mu = mu_ff + mu_ocean
```

This is the current loop-sum strategy. The same contribution list could later
be lowered as a stacked design matrix if the state dimensions are compatible or
can be represented with gathered state indices.

### Future Special Cases

Single-sector multi-source:

```text
contribution total:
    input source group = [source_a, source_b, ...]
    state = x_total
    mu = H_grouped @ x_total
```

This is not equivalent to independently optimized sectors summed after the
fact. It represents the assumption that one scale state applies to the grouped
prior flux.

Virtual total:

```text
aggregate total:
    mu_total = sum(mu_contribution)
    flux_total = sum(reconstructed_flux_contribution)
```

`total` should usually be a virtual aggregate, not an OpenGHG source and not a
sampled sector state. A derived `x_total` is only meaningful under explicit
conditions. For example, if all contributions share basis regions, an output
could compute a flux-weighted total scale factor, but that is an output
diagnostic rather than the latent variable sampled by the model.

High-resolution inner/outer:

```text
contribution outer:
    state = x_outer
    mu = H_outer @ x_outer

contribution inner:
    state = x_inner
    mu = H_inner @ x_inner

aggregate total:
    mu = mu_outer + mu_inner
```

This is structurally the same as multisector loop-sum, but the contribution
identity is domain partition rather than emission sector. Milestone 10 asks for
this distinction to survive into outputs, so flattening everything into a
single anonymous state would be a step backward even if it samples correctly.

Tracer linked-state:

```text
contribution ch4_ff_state:
    state = x_ff

forward co2:
    mu_co2_ff = H_co2_ff @ x_ff

forward o2:
    mu_o2_ff = conversion_factor * H_o2_ff @ x_ff
```

This is why the builder seam should not be only "loop versus stacked dot". The
larger abstraction is a contribution/state graph that can lower to PyMC in more
than one shape.

## Source, Sector, And Output Vocabulary

The current names are close, but the responsibilities should be sharper.

Recommended vocabulary:

- `flux_source`: OpenGHG input/source coordinate used for object-store lookup
  and data provenance.
- `source_group`: one or more `flux_source` values backing a model
  contribution.
- `model_contribution`: an optimized model component, currently usually called
  a sector.
- `model_sector`: a contribution whose scientific meaning is an emission or
  uptake sector.
- `model_variable_suffix`: private PyMC/trace suffix used for variable names.
- `output_sector_label`: product-neutral label for output diagnostics.
- `paris_sector_code`: PARIS schema-safe label used in PARIS variable names and
  the PARIS `sector` coordinate.
- `virtual_total`: aggregate over contributions, not a real source.

If the mapping is one-to-one and the names are identical, `sector_sources` adds
little semantic value. It is just a renaming layer. That is acceptable as a
transition, but the long-term model should not require users to invent a sector
mapping when all they have is a list of OpenGHG sources.

The first non-renaming case is likely:

```text
model sector "anthro":
    source_group = ["ff-inventory", "industry-inventory", "waste-inventory"]
```

That removes the one-to-one mapping and makes the sector/source split worth
having.

## PR 472 Output Flow

PR #472 is useful because it exercises the whole path from model-sector
metadata to product variables.

Good direction:

- It adds per-sector latest-PARIS flux variables alongside total variables.
- It keeps total flux outputs derived from reconstructed sector flux fields,
  not from summed scale factors.
- It adds a PARIS `sector` coordinate and sector country covariance diagnostics.

Risk:

- PARIS sector names are derived from `variable_suffix`. That suffix is
  documented as the PyMC-safe model variable suffix, not as a product-facing
  code. Changing a model variable suffix would therefore change public PARIS
  variable names.
- The PR adds a local sector metadata parser in `make_paris_outputs.py`, while
  `make_outputs.py` already has `OutputSector` metadata resolution. That is a
  duplication point.
- The current flow becomes:

```text
OpenGHG source
-> SectorSpec.name
-> variable_suffix
-> generic flux_<suffix> variables
-> PARIS flux_<paris_name> variables
```

That is understandable for #405, but it will get harder when milestone 10 adds
inner/outer outputs and milestone 11 adds species/tracer-aware outputs.

The output boundary should name the stages explicitly:

```text
model identity
-> product-neutral output label
-> product-specific variable name
```

PARIS-specific sanitisation belongs in the PARIS adapter. It should not leak
back into model variable names or model specs.

## Alignment With The Semantic IR Plan

The uploaded semantic IR plan argues for this pattern:

```text
TOML config
-> semantic intermediate representation
-> required-data / prepared-data layer
-> backend-specific compilation
-> PyMC graph / analytic model / visualization
```

#402 and #403 should not try to implement that whole architecture. They can,
however, avoid choices that would fight it.

Useful near-term alignment:

- Treat the RHIME model spec as strategy-independent.
- Treat `run_rhime` and `run_rhime_multisector` as convenience constructors for
  contribution shapes, not fundamentally separate pipelines.
- Keep source provenance separate from model contribution identity.
- Keep virtual aggregates such as `total` out of the input-source coordinate.
- Put product naming in output adapters, not in PyMC variable suffixes.
- Think of the builder strategy as a small `CompilationPlan` precursor: a
  recorded decision about how semantic contributions are lowered into a backend
  graph.

The near-term contribution plan is therefore a bridge:

```text
RhimeModelSpec
-> contribution plan
-> builder strategy
-> PyMC model
```

A later semantic model can generalize the same path:

```text
SemanticModel
-> prepared component data
-> compilation plan
-> backend model
```

## Practical Design Tests

These are not an implementation checklist. They are useful questions to keep
future edits honest.

- Can a single run have multiple input `flux_sources` but one optimized model
  contribution?
- Can a model contribution be backed by more than one OpenGHG source?
- Can `total` be present in outputs without being present as an OpenGHG source
  or sampled sector?
- Can the same contribution list be compiled as loop-sum or stacked-dot without
  changing the model spec?
- Can model variable suffixes change without changing PARIS variable names?
- Can inner/outer domain identity survive both sampling and output?
- Can one latent state contribute to both primary-species and tracer
  likelihoods without duplicating the latent?

If the answer to any of these is "no", the current design is probably encoding
too much of the current shared-basis multisector implementation as permanent
architecture.

## Summary

The current code likely contains most of the minimum needed to close the narrow
parts of #402 and #403. The larger opportunity is to use those issues to stop
treating `run_rhime` and `run_rhime_multisector` as separate model families.

The better abstraction is:

```text
input sources are provenance
model contributions are optimized state-bearing components
builder strategies lower contributions into PyMC graphs
virtual totals aggregate contributions
output adapters own product names
```

That is small enough to fit the current milestones and compatible with the
semantic IR direction.
