# `adding_multi_gas_model`: historical comparison and modern port

## Decision

Do not merge or cherry-pick `origin/adding_multi_gas_model`. The branch is four
commits ahead of its 2023 base but more than 1,500 commits behind the current
`devel` branch. Its useful scientific behavior is now represented by the
isolated experimental module
`openghg_inversions.experimental.ramsden2022`.

The experimental module ports only the Ramsden et al. (2022) methane/ethane
model:

- one labelled methane scaling state per sector;
- the same fossil-fuel state contributes to methane and ethane likelihoods;
- ethane has no non-fossil contribution;
- the ethane:methane ratio can be fixed, scalar, or spatially varying;
- methane and ethane retain independent sites, times, boundary states, and
  absolute model-error terms;
- sampling uses the current `RhimeSampler`.

It deliberately does not port the branch's ACRG/OpenGHG loaders, pickle cache,
custom Metropolis-Hastings sampler, configuration parser, isotope extensions,
or bespoke NetCDF/country postprocessing.

The paper is:

> Ramsden, A. E. et al. (2022), "Quantifying fossil fuel methane emissions
> using observations of atmospheric ethane and an uncertain emission ratio",
> *Atmospheric Chemistry and Physics* 22, 3911-3929,
> <https://doi.org/10.5194/acp-22-3911-2022>.

## Model comparison

For fossil-fuel (`FF`) and non-fossil (`nonFF`) methane states, the paper model
is

```text
mu_CH4 =
    H_CH4,FF @ x_FF
  + H_CH4,nonFF @ x_nonFF
  + Hbc_CH4 @ bc_CH4

mu_C2H6 =
    H_C2H6,FF @ (R * x_FF)
  + Hbc_C2H6 @ bc_C2H6
```

The gas likelihoods are conditionally independent. Model error is absolute in
each gas's observation units:

```text
epsilon_g = max(
    sqrt(measurement_error_g**2 + sigma_g(site, period)**2),
    min_error_g,
)
```

This is not the same as modern standard RHIME's enhancement-proportional error
term, so the experimental builder has a small namespaced two-channel
likelihood rather than calling the current single-channel likelihood helper
twice. Set `min_error=0` in prepared inputs for an exact comparison with the
paper; a nonzero value intentionally retains the modern RHIME safety floor.
Sigma priors are required explicitly, as are boundary priors for channels with
boundary states; the experimental API does not silently substitute modern
RHIME defaults for the paper's settings.

### Direct ratio versus historical multiplier

The paper samples the direct molar ratio `R`. In contrast, the historical
branch constructs ethane sensitivity from an inventory that already contains
a reference ratio (0.075 in the supplied configuration) and samples a
dimensionless multiplier around that inventory:

```text
R_physical = 0.075 * ratio_multiplier
```

The new `RamsdenSectorSpec` makes this distinction explicit:

- `reference_ratio=None`: `tracer.H` must be ratio-free and the sampled/fixed
  value is the direct emission ratio;
- `reference_ratio=<positive value>`: `tracer.H` must already contain that
  ratio and the sampled/fixed value is a dimensionless multiplier. The model
  exposes both `ratio_multiplier_<sector>` and
  `emission_ratio_<sector>`.

Sampled ratios must use priors with non-negative support, and sectors must bind
unique primary and tracer sensitivity sources so that duplicate,
non-identifiable states fail during validation.

`RamsdenPreparedInputs.tracer_design_reference_ratios` independently records
which reference ratio, if any, is already present in every tracer design. The
builder requires those values to agree with the model spec. This prevents a
ratio from being silently applied twice.

## Why the historical branch cannot be run unchanged

The branch is not only dependency-old; it contains correctness defects that
make it unsuitable as a production fallback:

- imports refer to removed modules and obsolete OpenGHG/ACRG APIs;
- pseudo-data setup references an undefined emissions-uncertainty variable;
- site/time matching checks site and timestamp membership separately rather
  than matching `(site, time)` pairs;
- the real-data boundary proposal can score stale proposed state from another
  parameter update, violating the intended Metropolis transition;
- posterior modelled-observation slicing mixes burn-in and posterior draws and
  can create inconsistent output dimensions;
- ethane gridded/country outputs omit the sampled ratio multiplier;
- basis counts and sector layouts are inferred positionally without labelled
  alignment;
- there are no multi-gas tests.

The old code remains useful as a source for equations and tiny repaired
synthetic-oracle comparisons.

## Relationship to milestones 10 and 11

Milestone 10 ("high-resolution RHIME and parallel extension hooks") includes
issue #410, which calls for isolated model variants using the modern
run-spec/preparation/builder/result pattern and explicitly avoids new model
logic in `hbmcmc`.

Milestone 11 ("linked CO2/O2 tracer inversion") will provide the durable,
general contracts that this experiment intentionally does not invent:

- serialisable primary/tracer coupling specs (#411);
- channel-aware preparation and shared-state model construction (#412);
- tracer-aware postprocessing and outputs (#413).

The Ramsden module is therefore a running historical comparison and a
requirements fixture for milestone 11, not the proposed final generic tracer
API. It has a dedicated result carrier and does not alter `RhimeResult` or
`InversionOutput`.

## Prepared-input usage

Both datasets use the canonical RHIME variable names (`H`, `mf`, `mf_error`,
`min_error`, `site_indicator`, and optional `H_bc`). They can have different
observation axes but coupled sector designs must have exactly matching labelled
state coordinates. Retain and pass both `BasisFunctions` objects when using
ordinary numeric region labels; the builder then compares the actual spatial
basis maps rather than treating two `0..N-1` indexes as proof of alignment.

```python
from openghg_inversions.experimental.ramsden2022 import (
    RamsdenChannelSpec,
    RamsdenModelSpec,
    RamsdenPreparedInputs,
    RamsdenSectorSpec,
    run_ramsden_from_prepared_inputs,
)
from openghg_inversions.rhime import RhimeSampler

spec = RamsdenModelSpec(
    primary=RamsdenChannelSpec(
        species="ch4",
        observation_units="ppb",
        sigma_prior={"pdf": "uniform", "lower": 10.0, "upper": 50.0},
        sigma_frequency="monthly",
        use_bc=True,
        bc_prior={
            "pdf": "truncatednormal",
            "mu": 1.0,
            "sigma": 0.05,
            "lower": 0.0,
        },
    ),
    tracer=RamsdenChannelSpec(
        species="c2h6",
        observation_units="ppt",
        sigma_prior={"pdf": "uniform", "lower": 20.0, "upper": 50.0},
        sigma_frequency="monthly",
        use_bc=True,
        bc_prior={
            "pdf": "truncatednormal",
            "mu": 1.0,
            "sigma": 0.5,
            "lower": 0.0,
        },
    ),
    sectors=(
        RamsdenSectorSpec(
            name="fossil_fuel",
            primary_flux_source="ch4_ff",
            tracer_flux_source="c2h6_ff_r0075",
            x_prior={
                "pdf": "truncatednormal",
                "mu": 1.0,
                "sigma": 0.5,
                "lower": 0.0,
            },
            ratio_prior={"pdf": "uniform", "lower": 0.1, "upper": 2.7},
            ratio_resolution="spatial",
            reference_ratio=0.075,
        ),
        RamsdenSectorSpec(
            name="non_fossil",
            primary_flux_source="ch4_nonff",
            x_prior={
                "pdf": "truncatednormal",
                "mu": 1.0,
                "sigma": 0.5,
                "lower": 0.0,
            },
        ),
    ),
)

result = run_ramsden_from_prepared_inputs(
    prepared_inputs=RamsdenPreparedInputs(
        primary=prepared_ch4.inv_inputs,
        tracer=prepared_c2h6.inv_inputs,
        tracer_design_reference_ratios={"c2h6_ff_r0075": 0.075},
        primary_basis_functions=prepared_ch4.basis_functions,
        tracer_basis_functions=prepared_c2h6.basis_functions,
    ),
    model_spec=spec,
    sampler=RhimeSampler(
        draws=1000,
        tune=1000,
        chains=4,
        nuts_sampler="numpyro",
        sample_posterior_predictive=("y_ch4", "y_c2h6"),
    ),
)
```

The builder compares supported mole-fraction unit declarations on `mf` (and on
other arrays when present) with `observation_units`. The caller remains
responsible for expressing numeric sigma-prior values and unitless forward
designs on that same scale. In particular, the paper labels the ethane
model-error values as ppb, while the historical data/configuration strongly
indicate ppt; this should not be hidden behind an implicit conversion.
