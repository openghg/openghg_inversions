# Biosphere control-space prototypes

These standalone experiments compare coordinates and one alternative prior for
cancelling gross primary production (GPP) and total ecosystem respiration (TER)
templates. They do not change a production recipe or default. Run them from a
source checkout with its OpenGHG Inversions, PyMC, NumPyro and pytest dependencies.
The model code follows the [RHIME model-development conventions](../docs/development/rhime_model_development.rst).

## Models and interpretation

| `parameterization` | Prior and controls |
| --- | --- |
| `lognormal` | Existing correlated, mean-preserving lognormal state, whitened in log space. |
| `rotated-lognormal` | Exactly the same prior after a complete orthogonal rotation; no modes are discarded. |
| `net-gross-lognormal` | Exactly the same joint prior expressed through net flux, log geometric gross magnitude and remaining log scales, including its Jacobian. |
| `gaussian-net-shape` | A new Gaussian biosphere prior with the original arithmetic mean/covariance, retaining two temporal templates. Other sources, including fossil fuel (FF), retain their lognormal prior. |

For positive reference magnitudes `G0, T0`, the net control is
`N = T0 * alpha_TER - G0 * alpha_GPP`. The second control retains common
gross activity in the lognormal model and the second net-field shape in the
Gaussian model. Both original transported biosphere templates remain present.
Keeping a monthly integrated net fixed does not generally keep every modelled
concentration fixed.

The Gaussian biosphere coefficients can be negative. They are signed template
weights for a net field, not negative physical GPP or TER magnitudes. This
prototype requires the biosphere pair to be prior-independent of other sources;
it rejects cross-source covariance rather than silently dropping it. The exact
lognormal transformations preserve all supplied correlations.

## Callable components

`biosphere_controls.py` contains ordinary functions:

- `build_biosphere_model(H, *, retained_prior, fixed_prior_contribution,
  observations, observation_error, aggregation_error, parameterization,
  rotation=None, gross_flux_weights=(G0, T0), gpp_state="GPP", ter_state="TER",
  fixed_model_mismatch=None)` returns a registered PyMC model using the public
  OGI additive Gaussian likelihood. Mismatch, when supplied, is a fixed labelled
  standard deviation; aggregation error enters exactly once.
- `add_biosphere_state(...)` builds the alternative state inside
  `registered_model()`. Baseline and rotation accept any state-vector length;
  the net variants act on one explicitly named pair.
- `likelihood_rotation(H, prior, R)` returns a complete orthogonal alignment
  from the error-whitened log-state Jacobian at the arithmetic prior mean.
  `R` is the complete fixed observation covariance.
- `net_gross_coordinates(prior, scales, pair=(g, t), gross_flux_weights=(G0, T0))`
  converts physical scales to the standardized sampling coordinate.
- `linear_gaussian_posterior(H, y, R, m, B)` is an equation oracle. For the mixed
  Gaussian-biosphere/lognormal-FF model it applies conditional on a declared FF
  signal, after subtracting that signal and the affine intercept from `y`.

The state dimension and labels come from `CorrelatedLognormalPrior`; `H` uses
that same dimension plus `nmeasure`. Arrays must already be materialized and
aligned, with consistent concentration units. `fixed_prior_contribution` is the
affine intercept, not the total prior prediction. Gross weights are finite
positive flux magnitudes on the same declared support, in common flux units.
The model exposes `flux_scaling` in the original state order and
`modelled_concentration`. The net variants also expose `biosphere_net` and
`biosphere_shape_amplitude` in the gross weights' units.

Use double precision before importing PyMC or OGI. Each worker should have a
separate writable compilation cache. In a clean process, for example:

```bash
export PYTENSOR_FLAGS="floatX=float64,base_compiledir=${TMPDIR:-/tmp}/ogi-biosphere-${SLURM_JOB_ID:-local}"
```

The sampling variable is `flux_scaling_latent`, except in the net/gross arm,
where it is `net_gross_coordinates`. All variants initialize physical scales
at their arithmetic prior mean. Use `RhimeSampler.sample(model)` or ordinary
`pm.sample`. **Disable prior-predictive sampling for `net-gross-lognormal`:** its
proper induced density is encoded by a Potential over Flat coordinates, so
ordinary forward sampling would not draw its intended prior. Generate those
prior draws from the original lognormal and transform them instead.

## Frozen three-source probe

`run_biosphere_probe.py` consumes a prepared NetCDF file. It expects the ordered
`flux_state = [GPP, TER, FF]`, covariance axis `flux_state_cov`, scenarios `base`
and `ff10`, and variables `H`, `observations`, `observation_error`,
`fixed_prior_contribution`, `prior_mean`, `prior_covariance`, `reference_flux`,
`truth_scaling`, `wur_available`, `day_equal_count`, and `night_equal_count`.
GPP's `reference_flux` is negative, while TER and FF are positive. The input owns
the scientific identities, support, units, known noise and truth definitions.
The runner uses fixed independent observation error and no aggregation error;
it is not a replay of the full spatial production model.

```bash
python -m prototypes.run_biosphere_probe --input probe.nc --output results --information-only
python -m prototypes.run_biosphere_probe --input probe.nc --output results --index 0 --build-only
python -m prototypes.run_biosphere_probe --input probe.nc --output results --index 0 --draws 1000 --tune 1000
```

Indices 0–3 select the four models on the availability mask for `base`; 4–7 use
all hours. Indices 8–15 repeat those combinations for `ff10`. Each fit uses four
chains and writes posterior draws, flux functionals and a diagnostic summary.
The information-only comparison is an arithmetic-moment Gaussian reference,
not an exact lognormal posterior. Day/night support comparisons assume the
declared fixed error model; they do not establish realistic night-time
transport accuracy.

## Equation checks

```bash
python -m pytest --noconftest tests/test_biosphere_controls_prototype.py -q
```

These self-contained checks do not use repository-wide data fixtures, hence
`--noconftest`. They verify transformed densities with an independent numerical
Jacobian, predictions, complete orthogonal rotations, retained Gaussian template
rank, preservation of the FF prior, and the conditional Gaussian oracle. They
do not run sampling. Passing them establishes the equations; sampler efficiency
and flux recovery require the separately declared experiment.
