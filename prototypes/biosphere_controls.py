"""OPE-194: small, explicit biosphere control-space experiments.

The first three parameterizations preserve one correlated lognormal target.
``gaussian-net-shape`` instead replaces only the paired biosphere prior by a
Gaussian with matching arithmetic moments. Its two signed coefficients retain
both temporal templates; they are not physical GPP/TER magnitudes. Other
sources retain their lognormal prior and must be prior-independent of this pair.

These are direct model components, not production defaults. The net/gross
lognormal uses a density Potential: do not call prior-predictive sampling for
that arm. Generate its prior draws from the original lognormal instead.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from openghg_inversions.correlated_state import CorrelatedLognormalPrior
from openghg_inversions.models.additive_sigma import add_additive_sigma_likelihood
from openghg_inversions.models.components import (
    add_coherent_affine_component,
    add_correlated_lognormal_state,
    apply_linear_sensitivity,
)
from openghg_inversions.models.coords import add_coords, registered_model
from openghg_inversions.models.state_activity import prepare_linear_sensitivity
from openghg_inversions.observation_error import AggregationError


PARAMETERIZATIONS = (
    "lognormal",
    "rotated-lognormal",
    "net-gross-lognormal",
    "gaussian-net-shape",
)


def likelihood_rotation(
    flux_sensitivity: xr.DataArray,
    prior: CorrelatedLognormalPrior,
    observation_covariance: np.ndarray,
) -> np.ndarray:
    """Return a complete orthogonal rotation at the arithmetic prior mean.

    This is a fixed local alignment, without mode truncation or eigenvalue scaling.
    The supplied covariance is the complete conditional observation covariance.
    """
    sensitivity, _ = xr.align(flux_sensitivity, prior.mean, join="exact")
    design = np.asarray(sensitivity.transpose("nmeasure", prior.state_dim).values)
    jacobian = (design * prior.mean.values) @ prior.latent_cholesky.values
    whitened = np.linalg.solve(np.linalg.cholesky(observation_covariance), jacobian)
    return np.linalg.svd(whitened, full_matrices=whitened.shape[0] < whitened.shape[1])[2].T


def net_shape_matrix(gpp_weight: float, ter_weight: float) -> np.ndarray:
    """Map paired template coefficients to net and common-shape amplitudes."""
    weights = np.asarray([gpp_weight, ter_weight], dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Gross-flux weights must be finite and strictly positive.")
    return np.asarray([[-gpp_weight, ter_weight], [gpp_weight, ter_weight]])


def net_gross_coordinates(
    prior: CorrelatedLognormalPrior,
    scales: np.ndarray,
    *,
    pair: tuple[int, int],
    gross_flux_weights: tuple[float, float],
) -> np.ndarray:
    """Map physical scales to the standardized net/gross sampling coordinate.

    This also supplies reproducible initial points and density-equivalence checks.
    The pair contains positions in the already-labelled prior state.
    """
    g, t = pair
    g0, t0 = gross_flux_weights
    transform = net_shape_matrix(g0, t0)
    values = np.asarray(scales, dtype=float)
    if values.shape != prior.mean.shape or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Scales must be a finite positive vector in the prior state order.")
    log_mean = prior.latent_mean.values
    log_covariance = prior.latent_covariance.values
    log_sd = np.sqrt(np.diag(log_covariance))
    pair_covariance = prior.arithmetic_covariance.values[np.ix_(pair, pair)]
    net_sd = np.sqrt((transform @ pair_covariance @ transform.T)[0, 0])
    q_sd = np.sqrt(log_covariance[np.ix_(pair, pair)].sum() / 4.0)
    coordinate = (np.log(values) - log_mean) / log_sd
    coordinate[g] = (
        t0 * (values[t] - prior.mean.values[t]) - g0 * (values[g] - prior.mean.values[g])
    ) / net_sd
    coordinate[t] = (0.5 * np.log(values[list(pair)]).sum() - 0.5 * log_mean[list(pair)].sum()) / q_sd
    return coordinate


def add_biosphere_state(
    prior: CorrelatedLognormalPrior,
    *,
    parameterization: str = "lognormal",
    rotation: np.ndarray | None = None,
    gross_flux_weights: tuple[float, float] = (1.0, 1.0),
    gpp_state: str = "GPP",
    ter_state: str = "TER",
):
    """Construct one labelled state inside an OGI ``registered_model``.

    The public output ``flux_scaling`` preserves the original order. In the Gaussian
    arm its biosphere entries are signed template weights, while other entries
    remain positive source multipliers. Baseline and rotation support any state
    size. The two net arms require one explicitly identified GPP/TER pair.
    All arms initialize at the arithmetic prior mean.
    """
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(f"Unknown parameterization: {parameterization!r}.")
    if rotation is not None and parameterization != "rotated-lognormal":
        raise ValueError("A rotation is used only by rotated-lognormal.")
    mean = prior.mean.values
    log_mean = prior.latent_mean.values
    log_cholesky = prior.latent_cholesky.values
    nstate = mean.size
    prior_mean_latent = np.linalg.solve(log_cholesky, np.log(mean) - log_mean)
    if parameterization == "lognormal":
        result = add_correlated_lognormal_state(prior, var_name="flux_scaling")
        pm.modelcontext(None).set_initval(result.latent, pm.floatX(prior_mean_latent))
        return result.state

    if parameterization == "rotated-lognormal":
        if rotation is None:
            raise ValueError("rotated-lognormal requires a complete orthogonal rotation.")
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (nstate, nstate) or not np.isfinite(rotation).all():
            raise ValueError("Rotation must be a finite square matrix spanning every state.")
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(nstate), rtol=1e-10, atol=1e-10)
        add_coords(prior.mean.coords, model_dims=(prior.state_dim,))
        latent = pm.Normal(
            "flux_scaling_latent",
            0.0,
            1.0,
            dims=prior.state_dim,
            initval=pm.floatX(rotation.T @ prior_mean_latent),
        )
        return pm.Deterministic(
            "flux_scaling",
            pt.exp(log_mean + (log_cholesky @ rotation) @ latent),
            dims=prior.state_dim,
        )

    labels = prior.mean.coords[prior.state_dim].to_index()
    pair = (int(labels.get_loc(gpp_state)), int(labels.get_loc(ter_state)))
    g, t = pair
    if g == t:
        raise ValueError("GPP and TER must identify different states.")
    other = np.asarray([index for index in range(nstate) if index not in pair], dtype=int)
    g0, t0 = gross_flux_weights
    transform = net_shape_matrix(g0, t0)
    covariance = prior.arithmetic_covariance.values
    log_covariance = prior.latent_covariance.values
    pair_covariance = covariance[np.ix_(pair, pair)]
    control_mean = transform @ mean[list(pair)]
    control_covariance = transform @ pair_covariance @ transform.T
    if parameterization == "gaussian-net-shape" and np.any(covariance[np.ix_(pair, other)] != 0):
        raise ValueError("Gaussian biosphere prototype requires prior independence from other sources.")
    add_coords(prior.mean.coords, model_dims=(prior.state_dim,))

    if parameterization == "net-gross-lognormal":
        net_sd = np.sqrt(control_covariance[0, 0])
        q_sd = np.sqrt(log_covariance[np.ix_(pair, pair)].sum() / 4.0)
        log_sd = np.sqrt(np.diag(log_covariance))
        coordinate = pm.Flat(
            "net_gross_coordinates",
            dims=prior.state_dim,
            initval=pm.floatX(
                net_gross_coordinates(prior, mean, pair=pair, gross_flux_weights=gross_flux_weights)
            ),
        )
        net = control_mean[0] + net_sd * coordinate[g]
        q = 0.5 * log_mean[list(pair)].sum() + q_sd * coordinate[t]
        geometric_gross = np.sqrt(g0 * t0) * pt.exp(q)
        contrast = pt.arcsinh(net / (2.0 * geometric_gross))
        log_scales = pt.as_tensor_variable(log_mean) + log_sd * coordinate
        log_scales = pt.set_subtensor(log_scales[g], q + 0.5 * np.log(t0 / g0) - contrast)
        log_scales = pt.set_subtensor(log_scales[t], q + 0.5 * np.log(g0 / t0) + contrast)
        total_gross = pt.sqrt(net**2 + 4.0 * geometric_gross**2)
        log_jacobian = np.log(2.0 * net_sd * q_sd) - pt.log(total_gross) + np.log(log_sd[other]).sum()
        pm.Potential(
            "induced_lognormal_prior",
            pm.logp(pm.MvNormal.dist(mu=log_mean, chol=log_cholesky), log_scales) + log_jacobian,
        )
        state = pt.exp(log_scales)
        shape_amplitude = total_gross
    else:
        latent = pm.Normal(
            "flux_scaling_latent",
            0.0,
            1.0,
            dims=prior.state_dim,
            initval=pm.floatX(np.zeros(nstate)),
        )
        control = control_mean + np.linalg.cholesky(control_covariance) @ latent[list(pair)]
        coefficients = np.linalg.inv(transform) @ control
        state = pt.as_tensor_variable(np.zeros(nstate))
        state = pt.set_subtensor(state[list(pair)], coefficients)
        if other.size:
            other_cholesky = np.linalg.cholesky(log_covariance[np.ix_(other, other)])
            state = pt.set_subtensor(state[other], pt.exp(log_mean[other] + other_cholesky @ latent[other]))
            initial = np.zeros(nstate)
            initial[other] = np.linalg.solve(other_cholesky, np.log(mean[other]) - log_mean[other])
            pm.modelcontext(None).set_initval(latent, pm.floatX(initial))
        net, shape_amplitude = control[0], control[1]
    pm.Deterministic("biosphere_net", net)
    pm.Deterministic("biosphere_shape_amplitude", shape_amplitude)
    return pm.Deterministic("flux_scaling", state, dims=prior.state_dim)


def build_biosphere_model(
    flux_sensitivity: xr.DataArray,
    *,
    retained_prior: CorrelatedLognormalPrior,
    fixed_prior_contribution: xr.DataArray,
    observations: xr.DataArray,
    observation_error: xr.DataArray,
    aggregation_error: AggregationError,
    parameterization: str = "lognormal",
    rotation: np.ndarray | None = None,
    gross_flux_weights: tuple[float, float] = (1.0, 1.0),
    gpp_state: str = "GPP",
    ter_state: str = "TER",
    fixed_model_mismatch: xr.DataArray | None = None,
) -> pm.Model:
    """Build an explicit fixed-covariance prototype using public OGI components.

    Inputs must already be materialized, scientifically coherent, and in matching
    units. The fixed contribution is an affine intercept, not the total prior mean.
    No state fixing is performed; exact-zero columns retain their prior state.
    """
    sensitivity, _ = xr.align(flux_sensitivity, retained_prior.mean, join="exact")
    prepared_flux = prepare_linear_sensitivity(sensitivity, output_dim="nmeasure")
    with registered_model() as model:
        state = add_biosphere_state(
            retained_prior,
            parameterization=parameterization,
            rotation=rotation,
            gross_flux_weights=gross_flux_weights,
            gpp_state=gpp_state,
            ter_state=ter_state,
        )
        signal = apply_linear_sensitivity(
            prepared_flux,
            state,
            data_name="co2_sensitivity",
            output_name="scaled_flux_contribution",
            compute_deterministic=False,
        )
        mean = add_coherent_affine_component(
            fixed_prior_contribution, signal, output_name="modelled_concentration"
        )
        add_additive_sigma_likelihood(
            observations=observations,
            observation_error=observation_error,
            aggregation_error=aggregation_error,
            mean=mean,
            fixed_model_mismatch=fixed_model_mismatch,
        )
    return model


def linear_gaussian_posterior(
    design: np.ndarray,
    observations: np.ndarray,
    observation_covariance: np.ndarray,
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Gaussian equation oracle; subtract fixed/FF contributions first.

    For the mixed Gaussian-biosphere/lognormal-FF arm this is conditional on the
    declared FF signal, not the marginal posterior after integrating FF uncertainty.
    """
    cross_covariance = prior_covariance @ design.T
    innovation_covariance = observation_covariance + design @ cross_covariance
    gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T
    mean = prior_mean + gain @ (observations - design @ prior_mean)
    covariance = prior_covariance - gain @ cross_covariance.T
    return mean, (covariance + covariance.T) / 2.0
