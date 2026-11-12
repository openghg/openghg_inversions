"""Dense Gaussian oracles for complete native functionals of a coherent reduction."""

import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.affine_flux_map import AffineFluxMap
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact
from openghg_inversions.basis.covariance_products import SuppliedRestriction, project_native_covariance
from openghg_inversions.coherent_reduction import (
    NativeFunctionalProducts,
    marginalise_functional_states,
    project_native_functionals,
    reduce_native_gaussian,
)
from openghg_inversions.models.fixed_ou import prepare_fixed_ou_low_rank
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.postprocessing.co2_flux_outputs import (
    co2_country_flux_outputs,
    co2_country_functionals,
)
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.postprocessing.unresolved_functionals import (
    UnresolvedFunctionalConditioning,
    complete_functional_draws,
    dense_solve,
    fixed_ou_solve,
)
from openghg_inversions.rhime.co2.co2_affine_output import BoundCo2AffineFluxMap
from openghg_inversions.source_covariance import IndependentSourceCovariance

SOURCES = ("fossil", "biosphere")
SOURCE_DIM = "native_source"
NATIVE_DIMS = (SOURCE_DIM, "lat", "lon")
LATITUDE = np.array([50.0, 51.0, 52.0])
LONGITUDE = np.array([-2.0, -1.0, 0.0, 1.0])
SIGMAS = {"fossil": 0.6, "biosphere": 0.9}
LENGTHS = {"fossil": 0.9, "biosphere": 1.6}
# Country borders cross every retained region of both partitions.
COUNTRY_MAP = np.array([[0, 0, 1, 1], [0, 2, 1, 1], [2, 2, 2, 1]])
COUNTRY_NAMES = ["FRANCE", "GERMANY", "SPAIN"]
# Two partitions of the same native grid, per source. Both use the same
# retained-state labels, so only the products' values tell them apart.
PARTITIONS = {
    "columns": {"fossil": np.array([[0, 0, 1, 1]] * 3), "biosphere": np.array([[0, 1, 1, 2]] * 3)},
    "rows": {
        "fossil": np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
        "biosphere": np.array([[0, 0, 0, 0], [0, 0, 1, 1], [2, 2, 2, 2]]),
    },
}
N_OBSERVATION = 7
OBSERVATIONS = [f"obs-{index}" for index in range(N_OBSERVATION)]
SITES = np.array([0, 0, 0, 1, 1, 1, 1])
HOURS = np.array([0.0, 2.0, 5.0, 0.0, 1.0, 4.0, 9.0])
TAU = {"MHD": 4.0, "TAC": 2.5}
NOISE_DIAGONAL = np.array([0.2, 0.25, 0.3, 0.15, 0.2, 0.35, 0.25])
MARGINALISED = ["fossil-1", "biosphere-2"]


def _labelled(values, dims, **extra_coords) -> xr.DataArray:
    """Attach the shared native coordinates."""
    coords = {SOURCE_DIM: list(SOURCES), "lat": LATITUDE, "lon": LONGITUDE}
    return xr.DataArray(
        values, dims=dims, coords={**{d: coords[d] for d in dims if d in coords}, **extra_coords}
    )


def _covariance() -> tuple[IndependentSourceCovariance, np.ndarray]:
    """Return the source-blocked native covariance action and its dense oracle."""
    latitude = xr.DataArray(LATITUDE, dims="lat", coords={"lat": LATITUDE})
    longitude = xr.DataArray(LONGITUDE, dims="lon", coords={"lon": LONGITUDE})
    components = {}
    blocks = []
    for source in SOURCES:
        components[source] = SeparableExponentialCovariance(
            latitude, longitude, sigma=SIGMAS[source], correlation_length=LENGTHS[source]
        )
        k_lat = np.exp(-np.abs(LATITUDE[:, None] - LATITUDE[None, :]) / LENGTHS[source])
        k_lon = np.exp(-np.abs(LONGITUDE[:, None] - LONGITUDE[None, :]) / LENGTHS[source])
        blocks.append(SIGMAS[source] ** 2 * np.kron(k_lat, k_lon))
    cells = blocks[0].shape[0]
    dense = np.zeros((2 * cells, 2 * cells))
    dense[:cells, :cells] = blocks[0]
    dense[cells:, cells:] = blocks[1]
    return IndependentSourceCovariance(components, source_dim=SOURCE_DIM), dense


def _restriction(partition: str) -> xr.DataArray:
    """Return the regional-mean restriction of one partition with plain state labels."""
    rows, labels = [], []
    for index, source in enumerate(SOURCES):
        regions = PARTITIONS[partition][source]
        for region in np.unique(regions):
            row = np.zeros((len(SOURCES), *regions.shape))
            row[index] = regions == region
            rows.append(row / row.sum())
            labels.append(f"{source}-{region}")
    return _labelled(np.array(rows), ("state", *NATIVE_DIMS), state=labels).assign_attrs(units="1")


def _ou(amplitudes) -> np.ndarray:
    """Return the within-site OU covariance for site amplitudes ordered like ``TAU``."""
    covariance = np.zeros((N_OBSERVATION, N_OBSERVATION))
    for site, (amplitude, tau) in enumerate(zip(amplitudes, TAU.values(), strict=True)):
        rows = np.flatnonzero(SITES == site)
        lag = np.abs(HOURS[rows, None] - HOURS[None, rows])
        covariance[np.ix_(rows, rows)] = amplitude**2 * np.exp(-lag / tau)
    return covariance


def _problem():
    """Return one reproducible native problem with an extra Gaussian boundary term."""
    rng = np.random.default_rng(1077)
    covariance, dense_b = _covariance()
    mean = _labelled(1.0 + 0.2 * rng.normal(size=(2, 3, 4)), NATIVE_DIMS).assign_attrs(units="1")
    flux = _labelled(
        np.stack([rng.uniform(1.0, 3.0, size=(3, 4)), rng.uniform(-3.0, 1.0, size=(3, 4))]), NATIVE_DIMS
    ).assign_attrs(units="mol m-2 s-1")
    sensitivity = _labelled(
        rng.uniform(0.0, 1.0, size=(N_OBSERVATION, 2, 3, 4)),
        ("observation", *NATIVE_DIMS),
        observation=OBSERVATIONS,
    ).assign_attrs(units="ppm")
    noise = np.diag(NOISE_DIAGONAL) + _ou([0.6, 0.4])
    boundary = rng.normal(size=(N_OBSERVATION, 2))
    boundary_prior = np.diag([0.5, 0.3])
    observations = rng.normal(size=N_OBSERVATION) + sensitivity.values.reshape(N_OBSERVATION, -1) @ (
        mean.values.reshape(-1) + 0.5
    )
    return {
        "covariance": covariance,
        "B": dense_b,
        "mean": mean,
        "flux": flux,
        "sensitivity": sensitivity,
        "R": noise,
        "G": boundary,
        "P_beta": boundary_prior,
        "y": observations,
    }


def _without_boundary(problem):
    """Return the problem with no boundary term."""
    return {**problem, "G": np.zeros((N_OBSERVATION, 0)), "P_beta": np.zeros((0, 0))}


def _countries() -> Countries:
    """Return a three-country map on the native grid."""
    return Countries(
        xr.Dataset(
            {"country": (("lat", "lon"), COUNTRY_MAP), "name": ("ncountries", COUNTRY_NAMES)},
            coords={"lat": LATITUDE, "lon": LONGITUDE},
        )
    )


def _reduce(
    problem, restriction: xr.DataArray, functionals: xr.DataArray, functional_dim: str = "functional"
):
    """Reduce one restriction with its functional products."""
    return reduce_native_gaussian(
        covariance=problem["covariance"],
        strategy=SuppliedRestriction(restriction),
        state_dim="state",
        native_mean=problem["mean"],
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
        observation_batch_size=3,
        functionals=functionals,
        functional_dim=functional_dim,
    )


def _generic_functionals(problem) -> xr.DataArray:
    """Return one country total per source, plus one border-free functional."""
    rows, labels = [], []
    for index, source in enumerate(SOURCES):
        for country, name in enumerate(COUNTRY_NAMES):
            row = np.zeros((2, 3, 4))
            row[index] = (COUNTRY_MAP == country) * problem["flux"].values[index]
            rows.append(row)
            labels.append(f"{name}/{source}")
    rows.append(np.ones((2, 3, 4)))
    labels.append("domain")
    return _labelled(np.array(rows), ("functional", *NATIVE_DIMS), functional=labels).assign_attrs(
        units="g yr-1"
    )


def _flat(array: xr.DataArray, leading: str) -> np.ndarray:
    """Return a dense matrix with one native column per cell."""
    return np.asarray(to_dense(array.transpose(leading, *NATIVE_DIMS)).values).reshape(
        array.sizes[leading], -1
    )


def _observation_covariance(values: np.ndarray) -> xr.DataArray:
    """Label a dense observation covariance."""
    return xr.DataArray(
        values,
        dims=("observation", "observation_cov"),
        coords={"observation": OBSERVATIONS, "observation_cov": OBSERVATIONS},
    )


def _samples(values: np.ndarray, dim: str, labels) -> xr.DataArray:
    """Label ``(chain, draw, dim)`` sample values."""
    return xr.DataArray(
        values,
        dims=("chain", "draw", dim),
        coords={"chain": np.arange(values.shape[0]), "draw": np.arange(values.shape[1]), dim: labels},
    )


def _residual(values: np.ndarray) -> xr.DataArray:
    """Label residual draws in the sensitivity units."""
    return _samples(values, "observation", OBSERVATIONS).assign_attrs(units="ppm")


def _native_posterior(
    problem, q: np.ndarray, noise: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the native-grid Gaussian posterior of ``Q x`` with the boundary marginalised."""
    b, h, m = problem["B"], _flat(problem["sensitivity"], "observation"), problem["mean"].values.reshape(-1)
    noise = problem["R"] if noise is None else noise
    s = h @ b @ h.T + problem["G"] @ problem["P_beta"] @ problem["G"].T + noise
    gain = q @ b @ h.T @ np.linalg.inv(s)
    return q @ m + gain @ (problem["y"] - h @ m), q @ b @ q.T - gain @ h @ b @ q.T


def _native_conditional(problem, q, pi, delta, beta, noise):
    """Return the native ``E[Q x | Pi x, beta, y]`` and ``Cov[Q x | Pi x, beta, y]``."""
    b, h, m = problem["B"], _flat(problem["sensitivity"], "observation"), problem["mean"].values.reshape(-1)
    joint = np.vstack([pi, h])
    observed_covariance = joint @ b @ joint.T
    observed_covariance[pi.shape[0] :, pi.shape[0] :] += noise
    gain = q @ b @ joint.T @ np.linalg.inv(observed_covariance)
    observed = np.concatenate([delta, problem["y"] - problem["G"] @ beta - h @ m])
    return q @ m + gain @ observed, q @ b @ q.T - gain @ joint @ b @ q.T


def _reduced_posterior(problem, reduction):
    """Return the exact Gaussian posterior of ``z = (alpha - Pi m, beta)`` in the reduced model."""
    h_alpha = reduction.effective_observation_operator.values
    jacobian = np.hstack([h_alpha, problem["G"]])
    prior = np.zeros((jacobian.shape[1], jacobian.shape[1]))
    n_state = h_alpha.shape[1]
    prior[:n_state, :n_state] = reduction.retained_covariance.values
    prior[n_state:, n_state:] = problem["P_beta"]
    weight = np.linalg.inv(reduction.unresolved_observation_covariance.values + problem["R"])
    innovation = problem["y"] - reduction.native_observation_mean.values
    covariance = np.linalg.inv(np.linalg.inv(prior) + jacobian.T @ weight @ jacobian)
    return covariance @ jacobian.T @ weight @ innovation, covariance, jacobian, weight, innovation


def _complete_moments(problem, reduction) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return complete and retained-only functional moments from the reduction's products."""
    products = reduction.functional_products
    mean_z, covariance_z, jacobian, weight, innovation = _reduced_posterior(problem, reduction)
    d = products.retained_response.values
    n_state = d.shape[1]
    gain = products.unresolved_observation_covariance.values @ weight
    linear = np.hstack([d, np.zeros((d.shape[0], jacobian.shape[1] - n_state))]) - gain @ jacobian
    offset = products.functional_mean.values + gain @ innovation
    conditional = (
        products.unresolved_covariance.values - gain @ products.unresolved_observation_covariance.values.T
    )
    complete_mean = offset + linear @ mean_z
    complete_covariance = linear @ covariance_z @ linear.T + conditional
    retained_mean = products.functional_mean.values + d @ mean_z[:n_state]
    retained_covariance = d @ covariance_z[:n_state, :n_state] @ d.T
    return complete_mean, complete_covariance, retained_mean, retained_covariance


def _exact_solve(problem, reduction):
    """Return the dense exact ``A + R`` solve of one reduction."""
    return dense_solve(
        _observation_covariance(reduction.unresolved_observation_covariance.values + problem["R"]),
        description="exact dense A + dense R",
    )


def test_two_partitions_give_the_native_posterior_of_every_functional() -> None:
    """With exact A, complete functionals equal the native posterior for any partition."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    native_mean, native_covariance = _native_posterior(problem, _flat(functionals, "functional"))
    scale = np.abs(native_covariance).max()

    complete = {}
    for partition in PARTITIONS:
        reduction = _reduce(problem, _restriction(partition), functionals)
        products = reduction.functional_products
        assert products.functional_mean.get_index("functional").equals(functionals.get_index("functional"))
        assert products.unresolved_observation_covariance.dims == ("functional", "observation")
        assert products.unresolved_covariance.attrs["units"] == "(g yr-1)^2"
        assert np.linalg.eigvalsh(products.unresolved_covariance.values).min() > -1e-12 * scale
        complete_mean, complete_covariance, retained_mean, retained_covariance = _complete_moments(
            problem, reduction
        )
        np.testing.assert_allclose(
            complete_mean, native_mean, rtol=1e-10, atol=1e-10 * np.abs(native_mean).max()
        )
        np.testing.assert_allclose(complete_covariance, native_covariance, atol=1e-10 * scale)
        # The production reconstruction is linear in (alpha, beta), so evaluating it
        # at the posterior mean of (alpha, beta) gives the complete posterior mean.
        mean_z, _, jacobian, _, innovation = _reduced_posterior(problem, reduction)
        n_state = products.retained_response.sizes["state"]
        at_mean = complete_functional_draws(
            _samples(mean_z[None, None, :n_state], "state", products.retained_response.state),
            products,
            residual=_residual((innovation - jacobian @ mean_z)[None, None]),
            solve=_exact_solve(problem, reduction),
        )
        np.testing.assert_allclose(
            at_mean.values[0, 0], native_mean, rtol=1e-10, atol=1e-10 * np.abs(native_mean).max()
        )
        # Retained-only output misses both a mean update and unresolved variance.
        assert np.abs(retained_mean - native_mean).max() > 1e-2 * np.abs(native_mean).max()
        assert np.abs(retained_covariance - native_covariance).max() > 1e-2 * scale
        complete[partition] = (complete_mean, complete_covariance)

    np.testing.assert_allclose(complete["columns"][0], complete["rows"][0], rtol=1e-10)
    np.testing.assert_allclose(complete["columns"][1], complete["rows"][1], atol=1e-10 * scale)


def test_products_match_dense_definitions() -> None:
    """Each product equals its dense definition from B, H, Pi and Q."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    reduction = _reduce(problem, _restriction("columns"), functionals)
    products = reduction.functional_products
    b, h, q = problem["B"], _flat(problem["sensitivity"], "observation"), _flat(functionals, "functional")
    pi = _flat(_restriction("columns"), "state")
    c = pi @ b @ pi.T
    k = b @ pi.T @ np.linalg.inv(c)
    b_perp = b - k @ c @ k.T

    np.testing.assert_allclose(products.functional_mean.values, q @ problem["mean"].values.reshape(-1))
    np.testing.assert_allclose(products.retained_response.values, q @ k, atol=1e-12)
    np.testing.assert_allclose(
        products.unresolved_observation_covariance.values, q @ b_perp @ h.T, atol=1e-11
    )
    np.testing.assert_allclose(products.unresolved_covariance.values, q @ b_perp @ q.T, atol=1e-11)
    np.testing.assert_allclose(products.retained_covariance.values, reduction.retained_covariance.values)
    assert products.effective_observation_operator.attrs["units"] == "ppm"
    assert products.marginalised is None


def test_per_draw_conditional_mean_is_the_native_conditional_mean() -> None:
    """Without conditional noise, each draw equals E[Q x | Pi x, beta, y] on the native grid."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    reduction = _reduce(problem, _restriction("rows"), functionals)
    products = reduction.functional_products
    q = _flat(functionals, "functional")
    pi = _flat(_restriction("rows"), "state")
    rng = np.random.default_rng(3)
    delta = rng.normal(size=(2, 3, pi.shape[0]))
    beta = rng.normal(size=(2, 3, 2))
    residual = (
        problem["y"]
        - reduction.native_observation_mean.values
        - delta @ reduction.effective_observation_operator.values.T
        - beta @ problem["G"].T
    )

    draws = complete_functional_draws(
        _samples(delta, "state", products.retained_response.state),
        products,
        residual=_residual(residual),
        solve=_exact_solve(problem, reduction),
    )

    for chain in range(2):
        for draw in range(3):
            expected, _ = _native_conditional(
                problem, q, pi, delta[chain, draw], beta[chain, draw], problem["R"]
            )
            np.testing.assert_allclose(draws.values[chain, draw], expected, rtol=1e-9)
    assert draws.dims == ("chain", "draw", "functional")
    assert draws.attrs["units"] == "g yr-1"


def test_conditional_draws_reproduce_the_native_posterior_by_monte_carlo() -> None:
    """Draws from the reduced posterior plus conditional noise sample the native posterior."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    native_mean, native_covariance = _native_posterior(problem, _flat(functionals, "functional"))
    reduction = _reduce(problem, _restriction("columns"), functionals)
    products = reduction.functional_products
    mean_z, covariance_z, jacobian, _, innovation = _reduced_posterior(problem, reduction)
    count = 40_000
    z = np.random.default_rng(11).multivariate_normal(mean_z, covariance_z, size=count)[None]
    n_state = products.retained_response.sizes["state"]

    draws = complete_functional_draws(
        _samples(z[..., :n_state], "state", products.retained_response.state),
        products,
        residual=_residual(innovation - z @ jacobian.T),
        solve=_exact_solve(problem, reduction),
        rng=np.random.default_rng(12),
    ).values[0]

    standard_error = np.sqrt(np.diag(native_covariance) / count)
    np.testing.assert_array_less(np.abs(draws.mean(axis=0) - native_mean), 5 * standard_error)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), native_covariance, atol=0.04 * np.abs(native_covariance).max()
    )


def _fixed_ou(reduction, amplitude: xr.DataArray):
    """Return an LRPD fixed-OU solve whose factor reproduces the exact A of ``reduction``."""
    eigenvalues, vectors = np.linalg.eigh(reduction.unresolved_observation_covariance.values)
    keep = eigenvalues > 1e-12 * eigenvalues.max()
    factor = vectors[:, keep] * np.sqrt(eigenvalues[keep])
    covariance = prepare_fixed_ou_low_rank(
        factor,
        NOISE_DIAGONAL,
        np.datetime64("2021-01-01T00", "ns") + (HOURS * 3600e9).astype("timedelta64[ns]"),
        SITES,
        TAU,
        site_labels=list(TAU),
    )
    return fixed_ou_solve(
        covariance,
        amplitude,
        site_dim="site",
        observation_index=reduction.effective_observation_operator.indexes["observation"],
    )


def test_low_rank_fixed_ou_draws_use_each_draws_noise() -> None:
    """LRPD A with draw-specific OU noise conditions each draw on its own R, means and covariances."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    q = _flat(functionals, "functional")
    pi = _flat(_restriction("columns"), "state")
    reduction = _reduce(problem, _restriction("columns"), functionals)
    products = reduction.functional_products
    count = 20_000
    amplitudes = np.where((np.arange(count) % 2 == 0)[:, None], [[0.1, 0.1]], [[2.5, 2.0]])
    # Amplitudes are stored with sites and draws in a different order from the trace.
    amplitude = xr.DataArray(
        amplitudes[None, ::-1, ::-1],
        dims=("chain", "draw", "site"),
        coords={"chain": [0], "draw": np.arange(count)[::-1], "site": list(TAU)[::-1]},
    )
    solve = _fixed_ou(reduction, amplitude)
    delta = np.zeros((1, count, pi.shape[0]))
    beta = np.array([0.4, -0.2])
    residual = np.broadcast_to(
        problem["y"] - reduction.native_observation_mean.values - problem["G"] @ beta,
        (1, count, N_OBSERVATION),
    )
    state = _samples(delta, "state", products.retained_response.state)

    means = complete_functional_draws(state, products, residual=_residual(residual), solve=solve).values[0]
    draws = complete_functional_draws(
        state, products, residual=_residual(residual), solve=solve, rng=np.random.default_rng(2)
    ).values[0]

    moments = []
    for parity in (0, 1):
        noise = np.diag(NOISE_DIAGONAL) + _ou(amplitudes[parity])
        expected_mean, expected_covariance = _native_conditional(problem, q, pi, delta[0, 0], beta, noise)
        np.testing.assert_allclose(
            means[parity::2], np.broadcast_to(expected_mean, means[parity::2].shape), rtol=1e-8
        )
        np.testing.assert_allclose(
            np.cov(draws[parity::2], rowvar=False),
            expected_covariance,
            atol=0.05 * np.abs(expected_covariance).max(),
        )
        moments.append((expected_mean, expected_covariance))
    # The two noise levels give conditional covariances that differ by more than
    # the Monte Carlo tolerance above, so each draw must use its own noise.
    assert np.abs(moments[0][1] - moments[1][1]).max() > 0.15 * np.abs(moments[0][1]).max()


def test_prior_draws_add_the_unresolved_prior_covariance() -> None:
    """Prior functional draws have covariance Q B Q.T, not only D C_alpha D.T."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    q = _flat(functionals, "functional")
    reduction = _reduce(problem, _restriction("rows"), functionals)
    products = reduction.functional_products
    count = 40_000
    delta = np.random.default_rng(5).multivariate_normal(
        np.zeros(reduction.retained_covariance.shape[0]), reduction.retained_covariance.values, size=count
    )
    draws = complete_functional_draws(
        _samples(delta[None], "state", products.retained_response.state),
        products,
        rng=np.random.default_rng(6),
    ).values[0]

    prior = q @ problem["B"] @ q.T
    np.testing.assert_allclose(np.cov(draws, rowvar=False), prior, atol=0.04 * np.abs(prior).max())
    d = products.retained_response.values
    retained = d @ reduction.retained_covariance.values @ d.T
    assert np.abs(retained - prior).max() > 0.1 * np.abs(prior).max()


def test_marginalised_states_move_into_the_unresolved_products() -> None:
    """Marginalising states equals reducing with the kept rows of Pi, with zero D columns."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    restriction = _restriction("rows")
    full = _reduce(problem, restriction, functionals)
    marginalised = restriction.state.isin(MARGINALISED)
    kept = _reduce(problem, restriction.sel(state=~marginalised), functionals).functional_products

    result = marginalise_functional_states(full.functional_products, marginalised)

    np.testing.assert_allclose(
        result.retained_response.sel(state=~marginalised), kept.retained_response, atol=1e-12
    )
    np.testing.assert_array_equal(result.retained_response.sel(state=marginalised), 0.0)
    np.testing.assert_allclose(
        result.unresolved_observation_covariance, kept.unresolved_observation_covariance, atol=1e-11
    )
    np.testing.assert_allclose(result.unresolved_covariance, kept.unresolved_covariance, atol=1e-11)
    assert result.marginalised.values.tolist() == marginalised.values.tolist()
    assert marginalise_functional_states(full.functional_products, xr.zeros_like(marginalised)) is (
        full.functional_products
    )
    with pytest.raises(ValueError, match="already marginalised"):
        marginalise_functional_states(result, marginalised)


def test_complete_draws_validate_units_labels_and_order() -> None:
    """Residual units, solve observation order and sample labels are checked."""
    problem = _problem()
    reduction = _reduce(problem, _restriction("rows"), _generic_functionals(problem))
    products = reduction.functional_products
    state = _samples(
        np.zeros((1, 2, products.retained_response.sizes["state"])), "state", products.retained_response.state
    )
    residual = _residual(np.zeros((1, 2, N_OBSERVATION)))
    solve = _exact_solve(problem, reduction)

    with pytest.raises(ValueError, match="differ in scale"):
        complete_functional_draws(state, products, residual=residual.assign_attrs(units="ppb"), solve=solve)
    unitless = residual.copy()
    unitless.attrs = {}
    with pytest.raises(ValueError, match="must carry units"):
        complete_functional_draws(state, products, residual=unitless, solve=solve)
    reversed_solve = dense_solve(
        _observation_covariance(np.eye(N_OBSERVATION)).isel(
            observation=slice(None, None, -1), observation_cov=slice(None, None, -1)
        )
    )
    with pytest.raises(ValueError, match="different observation labels or order"):
        complete_functional_draws(state, products, residual=residual, solve=reversed_solve)
    with pytest.raises(ValueError, match="'draw' labels"):
        complete_functional_draws(state, products, residual=residual.assign_coords(draw=[5, 6]), solve=solve)


def test_functional_products_reject_mismatched_dimensions() -> None:
    """Functionals must use exactly the covariance's native dimensions."""
    problem = _problem()
    functionals = _generic_functionals(problem).isel({SOURCE_DIM: 0})
    with pytest.raises(ValueError, match="Functionals must have dimensions"):
        _reduce(problem, _restriction("rows"), functionals)


def _co2_setup(partition: str, countries: Countries | None = None, functional_partition: str | None = None):
    """Return a bound CO2 map for ``partition`` and country products for ``functional_partition``."""
    problem = _problem()
    countries = _countries() if countries is None else countries

    def products_for(name: str):
        """Return covariance products of one partition."""
        return project_native_covariance(
            covariance=problem["covariance"],
            strategy=SuppliedRestriction(_restriction(name)),
            state_dim="state",
            native_sensitivity=problem["sensitivity"],
            observation_dim="observation",
        )

    covariance_products = products_for(partition)
    reduction = _reduce(problem, _restriction(partition), _generic_functionals(problem))
    mapping = AffineFluxMap(
        problem["mean"], problem["flux"], covariance_products.prolongation, state_dim="state"
    )
    bound = BoundCo2AffineFluxMap(
        AffineFluxMapArtifact(mapping, "fixture", {"projection_strategy": "supplied"}, {}),
        reduction.retained_mean,
    )
    functionals = co2_country_functionals(bound, countries, source_dim=SOURCE_DIM)
    products = project_native_functionals(
        covariance=problem["covariance"],
        products=products_for(functional_partition or partition),
        functionals=functionals,
        functional_dim="country_functional",
        native_mean=problem["mean"],
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
    )
    return problem, reduction, bound, countries, functionals, products


def _co2_trace(
    problem, reduction, count: int, *, fixed: xr.DataArray | None = None, record_activity: bool = True
) -> tuple[xr.DataTree, xr.DataArray]:
    """Return a CO2 trace with exact reduced posterior draws and their residuals (no boundary).

    ``reduction`` is the reduction the fit used. With ``fixed``, the trace
    holds those states at their prior mean and, with ``record_activity``,
    records them as inactive in constant data.
    """
    state_labels = reduction.retained_mean.state
    active = (
        np.ones(state_labels.size, dtype=bool) if fixed is None else ~np.asarray(fixed.values, dtype=bool)
    )
    h_alpha = reduction.effective_observation_operator.values[:, active]
    c = reduction.retained_covariance.values[np.ix_(active, active)]
    weight = np.linalg.inv(reduction.unresolved_observation_covariance.values + problem["R"])
    innovation = problem["y"] - reduction.native_observation_mean.values
    covariance = np.linalg.inv(np.linalg.inv(c) + h_alpha.T @ weight @ h_alpha)
    mean = covariance @ h_alpha.T @ weight @ innovation
    rng = np.random.default_rng(21)
    posterior = np.zeros((1, count, state_labels.size))
    posterior[..., active] = rng.multivariate_normal(mean, covariance, size=count)
    prior = np.zeros((1, count, state_labels.size))
    prior[..., active] = rng.multivariate_normal(np.zeros_like(mean), c, size=count)
    groups = {}
    for group, delta in {"posterior": posterior, "prior": prior}.items():
        draws = _samples(reduction.retained_mean.values + delta, "state", state_labels).assign_attrs(
            units="1"
        )
        groups[group] = xr.Dataset({"flux_scaling": draws})
    if fixed is not None and record_activity:
        groups["constant_data"] = xr.Dataset(
            {"flux_scaling_is_active": xr.DataArray(active, dims="state", coords={"state": state_labels})}
        )
    groups["/"] = xr.Dataset(
        attrs={
            "rhime_model_metadata": json.dumps({"recipe": "co2"}),
            "rhime_variable_roles": json.dumps({"flux_scale": "flux_scaling"}),
        }
    )
    return xr.DataTree.from_dict(groups), _residual(innovation - posterior[..., active] @ h_alpha.T)


def _country_totals(functionals: xr.DataArray) -> np.ndarray:
    """Return the matrix summing (country, source) rows into country totals."""
    labels = functionals.get_index("country_functional")
    return np.array([[label[0] == country for label in labels] for country in COUNTRY_NAMES], dtype=float)


def _assert_matches_native(result, problem, functionals, count: int) -> None:
    """Complete country totals sample the native posterior within Monte Carlo error."""
    q = _flat(functionals, "country_functional")
    native_mean, native_covariance = _native_posterior(_without_boundary(problem), q)
    total = _country_totals(functionals)
    total_mean = total @ native_mean
    total_sd = np.sqrt(np.diag(total @ native_covariance @ total.T))
    np.testing.assert_array_less(
        np.abs(result.country_total_posterior_mean.sel(country=COUNTRY_NAMES).values - total_mean),
        5 * total_sd / np.sqrt(count),
    )
    np.testing.assert_allclose(
        result.country_total_posterior_stdev.sel(country=COUNTRY_NAMES), total_sd, rtol=0.03
    )


def test_co2_country_outputs_report_complete_native_country_totals() -> None:
    """Complete multisource country totals sample the native posterior; retained-only does not."""
    problem, reduction, bound, countries, functionals, products = _co2_setup("columns")
    count = 20_000
    trace, residual = _co2_trace(problem, reduction, count)
    q = _flat(functionals, "country_functional")
    native_mean, native_covariance = _native_posterior(_without_boundary(problem), q)
    labels = functionals.get_index("country_functional")
    unresolved = UnresolvedFunctionalConditioning(
        products=products, residual=residual, solve=_exact_solve(problem, reduction), seed=4
    )

    result = co2_country_flux_outputs(trace, bound, countries, unresolved=unresolved)
    retained_only = co2_country_flux_outputs(trace, bound, countries)

    assert result.attrs["uncertainty_scope"] == "complete_native"
    assert result.attrs["unresolved_conditioning"] == "exact dense A + dense R"
    assert result.attrs["marginalised_state_count"] == "0"
    assert retained_only.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert "unresolved_conditioning" not in retained_only.attrs
    for variable in result.data_vars.values():
        assert variable.attrs["uncertainty_scope"] == "complete_native"
        assert variable.attrs["units"] == "g yr-1"
    assert result.country.values.tolist() == COUNTRY_NAMES

    _assert_matches_native(result, problem, functionals, count)
    per_source = result.country_posterior_mean.transpose("country", SOURCE_DIM).values
    expected = np.array([[native_mean[labels.get_loc((c, s))] for s in SOURCES] for c in COUNTRY_NAMES])
    source_sd = np.sqrt(np.diag(native_covariance))
    expected_sd = np.array([[source_sd[labels.get_loc((c, s))] for s in SOURCES] for c in COUNTRY_NAMES])
    np.testing.assert_array_less(np.abs(per_source - expected), 5 * expected_sd / np.sqrt(count))

    total = _country_totals(functionals)
    prior_sd = np.sqrt(np.diag(total @ q @ problem["B"] @ q.T @ total.T))
    np.testing.assert_allclose(
        result.country_total_prior_stdev.sel(country=COUNTRY_NAMES), prior_sd, rtol=0.03
    )
    # The retained-only output is materially different when unresolved variance is non-zero.
    total_sd = np.sqrt(np.diag(total @ native_covariance @ total.T))
    retained_sd = retained_only.country_total_posterior_stdev.sel(country=COUNTRY_NAMES).values
    assert np.abs(retained_sd - total_sd).max() > 0.1 * total_sd.max()


def test_co2_complete_outputs_agree_across_partitions() -> None:
    """Two partitions give the same complete country totals up to Monte Carlo error."""
    summaries = []
    for partition in PARTITIONS:
        problem, reduction, bound, countries, _, products = _co2_setup(partition)
        trace, residual = _co2_trace(problem, reduction, 20_000)
        unresolved = UnresolvedFunctionalConditioning(
            products=products, residual=residual, solve=_exact_solve(problem, reduction)
        )
        summaries.append(co2_country_flux_outputs(trace, bound, countries, unresolved=unresolved))
    first, second = summaries
    sd = first.country_total_posterior_stdev
    assert (abs(first.country_total_posterior_mean - second.country_total_posterior_mean) < 0.1 * sd).all()
    xr.testing.assert_allclose(sd, second.country_total_posterior_stdev, rtol=0.05)


def test_co2_country_outputs_reject_products_for_another_functional_or_partition() -> None:
    """Products prepared for a different functional, mean or partition fail clearly."""
    problem, reduction, bound, countries, _, products = _co2_setup("rows")
    trace, residual = _co2_trace(problem, reduction, 4)

    def conditioning(candidate):
        """Wrap candidate products with this fit's residual and solve."""
        return UnresolvedFunctionalConditioning(
            products=candidate, residual=residual, solve=_exact_solve(problem, reduction)
        )

    wrong_mean = replace(products, functional_mean=products.functional_mean * 2.0)
    with pytest.raises(ValueError, match="different country functional or native mean"):
        co2_country_flux_outputs(trace, bound, countries, unresolved=conditioning(wrong_mean))
    # Same state labels, same Q m, different partition.
    *_, other_partition = _co2_setup("rows", functional_partition="columns")
    with pytest.raises(ValueError, match="different retained partition or covariance"):
        co2_country_flux_outputs(trace, bound, countries, unresolved=conditioning(other_partition))
    marginalised = marginalise_functional_states(
        products, products.retained_response.state.isin(MARGINALISED)
    )
    with pytest.raises(ValueError, match="full-restriction functional products"):
        co2_country_flux_outputs(trace, bound, countries, unresolved=conditioning(marginalised))


@pytest.mark.parametrize("record_activity", [True, False])
def test_co2_inactive_states_require_and_use_declared_marginalisation(record_activity: bool) -> None:
    """A fit that marginalised states into A gives the native posterior only when they are declared.

    Without a recorded activity mask, fixed states are found from constant posterior draws.
    """
    problem, full, bound, countries, functionals, products = _co2_setup("rows")
    marginalised = full.retained_mean.state.isin(MARGINALISED)
    # The fit used the kept-row reduction: its A absorbed the marginalised states.
    kept = _reduce(problem, _restriction("rows").sel(state=~marginalised), _generic_functionals(problem))
    fit = replace(
        full,
        effective_observation_operator=kept.effective_observation_operator.reindex(
            state=full.retained_mean.state, fill_value=0.0
        ),
        unresolved_observation_covariance=kept.unresolved_observation_covariance,
    )
    count = 20_000
    trace, residual = _co2_trace(problem, fit, count, fixed=marginalised, record_activity=record_activity)
    assert ("constant_data" in trace.children) is record_activity
    solve = _exact_solve(problem, kept)

    with pytest.raises(ValueError, match="inactive states must be exactly the states marginalised"):
        co2_country_flux_outputs(
            trace, bound, countries, unresolved=UnresolvedFunctionalConditioning(products, residual, solve)
        )
    result = co2_country_flux_outputs(
        trace,
        bound,
        countries,
        unresolved=UnresolvedFunctionalConditioning(products, residual, solve, marginalised=marginalised),
    )

    assert result.attrs["marginalised_state_count"] == "2"
    _assert_matches_native(result, problem, functionals, count)


def test_country_membership_may_vary_by_source() -> None:
    """Source-specific membership gives each source its own country rows."""
    countries = _countries()
    by_source = copy.copy(countries)
    biosphere_map = xr.DataArray(
        (np.arange(12).reshape(3, 4) % 3 == np.arange(3)[:, None, None]).astype(float),
        dims=("country", "lat", "lon"),
        coords={"country": countries.matrix.country.values, "lat": LATITUDE, "lon": LONGITUDE},
    )
    by_source.matrix = xr.concat(
        [to_dense(countries.matrix).transpose("country", "lat", "lon"), biosphere_map],
        dim=xr.DataArray(list(SOURCES), dims=SOURCE_DIM),
    )
    problem, reduction, bound, _, functionals, products = _co2_setup("rows", countries=by_source)

    rows = to_dense(functionals).sel(country_functional=("SPAIN", "biosphere"))
    weights = to_dense(by_source.matrix * by_source.area_grid).sel(
        {"country": "SPAIN", SOURCE_DIM: "biosphere"}
    )
    flux = problem["flux"].sel({SOURCE_DIM: "biosphere"})
    np.testing.assert_allclose(
        rows.sel({SOURCE_DIM: "biosphere"}).values,
        (weights * flux).values * 44.01 * 365 * 24 * 3600,
        rtol=1e-4,
    )
    np.testing.assert_array_equal(rows.sel({SOURCE_DIM: "fossil"}).values, 0.0)
    trace, residual = _co2_trace(problem, reduction, 50)
    retained = co2_country_flux_outputs(trace, bound, by_source)
    complete = co2_country_flux_outputs(
        trace,
        bound,
        by_source,
        unresolved=UnresolvedFunctionalConditioning(products, residual, _exact_solve(problem, reduction)),
    )
    assert (
        retained.country_total_posterior_mean.dims
        == complete.country_total_posterior_mean.dims
        == ("country",)
    )


def test_single_source_country_functionals_have_one_row_per_country() -> None:
    """A lat/lon map gives one functional per country and complete outputs keep the country axis."""
    problem = _problem()
    flux = problem["flux"].isel({SOURCE_DIM: 0}, drop=True)
    mean = problem["mean"].isel({SOURCE_DIM: 0}, drop=True)
    prolongation = xr.DataArray(
        np.random.default_rng(2).normal(size=(3, 4, 2)),
        dims=("lat", "lon", "state"),
        coords={"lat": LATITUDE, "lon": LONGITUDE, "state": ["a", "b"]},
        attrs={"units": "1"},
    )
    reference = xr.DataArray([1.0, 1.0], dims="state", coords={"state": ["a", "b"]}, attrs={"units": "1"})
    bound = BoundCo2AffineFluxMap(
        AffineFluxMapArtifact(AffineFluxMap(mean, flux, prolongation, state_dim="state"), "fixture", {}, {}),
        reference,
    )
    countries = _countries()
    functionals = co2_country_functionals(bound, countries)
    assert functionals.dims == ("country_functional", "lat", "lon")
    assert functionals.get_index("country_functional").names == ["country"]
    q = np.asarray(to_dense(functionals).values).reshape(3, -1)
    weights = to_dense(countries.matrix * countries.area_grid).transpose("country", "lat", "lon").values
    np.testing.assert_allclose(q, (weights * flux.values).reshape(3, -1) * 44.01 * 365 * 24 * 3600, rtol=1e-4)

    functional_coords = functionals.country_functional.coords
    products = NativeFunctionalProducts(
        functional_mean=xr.DataArray(
            q @ mean.values.reshape(-1), dims="country_functional", coords=functional_coords
        ),
        retained_response=xr.DataArray(
            q @ prolongation.values.reshape(-1, 2),
            dims=("country_functional", "state"),
            coords={**functional_coords, "state": ["a", "b"]},
        ),
        unresolved_observation_covariance=xr.DataArray(
            np.zeros((3, 1)),
            dims=("country_functional", "observation"),
            coords={**functional_coords, "observation": ["obs"]},
        ),
        unresolved_covariance=xr.DataArray(
            np.zeros((3, 3)), dims=("country_functional", "country_functional_cov")
        ),
        retained_covariance=xr.DataArray(np.eye(2), dims=("state", "state_cov")),
        effective_observation_operator=xr.DataArray(
            np.zeros((1, 2)),
            dims=("observation", "state"),
            coords={"observation": ["obs"], "state": ["a", "b"]},
        ),
    )
    state = _samples(1.0 + 0.1 * np.arange(6.0).reshape(1, 3, 2), "state", ["a", "b"]).assign_attrs(units="1")
    trace = xr.DataTree.from_dict(
        {
            "posterior": xr.Dataset({"flux_scaling": state}),
            "prior": xr.Dataset({"flux_scaling": state}),
            "/": xr.Dataset(
                attrs={
                    "rhime_model_metadata": json.dumps({"recipe": "co2"}),
                    "rhime_variable_roles": json.dumps({"flux_scale": "flux_scaling"}),
                }
            ),
        }
    )
    solve = dense_solve(
        xr.DataArray(np.eye(1), dims=("observation", "observation_cov"), coords={"observation": ["obs"]})
    )
    residual = _samples(np.zeros((1, 3, 1)), "observation", ["obs"])
    unresolved = UnresolvedFunctionalConditioning(products, residual, solve)
    complete = co2_country_flux_outputs(trace, bound, countries, unresolved=unresolved)
    retained = co2_country_flux_outputs(trace, bound, countries)
    assert complete.country_total_posterior_mean.dims == ("country",)
    xr.testing.assert_allclose(
        complete.country_total_posterior_mean.sel(country=retained.country),
        retained.country_total_posterior_mean,
        rtol=1e-6,
    )


def _float32_bound(bound: BoundCo2AffineFluxMap) -> BoundCo2AffineFluxMap:
    """Return the same map with float32 native mean, flux, prolongation and reference state."""
    affine = bound.affine_map
    mapping = AffineFluxMap(
        affine.native_mean.astype("float32"),
        affine.flux.astype("float32"),
        affine.prolongation.astype("float32"),
        state_dim=affine.state_dim,
    )
    return BoundCo2AffineFluxMap(
        replace(bound.artifact, affine_map=mapping), bound.reference_state.astype("float32")
    )


def test_co2_identity_checks_accept_float32_inputs_but_not_another_partition() -> None:
    """Float32 storage of the map and draws passes the identity checks; a different partition does not."""
    problem, reduction, bound, countries, _, products = _co2_setup("rows")
    trace, residual = _co2_trace(problem, reduction, 2_000)
    float32_trace = trace.copy()
    for group in ("prior", "posterior"):
        float32_trace[group]["flux_scaling"] = trace[group]["flux_scaling"].astype("float32")
    float32_bound = _float32_bound(bound)
    solve = _exact_solve(problem, reduction)

    result = co2_country_flux_outputs(
        float32_trace,
        float32_bound,
        countries,
        unresolved=UnresolvedFunctionalConditioning(products, residual, solve),
    )
    expected = co2_country_flux_outputs(
        trace, bound, countries, unresolved=UnresolvedFunctionalConditioning(products, residual, solve)
    )

    assert result.attrs["uncertainty_scope"] == "complete_native"
    sd = expected.country_total_posterior_stdev
    assert (
        abs(result.country_total_posterior_mean - expected.country_total_posterior_mean) < 1e-3 * sd
    ).all()
    *_, other_partition = _co2_setup("rows", functional_partition="columns")
    with pytest.raises(ValueError, match="different retained partition or covariance .*input precision"):
        co2_country_flux_outputs(
            float32_trace,
            float32_bound,
            countries,
            unresolved=UnresolvedFunctionalConditioning(other_partition, residual, solve),
        )


def test_fixed_states_cannot_be_detected_from_one_draw_without_a_mask() -> None:
    """A single posterior draw without an activity mask cannot show which states are fixed."""
    problem, reduction, bound, countries, _, products = _co2_setup("rows")
    trace, residual = _co2_trace(problem, reduction, 1)
    with pytest.raises(ValueError, match="cannot be detected from one posterior draw"):
        co2_country_flux_outputs(
            trace,
            bound,
            countries,
            unresolved=UnresolvedFunctionalConditioning(products, residual, _exact_solve(problem, reduction)),
        )
