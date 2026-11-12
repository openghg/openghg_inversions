"""Dense Gaussian oracles for complete native functionals of a coherent reduction."""

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
# Two partitions of the same native grid, per source.
PARTITIONS = {
    "columns": {"fossil": np.array([[0, 0, 1, 1]] * 3), "biosphere": np.array([[0, 1, 1, 2]] * 3)},
    "rows": {
        "fossil": np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]),
        "biosphere": np.array([[0, 0, 0, 0], [0, 0, 1, 1], [2, 2, 2, 2]]),
    },
}
N_OBSERVATION = 7


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
        observation=[f"obs-{index}" for index in range(N_OBSERVATION)],
    ).assign_attrs(units="ppm")
    noise_root = 0.3 * rng.normal(size=(N_OBSERVATION, N_OBSERVATION))
    noise = noise_root @ noise_root.T + 0.2 * np.eye(N_OBSERVATION)
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


def _countries() -> Countries:
    """Return a three-country map on the native grid."""
    return Countries(
        xr.Dataset(
            {"country": (("lat", "lon"), COUNTRY_MAP), "name": ("ncountries", COUNTRY_NAMES)},
            coords={"lat": LATITUDE, "lon": LONGITUDE},
        )
    )


def _reduce(problem, partition: str, functionals: xr.DataArray, functional_dim: str):
    """Reduce one partition with its functional products."""
    return reduce_native_gaussian(
        covariance=problem["covariance"],
        strategy=SuppliedRestriction(_restriction(partition)),
        state_dim="state",
        native_mean=problem["mean"],
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
        observation_batch_size=3,
        functionals=functionals,
        functional_dim=functional_dim,
    )


def _generic_functionals(problem) -> xr.DataArray:
    """Return one country total per source as plain native functionals."""
    rows, labels = [], []
    for index, source in enumerate(SOURCES):
        for country, name in enumerate(COUNTRY_NAMES):
            row = np.zeros((2, 3, 4))
            row[index] = (COUNTRY_MAP == country) * problem["flux"].values[index]
            rows.append(row)
            labels.append(f"{name}/{source}")
    # A third, border-free functional checks that rows need not be country-shaped.
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


def _native_posterior(problem, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the native-grid Gaussian posterior of ``Q x`` with the boundary marginalised."""
    b, h, m = problem["B"], _flat(problem["sensitivity"], "observation"), problem["mean"].values.reshape(-1)
    s = h @ b @ h.T + problem["G"] @ problem["P_beta"] @ problem["G"].T + problem["R"]
    gain = q @ b @ h.T @ np.linalg.inv(s)
    return q @ m + gain @ (problem["y"] - h @ m), q @ b @ q.T - gain @ h @ b @ q.T


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


def test_two_partitions_give_the_native_posterior_of_every_functional() -> None:
    """With exact A, complete functionals equal the native posterior for any partition."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    native_mean, native_covariance = _native_posterior(problem, _flat(functionals, "functional"))
    scale = np.abs(native_covariance).max()

    complete = {}
    for partition in PARTITIONS:
        reduction = _reduce(problem, partition, functionals, "functional")
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
    reduction = _reduce(problem, "columns", functionals, "functional")
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


def test_per_draw_conditional_mean_is_the_native_conditional_mean() -> None:
    """Without conditional noise, each draw equals E[Q x | Pi x, beta, y] on the native grid."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    reduction = _reduce(problem, "rows", functionals, "functional")
    products = reduction.functional_products
    b, h, q = problem["B"], _flat(problem["sensitivity"], "observation"), _flat(functionals, "functional")
    pi = _flat(_restriction("rows"), "state")
    m = problem["mean"].values.reshape(-1)
    rng = np.random.default_rng(3)
    delta = rng.normal(size=(2, 3, pi.shape[0]))
    beta = rng.normal(size=(2, 3, 2))
    residual = (
        problem["y"]
        - reduction.native_observation_mean.values
        - delta @ reduction.effective_observation_operator.values.T
        - beta @ problem["G"].T
    )
    samples = {"chain": [0, 1], "draw": [0, 1, 2]}
    observation = reduction.effective_observation_operator.observation

    draws = complete_functional_draws(
        xr.DataArray(
            delta,
            dims=("chain", "draw", "state"),
            coords={**samples, "state": products.retained_response.state},
        ),
        products,
        residual=xr.DataArray(
            residual, dims=("chain", "draw", "observation"), coords={**samples, "observation": observation}
        ),
        solve=dense_solve(reduction.unresolved_observation_covariance.values + problem["R"]),
    )

    joint = np.vstack([pi, h])
    observed_covariance = joint @ b @ joint.T
    observed_covariance[pi.shape[0] :, pi.shape[0] :] += problem["R"]
    gain = q @ b @ joint.T @ np.linalg.inv(observed_covariance)
    for chain in range(2):
        for draw in range(3):
            observed = np.concatenate(
                [delta[chain, draw], problem["y"] - beta[chain, draw] @ problem["G"].T - h @ m]
            )
            np.testing.assert_allclose(draws.values[chain, draw], q @ m + gain @ observed, rtol=1e-9)
    assert draws.dims == ("chain", "draw", "functional")
    assert draws.attrs["units"] == "g yr-1"


def test_conditional_draws_reproduce_the_native_posterior_by_monte_carlo() -> None:
    """Draws from the reduced posterior plus conditional noise sample the native posterior."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    native_mean, native_covariance = _native_posterior(problem, _flat(functionals, "functional"))
    reduction = _reduce(problem, "columns", functionals, "functional")
    products = reduction.functional_products
    mean_z, covariance_z, jacobian, _, innovation = _reduced_posterior(problem, reduction)
    rng = np.random.default_rng(11)
    count = 40_000
    z = rng.multivariate_normal(mean_z, covariance_z, size=count)[None]
    n_state = products.retained_response.sizes["state"]
    samples = {"chain": [0], "draw": np.arange(count)}

    draws = complete_functional_draws(
        xr.DataArray(
            z[..., :n_state],
            dims=("chain", "draw", "state"),
            coords={**samples, "state": products.retained_response.state},
        ),
        products,
        residual=xr.DataArray(
            innovation - z @ jacobian.T,
            dims=("chain", "draw", "observation"),
            coords={**samples, "observation": reduction.effective_observation_operator.observation},
        ),
        solve=dense_solve(reduction.unresolved_observation_covariance.values + problem["R"]),
        noise_varies_by_draw=False,
        rng=np.random.default_rng(12),
    ).values[0]

    standard_error = np.sqrt(np.diag(native_covariance) / count)
    np.testing.assert_array_less(np.abs(draws.mean(axis=0) - native_mean), 5 * standard_error)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), native_covariance, atol=0.04 * np.abs(native_covariance).max()
    )


def test_prior_draws_add_the_unresolved_prior_covariance() -> None:
    """Prior functional draws have covariance Q B Q.T, not only D C_alpha D.T."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    q = _flat(functionals, "functional")
    reduction = _reduce(problem, "rows", functionals, "functional")
    products = reduction.functional_products
    rng = np.random.default_rng(5)
    count = 40_000
    delta = rng.multivariate_normal(
        np.zeros(reduction.retained_covariance.shape[0]), reduction.retained_covariance.values, size=count
    )
    draws = complete_functional_draws(
        xr.DataArray(
            delta[None],
            dims=("chain", "draw", "state"),
            coords={"chain": [0], "draw": np.arange(count), "state": products.retained_response.state},
        ),
        products,
        rng=np.random.default_rng(6),
    ).values[0]

    prior = q @ problem["B"] @ q.T
    np.testing.assert_allclose(np.cov(draws, rowvar=False), prior, atol=0.04 * np.abs(prior).max())
    retained = (
        products.retained_response.values
        @ reduction.retained_covariance.values
        @ products.retained_response.values.T
    )
    assert np.abs(retained - prior).max() > 0.1 * np.abs(prior).max()


def test_marginalised_states_move_into_the_unresolved_products() -> None:
    """Marginalising states equals reducing with the kept rows of Pi, with zero D columns."""
    problem = _problem()
    functionals = _generic_functionals(problem)
    full = _reduce(problem, "rows", functionals, "functional")
    restriction = _restriction("rows")
    marginalised = restriction.state.isin(["fossil-1", "biosphere-2"])
    kept = reduce_native_gaussian(
        covariance=problem["covariance"],
        strategy=SuppliedRestriction(restriction.sel(state=~marginalised)),
        state_dim="state",
        native_mean=problem["mean"],
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
        functionals=functionals,
    ).functional_products

    result = marginalise_functional_states(
        full.functional_products,
        retained_covariance=full.retained_covariance,
        effective_observation_operator=full.effective_observation_operator,
        marginalised=marginalised,
    )

    np.testing.assert_allclose(
        result.retained_response.sel(state=~marginalised), kept.retained_response, atol=1e-12
    )
    np.testing.assert_array_equal(result.retained_response.sel(state=marginalised), 0.0)
    np.testing.assert_allclose(
        result.unresolved_observation_covariance, kept.unresolved_observation_covariance, atol=1e-11
    )
    np.testing.assert_allclose(result.unresolved_covariance, kept.unresolved_covariance, atol=1e-11)
    unchanged = marginalise_functional_states(
        full.functional_products,
        retained_covariance=full.retained_covariance,
        effective_observation_operator=full.effective_observation_operator,
        marginalised=xr.zeros_like(marginalised),
    )
    assert unchanged is full.functional_products


def test_functional_products_reject_mismatched_dimensions() -> None:
    """Functionals must use exactly the covariance's native dimensions."""
    problem = _problem()
    functionals = _generic_functionals(problem).isel({SOURCE_DIM: 0})
    with pytest.raises(ValueError, match="Functionals must have dimensions"):
        _reduce(problem, "rows", functionals, "functional")


def _co2_setup(partition: str):
    """Return a bound CO2 map, its country functionals and products for one partition."""
    problem = _problem()
    covariance_products = project_native_covariance(
        covariance=problem["covariance"],
        strategy=SuppliedRestriction(_restriction(partition)),
        state_dim="state",
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
    )
    reduction = _reduce(problem, partition, _generic_functionals(problem), "functional")
    mapping = AffineFluxMap(
        problem["mean"], problem["flux"], covariance_products.prolongation, state_dim="state"
    )
    bound = BoundCo2AffineFluxMap(
        AffineFluxMapArtifact(mapping, "fixture", {"projection_strategy": "supplied"}, {}),
        reduction.retained_mean,
    )
    countries = _countries()
    functionals = co2_country_functionals(bound, countries)
    products = project_native_functionals(
        covariance=problem["covariance"],
        products=covariance_products,
        functionals=functionals,
        functional_dim="country_functional",
        native_mean=problem["mean"],
        native_sensitivity=problem["sensitivity"],
        observation_dim="observation",
    )
    return problem, reduction, bound, countries, functionals, products


def _co2_trace(problem, reduction, count: int) -> tuple[xr.DataTree, xr.DataArray]:
    """Return a CO2 trace with exact reduced posterior draws and their residuals (no boundary)."""
    h_alpha = reduction.effective_observation_operator.values
    c = reduction.retained_covariance.values
    weight = np.linalg.inv(reduction.unresolved_observation_covariance.values + problem["R"])
    innovation = problem["y"] - reduction.native_observation_mean.values
    covariance = np.linalg.inv(np.linalg.inv(c) + h_alpha.T @ weight @ h_alpha)
    mean = covariance @ h_alpha.T @ weight @ innovation
    rng = np.random.default_rng(21)
    states = reduction.retained_mean.values
    posterior = rng.multivariate_normal(mean, covariance, size=count)[None]
    prior = rng.multivariate_normal(np.zeros_like(mean), c, size=count)[None]
    samples = {"chain": [0], "draw": np.arange(count)}
    groups = {}
    for group, delta in {"posterior": posterior, "prior": prior}.items():
        groups[group] = xr.Dataset(
            {
                "flux_scaling": xr.DataArray(
                    states + delta,
                    dims=("chain", "draw", "state"),
                    coords={**samples, "state": reduction.retained_mean.state},
                    attrs={"units": "1"},
                )
            }
        )
    groups["/"] = xr.Dataset(
        attrs={
            "rhime_model_metadata": json.dumps({"recipe": "co2"}),
            "rhime_variable_roles": json.dumps({"flux_scale": "flux_scaling"}),
        }
    )
    residual = xr.DataArray(
        innovation - posterior @ h_alpha.T,
        dims=("chain", "draw", "observation"),
        coords={**samples, "observation": reduction.effective_observation_operator.observation},
    )
    return xr.DataTree.from_dict(groups), residual


def test_co2_country_outputs_report_complete_native_country_totals() -> None:
    """Complete multisource country totals sample the native posterior; retained-only does not."""
    problem, reduction, bound, countries, functionals, products = _co2_setup("columns")
    count = 20_000
    trace, residual = _co2_trace(problem, reduction, count)
    q = _flat(functionals, "country_functional")
    native_mean, native_covariance = _native_posterior(
        {**problem, "G": np.zeros((N_OBSERVATION, 0)), "P_beta": np.zeros((0, 0))}, q
    )
    # Totals over sources: rows are (country, source) in stacked order.
    labels = functionals.get_index("country_functional")
    total = np.array([[label[0] == country for label in labels] for country in COUNTRY_NAMES], dtype=float)
    unresolved = UnresolvedFunctionalConditioning(
        products=products,
        residual=residual,
        solve=dense_solve(reduction.unresolved_observation_covariance.values + problem["R"]),
        description="exact dense A + fixed dense R",
        noise_varies_by_draw=False,
        seed=4,
    )

    result = co2_country_flux_outputs(trace, bound, countries, unresolved=unresolved)
    retained_only = co2_country_flux_outputs(trace, bound, countries)

    assert result.attrs["uncertainty_scope"] == "complete_native"
    assert result.attrs["unresolved_conditioning"] == "exact dense A + fixed dense R"
    assert retained_only.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert "unresolved_conditioning" not in retained_only.attrs
    for variable in result.data_vars.values():
        assert variable.attrs["uncertainty_scope"] == "complete_native"
        assert variable.attrs["units"] == "g yr-1"
    assert result.country.values.tolist() == COUNTRY_NAMES

    total_mean = total @ native_mean
    total_sd = np.sqrt(np.diag(total @ native_covariance @ total.T))
    np.testing.assert_array_less(
        np.abs(result.country_total_posterior_mean.sel(country=COUNTRY_NAMES).values - total_mean),
        5 * total_sd / np.sqrt(count),
    )
    np.testing.assert_allclose(
        result.country_total_posterior_stdev.sel(country=COUNTRY_NAMES), total_sd, rtol=0.03
    )
    per_source = result.country_posterior_mean.transpose("country", SOURCE_DIM).values
    expected_per_source = np.array(
        [[native_mean[labels.get_loc((c, s))] for s in SOURCES] for c in COUNTRY_NAMES]
    )
    source_sd = np.sqrt(np.diag(native_covariance))
    expected_sd = np.array([[source_sd[labels.get_loc((c, s))] for s in SOURCES] for c in COUNTRY_NAMES])
    np.testing.assert_array_less(np.abs(per_source - expected_per_source), 5 * expected_sd / np.sqrt(count))

    prior_sd = np.sqrt(np.diag(total @ q @ problem["B"] @ q.T @ total.T))
    np.testing.assert_allclose(
        result.country_total_prior_stdev.sel(country=COUNTRY_NAMES), prior_sd, rtol=0.03
    )
    # The retained-only output is materially different when unresolved variance is non-zero.
    retained_sd = retained_only.country_total_posterior_stdev.sel(country=COUNTRY_NAMES).values
    assert np.abs(retained_sd - total_sd).max() > 0.1 * total_sd.max()


def test_co2_complete_outputs_agree_across_partitions() -> None:
    """Two partitions give the same complete country totals up to Monte Carlo error."""
    summaries = []
    for partition in PARTITIONS:
        problem, reduction, bound, countries, _, products = _co2_setup(partition)
        trace, residual = _co2_trace(problem, reduction, 20_000)
        unresolved = UnresolvedFunctionalConditioning(
            products=products,
            residual=residual,
            solve=dense_solve(reduction.unresolved_observation_covariance.values + problem["R"]),
            description="exact",
            noise_varies_by_draw=False,
        )
        summaries.append(co2_country_flux_outputs(trace, bound, countries, unresolved=unresolved))
    first, second = summaries
    sd = first.country_total_posterior_stdev
    assert (abs(first.country_total_posterior_mean - second.country_total_posterior_mean) < 0.1 * sd).all()
    xr.testing.assert_allclose(sd, second.country_total_posterior_stdev, rtol=0.05)


def test_co2_country_outputs_reject_products_for_another_functional() -> None:
    """Products prepared for a different functional or mean fail clearly."""
    problem, reduction, bound, countries, _, products = _co2_setup("rows")
    trace, residual = _co2_trace(problem, reduction, 4)
    wrong = NativeFunctionalProducts(
        functional_mean=products.functional_mean * 2.0,
        retained_response=products.retained_response,
        unresolved_observation_covariance=products.unresolved_observation_covariance,
        unresolved_covariance=products.unresolved_covariance,
    )
    unresolved = UnresolvedFunctionalConditioning(
        products=wrong,
        residual=residual,
        solve=dense_solve(np.eye(N_OBSERVATION)),
        description="wrong",
    )
    with pytest.raises(ValueError, match="different country functional or native mean"):
        co2_country_flux_outputs(trace, bound, countries, unresolved=unresolved)


def test_fixed_ou_solve_uses_each_draws_site_amplitudes() -> None:
    """The fixed-OU solve inverts LRPD A plus each draw's OU noise, in the covariance's site order."""
    rng = np.random.default_rng(8)
    times = np.array(["2021-01-01T00", "2021-01-01T03", "2021-01-01T00", "2021-01-01T07", "2021-01-01T09"])
    times = times.astype("datetime64[ns]")
    site_index = np.array([0, 0, 1, 1, 0])
    factor = rng.normal(size=(5, 2))
    diagonal = rng.uniform(0.1, 0.3, size=5)
    covariance = prepare_fixed_ou_low_rank(
        factor, diagonal, times, site_index, {"MHD": 5.0, "TAC": 2.0}, site_labels=["MHD", "TAC"]
    )
    # Amplitudes are stored in a different site order from the covariance.
    amplitude = xr.DataArray(
        rng.uniform(0.5, 1.5, size=(2, 3, 2)),
        dims=("chain", "draw", "site"),
        coords={"chain": [0, 1], "draw": [0, 1, 2], "site": ["TAC", "MHD"]},
    )
    solve = fixed_ou_solve(covariance, amplitude, site_dim="site")
    rhs = rng.normal(size=(5, 3))
    hours = (times - times[0]) / np.timedelta64(1, "h")
    for chain in range(2):
        for draw in range(3):
            dense = factor @ factor.T + np.diag(diagonal)
            for site, tau in (("MHD", 5.0), ("TAC", 2.0)):
                rows = np.flatnonzero(site_index == ["MHD", "TAC"].index(site))
                lag = np.abs(hours[rows, None] - hours[None, rows])
                dense[np.ix_(rows, rows)] += float(
                    amplitude.sel(site=site).values[chain, draw]
                ) ** 2 * np.exp(-lag / tau)
            np.testing.assert_allclose(solve(rhs, chain, draw), np.linalg.solve(dense, rhs), rtol=1e-9)


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
    weights = to_dense(countries.matrix * countries.area_grid).transpose("country", "lat", "lon").values
    np.testing.assert_allclose(
        _two_dimensional(functionals),
        (weights * flux.values).reshape(3, -1) * 44.01 * 365 * 24 * 3600,
        rtol=1e-4,
    )
    count = 3
    products = NativeFunctionalProducts(
        functional_mean=xr.DataArray(
            _two_dimensional(functionals) @ mean.values.reshape(-1),
            dims="country_functional",
            coords=functionals.country_functional.coords,
        ),
        retained_response=xr.DataArray(
            _two_dimensional(functionals) @ prolongation.values.reshape(-1, 2),
            dims=("country_functional", "state"),
            coords={**functionals.country_functional.coords, "state": ["a", "b"]},
        ),
        unresolved_observation_covariance=xr.DataArray(
            np.zeros((3, 1)),
            dims=("country_functional", "observation"),
            coords={**functionals.country_functional.coords, "observation": ["obs"]},
        ),
        unresolved_covariance=xr.DataArray(
            np.zeros((3, 3)), dims=("country_functional", "country_functional_cov")
        ),
    )
    samples = {"chain": [0], "draw": np.arange(count)}
    state = xr.DataArray(
        np.ones((1, count, 2)),
        dims=("chain", "draw", "state"),
        coords={**samples, "state": ["a", "b"]},
        attrs={"units": "1"},
    )
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
    unresolved = UnresolvedFunctionalConditioning(
        products=products,
        residual=xr.DataArray(
            np.zeros((1, count, 1)),
            dims=("chain", "draw", "observation"),
            coords={**samples, "observation": ["obs"]},
        ),
        solve=dense_solve(np.eye(1)),
        description="zero unresolved",
    )
    complete = co2_country_flux_outputs(trace, bound, countries, unresolved=unresolved)
    retained = co2_country_flux_outputs(trace, bound, countries)
    assert complete.country_total_posterior_mean.dims == ("country",)
    xr.testing.assert_allclose(
        complete.country_total_posterior_mean.sel(country=retained.country),
        retained.country_total_posterior_mean,
        rtol=1e-6,
    )


def _two_dimensional(functionals: xr.DataArray) -> np.ndarray:
    """Return a single-source functional matrix with one column per cell."""
    return np.asarray(to_dense(functionals).values).reshape(functionals.sizes["country_functional"], -1)
