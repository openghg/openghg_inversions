"""Staged CO2 country outputs with persisted complete-functional products.

A small multisource native Gaussian problem is reduced with country
functionals, prepared and saved, fitted by an exact reduced-posterior trace
fixture, saved as samples, and replayed through ``handoffs.postprocess``. With
the exact dense ``A`` and the fit's noise, complete country totals sample the
native posterior; retained-only totals do not.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.covariance_products import SuppliedRestriction, project_native_covariance
from openghg_inversions.coherent_reduction import reduce_native_gaussian
from openghg_inversions.inversion_data import PreparedInputs
from openghg_inversions.postprocessing.co2_flux_outputs import (
    COUNTRY_FUNCTIONAL_DIM,
    co2_flux_country_functionals,
)
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.rhime.co2 import handoffs
from openghg_inversions.rhime.co2.co2_affine_output import (
    BoundCo2AffineFluxMap,
    import_explicit_affine_flux_map,
)
from openghg_inversions.rhime.co2.co2_outputs import Co2OutputOptions
from openghg_inversions.rhime.co2.co2_preparation import (
    COUNTRY_FUNCTIONAL_DIM as PREPARED_COUNTRY_FUNCTIONAL_DIM,
    CO2_PREPARED_INPUTS_SCHEMA_VERSION,
    Co2PreparedInputs,
    prepare_co2_inputs,
)
from openghg_inversions.rhime.co2.co2_unresolved import _observation_solve, co2_unresolved_conditioning
from test_complete_functionals import (
    COUNTRY_MAP,
    COUNTRY_NAMES,
    HOURS,
    LATITUDE,
    LONGITUDE,
    N_OBSERVATION,
    NATIVE_DIMS,
    NOISE_DIAGONAL,
    SITES,
    SOURCE_DIM,
    SOURCES,
    TAU,
    _covariance,
    _labelled,
    _ou,
)

PARTITION = {"fossil": np.array([[1, 1, 2, 2]] * 3), "biosphere": np.array([[1, 2, 2, 3]] * 3)}
SITE_NAMES = list(TAU)
START = pd.Timestamp("2020-01-01")
GRID = {"lat": LATITUDE, "lon": LONGITUDE}
COUNT = 20_000


def _observation_index() -> pd.MultiIndex:
    """Label observations by site and time."""
    return pd.MultiIndex.from_arrays(
        [[SITE_NAMES[site] for site in SITES], START + pd.to_timedelta(HOURS, unit="h")], names=("site", "time")
    )


def _basis(flux: xr.DataArray) -> BasisFunctions:
    """Return the multisource bucket basis whose regions cross every country border."""
    labels = {source: xr.DataArray(PARTITION[source], dims=("lat", "lon"), coords=GRID) for source in SOURCES}
    fluxes = {source: flux.sel({SOURCE_DIM: source}, drop=True) for source in SOURCES}
    return BasisFunctions.from_multi_source_flat_basis(labels, fluxes, operator_kwargs={"state_dim": "region"})


def _restriction(basis: BasisFunctions) -> xr.DataArray:
    """Return the regional-mean restriction on the basis's (source, region) states."""
    index = basis.operator.basis_matrix.indexes["region"]
    rows = []
    for source, region in index:
        row = np.zeros((len(SOURCES), *COUNTRY_MAP.shape))
        member = PARTITION[source] == region + 1
        row[SOURCES.index(source)] = member / member.sum()
        rows.append(row)
    coords = xr.Coordinates.from_pandas_multiindex(index, "region")
    return xr.DataArray(
        np.array(rows),
        dims=("region", *NATIVE_DIMS),
        coords={**coords, SOURCE_DIM: list(SOURCES), **GRID},
        attrs={"units": "1"},
    )


def _countries() -> Countries:
    """Return the shared three-country map."""
    return Countries(_country_dataset())


def _country_dataset() -> xr.Dataset:
    """Return the country file contents."""
    return xr.Dataset(
        {"country": (("lat", "lon"), COUNTRY_MAP), "name": ("ncountries", COUNTRY_NAMES)}, coords=GRID
    )


def _case(*, functionals: bool = True, aggregation_error_rank: int | None = None):
    """Prepare the coherent CO2 case and its affine reconstruction, unsaved."""
    rng = np.random.default_rng(68)
    covariance, dense_b = _covariance()
    mean = _labelled(1.0 + 0.2 * rng.normal(size=(2, 3, 4)), NATIVE_DIMS).assign_attrs(units="1")
    flux = _labelled(
        np.stack([rng.uniform(1.0, 3.0, size=(3, 4)), rng.uniform(-3.0, 1.0, size=(3, 4))]), NATIVE_DIMS
    ).assign_attrs(units="mol m-2 s-1")
    index = _observation_index()
    observation_coords = xr.Coordinates.from_pandas_multiindex(index, "observation")
    sensitivity = xr.DataArray(
        rng.uniform(0.0, 1.0, size=(N_OBSERVATION, 2, 3, 4)),
        dims=("observation", *NATIVE_DIMS),
        coords={**observation_coords, SOURCE_DIM: list(SOURCES), **GRID},
        attrs={"units": "ppm"},
    )
    h = sensitivity.values.reshape(N_OBSERVATION, -1)
    observed = rng.normal(size=N_OBSERVATION) + h @ (mean.values.reshape(-1) + 0.5) + 400.0
    basis = _basis(flux)
    restriction = _restriction(basis)
    q = co2_flux_country_functionals(flux, _countries()) if functionals else None
    reduction = reduce_native_gaussian(
        covariance=covariance,
        strategy=SuppliedRestriction(restriction),
        state_dim="region",
        native_mean=mean,
        native_sensitivity=sensitivity,
        observation_dim="observation",
        observation_batch_size=3,
        functionals=q,
        functional_dim=COUNTRY_FUNCTIONAL_DIM,
    )
    # The reduction's observation intercept is H m; place the observations
    # 400 ppm above it so the residual check uses absolute concentrations.
    reduction = _with_offset(reduction, 400.0)
    state = basis.operator.basis_matrix.indexes["region"]
    nmeasure = xr.Coordinates.from_pandas_multiindex(index, "nmeasure")
    inputs = xr.Dataset(
        {
            "H": xr.DataArray(
                np.zeros((N_OBSERVATION, state.size)),
                dims=("nmeasure", "region"),
                coords={**nmeasure, **xr.Coordinates.from_pandas_multiindex(state, "region")},
            ),
            "mf": ("nmeasure", observed),
            "mf_error": ("nmeasure", np.sqrt(NOISE_DIAGONAL)),
        },
        coords=nmeasure,
    )
    for name in ("H", "mf", "mf_error"):
        inputs[name].attrs["units"] = "ppm"
    canonical = PreparedInputs(
        inputs, basis, xr.Dataset({"averaging_period": ("site", ["1h"] * 2)}, coords={"site": SITE_NAMES})
    )
    prepared = prepare_co2_inputs(
        canonical,
        reduction,
        aggregation_error_rank=aggregation_error_rank,
        provenance={"start_date": "2020-01-01", "end_date": "2020-01-02", "domain": "EUROPE"},
    )
    prolongation = project_native_covariance(
        covariance=covariance,
        strategy=SuppliedRestriction(restriction),
        state_dim="region",
        native_sensitivity=sensitivity,
        observation_dim="observation",
    ).prolongation
    affine = import_explicit_affine_flux_map(
        prepared, mean, flux, prolongation, prepared_inputs_id="unsaved"
    )
    problem = {"B": dense_b, "h": h, "m": mean.values.reshape(-1), "y": observed - 400.0, "canonical": canonical}
    return problem, reduction, prepared, affine


def _with_offset(reduction, offset: float):
    """Shift the native observation mean and intercept by a constant background."""
    from dataclasses import replace

    return replace(
        reduction,
        native_observation_mean=(reduction.native_observation_mean + offset).assign_attrs(units="ppm"),
        observation_intercept=(reduction.observation_intercept + offset).assign_attrs(units="ppm"),
    )


def _fit_fixture(prepared: Co2PreparedInputs, count: int = COUNT, *, noise: np.ndarray | None = None) -> xr.DataTree:
    """Return a saved-shape additive-sigma CO2 trace of exact reduced-posterior draws.

    The fit's observation covariance is ``A + diag(noise)`` with the
    prepared dense ``A``; ``epsilon`` records its marginal standard deviation.
    """
    inputs = prepared.inv_inputs
    noise = NOISE_DIAGONAL if noise is None else noise
    h = inputs["H"].values
    a = inputs["aggregation_error_covariance"].values
    c = inputs["alpha_prior_covariance"].values
    prior_mean = inputs["alpha_prior_mean"].values
    fixed = inputs["fixed_prior_contribution"].values
    weight = np.linalg.inv(a + np.diag(noise))
    covariance = np.linalg.inv(np.linalg.inv(c) + h.T @ weight @ h)
    # The fit's pollution is fixed + H alpha, so the innovation is y - H m.
    innovation = inputs["mf"].values - fixed - h @ prior_mean
    posterior_mean = covariance @ h.T @ weight @ innovation
    rng = np.random.default_rng(1084)
    draws = {
        "posterior": rng.multivariate_normal(posterior_mean, covariance, size=count),
        "prior": rng.multivariate_normal(np.zeros_like(prior_mean), c, size=count),
    }
    observation_coords = xr.Coordinates.from_pandas_multiindex(inputs.indexes["nmeasure"], "nmeasure")
    state_coords = xr.Coordinates.from_pandas_multiindex(inputs.indexes["region"], "region")
    sample_coords = {"chain": [0], "draw": np.arange(count)}
    groups = {}
    for group, delta in draws.items():
        modelled = fixed + (prior_mean + delta) @ h.T
        epsilon = np.broadcast_to(np.sqrt(np.diag(a) + noise), modelled.shape)
        groups[group] = xr.Dataset(
            {
                "flux_scaling": xr.DataArray(
                    (prior_mean + delta)[None], dims=("chain", "draw", "region"),
                    coords={**sample_coords, **state_coords}, attrs={"units": "1"},
                ),
                "modelled_concentration": xr.DataArray(
                    modelled[None], dims=("chain", "draw", "nmeasure"),
                    coords={**sample_coords, **observation_coords}, attrs={"units": "ppm"},
                ),
                "co2_flux_contribution": xr.DataArray(
                    modelled[None], dims=("chain", "draw", "nmeasure"),
                    coords={**sample_coords, **observation_coords}, attrs={"units": "ppm"},
                ),
                "epsilon": xr.DataArray(
                    epsilon[None], dims=("chain", "draw", "nmeasure"),
                    coords={**sample_coords, **observation_coords}, attrs={"units": "ppm"},
                ),
            }
        )
    groups["/"] = xr.Dataset(
        attrs={
            "rhime_boundary_format_version": 2,
            "rhime_model_metadata": json.dumps({"recipe": "co2"}),
            "rhime_variable_roles": json.dumps(
                {
                    "flux_scale": "flux_scaling",
                    "model_mean": "modelled_concentration",
                    "pollution_concentration": "co2_flux_contribution",
                    "model_error": "epsilon",
                }
            ),
        }
    )
    return xr.DataTree.from_dict(groups)


def _native_country_totals(problem, functionals: xr.DataArray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the native posterior mean and SD of country totals (sources summed)."""
    q = np.asarray(to_dense(functionals.transpose(COUNTRY_FUNCTIONAL_DIM, *NATIVE_DIMS)).values).reshape(
        functionals.sizes[COUNTRY_FUNCTIONAL_DIM], -1
    )
    labels = functionals.get_index(COUNTRY_FUNCTIONAL_DIM)
    total = np.array([[label[0] == country for label in labels] for country in COUNTRY_NAMES], dtype=float)
    b, h, m = problem["B"], problem["h"], problem["m"]
    gain = q @ b @ h.T @ np.linalg.inv(h @ b @ h.T + np.diag(noise))
    mean = q @ m + gain @ (problem["y"] - h @ m)
    covariance = q @ b @ q.T - gain @ h @ b @ q.T
    return total @ mean, np.sqrt(np.diag(total @ covariance @ total.T))


def _staged(tmp_path, prepared, affine, trace, *, source_to_sector=None, **options):
    """Save preparation and samples, then replay the products from disk."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    country_file = tmp_path / "countries.nc"
    if not country_file.exists():
        _country_dataset().to_netcdf(country_file)
    manifest = handoffs.save_preparation(prepared, output_dir=tmp_path / "prepare", affine_artifact=affine)
    generation = handoffs.load_preparation(manifest).generation
    samples = handoffs.save_samples(
        manifest, trace, actual_inference={"algorithm": "fixture"}, preparation_generation=generation,
        output_dir=tmp_path / "sample",
    )
    output = Co2OutputOptions(
        output_format="basic", output_path=tmp_path / "products", country_file=country_file, **options
    )
    return handoffs.postprocess(samples, output=output, source_to_sector=source_to_sector)


def test_staged_country_outputs_are_complete_native_from_prepare_to_replay(tmp_path) -> None:
    """Persisted products give complete country totals that sample the native posterior."""
    problem, _, prepared, affine = _case()
    trace = _fit_fixture(prepared)
    mapping = {"fossil": "anthropogenic", "biosphere": "natural"}

    result = _staged(tmp_path, prepared, affine, trace, source_to_sector=mapping)

    country = result.outputs["country_flux"]
    assert country.attrs["uncertainty_scope"] == "complete_native"
    assert result.output_metadata["country_uncertainty_scope"] == "complete_native"
    assert result.output_metadata["uncertainty_scope"] == "retained_state_conditional"
    assert "dense A" in country.attrs["unresolved_conditioning"]
    assert country.attrs["marginalised_state_count"] == "0"
    assert float(country.attrs["conditional_covariance_min_eigenvalue"]) > -1e-8
    for variable in country.data_vars.values():
        assert variable.attrs["uncertainty_scope"] == "complete_native"

    functionals = co2_flux_country_functionals(affine.affine_map.flux, _countries())
    native_mean, native_sd = _native_country_totals(problem, functionals, NOISE_DIAGONAL)
    posterior_mean = country.country_total_posterior_mean.sel(country=COUNTRY_NAMES).values
    posterior_sd = country.country_total_posterior_stdev.sel(country=COUNTRY_NAMES).values
    np.testing.assert_array_less(np.abs(posterior_mean - native_mean), 5 * native_sd / np.sqrt(COUNT))
    np.testing.assert_allclose(posterior_sd, native_sd, rtol=0.03)

    # Sector totals come from the same complete (country, source) draws.
    for source, sector in mapping.items():
        np.testing.assert_allclose(
            country[f"country_{sector}_posterior_mean"],
            country.country_posterior_mean.sel({SOURCE_DIM: source}, drop=True),
        )
        np.testing.assert_allclose(
            country[f"country_{sector}_posterior_stdev"],
            country.country_posterior_stdev.sel({SOURCE_DIM: source}, drop=True),
        )
    saved = xr.open_dataset(result.output_metadata["country_flux_path"])
    assert saved.attrs["uncertainty_scope"] == "complete_native"

    # The same fit without persisted products stays retained-only, and its
    # spread is materially different.
    _, _, retained_prepared, retained_affine = _case(functionals=False)
    retained = _staged(tmp_path / "retained", retained_prepared, retained_affine, trace)
    retained_country = retained.outputs["country_flux"]
    assert retained_country.attrs["uncertainty_scope"] == "retained_state_conditional"
    assert "unresolved_conditioning" not in retained_country.attrs
    retained_sd = retained_country.country_total_posterior_stdev.sel(country=COUNTRY_NAMES).values
    assert np.abs(retained_sd - native_sd).max() > 0.1 * native_sd.max()


def test_prepared_country_functionals_round_trip(tmp_path) -> None:
    """Products survive NetCDF with their MultiIndex labels and the artifact's axes."""
    _, reduction, prepared, _ = _case()
    assert CO2_PREPARED_INPUTS_SCHEMA_VERSION == 4
    assert PREPARED_COUNTRY_FUNCTIONAL_DIM == COUNTRY_FUNCTIONAL_DIM
    products = prepared.country_functionals
    assert products is not None
    xr.testing.assert_identical(products.effective_observation_operator, prepared.inv_inputs["H"])
    xr.testing.assert_identical(products.retained_covariance, prepared.inv_inputs["alpha_prior_covariance"])
    assert prepared.provenance["country_functionals"]["count"] == len(COUNTRY_NAMES) * len(SOURCES)
    np.testing.assert_array_equal(
        products.unresolved_covariance.values, reduction.functional_products.unresolved_covariance.values
    )

    path = tmp_path / "prepared.nc"
    prepared.save(path)
    loaded = Co2PreparedInputs.load(path).country_functionals
    assert loaded is not None
    for name in (
        "functional_mean",
        "retained_response",
        "unresolved_observation_covariance",
        "unresolved_covariance",
    ):
        expected, actual = getattr(products, name), getattr(loaded, name)
        np.testing.assert_array_equal(actual.values, expected.values)
        assert actual.dims == expected.dims
        for dim in actual.dims:
            assert actual.indexes[dim].equals(expected.indexes[dim])
    assert list(loaded.functional_mean.indexes[COUNTRY_FUNCTIONAL_DIM].names) == ["country", "functional_source"]
    assert loaded.functional_mean.attrs["units"] == "g yr-1"

    without = tmp_path / "without.nc"
    _case(functionals=False)[2].save(without)
    assert Co2PreparedInputs.load(without).country_functionals is None


def test_requested_country_subset_uses_the_matching_product_rows(tmp_path) -> None:
    """Products restricted to selected countries equal the products of those rows of Q."""
    _, _, prepared, affine = _case()
    bound = BoundCo2AffineFluxMap(affine, prepared.inv_inputs["alpha_prior_mean"])
    trace = _fit_fixture(prepared, count=4)
    subset = Countries(_country_dataset(), country_selections=["SPAIN", "FRANCE"])
    conditioning = co2_unresolved_conditioning(trace, prepared, bound=bound, countries=subset)
    full = prepared.country_functionals
    labels = conditioning.products.functional_mean.indexes[COUNTRY_FUNCTIONAL_DIM]
    positions = full.functional_mean.indexes[COUNTRY_FUNCTIONAL_DIM].get_indexer(labels)
    assert sorted({label[0] for label in labels}) == ["FRANCE", "SPAIN"]
    np.testing.assert_array_equal(
        conditioning.products.unresolved_covariance.values,
        full.unresolved_covariance.values[np.ix_(positions, positions)],
    )
    np.testing.assert_array_equal(
        conditioning.products.retained_response.values, full.retained_response.values[positions]
    )

    result = _staged(
        tmp_path, prepared, affine, trace, paris_postprocessing_kwargs={"country_selections": ["SPAIN", "FRANCE"]}
    )
    assert result.outputs["country_flux"].attrs["uncertainty_scope"] == "complete_native"


def test_unprepared_country_is_rejected() -> None:
    """A country without prepared products cannot be reported as complete."""
    _, _, prepared, affine = _case()
    bound = BoundCo2AffineFluxMap(affine, prepared.inv_inputs["alpha_prior_mean"])
    dataset = _country_dataset()
    dataset["name"] = ("ncountries", ["FRANCE", "GERMANY", "PORTUGAL"])
    with pytest.raises(ValueError, match="were not prepared for"):
        co2_unresolved_conditioning(_fit_fixture(prepared, count=2), prepared, bound=bound, countries=Countries(dataset))


@pytest.mark.parametrize("rank", [None, 3])
def test_additive_sigma_solve_uses_each_draws_independent_variance(rank) -> None:
    """``(A + R)^-1`` uses the stored A and ``epsilon**2 - diag(A)`` at each draw."""
    _, _, prepared, _ = _case(aggregation_error_rank=rank)
    trace = _fit_fixture(_case()[2], count=3)
    a = _stored_a(prepared)
    variance = np.array([[0.1, 0.4, 0.9]])[..., None] * NOISE_DIAGONAL
    trace["posterior"]["epsilon"] = trace["posterior"]["epsilon"].copy(data=np.sqrt(np.diag(a) + variance))
    solve = _observation_solve(trace, prepared, json.loads(trace.attrs["rhime_variable_roles"]))
    assert solve.varies_by_draw
    rhs = np.random.default_rng(3).normal(size=(N_OBSERVATION, 2))
    for draw in range(3):
        np.testing.assert_allclose(
            solve.solve(rhs, 0, draw), np.linalg.solve(a + np.diag(variance[0, draw]), rhs), rtol=1e-9
        )

    constant = _fit_fixture(_case()[2], count=2)
    assert not _observation_solve(constant, prepared, json.loads(constant.attrs["rhime_variable_roles"])).varies_by_draw


def _stored_a(prepared: Co2PreparedInputs) -> np.ndarray:
    """Return the dense covariance of the stored aggregation-error representation."""
    inputs = prepared.inv_inputs
    if prepared.aggregation_error_mode == "dense":
        return inputs["aggregation_error_covariance"].values
    factor = inputs["low_rank_factor"].values
    return factor @ factor.T + np.diag(inputs["diagonal_residual_variance"].values)


@pytest.mark.parametrize("inferred", [True, False])
def test_fixed_ou_solve_rebuilds_the_fits_covariance(inferred: bool) -> None:
    """Fixed-OU fits condition on A + error**2 + OU at each draw's (or the fixed) amplitudes."""
    _, _, prepared, _ = _case(aggregation_error_rank=3)
    trace = _fit_fixture(_case()[2], count=2)
    observations = prepared.inv_inputs["mf"]
    error = 0.5 * np.sqrt(NOISE_DIAGONAL)
    amplitudes = np.array([[[0.6, 0.4], [0.2, 0.9]]])
    site = xr.DataArray(SITE_NAMES, dims="ou_site")
    trace["constant_data"] = xr.DataTree(
        xr.Dataset(
            {
                "error": observations.copy(data=error),
                "ou_site_index": observations.copy(data=SITES),
                "ou_tau_hours": xr.DataArray(list(TAU.values()), dims="ou_site", coords={"ou_site": site}),
            }
        )
    )
    if inferred:
        trace["posterior"]["ou_site_amplitude"] = xr.DataArray(
            amplitudes[:, :, ::-1], dims=("chain", "draw", "ou_site"),
            coords={"chain": [0], "draw": [0, 1], "ou_site": SITE_NAMES[::-1]},
        )
    else:
        trace["constant_data"]["ou_site_amplitude"] = xr.DataArray(
            amplitudes[0, 0], dims="ou_site", coords={"ou_site": SITE_NAMES}
        )
    trace.attrs["rhime_likelihood_builder"] = json.dumps(
        {"module": "openghg_inversions.models.fixed_ou", "qualname": "add_fixed_ou_gaussian_likelihood"}
    )
    solve = _observation_solve(trace, prepared, json.loads(trace.attrs["rhime_variable_roles"]))
    rhs = np.random.default_rng(5).normal(size=(N_OBSERVATION, 3))
    a = _stored_a(prepared)
    for draw in range(2):
        amplitude = amplitudes[0, draw] if inferred else amplitudes[0, 0]
        expected = np.linalg.solve(a + np.diag(error**2) + _ou(amplitude), rhs)
        np.testing.assert_allclose(solve.solve(rhs, 0, draw), expected, rtol=1e-8)


def test_unsupported_likelihood_is_rejected() -> None:
    """Likelihoods whose per-draw covariance is not rebuilt cannot claim complete outputs."""
    _, _, prepared, _ = _case()
    trace = _fit_fixture(prepared, count=2)
    trace.attrs["rhime_likelihood_builder"] = json.dumps(
        {"module": "openghg_inversions.models.scalar_sigma", "qualname": "add_scalar_sigma_eigen_likelihood"}
    )
    with pytest.raises(ValueError, match="do not support the recorded likelihood"):
        _observation_solve(trace, prepared, json.loads(trace.attrs["rhime_variable_roles"]))


def test_preparation_rejects_non_country_functionals() -> None:
    """Only country functional products belong in the CO2 artifact."""
    from dataclasses import replace

    problem, reduction, _, _ = _case()
    products = reduction.functional_products
    renamed = replace(
        products,
        functional_mean=products.functional_mean.rename({COUNTRY_FUNCTIONAL_DIM: "functional"}),
    )
    with pytest.raises(ValueError, match="persists only country functional products"):
        prepare_co2_inputs(
            problem["canonical"], replace(reduction, functional_products=renamed), aggregation_error_rank=None
        )
    with pytest.raises(ValueError, match="full-restriction"):
        replace(_case()[2], country_functionals=replace(products, marginalised=xr.zeros_like(
            reduction.retained_mean, dtype=bool)))
