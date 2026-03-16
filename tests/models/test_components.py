import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from openghg_inversions.models.components import (
    LinearComponentResult,
    add_inferpymc_likelihood_component,
    add_linear_component,
    add_model_data,
    add_offset_component,
    add_sigma_component,
)
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry


def _obs_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [["MHD", "MHD", "TAC", "TAC"], pd.to_datetime(["2019-01-01", "2019-01-02", "2019-02-01", "2019-02-02"])],
        names=["site", "time"],
    )


def _site_indicator() -> xr.DataArray:
    index = _obs_index()
    return xr.DataArray(
        np.array([0, 0, 1, 1]),
        dims=("nmeasure",),
        coords={"nmeasure": index},
        name="site_indicator",
    )


def _likelihood_dataset() -> xr.Dataset:
    index = _obs_index()
    return xr.Dataset(
        data_vars={
            "mf": ("nmeasure", np.array([1.0, 2.0, 3.0, 4.0])),
            "mf_error": ("nmeasure", np.full(4, 0.1)),
            "site_indicator": ("nmeasure", np.array([0, 0, 1, 1])),
            "sigma_freq_index": ("nmeasure", np.array([0, 0, 1, 1])),
            "min_error": ("nmeasure", np.full(4, 0.01)),
        },
        coords={
            "nmeasure": index,
        },
    )


def test_add_model_data_uses_add_coords() -> None:
    data = xr.DataArray([1.0, 2.0], dims=("nmeasure",), coords={"nmeasure": [10, 11]}, name="Y")

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        add_model_data(data)

    assert "Y" in model.named_vars
    assert "nmeasure" in model.coords


def test_add_linear_component_creates_expected_named_vars() -> None:
    data = xr.DataArray(
        np.ones((4, 2)),
        dims=("nmeasure", "nx"),
        coords={"nmeasure": np.arange(4), "nx": np.arange(2)},
        name="H",
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_linear_component(
            data,
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            var_name="x",
            output_name="mu",
        )

    assert isinstance(result, LinearComponentResult)
    assert {"hx", "x", "mu"}.issubset(model.named_vars)
    assert result.data is model.named_vars["hx"]
    assert result.latent is model.named_vars["x"]
    assert result.output is model.named_vars["mu"]


def test_add_linear_component_returns_effective_reparameterised_latent() -> None:
    data = xr.DataArray(
        np.ones((4, 2)),
        dims=("nmeasure", "nx"),
        coords={"nmeasure": np.arange(4), "nx": np.arange(2)},
        name="H",
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_linear_component(
            data,
            data_name="hx",
            prior_args={"pdf": "lognormal", "mean": 1.5, "stdev": 0.2, "reparameterise": True},
            var_name="x",
            output_name="mu",
        )

    assert "x_latent" in model.named_vars
    assert "x" in model.named_vars
    assert result.latent is model.named_vars["x_latent"]


def test_add_sigma_component_supports_explicit_and_derived_freq() -> None:
    site_indicator = _site_indicator()
    sigma_freq_index = xr.DataArray([0, 0, 1, 1], dims=("nmeasure",), coords=site_indicator.coords)

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_sigma_component(site_indicator, prior_args={"pdf": "uniform", "lower": 0.1, "upper": 1.0}, sigma_freq_index=sigma_freq_index)
        assert "sigma" in model.named_vars
        assert "sigma_freq_index" in model.named_vars

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_sigma_component(
            site_indicator,
            prior_args={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_freq="monthly",
            per_site=False,
        )
        assert model.named_vars["sigma"].eval().shape[0] == 1
        assert "sigma_freq_index" in model.named_vars


def test_add_offset_component_supports_manual_and_derived_freq() -> None:
    site_indicator = _site_indicator()
    manual_freq = xr.DataArray([0, 0, 1, 1], dims=("nmeasure",), coords=site_indicator.coords)

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_offset_component(
            site_indicator,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            offset_freq_indicator=manual_freq,
            output_name="offset",
        )
        assert "offset" in model.named_vars
        assert "offset_freq_indicator" in model.named_vars

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_offset_component(
            site_indicator,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            offset_freq="monthly",
            output_name="offset",
        )
        assert "offset" in model.named_vars
        assert "offset_freq_indicator" in model.named_vars


def test_add_inferpymc_likelihood_component_adds_epsilon_and_y() -> None:
    ds = _likelihood_dataset()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        mu_bc = pm.Data("mu_bc_input", np.zeros(4), dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=mu_bc,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_per_site=True,
        )

    assert {"epsilon", "y", "sigma"}.issubset(model.named_vars)
