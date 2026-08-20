import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pytensor.compile.mode import Mode

from openghg_inversions.models.components import (
    LinearComponentResult,
    add_inferpymc_likelihood_component,
    add_linear_component,
    add_model_data,
    add_offset_component,
    add_sigma_component,
    resolve_model_variable,
)
from openghg_inversions.models.coords import CoordRegistry, attach_coord_registry
from openghg_inversions.models.state_activity import prepare_linear_design
from openghg_inversions.sigma import SigmaAlignment


def _obs_index() -> pd.MultiIndex:
    """Create a stacked observation index used by component tests."""
    return pd.MultiIndex.from_arrays(
        [
            ["MHD", "MHD", "TAC", "TAC"],
            pd.to_datetime(["2019-01-01", "2019-01-02", "2019-02-01", "2019-02-02"]),
        ],
        names=["site", "time"],
    )


def _obs_coords() -> xr.Coordinates:
    """Create explicit xarray coordinates for the stacked observation index."""
    return xr.Coordinates.from_pandas_multiindex(_obs_index(), "nmeasure")


def _site_indicator() -> xr.DataArray:
    """Create a simple site-indicator DataArray aligned to the test index."""
    return xr.DataArray(
        np.array([0, 0, 1, 1]),
        dims=("nmeasure",),
        coords=_obs_coords(),
        name="site_indicator",
    )


def _likelihood_dataset() -> xr.Dataset:
    """Create a minimal canonical-style dataset for likelihood tests."""
    return xr.Dataset(
        data_vars={
            "mf": ("nmeasure", np.array([1.0, 2.0, 3.0, 4.0])),
            "mf_error": ("nmeasure", np.full(4, 0.1)),
            "site_indicator": ("nmeasure", np.array([0, 0, 1, 1])),
            "min_error": ("nmeasure", np.full(4, 0.01)),
        },
        coords=_obs_coords(),
    )


def _sigma_alignment(data: xr.Dataset, *, per_site: bool = True) -> SigmaAlignment:
    """Create prepared sigma alignment data for likelihood tests."""
    period_index = xr.DataArray(
        np.array([0, 0, 1, 1]),
        dims=("nmeasure",),
        coords=data["site_indicator"].coords,
    )
    return SigmaAlignment.from_indices(
        data["site_indicator"],
        period_index,
        per_site=per_site,
    )


def test_add_model_data_uses_add_coords() -> None:
    """Check add_model_data registers model coords through the shared coord helper."""
    data = xr.DataArray([1.0, 2.0], dims=("nmeasure",), coords={"nmeasure": [10, 11]}, name="Y")

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        add_model_data(data)

    assert "Y" in model.named_vars
    assert "nmeasure" in model.coords


def test_add_linear_component_creates_expected_named_vars() -> None:
    """Check add_linear_component returns the created PyMC objects explicitly."""
    data = xr.DataArray(
        np.ones((4, 2)),
        dims=("nmeasure", "nx"),
        coords={"nmeasure": np.arange(4), "nx": np.arange(2)},
        name="H",
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_linear_component(
            prepare_linear_design(data),
            data_name="hx",
            prior_args={"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            var_name="x",
            output_name="mu",
        )

    assert {"hx", "x", "mu"}.issubset(model.named_vars)
    assert isinstance(result, LinearComponentResult)
    assert result.data is model.named_vars["hx"]
    assert result.latent is model.named_vars["x"]
    assert result.state is model.named_vars["x"]
    assert result.output is model.named_vars["mu"]


def test_add_linear_component_returns_effective_reparameterised_latent() -> None:
    """Check the component result exposes the true reparameterized latent variable."""
    data = xr.DataArray(
        np.ones((4, 2)),
        dims=("nmeasure", "nx"),
        coords={"nmeasure": np.arange(4), "nx": np.arange(2)},
        name="H",
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        result = add_linear_component(
            prepare_linear_design(data),
            data_name="hx",
            prior_args={"pdf": "lognormal", "mean": 1.5, "stdev": 0.2, "reparameterise": True},
            var_name="x",
            output_name="mu",
        )

    assert "x_latent" in model.named_vars
    assert "x" in model.named_vars
    assert result.latent is model.named_vars["x_latent"]


def test_resolve_model_variable_prefers_latent() -> None:
    """Check shared model-variable resolution prefers the reparameterised latent."""
    data = xr.DataArray(
        np.ones((4, 2)),
        dims=("nmeasure", "nx"),
        coords={"nmeasure": np.arange(4), "nx": np.arange(2)},
        name="H",
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        add_linear_component(
            prepare_linear_design(data),
            data_name="hx",
            prior_args={"pdf": "lognormal", "mean": 1.5, "stdev": 0.2, "reparameterise": True},
            var_name="x",
            output_name="mu",
        )

    assert resolve_model_variable(model, "x") is model.named_vars["x_latent"]
    assert resolve_model_variable(model, "missing") is None


def test_add_sigma_component_uses_prepared_alignment() -> None:
    """Check the PyMC component only consumes backend-neutral prepared alignment."""
    data = _likelihood_dataset()
    alignment = _sigma_alignment(data)

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_sigma_component(
            alignment,
            prior_args={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            compute_deterministic=True,
        )
        assert "sigma" in model.named_vars
        assert "sigma_site_index" in model.named_vars
        assert "sigma_period_index" in model.named_vars
        assert "sigma_aligned" in model.named_vars

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_sigma_component(
            _sigma_alignment(data, per_site=False),
            prior_args={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
        )
        assert model.named_vars["sigma"].eval().shape[0] == 1
        assert "site_indicator" not in model.named_vars
        assert np.array_equal(model.named_vars["sigma_site_index"].eval(), np.zeros(4))
        assert "sigma_period_index" in model.named_vars


def test_add_offset_component_supports_manual_and_derived_freq() -> None:
    """Check offsets accept explicit or internally derived frequency indicators."""
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


def test_add_offset_component_supports_one_global_scalar() -> None:
    """A global offset has one latent value broadcast over observations."""
    site_indicator = _site_indicator()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        offset = add_offset_component(
            site_indicator,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            per_site=False,
        )

    assert model.named_vars["offset_latent"].ndim == 0
    assert offset.eval().shape == (4,)
    assert "offset_design" not in model.named_vars


@pytest.mark.parametrize("invalid_args", [{"offset_freq": "monthly"}, {"drop_first": True}])
def test_global_offset_rejects_site_period_options(invalid_args: dict[str, object]) -> None:
    """Global offsets reject options that only have site-design semantics."""
    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        with pytest.raises(ValueError, match="Global offsets"):
            add_offset_component(
                _site_indicator(),
                prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                per_site=False,
                **invalid_args,
            )


def test_add_offset_component_drop_first_and_freq_builds_expected_design() -> None:
    """Check drop-first offsets still build the expected site-period design."""
    site_indicator = _site_indicator()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_offset_component(
            site_indicator,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            offset_freq="monthly",
            output_name="offset",
            drop_first=True,
        )

    offset_design = model.named_vars["offset_design"].eval()
    assert offset_design.shape == (4, 2)
    np.testing.assert_array_equal(offset_design[:2], np.zeros((2, 2)))
    np.testing.assert_array_equal(offset_design[2:], np.array([[0, 1], [0, 1]]))


def test_add_inferpymc_likelihood_component_adds_epsilon_and_y() -> None:
    """Check the likelihood helper adds epsilon, y, and sigma variables."""
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
            sigma_alignment=_sigma_alignment(ds),
        )

    assert {"epsilon", "y", "sigma"}.issubset(model.named_vars)


def test_likelihood_no_model_error_uses_observation_error() -> None:
    """Check no-model-error mode bypasses pollution-event model error."""
    ds = _likelihood_dataset().copy()
    ds["min_error"] = ("nmeasure", np.full(ds.sizes["nmeasure"], 999.0))

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            no_model_error=True,
            sigma_alignment=_sigma_alignment(ds, per_site=False),
        )

    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), ds["mf_error"].values)


def test_likelihood_pollution_events_from_obs_can_run_without_boundary_conditions() -> None:
    """Check obs-derived pollution-event scaling does not require BC terms."""
    ds = _likelihood_dataset().copy()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.zeros(4), dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.5, "upper": 1.5},
            pollution_events_from_obs=True,
            sigma_alignment=SigmaAlignment.from_frequency(
                ds["site_indicator"],
                frequency=None,
                per_site=False,
            ),
            power=2.0,
        )

    epsilon = model.named_vars["epsilon"].eval(mode=Mode(linker="py", optimizer="fast_run"))
    assert np.all(np.diff(epsilon) > 0)
    assert "y" in model.named_vars


def test_likelihood_samples_prior_predictive_with_shared_sigma_and_registered_site_indicator() -> None:
    """Check shared sigma indexing still works after offsets register site data."""
    ds = _likelihood_dataset()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        mu_bc = pm.Data("mu_bc_input", np.zeros(4), dims="nmeasure")
        offset = add_offset_component(
            ds["site_indicator"],
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            output_name="offset",
        )
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=mu_bc,
            offset=offset,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=_sigma_alignment(ds, per_site=False),
        )

        assert model.named_vars["sigma"].eval().shape[0] == 1
        pm.sample_prior_predictive(draws=1, model=model, random_seed=123)
