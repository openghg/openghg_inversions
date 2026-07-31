from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

import openghg_inversions.models.components as components_module
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
            data,
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
            data,
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


def test_likelihood_additive_model_error_uses_response_independent_sigma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Additive Gaussian mode combines observation error and sigma in quadrature."""
    observation_error = np.array([-0.1, 0.2, -0.4, 0.3])
    min_error = np.array([0.05, 0.6, 0.1, 0.2])
    ds = _likelihood_dataset().assign(
        mf_error=("nmeasure", observation_error),
        min_error=("nmeasure", min_error),
    )
    sigma_values = pm.floatX(np.array([0.5, 0.5, 0.25, 0.25]))

    def fixed_sigma(*args, **kwargs):
        """Return fixed observation-aligned additive errors."""
        return pt.as_tensor_variable(sigma_values)

    monkeypatch.setattr(components_module, "add_sigma_component", fixed_sigma)

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=_sigma_alignment(ds, per_site=False),
            additive_model_error=True,
        )

    expected = np.maximum(np.sqrt(observation_error**2 + sigma_values**2), min_error)
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected)
    assert "Normal" in type(model.named_vars["y"].owner.op).__name__


@pytest.mark.parametrize("nu", [4.0, 8.0])
def test_likelihood_additive_student_t_uses_same_response_independent_scale(
    monkeypatch: pytest.MonkeyPatch,
    nu: float,
) -> None:
    """Additive Student-t changes the tails without changing epsilon."""
    observation_error = np.array([-0.1, 0.2, -0.4, 0.3])
    min_error = np.array([0.05, 0.6, 0.1, 0.2])
    ds = _likelihood_dataset().assign(
        mf_error=("nmeasure", observation_error),
        min_error=("nmeasure", min_error),
    )
    sigma_values = pm.floatX(np.array([0.5, 0.5, 0.25, 0.25]))

    def fixed_sigma(*args, **kwargs):
        """Return fixed observation-aligned additive errors."""
        return pt.as_tensor_variable(sigma_values)

    monkeypatch.setattr(components_module, "add_sigma_component", fixed_sigma)

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=_sigma_alignment(ds, per_site=False),
            additive_model_error=True,
            additive_student_t_nu=nu,
        )

    expected = np.maximum(np.sqrt(observation_error**2 + sigma_values**2), min_error)
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected)
    assert "StudentT" in type(model.named_vars["y"].owner.op).__name__


@pytest.mark.parametrize("nu", [True, 2.0, float("inf"), "4"])
def test_additive_student_t_rejects_invalid_degrees_of_freedom(nu: Any) -> None:
    """Student-t degrees of freedom must be fixed, numeric, and finite-variance."""
    ds = _likelihood_dataset()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        with pytest.raises(ValueError, match="additive_student_t_nu"):
            add_inferpymc_likelihood_component(
                ds,
                mu=mu,
                mu_bc=None,
                sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                sigma_alignment=_sigma_alignment(ds, per_site=False),
                additive_model_error=True,
                additive_student_t_nu=nu,
            )


def test_additive_student_t_requires_additive_model_error() -> None:
    """Student-t cannot be attached to either pollution-event likelihood."""
    ds = _likelihood_dataset()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        with pytest.raises(ValueError, match="additive_model_error=True"):
            add_inferpymc_likelihood_component(
                ds,
                mu=mu,
                mu_bc=None,
                sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                sigma_alignment=_sigma_alignment(ds, per_site=False),
                additive_student_t_nu=4.0,
            )


@pytest.mark.parametrize(
    "incompatible_option",
    [
        "pollution_events_from_obs",
        "pollution_events_from_obs_one_sided",
        "pollution_events_from_obs_johnson_su",
        "no_model_error",
    ],
)
def test_additive_model_error_rejects_incompatible_options(incompatible_option: str) -> None:
    """Additive sigma cannot be mixed with dimensionless pollution-event modes."""
    ds = _likelihood_dataset()
    options = {"additive_model_error": True, incompatible_option: True}

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        with pytest.raises(ValueError, match=incompatible_option):
            add_inferpymc_likelihood_component(
                ds,
                mu=mu,
                mu_bc=None,
                sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                sigma_alignment=_sigma_alignment(ds, per_site=False),
                power=2.0,
                **options,
            )


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

    epsilon = model.named_vars["epsilon"].eval()
    assert np.all(np.diff(epsilon) > 0)
    assert "y" in model.named_vars


def _epsilon_for_obs_pollution_events(
    monkeypatch: pytest.MonkeyPatch,
    *,
    one_sided: bool | None,
) -> np.ndarray:
    """Build and evaluate an observation-derived likelihood with fixed sigma.

    Args:
        monkeypatch: Pytest helper used to replace the sampled sigma component.
        one_sided: Whether to enable one-sided PEFO. ``None`` omits the option
            so the public default is exercised.

    Returns:
        Evaluated observation-aligned epsilon values.
    """
    ds = _likelihood_dataset()
    mu_bc_values = np.array([2.0, 1.0, 4.0, 2.0])

    def fixed_sigma(*args, **kwargs):
        """Return deterministic observation-aligned sigma values for this regression test."""
        return pt.full((ds.sizes["nmeasure"],), 0.5)

    monkeypatch.setattr(components_module, "add_sigma_component", fixed_sigma)

    likelihood_options = {"pollution_events_from_obs": True}
    if one_sided is not None:
        likelihood_options["pollution_events_from_obs_one_sided"] = one_sided

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.zeros(4), dims="nmeasure")
        mu_bc = pm.Data("mu_bc_input", mu_bc_values, dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=mu_bc,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=_sigma_alignment(ds, per_site=False),
            power=2.0,
            **likelihood_options,
        )

    return model.named_vars["epsilon"].eval()


def test_obs_pollution_events_remain_two_sided_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PEFO retains absolute observed enhancements when one-sided mode is omitted."""
    epsilon = _epsilon_for_obs_pollution_events(monkeypatch, one_sided=None)

    pollution_events = np.array([1.0, 1.0, 1.0, 2.0])
    expected = np.sqrt(0.1**2 + (0.5 * pollution_events) ** 2)
    np.testing.assert_allclose(epsilon, expected)


def test_obs_pollution_events_one_sided_clip_below_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-sided PEFO clips below-baseline values while retaining positive enhancements."""
    epsilon = _epsilon_for_obs_pollution_events(monkeypatch, one_sided=True)

    pollution_events = np.array([0.0, 1.0, 0.0, 2.0])
    expected = np.sqrt(0.1**2 + (0.5 * pollution_events) ** 2)
    np.testing.assert_allclose(epsilon, expected)


@pytest.mark.parametrize(
    ("one_sided", "pollution_events"),
    [
        (None, np.array([3.999999, 1.999999, -0.000001, 1.999999])),
        (True, np.array([0.000001, 0.000001, 0.000001, 2.000001])),
    ],
)
def test_obs_pollution_events_without_bc_preserve_default_and_clip_one_sided(
    monkeypatch: pytest.MonkeyPatch,
    one_sided: bool | None,
    pollution_events: np.ndarray,
) -> None:
    """No-BC PEFO preserves its default stabilizer while one-sided mode stays nonnegative."""
    ds = _likelihood_dataset().assign(mf=("nmeasure", np.array([-4.0, -2.0, 0.0, 2.0])))

    def fixed_sigma(*args, **kwargs):
        """Return deterministic observation-aligned sigma values for this regression test."""
        return pt.full((ds.sizes["nmeasure"],), 0.5)

    monkeypatch.setattr(components_module, "add_sigma_component", fixed_sigma)
    likelihood_options = {"pollution_events_from_obs": True}
    if one_sided is not None:
        likelihood_options["pollution_events_from_obs_one_sided"] = one_sided

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.zeros(4), dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=None,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=_sigma_alignment(ds, per_site=False),
            power=2.0,
            **likelihood_options,
        )

    expected = np.sqrt(0.1**2 + (0.5 * pollution_events) ** 2)
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {
                "pollution_events_from_obs": False,
                "pollution_events_from_obs_johnson_su": True,
                "power": 2.0,
            },
            "pollution_events_from_obs",
        ),
        (
            {
                "pollution_events_from_obs": True,
                "pollution_events_from_obs_one_sided": True,
                "pollution_events_from_obs_johnson_su": True,
                "power": 2.0,
            },
            "one_sided",
        ),
        (
            {
                "pollution_events_from_obs": True,
                "pollution_events_from_obs_johnson_su": True,
                "no_model_error": True,
                "power": 2.0,
            },
            "no_model_error",
        ),
        (
            {
                "pollution_events_from_obs": True,
                "pollution_events_from_obs_johnson_su": True,
                "power": 1.99,
            },
            "power.*2",
        ),
    ],
)
def test_johnson_su_pollution_events_reject_invalid_combinations(
    options: dict[str, bool | float],
    message: str,
) -> None:
    """Johnson-SU PEFO rejects modes that do not define its transformed distribution."""
    ds = _likelihood_dataset()

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        mu_bc = pm.Data("mu_bc_input", np.zeros(4), dims="nmeasure")
        with pytest.raises(ValueError, match=message):
            add_inferpymc_likelihood_component(
                ds,
                mu=mu,
                mu_bc=mu_bc,
                sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                sigma_alignment=_sigma_alignment(ds, per_site=False),
                **options,
            )


def test_johnson_su_likelihood_uses_full_baseline_and_coherent_predictive_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Johnson-SU mode uses BC plus offset and never reuses observed Y when drawing."""
    observed = np.array([1.2, 2.0, 3.1, 4.4])
    observation_error = np.array([-0.1, 0.05, -0.5, 0.2])
    min_error = np.array([0.2, 0.1, 0.1, 0.3])
    ds = _likelihood_dataset().assign(
        mf=("nmeasure", observed),
        mf_error=("nmeasure", observation_error),
        min_error=("nmeasure", min_error),
    )
    mu_values = np.array([0.2, 0.5, 1.1, -0.1])
    mu_bc_values = np.array([0.8, 1.0, 1.5, 2.0])
    offset_values = np.array([0.1, -0.2, 0.3, 0.4])
    sigma_values = pm.floatX(np.array([0.2, 0.25, 0.3, 0.15]))

    def fixed_sigma(*args, **kwargs):
        """Return deterministic observation-aligned sigma for component semantics."""
        return pt.as_tensor_variable(sigma_values)

    monkeypatch.setattr(components_module, "add_sigma_component", fixed_sigma)

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", mu_values, dims="nmeasure")
        mu_bc = pm.Data("mu_bc_input", mu_bc_values, dims="nmeasure")
        offset = pm.Data("offset_input", offset_values, dims="nmeasure")
        add_inferpymc_likelihood_component(
            ds,
            mu=mu,
            mu_bc=mu_bc,
            offset=offset,
            sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
            sigma_alignment=_sigma_alignment(ds, per_site=False),
            pollution_events_from_obs=True,
            pollution_events_from_obs_johnson_su=True,
            power=2.0,
        )

    baseline = mu_bc_values + offset_values
    effective_error = np.maximum(np.abs(observation_error), min_error)
    expected_epsilon = np.sqrt(effective_error**2 + (sigma_values * (observed - baseline)) ** 2)
    np.testing.assert_allclose(model.named_vars["epsilon"].eval(), expected_epsilon)
    assert "CustomDist" in type(model.named_vars["y"].owner.op).__name__

    first_draws = pm.draw(model.named_vars["y"], draws=7, random_seed=943)
    with model:
        pm.set_data({"Y": observed + 1000.0})
    second_draws = pm.draw(model.named_vars["y"], draws=7, random_seed=943)

    assert first_draws.shape == (7, 4)
    np.testing.assert_array_equal(first_draws, second_draws)


def test_johnson_su_likelihood_requires_positive_effective_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Johnson-SU mode rejects a singular zero observation-error transform."""
    ds = _likelihood_dataset().assign(
        mf_error=("nmeasure", np.zeros(4)),
        min_error=("nmeasure", np.zeros(4)),
    )

    def fixed_sigma(*args, **kwargs):
        """Return a harmless fixed sigma so the test isolates observation error."""
        return pt.full((ds.sizes["nmeasure"],), 0.2)

    monkeypatch.setattr(components_module, "add_sigma_component", fixed_sigma)

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        mu = pm.Data("mu_input", np.ones(4), dims="nmeasure")
        with pytest.raises(ValueError, match="finite.*positive|positive.*finite"):
            add_inferpymc_likelihood_component(
                ds,
                mu=mu,
                mu_bc=None,
                sigprior={"pdf": "uniform", "lower": 0.1, "upper": 1.0},
                sigma_alignment=_sigma_alignment(ds, per_site=False),
                pollution_events_from_obs=True,
                pollution_events_from_obs_johnson_su=True,
                power=2.0,
            )


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
