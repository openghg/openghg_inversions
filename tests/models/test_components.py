import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.models import add_coherent_affine_component
from openghg_inversions.models.components import (
    LinearComponentResult,
    add_linked_linear_component,
    add_linear_component,
    add_model_data,
    add_offset_component,
    add_sigma_component,
    resolve_model_variable,
)
from openghg_inversions.models.coords import (
    CoordRegistry,
    attach_coord_registry,
    get_coord_registry,
)
from openghg_inversions.models.state_activity import prepare_linear_sensitivity
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
            prepare_linear_sensitivity(data),
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
            prepare_linear_sensitivity(data),
            data_name="hx",
            prior_args={"pdf": "lognormal", "mean": 1.5, "stdev": 0.2, "reparameterise": True},
            var_name="x",
            output_name="mu",
        )

    assert "x_latent" in model.named_vars
    assert "x" in model.named_vars
    assert result.latent is model.named_vars["x_latent"]


def test_add_linked_linear_component_applies_an_already_constructed_state() -> None:
    """A linked component applies only the state expression supplied by its caller."""
    sensitivity = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("nmeasure", "state"),
        coords={"nmeasure": [0, 1], "state": ["a", "b"]},
    )
    prepared = prepare_linear_sensitivity(sensitivity)

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        state = add_model_data(
            xr.DataArray([2.0, 3.0], dims="state", coords={"state": ["a", "b"]}),
            "linked_state",
        )
        multiplier = add_model_data(
            xr.DataArray([0.5, 2.0], dims="state", coords={"state": ["a", "b"]}),
            "emission_ratio",
        )
        linked_state = state * multiplier
        output = add_linked_linear_component(
            prepared,
            linked_state,
            data_name="linked_sensitivity",
            output_name="linked_signal",
        )
        unscaled_output = add_linked_linear_component(
            prepared,
            state,
            data_name="linked_sensitivity",
            output_name="unscaled_linked_signal",
        )

    assert output is model["linked_signal"]
    np.testing.assert_allclose(output.eval(), [13.0, 27.0])
    np.testing.assert_allclose(unscaled_output.eval(), [8.0, 18.0])


def test_add_coherent_affine_component_registers_fixed_data_and_output() -> None:
    """The affine component owns only fixed data and the summed deterministic."""
    fixed = xr.DataArray(
        [10.0, 20.0],
        dims="observation",
        coords={"observation": ["co2", "o2"]},
        name="fixed_prior_contribution",
    )

    with pm.Model() as model:
        attach_coord_registry(model, CoordRegistry())
        linear_signal = add_model_data(
            fixed.copy(data=[1.5, -2.0]),
            "linear_signal",
        )
        output = add_coherent_affine_component(
            fixed,
            linear_signal,
            output_name="modelled_concentration",
        )

    assert output is model["modelled_concentration"]
    assert "fixed_prior_contribution" in model.named_vars
    np.testing.assert_allclose(output.eval(), [11.5, 18.0])


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
            prepare_linear_sensitivity(data),
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


def test_add_offset_component_derives_frequency_indicator() -> None:
    """Check offsets derive their frequency indicator from observation time."""
    observations = xr.DataArray(
        np.ones(4),
        dims="nmeasure",
        coords={
            "site": ("nmeasure", ["MHD", "MHD", "TAC", "TAC"]),
            "time": (
                "nmeasure",
                pd.to_datetime(["2019-01-01", "2019-01-02", "2019-02-01", "2019-02-02"]),
            ),
        },
    )

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_offset_component(
            observations,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            offset_freq="monthly",
            output_name="offset",
        )
        assert "offset" in model.named_vars
        assert "offset_freq_indicator" in model.named_vars

    np.testing.assert_array_equal(model["offset_freq_indicator"].eval(), [0, 0, 1, 1])


def test_add_offset_component_requires_time_for_frequency() -> None:
    """Frequency-based offsets require an observation-aligned time coordinate."""
    observations = xr.DataArray(
        np.ones(4),
        dims="nmeasure",
        coords={"site": ("nmeasure", ["MHD", "MHD", "TAC", "TAC"])},
    )

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        with pytest.raises(ValueError, match="no observation-aligned time coordinate"):
            add_offset_component(
                observations,
                prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                offset_freq="monthly",
            )
        assert "site_indicator" not in model.named_vars


def test_add_offset_component_requires_complete_times_for_frequency() -> None:
    """Reject missing timestamps before registering offset graph state."""
    observations = xr.DataArray(
        np.ones(2),
        dims="nmeasure",
        coords={
            "site": ("nmeasure", ["MHD", "MHD"]),
            "time": ("nmeasure", [np.datetime64("2019-01-01"), np.datetime64("NaT")]),
        },
    )

    with pm.Model(coords={"nmeasure": np.arange(2)}) as model:
        attach_coord_registry(model, CoordRegistry())
        with pytest.raises(ValueError, match="complete observation timestamps"):
            add_offset_component(
                observations,
                prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                offset_freq="monthly",
            )
        assert "site_indicator" not in model.named_vars


def test_add_offset_component_derives_site_indicator_from_observations() -> None:
    """Check the offset owns site coding from labelled observations."""
    observations = xr.DataArray(
        np.ones(4),
        dims="nmeasure",
        coords=_obs_coords(),
        name="mf",
    )

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_offset_component(
            observations,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
        )

    np.testing.assert_array_equal(model["site_indicator"].eval(), [0, 0, 1, 1])
    assert model.named_vars_to_dims["offset_latent"] == ("offset_term",)
    registry = get_coord_registry(model)
    assert registry is not None
    np.testing.assert_array_equal(registry.original_coords["offset_term"], ["MHD", "TAC"])


def test_add_offset_component_requires_observation_sites() -> None:
    """Reject offset inputs without labelled observation sites."""
    observations = xr.DataArray(np.ones(4), dims="nmeasure", name="mf")

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        with pytest.raises(ValueError, match="observation-aligned `site`"):
            add_offset_component(
                observations,
                prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            )


def test_add_offset_component_supports_one_global_scalar() -> None:
    """A global offset needs only observation length, not site metadata."""
    observations = xr.DataArray(np.ones(4), dims="nmeasure", name="mf")

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        offset = add_offset_component(
            observations,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            per_site=False,
        )

    assert model.named_vars["offset_latent"].ndim == 0
    assert offset.eval().shape == (4,)
    assert "offset_design" not in model.named_vars
    assert "site_indicator" not in model.named_vars


@pytest.mark.parametrize("invalid_args", [{"offset_freq": "monthly"}, {"drop_first": True}])
def test_global_offset_rejects_site_period_options(invalid_args: dict[str, object]) -> None:
    """Global offsets reject options that only have site-design semantics."""
    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        with pytest.raises(ValueError, match="Global offsets"):
            add_offset_component(
                _likelihood_dataset()["mf"],
                prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                per_site=False,
                **invalid_args,
            )


def test_site_offset_rejects_drop_first_when_only_one_site_exists() -> None:
    """Do not allow drop-first coding to remove the only offset site."""
    observations = xr.DataArray(
        np.ones(2),
        dims="nmeasure",
        coords={"nmeasure": [0, 1], "site": ("nmeasure", ["MHD", "MHD"])},
        name="mf",
    )

    with pm.Model(coords={"nmeasure": np.arange(2)}) as model:
        attach_coord_registry(model, CoordRegistry())
        with pytest.raises(ValueError, match="removes the only available offset site"):
            add_offset_component(
                observations,
                prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
                drop_first=True,
            )


def test_add_offset_component_drop_first_and_freq_builds_expected_design() -> None:
    """Check drop-first offsets still build the expected site-period design."""
    observations = _likelihood_dataset()["mf"]

    with pm.Model(coords={"nmeasure": np.arange(4)}) as model:
        attach_coord_registry(model, CoordRegistry())
        add_offset_component(
            observations,
            prior_args={"pdf": "normal", "mu": 0.0, "sigma": 1.0},
            offset_freq="monthly",
            output_name="offset",
            drop_first=True,
        )

    offset_design = model.named_vars["offset_design"].eval()
    assert offset_design.shape == (4, 2)
    np.testing.assert_array_equal(offset_design[:2], np.zeros((2, 2)))
    np.testing.assert_array_equal(offset_design[2:], np.array([[0, 1], [0, 1]]))
    registry = get_coord_registry(model)
    assert registry is not None
    assert registry.original_coords["offset_term"].equals(
        pd.MultiIndex.from_tuples(
            [("TAC", 0), ("TAC", 1)],
            names=("offset_site", "offset_period"),
        )
    )
