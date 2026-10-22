"""Model-owned alignments consume retained observations and reusable raw inputs."""

from dataclasses import replace
from types import SimpleNamespace

import dask.array as da
from dask import delayed
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_inputs import make_inv_inputs
from openghg_inversions.model_error import MinimumError
from openghg_inversions.rhime.nested import materialize_nested_inputs
from openghg_inversions.rhime.specs import (
    AdditiveSigmaSettings, FixedErrorSettings, PollutionEventSettings, RhimeModelSpec, SectorSpec,
)
from openghg_inversions.rhime.standard import materialize_standard_inputs


def _raw() -> xr.Dataset:
    """Keep nonmonotonic times and remove a valid-mf row with invalid H."""
    time = pd.to_datetime(["2020-02-01", "2020-01-01", "2020-01-02"])
    sites = {}
    for site, values in (("BBB", [6., 2., 1000.]), ("AAA", [12., 4., 2000.])):
        sites[site] = xr.Dataset(
            {
                "mf": ("time", values, {"units": "ppb"}),
                "mf_mod": ("time", np.zeros(3), {"units": "ppb"}),
                "bc_mod": ("time", np.full(3, .001), {"units": "ppm"}),
                "mf_error": ("time", np.ones(3)),
                "mf_repeatability": ("time", np.ones(3)),
                "mf_variability": ("time", np.zeros(3)),
                "H": (("region", "time"), [[1., 1., np.nan]]),
                "H_bc": (("bc_region", "time"), [[1., 2., 3.], [0., 0., 0.]]),
            },
            coords={"time": time, "region": [0], "bc_region": ["north", "unused"]},
        )
    result = make_inv_inputs(sites)
    result["bc_diagnostic"] = ("bc_region", [11., 22.])
    return result


def _spec(likelihood=None, **kwargs) -> RhimeModelSpec:
    return RhimeModelSpec(
        species="ch4", domain="EUROPE", sectors=(SectorSpec("total", "total", {"pdf": "normal", "mu": 1., "sigma": .2}, "total"),),
        likelihood=likelihood, **kwargs,
    )


def test_one_raw_preparation_supports_two_frequencies_and_preserves_bc_auxiliaries():
    raw = _raw()
    original = raw.copy(deep=True)
    for var in raw.data_vars.values():
        if isinstance(var.data, np.ndarray):
            var.data.flags.writeable = False
    prepared = SimpleNamespace(inv_inputs=raw)
    monthly, _ = materialize_standard_inputs(prepared, _spec(bc_freq="monthly"))
    constant, _ = materialize_standard_inputs(prepared, _spec())

    assert "min_error" not in raw
    assert raw.H_bc.attrs["boundary_sensitivity_preparation"] == "raw-spatial"
    assert monthly.sizes["bc_region"] == 4
    assert constant.sizes["bc_region"] == 2
    np.testing.assert_allclose(monthly.H_bc.sum("bc_region"), raw.H_bc.sum("bc_region"))
    xr.testing.assert_equal(monthly.bc_diagnostic.rename(raw_bc_region="bc_region"), raw.bc_diagnostic)
    assert list(monthly.raw_bc_region.values) == ["north", "unused"]
    xr.testing.assert_identical(raw, original)


@pytest.mark.parametrize(
    ("value", "options", "expected"),
    [
        (3., {}, [3., 3., 3., 3.]),
        ({"AAA": 5., "BBB": 7.}, {}, [7., 7., 5., 5.]),
        ("residual", {}, np.full(4, np.std([5., 1., 11., 3.]))),
        ("residual", {"by_site": True}, [2., 2., 4., 4.]),
        ("percentile", {}, [0., 0., 0., 0.]),
    ],
)
def test_floor_uses_only_retained_observations(value, options, expected):
    raw = _raw()
    spec = _spec(PollutionEventSettings(min_error=value, min_error_options=options))
    result, _ = materialize_standard_inputs(SimpleNamespace(inv_inputs=raw), spec)

    np.testing.assert_allclose(result.min_error, expected)
    assert result.min_error.attrs["units"] == "ppb"
    assert list(result.site.values) == ["BBB", "BBB", "AAA", "AAA"]
    np.testing.assert_array_equal(result.time, raw.time)
    assert "min_error" not in raw


def test_residual_converts_modelled_units_and_can_omit_bc_mod():
    observations = xr.Dataset(
        {"mf": ("nmeasure", [2., 8.], {"units": "ppb"}),
         "mf_mod": ("nmeasure", [.001, .003], {"units": "ppm"}),
         "bc_mod": ("nmeasure", [.001, .002], {"units": "ppm"})},
        coords={"site": ("nmeasure", ["AAA", "AAA"])},
    )
    result = MinimumError.from_observations(observations, "residual", by_site=False)
    np.testing.assert_allclose(result.values, [1.5, 1.5])
    result = MinimumError.from_observations(observations.drop_vars("bc_mod"), "residual")
    np.testing.assert_allclose(result.values, [2., 2.])


@pytest.mark.parametrize("likelihood", [None, FixedErrorSettings(), AdditiveSigmaSettings(min_error="residual")])
def test_unused_floor_and_disabled_bc_are_never_evaluated(monkeypatch, likelihood):
    raw = _raw().drop_vars("mf_mod")

    @delayed
    def unused():
        raise AssertionError("Unused model operands must remain lazy")

    raw["bc_mod"] = ("nmeasure", da.from_delayed(unused(), shape=(4,), dtype=float))
    raw["H_bc"] = (("bc_region", "nmeasure"), da.from_delayed(unused(), shape=(2, 4), dtype=float))

    def forbidden(*args, **kwargs):
        raise AssertionError("Floor estimator must not be called")

    monkeypatch.setattr(MinimumError, "from_observations", forbidden)
    result, _ = materialize_standard_inputs(SimpleNamespace(inv_inputs=raw), _spec(likelihood, use_bc=False))
    assert "min_error" not in result
    assert isinstance(result.H_bc.data, da.Array)
    assert isinstance(result.bc_mod.data, da.Array)
    assert "raw_bc_region" not in result.dims


def test_additive_floor_is_opt_in_and_percentile_does_not_select_modelled_arrays():
    raw = _raw().drop_vars(["mf_mod", "bc_mod"])
    result, _ = materialize_standard_inputs(
        SimpleNamespace(inv_inputs=raw),
        _spec(AdditiveSigmaSettings(use_minimum_error_floor=True, min_error="percentile")),
    )
    np.testing.assert_allclose(result.min_error, 0.)


def test_nested_materialization_uses_the_same_model_owned_alignments():
    raw = _raw()
    raw["H_inner"] = (("inner_region", "nmeasure"), raw.H.data)
    result, priors = materialize_nested_inputs(
        SimpleNamespace(inv_inputs=raw), _spec(PollutionEventSettings(min_error=2.), bc_freq="monthly"),
    )
    assert result.sizes["bc_region"] == 4
    np.testing.assert_allclose(result.min_error, 2.)
    assert {"x_outer", "x_inner"} <= priors.keys()


def test_in_memory_baked_boundary_input_is_rejected():
    raw = _raw()
    raw.H_bc.attrs["boundary_sensitivity_preparation"] = "sampled-period-expansion"
    with pytest.raises(ValueError, match="raw spatial H_bc"):
        materialize_standard_inputs(SimpleNamespace(inv_inputs=raw), _spec())


def test_percentile_uses_retained_nonmonotonic_monthly_samples():
    raw = _raw().reset_index("nmeasure")
    raw = raw.assign_coords(time=("nmeasure", pd.to_datetime([
        "2020-01-03", "2020-01-01", "2020-01-04", "2020-01-02",
    ])))
    raw = raw.set_index(nmeasure=["site", "time"])
    result, _ = materialize_standard_inputs(
        SimpleNamespace(inv_inputs=raw), _spec(PollutionEventSettings(min_error="percentile")),
    )
    np.testing.assert_allclose(result.min_error, [1.8, 1.8, 3.6, 3.6])
    xr.testing.assert_equal(result.nmeasure, raw.nmeasure)


def test_selected_operands_auxiliary_coordinates_and_priors_share_one_compute():
    executions = []

    @delayed
    def shared():
        executions.append("shared")
        return np.array([1., 2., 3., 4.])

    values = da.from_delayed(shared(), shape=(4,), dtype=float)
    time = np.asarray(["2020-01-01", "2020-01-02", "2020-02-01", "2020-02-02"], dtype="datetime64[ns]")
    raw = xr.Dataset(
        {
            "H": (("region", "nmeasure"), values[None, :]),
            "H_bc": (("bc_region", "nmeasure"), values[None, :]),
            "mf": ("nmeasure", values * 4),
            "mf_mod": ("nmeasure", values),
            "mf_error": ("nmeasure", values * .1),
            "extension": ("nmeasure", values + 20),
        },
        coords={
            "nmeasure": np.arange(4), "region": [0], "bc_region": ["north"],
            "site": ("nmeasure", da.from_array(np.array(["AAA"] * 4), chunks=2)),
            "time": ("nmeasure", da.from_array(time, chunks=2)),
        },
    )
    spec = _spec(PollutionEventSettings(min_error="residual"), bc_freq="monthly")
    spec = replace(spec, sectors=(replace(spec.sectors[0], x_prior={
        "pdf": "normal", "mu": values[0], "sigma": .2,
    }),))
    result, priors = materialize_standard_inputs(SimpleNamespace(inv_inputs=raw), spec)

    assert executions == ["shared"]
    assert not isinstance(result.H_bc.data, da.Array)
    assert not isinstance(result.time.data, da.Array)
    assert not isinstance(result.site.data, da.Array)
    assert isinstance(result.extension.data, da.Array)
    assert isinstance(raw.H.data, da.Array)
    assert priors["x"]["mu"] == 1.
    np.testing.assert_allclose(result.min_error, np.std([3., 6., 9., 12.]))
