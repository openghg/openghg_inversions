"""Regression tests for inversion-input preparation.

This module is intentionally transitional.

The committed ``.npz`` fixture stores a legacy-shaped projection of
``make_inv_inputs(...)`` output so that changes to the new inversion-input
pipeline can still be checked conservatively against a frozen reference used
during the refactor.

The marked ``create_frozen`` test is developer-only. Running it rewrites the
committed frozen data file and should only be done when the reference fixture is
being intentionally refreshed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask import array as da

from openghg_inversions.inversion_inputs import (
    add_min_error,
    add_site_indicator,
    concat_gather_datasets,
    make_inv_inputs,
)
from openghg_inversions.model_error import MinimumError
from openghg_inversions.sigma import SigmaAlignment


# Helpers for saving result of make_inv_inputs
def _as_numpy(obj):
    """Convert xarray/pandas/numpy-ish objects into numpy arrays for freezing."""
    if isinstance(obj, xr.DataArray):
        return obj.values
    if isinstance(obj, xr.Dataset):
        raise TypeError("Unexpected Dataset here")
    return np.asarray(obj)


def _freeze_dict(d):
    out = {}
    for k, v in d.items():
        # keep simple scalars as 0-d arrays too
        if isinstance(v, (int, float, bool, str)) or v is None:
            out[k] = np.asarray(v, dtype=object)
        else:
            out[k] = _as_numpy(v)
    return out


def save_frozen_npz(path: Path, *, mcmc_args: dict, post_process_args: dict):
    np.savez(
        path,
        **{f"mcmc__{k}": v for k, v in _freeze_dict(mcmc_args).items()},
        **{f"post__{k}": v for k, v in _freeze_dict(post_process_args).items()},
    )


def load_frozen_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    mcmc_args = {k.replace("mcmc__", "", 1): data[k] for k in data.files if k.startswith("mcmc__")}
    post_args = {k.replace("post__", "", 1): data[k] for k in data.files if k.startswith("post__")}
    return mcmc_args, post_args


def _attach_legacy_sigma_index(
    inv_inputs: xr.Dataset,
    *,
    frequency: str | None,
    anchor_time: str | None,
) -> xr.Dataset:
    """Attach the sigma index expected by the legacy hbmcmc compatibility path."""
    alignment = SigmaAlignment.from_frequency(
        inv_inputs["site_indicator"],
        frequency=frequency,
        anchor_time=anchor_time,
    )
    return inv_inputs.assign(sigma_freq_index=alignment.period_index.rename("sigma_freq_index"))


# Helpers for comparisons
def _assert_allclose_or_equal(a, b, rtol=0, atol=0):
    a_arr = np.asanyarray(a)
    b_arr = np.asanyarray(b)

    # datetime64 / timedelta64 should use exact equality (or int-view comparison)
    if a_arr.dtype.kind in ("M", "m") or b_arr.dtype.kind in ("M", "m"):
        np.testing.assert_array_equal(a_arr, b_arr)
        return

    np.testing.assert_allclose(a_arr, b_arr, rtol=rtol, atol=atol)


def _compare_with_frozen(result: dict, frozen: dict):
    # compare keys to help with debugging
    assert set(result.keys()) == set(frozen.keys())

    for k, v in result.items():
        result_v = _freeze_dict({k: v})[k]
        frozen_v = frozen[k]
        try:
            _assert_allclose_or_equal(result_v, frozen_v, rtol=0, atol=int(1e-12))
        except AssertionError as exc:
            raise AssertionError(f"Mismatch for key {k!r}") from exc


# Regression tests against frozen data
@pytest.fixture
def inv_inputs_args(mhd_and_tac_fp_data):
    return dict(
        fp_data=mhd_and_tac_fp_data,
        sites=["MHD", "TAC"],
        start_date="2019-01-01",
        use_bc=True,
        bc_freq="3h",
        sigma_freq="3h",
        min_error="percentile",
        calculate_min_error=None,
        min_error_options={},
    )


@pytest.mark.create_frozen
def test_inversion_input_create_frozen(raw_data_path, inv_inputs_args):
    """Regenerate the committed frozen compatibility fixture.

    This test is not intended for routine runs. It exists only so developers can
    intentionally refresh the committed ``.npz`` file when the frozen reference
    for this transitional regression check needs to change.
    """
    out_name = raw_data_path / "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz"
    inv_inputs = make_inv_inputs(
        **{
            k: v
            for k, v in inv_inputs_args.items()
            if k
            not in {
                "calculate_min_error",
                "use_bc",
                "min_error_options",
                "sigma_freq",
            }
        },
        min_error_per_site=inv_inputs_args["min_error_options"].get("by_site", False),
    )
    inv_inputs = _attach_legacy_sigma_index(
        inv_inputs,
        frequency=inv_inputs_args["sigma_freq"],
        anchor_time=inv_inputs_args["start_date"],
    )
    obs_prior_factor = (
        inv_inputs.mf_prior_factor.values
        if "mf_prior_factor" in inv_inputs
        else np.zeros_like(inv_inputs.mf.values)
    )
    obs_prior_upper_level_factor = (
        inv_inputs.mf_prior_upper_level_factor.values
        if "mf_prior_upper_level_factor" in inv_inputs
        else np.zeros_like(inv_inputs.mf.values)
    )

    save_frozen_npz(
        out_name,
        mcmc_args={
            "Hx": inv_inputs.H.values,
            "Y": inv_inputs.mf.values,
            "error": inv_inputs.mf_error.values,
            "siteindicator": inv_inputs.site_indicator.values,
            "sigma_freq_index": inv_inputs.sigma_freq_index.values,
            "min_error": inv_inputs.min_error.values,
            "Hbc": inv_inputs.H_bc.values,
        },
        post_process_args={
            "Ytime": inv_inputs.time.values,
            "obs_repeatability": inv_inputs.mf_repeatability.values,
            "obs_variability": inv_inputs.mf_variability.values,
            "obs_prior_factor": obs_prior_factor,
            "obs_prior_upper_level_factor": obs_prior_upper_level_factor,
        },
    )


def test_inversion_input_hbmcmc_matches_frozen(raw_data_path, inv_inputs_args):
    """Check the current compatibility projection matches the frozen fixture."""
    frozen_path = raw_data_path / "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz"

    frozen_mcmc, frozen_post = load_frozen_npz(frozen_path)

    inv_inputs = make_inv_inputs(
        fp_data=inv_inputs_args["fp_data"],
        sites=inv_inputs_args["sites"],
        start_date=inv_inputs_args["start_date"],
        bc_freq=inv_inputs_args["bc_freq"],
        min_error=inv_inputs_args["min_error"],
        min_error_per_site=inv_inputs_args["min_error_options"].get("by_site", False),
    )
    inv_inputs = _attach_legacy_sigma_index(
        inv_inputs,
        frequency=inv_inputs_args["sigma_freq"],
        anchor_time=inv_inputs_args["start_date"],
    )

    result_mcmc = {
        "Hx": inv_inputs.H.values,
        "Y": inv_inputs.mf.values,
        "error": inv_inputs.mf_error.values,
        "siteindicator": inv_inputs.site_indicator.values,
        "sigma_freq_index": inv_inputs.sigma_freq_index.values,
        "min_error": inv_inputs.min_error.values,
        "Hbc": inv_inputs.H_bc.values,
    }
    obs_prior_factor = (
        inv_inputs.mf_prior_factor.values
        if "mf_prior_factor" in inv_inputs
        else np.zeros_like(inv_inputs.mf.values)
    )
    obs_prior_upper_level_factor = (
        inv_inputs.mf_prior_upper_level_factor.values
        if "mf_prior_upper_level_factor" in inv_inputs
        else np.zeros_like(inv_inputs.mf.values)
    )
    result_post = {
        "Ytime": inv_inputs.time.values,
        "obs_repeatability": inv_inputs.mf_repeatability.values,
        "obs_variability": inv_inputs.mf_variability.values,
        "obs_prior_factor": obs_prior_factor,
        "obs_prior_upper_level_factor": obs_prior_upper_level_factor,
    }

    _compare_with_frozen(result_mcmc, frozen_mcmc)
    _compare_with_frozen(result_post, frozen_post)


def test_make_inv_inputs_preserves_requested_order_and_column_factors() -> None:
    """Mixed inputs retain requested site order and column-only factors."""

    def site_dataset(
        site: str,
        value: float,
        *,
        prior_factor: float | None = None,
        upper_factor: float | None = None,
    ) -> xr.Dataset:
        """Build one observation-aligned site dataset for gathering."""
        time = pd.date_range("2019-01-01", periods=1, freq="h")
        dataset = xr.Dataset(
            {
                "H": (("region", "time"), [[value]]),
                "mf": ("time", [value]),
                "mf_error": ("time", [0.1]),
                "mf_repeatability": ("time", [0.1]),
                "mf_variability": ("time", [0.0]),
            },
            coords={"region": [0], "time": time},
            attrs={"site": site},
        )
        if prior_factor is not None:
            dataset["mf_prior_factor"] = xr.DataArray(
                [prior_factor],
                dims="time",
                attrs={"long_name": "column prior factor"},
            )
        if upper_factor is not None:
            dataset["mf_prior_upper_level_factor"] = xr.DataArray(
                [upper_factor],
                dims="time",
                attrs={"long_name": "upper-level prior factor"},
            )
        return dataset

    fp_data = {
        "SURFACE": site_dataset("SURFACE", 1.0),
        "SATELLITE": site_dataset(
            "SATELLITE",
            2.0,
            prior_factor=0.2,
            upper_factor=0.3,
        ),
    }

    result = make_inv_inputs(
        fp_data,
        sites=["SATELLITE", "SURFACE"],
        min_error=0.0,
    )

    assert result["site_names"].values.tolist() == ["SATELLITE", "SURFACE"]
    assert result["site"].values.tolist() == ["SATELLITE", "SURFACE"]
    np.testing.assert_allclose(result["mf"].values, [2.0, 1.0])
    np.testing.assert_allclose(result["mf_prior_factor"].values, [0.2, 0.0])
    np.testing.assert_allclose(result["mf_prior_upper_level_factor"].values, [0.3, 0.0])
    assert result["mf_prior_factor"].attrs["long_name"] == "column prior factor"
    assert result["mf_prior_upper_level_factor"].attrs["long_name"] == "upper-level prior factor"


def test_make_inv_inputs_rejects_partial_column_factor_pair() -> None:
    """Column datasets cannot silently omit one prior-factor variable."""
    time = pd.date_range("2019-01-01", periods=1, freq="h")
    dataset = xr.Dataset(
        {
            "H": (("region", "time"), [[1.0]]),
            "mf": ("time", [2.0]),
            "mf_error": ("time", [0.1]),
            "mf_repeatability": ("time", [0.1]),
            "mf_variability": ("time", [0.0]),
            "mf_prior_factor": ("time", [0.2]),
        },
        coords={"region": [0], "time": time},
    )

    with pytest.raises(ValueError, match="must define both"):
        make_inv_inputs({"SATELLITE": dataset}, sites=["SATELLITE"], min_error=0.0)


# ----------------------------------------
# Tests for helper functions
# ----------------------------------------
@pytest.fixture
def gathered_ds(mhd_and_tac_fp_data) -> xr.Dataset:
    to_concat = {k: v for k, v in mhd_and_tac_fp_data.items() if not k.startswith(".")}
    return concat_gather_datasets(
        to_concat,
        key_dim="site",
        ragged_dim="time",
        stack_dim="nmeasure",
        missing_data_vars="drop",
    )


def test_add_site_indicator(gathered_ds):
    """Test adding site_indicator and site_names."""
    ds = add_site_indicator(gathered_ds, sort=False)

    assert list(ds.site_names.values) == ["MHD", "TAC"]

    assert np.all(ds.site_names.values[ds.site_indicator.values] == ds.site.values)


def _make_minimal_fp_site(*, mf_base: float, include_inlet_height: bool) -> xr.Dataset:
    """Create a tiny per-site dataset for `make_inv_inputs` tests."""
    time = xr.DataArray(pd.date_range("2020-01-01", periods=2, freq="1h"), dims="time", name="time")
    region = xr.DataArray(["r0", "r1"], dims="region", name="region")
    data_vars = {
        "mf": xr.DataArray(np.array([mf_base, mf_base + 1.0]), dims="time", coords={"time": time}),
        "mf_mod": xr.DataArray(np.array([mf_base, mf_base + 1.0]), dims="time", coords={"time": time}),
        "mf_error": xr.DataArray(np.array([0.1, 0.2]), dims="time", coords={"time": time}),
        "mf_repeatability": xr.DataArray(np.array([0.05, 0.05]), dims="time", coords={"time": time}),
        "mf_variability": xr.DataArray(np.array([0.05, 0.15]), dims="time", coords={"time": time}),
        "H": xr.DataArray(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            dims=("region", "time"),
            coords={"region": region, "time": time},
        ),
    }
    if include_inlet_height:
        data_vars["inlet_height"] = xr.DataArray(np.array([100.0, 100.0]), dims="time", coords={"time": time})

    return xr.Dataset(data_vars)


def test_make_inv_inputs_drops_non_shared_data_vars():
    """`make_inv_inputs` should drop optional non-shared vars before gathering."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=True),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }

    with pytest.warns(UserWarning, match="Dropping data variables.*inlet_height"):
        result = make_inv_inputs(fp_data=fp_data, sites=["AAA", "BBB"], min_error=0.0)

    assert "inlet_height" not in result
    assert {"H", "mf", "mf_error", "site_indicator", "site_names", "min_error"} <= set(result.data_vars)
    assert "sigma_freq_index" not in result


def test_make_inv_inputs_rejects_explicit_empty_sites() -> None:
    """An explicit empty site selection must not silently expand to all sites."""
    fp_data = {"AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False)}

    with pytest.raises(ValueError, match="sites.*at least one site"):
        make_inv_inputs(fp_data=fp_data, sites=[], min_error=0.0)


def test_make_inv_inputs_infers_sites_only_when_sites_is_none() -> None:
    """A None site selection still infers every non-metadata site."""
    fp_data = {
        ".species": "CH4",
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }

    result = make_inv_inputs(fp_data=fp_data, sites=None, min_error=0.0)

    assert list(result["site_names"].values) == ["AAA", "BBB"]


def test_make_inv_inputs_rejects_missing_requested_site() -> None:
    """A missing requested site raises a clear error before dataset gathering."""
    fp_data = {"AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False)}

    with pytest.raises(ValueError, match=r"missing requested site\(s\).*BBB"):
        make_inv_inputs(fp_data=fp_data, sites=["AAA", "BBB"], min_error=0.0)


def test_make_inv_inputs_rejects_mismatched_gathered_state_indexes() -> None:
    """Site gathering must not outer-align different source/region state layouts."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }
    state_indexes = {
        "AAA": pd.MultiIndex.from_tuples(
            [("ff", 0), ("ff", 1), ("ocean", 0)],
            names=["source", "region_in_source"],
        ),
        "BBB": pd.MultiIndex.from_tuples(
            [("ff", 0), ("ocean", 0)],
            names=["source", "region_in_source"],
        ),
    }
    for site, state_index in state_indexes.items():
        time = fp_data[site].coords["time"]
        fp_data[site]["H"] = xr.DataArray(
            np.ones((len(state_index), time.size)),
            dims=("state", "time"),
            coords={
                **xr.Coordinates.from_pandas_multiindex(state_index, "state"),
                "time": time,
            },
        )

    with pytest.raises(ValueError, match="identical indexes on every non-time dimension"):
        make_inv_inputs(fp_data=fp_data, sites=["AAA", "BBB"], min_error=0.0)


def test_make_inv_inputs_rejects_mismatched_state_dimension_names() -> None:
    """Site gathering must not broadcast differently named state dimensions."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }
    for site, state_dim in (("AAA", "state"), ("BBB", "region")):
        time = fp_data[site].coords["time"]
        fp_data[site]["H"] = xr.DataArray(
            np.ones((2, time.size)),
            dims=(state_dim, "time"),
            coords={state_dim: [0, 1], "time": time},
        )

    with pytest.raises(ValueError, match="variable 'H'.*same non-time dimensions"):
        make_inv_inputs(fp_data=fp_data, sites=["AAA", "BBB"], min_error=0.0)


def test_make_inv_inputs_retains_separate_aggregation_error_component() -> None:
    """Canonical inputs retain aggregation error without modifying raw mf_error."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }
    for site, aggregation_error in (("AAA", [0.3, 0.4]), ("BBB", [0.5, 0.6])):
        fp_data[site]["aggregation_error_sd"] = ("time", aggregation_error)
    raw_errors = np.concatenate([fp_data[site]["mf_error"].values for site in ("AAA", "BBB")])

    result = make_inv_inputs(fp_data=fp_data, sites=["AAA", "BBB"], min_error=0.2)

    np.testing.assert_allclose(result["mf_error"].values, raw_errors)
    np.testing.assert_allclose(result["aggregation_error_sd"].values, [0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(result["min_error"].values, 0.2)


def test_make_inv_inputs_raises_if_required_var_would_be_dropped():
    """`make_inv_inputs` should still fail clearly if a required var is not shared."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False).drop_vars("mf_error"),
    }

    with pytest.raises(ValueError, match="Required inversion data variables.*mf_error"):
        make_inv_inputs(fp_data=fp_data, sites=["AAA", "BBB"], min_error=0.0)


def test_make_inv_inputs_accepts_integer_min_error():
    """Integer min_error values should be treated as numeric scalar errors."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }

    result = make_inv_inputs(fp_data=fp_data, sites=["AAA", "BBB"], min_error=40)

    assert np.all(result.min_error.values == 40.0)


def test_make_inv_inputs_maps_dict_min_error_by_site():
    """Site-specific min_error mappings should align onto selected stacked observations."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
        "CCC": _make_minimal_fp_site(mf_base=30.0, include_inlet_height=False),
    }

    result = make_inv_inputs(
        fp_data=fp_data,
        sites=["AAA", "BBB"],
        min_error={"AAA": 1.5, "BBB": 2.5},
    )

    expected = np.where(result.site_indicator.values == 0, 1.5, 2.5)
    np.testing.assert_allclose(result.min_error.values, expected)


def test_add_min_error_can_use_site_coord_without_site_indicator():
    """Standalone min_error setup can derive site info from a site coordinate."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }
    ds = xr.Dataset(
        {"mf": xr.DataArray(np.ones(4), dims="nmeasure")},
        coords={"site": xr.DataArray(["AAA", "AAA", "BBB", "BBB"], dims="nmeasure")},
    )

    result = add_min_error(ds, fp_data=fp_data, min_error={"AAA": 1.5, "BBB": 2.5})

    np.testing.assert_allclose(result.min_error.values, [1.5, 1.5, 2.5, 2.5])


def test_make_inv_inputs_dict_min_error_missing_site_raises_clear_error():
    """Site-specific min_error mappings should fail clearly when a selected site is missing."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
    }

    with pytest.raises(ValueError, match="min_error mapping is missing values.*BBB"):
        make_inv_inputs(
            fp_data=fp_data,
            sites=["AAA", "BBB"],
            min_error={"AAA": 1.5},
        )


def test_make_inv_inputs_residual_min_error_can_be_site_specific():
    """Residual min_error values can be calculated per selected site."""
    fp_data = {
        "AAA": _make_minimal_fp_site(mf_base=10.0, include_inlet_height=False),
        "BBB": _make_minimal_fp_site(mf_base=20.0, include_inlet_height=False),
        "CCC": _make_minimal_fp_site(mf_base=30.0, include_inlet_height=False),
    }
    fp_data["AAA"]["mf_mod"] = fp_data["AAA"]["mf"] - xr.DataArray([0.0, 2.0], dims="time")
    fp_data["BBB"]["mf_mod"] = fp_data["BBB"]["mf"] - xr.DataArray([0.0, 4.0], dims="time")
    fp_data["CCC"]["mf_mod"] = fp_data["CCC"]["mf"] - xr.DataArray([0.0, 8.0], dims="time")

    result = make_inv_inputs(
        fp_data=fp_data,
        sites=["AAA", "BBB"],
        min_error="residual",
        min_error_per_site=True,
    )

    expected = np.where(result.site_indicator.values == 0, 1.0, 2.0)
    np.testing.assert_allclose(result.min_error.values, expected)


def test_minimum_error_uses_declared_site_order_and_records_provenance(tmp_path: Path):
    """Per-site values follow labels rather than mapping or alphabetical order."""
    observations = xr.Dataset(
        {"mf": ("nmeasure", [1.0, 2.0, 3.0])},
        coords={"site": ("nmeasure", ["BBB", "AAA", "BBB"])},
    )
    observations.mf.attrs["units"] = "ppb"

    result = MinimumError.prepare(observations, {}, {"AAA": 1.0, "BBB": 2.0})

    np.testing.assert_allclose(result.values, [2.0, 1.0, 2.0])
    assert result.sites == ("BBB", "AAA")
    assert result.values.attrs == {
        "units": "ppb",
        "minimum_error_method": "per_site",
        "minimum_error_by_site": 1,
        "minimum_error_sites": "BBB,AAA",
    }
    result.values.to_netcdf(tmp_path / "minimum_error.nc")


@pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
def test_minimum_error_rejects_invalid_values(value: float):
    observations = xr.Dataset({"mf": ("nmeasure", [1.0])})

    with pytest.raises(ValueError, match="finite and non-negative"):
        MinimumError.prepare(observations, {}, value)


def test_scalar_minimum_error_preserves_lazy_borrowed_observations():
    mf = xr.DataArray(da.from_array(np.ones(3, dtype=int), chunks=2), dims="nmeasure")
    observations = xr.Dataset({"mf": mf})

    result = MinimumError.prepare(observations, {}, 0.5)

    assert result.values.variable._data is not observations.mf.variable._data
    assert hasattr(result.values.data, "chunks")
    assert np.issubdtype(result.values.dtype, np.floating)
    np.testing.assert_allclose(result.values.compute(), 0.5)


def test_minimum_error_requires_preparation():
    with pytest.raises(TypeError, match=r"MinimumError\.prepare"):
        MinimumError()
