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
from numpy.testing import assert_almost_equal
import pytest
import xarray as xr

from openghg_inversions.inversion_inputs import add_site_indicator, concat_gather_datasets, make_inv_inputs


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
            if k != "calculate_min_error" and k != "use_bc" and k != "min_error_options"
        },
        min_error_per_site=inv_inputs_args["min_error_options"].get("by_site", False),
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
        sigma_freq=inv_inputs_args["sigma_freq"],
        min_error=inv_inputs_args["min_error"],
        min_error_per_site=inv_inputs_args["min_error_options"].get("by_site", False),
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


# ----------------------------------------
# Tests for helper functions
# ----------------------------------------
@pytest.fixture
def gathered_ds(mhd_and_tac_fp_data) -> xr.Dataset:
    to_concat = {k: v for k, v in mhd_and_tac_fp_data.items() if not k.startswith(".")}
    return concat_gather_datasets(to_concat, key_dim="site", ragged_dim="time", stack_dim="nmeasure")


def test_add_site_indicator(gathered_ds):
    """Test adding site_indicator and site_names."""
    ds = add_site_indicator(gathered_ds, sort=False)

    assert list(ds.site_names.values) == ["MHD", "TAC"]

    assert np.all(ds.site_names.values[ds.site_indicator.values] == ds.site.values)
