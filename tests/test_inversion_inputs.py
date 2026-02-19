"""Test functions for creating inputs for PyMC."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_inputs import _transform_bc_freq

from openghg_inversions.basis import basis_functions_wrapper
from openghg_inversions.inversion_data.get_data import data_processing_surface_notracer
from openghg_inversions.hbmcmc.hbmcmc import make_inv_inputs


@pytest.fixture
def fp_data(mhd_and_tac_ch4_data_args):
    fp_all, *_ = data_processing_surface_notracer(**mhd_and_tac_ch4_data_args)

    basis_args = {
        "species": "ch4",
        "domain": "EUROPE",
        "start_date": "2019-01-01",
        "emissions_name": ["total-ukghg-edgar7"],
        "nbasis": 20,
        "use_bc": True,
        "basis_algorithm": "weighted",
        "bc_basis_case": "NESW",
    }

    fp_data = basis_functions_wrapper(fp_all, **basis_args)

    return fp_data


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


def compare_with_frozen(result: dict, frozen: dict):
    # compare keys to help with debugging
    assert set(result.keys()) == set(frozen.keys())

    for k, v in result.items():
        result_v = _freeze_dict({k: v})[k]
        frozen_v = frozen[k]
        _assert_allclose_or_equal(result_v, frozen_v, rtol=0, atol=0)


# Regression tests against frozen data
@pytest.fixture
def inv_inputs_args(fp_data):
    return dict(
        fp_data=fp_data,
        sites=["MHD", "TAC"],
        dropped_sites=[],
        start_date="2019-01-01",
        end_date="2019-01-02",
        use_bc=True,
        bc_freq="3h",
        sigma_freq="3h",
        min_error="percentile",
        calculate_min_error=None,
        min_error_options={},
    )


@pytest.mark.create_frozen
def test_inversion_input_create_frozen(raw_data_path, inv_inputs_args):
    """This 'test' just regenerates frozen data for use in other tests."""
    mcmc_args, post_args = make_inv_inputs(**inv_inputs_args)

    out_name = raw_data_path / "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz"
    save_frozen_npz(out_name, mcmc_args=mcmc_args, post_process_args=post_args)


def test_inversion_input_hbmcmc_matches_frozen(raw_data_path, inv_inputs_args):
    """Test that result of make_inv_inputs from hbmcmc.py matches frozen data."""
    frozen_path = raw_data_path / "frozen_mhd_tac_make_inv_inputs_hbmcmc.npz"

    frozen_mcmc, frozen_post = load_frozen_npz(frozen_path)

    mcmc_args, post_args = make_inv_inputs(**inv_inputs_args)

    compare_with_frozen(mcmc_args, frozen_mcmc)
    compare_with_frozen(post_args, frozen_post)
