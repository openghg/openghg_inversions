"""Tests for reconstructing observation-aligned sigma posterior values."""

import arviz as az
import numpy as np
import xarray as xr

from openghg_inversions.postprocessing.sigma import reconstruct_sigma_aligned


def test_reconstruct_sigma_aligned_uses_registered_component_indexes() -> None:
    """Reconstruct sigma on observations while preserving posterior coordinates."""
    sigma = xr.DataArray(
        np.arange(12, dtype=float).reshape(1, 2, 2, 3),
        dims=("chain", "draw", "nsigma_site", "nsigma_time"),
        name="sigma",
    )
    model_data = xr.Dataset(
        {
            "sigma_site_index": ("nmeasure", np.array([0, 1, 0])),
            "sigma_period_index": ("nmeasure", np.array([2, 0, 1])),
        },
        coords={"nmeasure": ["MHD-1", "TAC-1", "MHD-2"]},
    )
    trace = az.InferenceData(
        posterior=xr.Dataset({"sigma": sigma}),
        constant_data=model_data,
    )

    actual = reconstruct_sigma_aligned(trace)
    expected = sigma.isel(
        nsigma_site=model_data["sigma_site_index"],
        nsigma_time=model_data["sigma_period_index"],
    ).rename("sigma_aligned")

    xr.testing.assert_identical(actual, expected)
