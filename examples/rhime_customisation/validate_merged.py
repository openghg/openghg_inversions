"""Inspect and validate acquired custom arrays without executing their graphs.

Run from the repository with ``uv run python examples/rhime_customisation/validate_merged.py``.
"""

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import xarray as xr

from openghg_inversions.inversion_data import MergedData, SiteOptions


def main() -> None:
    """Validate metadata, then demonstrate a useful site/variable error."""
    @delayed
    def deferred_response():
        raise AssertionError("This example must not execute numerical payloads.")

    site = xr.Dataset(
        {
            "mf": ("time", [400., 401.], {"units": "ppm"}),
            "mf_error": ("time", [.1, .2], {"units": "umol/mol"}),
            "fp_x_flux": (
                ("time", "lat", "lon"), da.from_delayed(deferred_response(), shape=(2, 2, 2), dtype=float),
            ),
            "custom_diagnostic": ("time", [1., 2.]),
        },
        coords={"time": pd.date_range("2020-01-01", periods=2, freq="h"),
                "lat": [50., 51.], "lon": [-2., -1.]},
    )
    flux = xr.DataArray(
        np.ones((2, 2)), dims=("lat", "lon"), coords={"lat": site.lat, "lon": site.lon},
        attrs={"units": "mol m-2 s-1"}, name="flux",
    )
    merged = MergedData(
        site_data={"TAC": site}, flux_data={"inventory": flux.to_dataset()},
        site_options=SiteOptions.from_inputs(sites=["TAC"], averaging_period="1h"),
    )
    schema = merged.site_schema(require_footprints=False, use_bc=False)
    print("Declared site fields:", tuple(schema.data_vars))
    choices = dict(split_by_sectors=False, require_footprints=False, use_bc=False, flux_sources=("inventory",))

    def unexpected_execution(*args):
        raise AssertionError("Metadata validation executed an array graph.")

    with Callback(pretask=unexpected_execution):
        merged.validate_scientific_inputs(**choices)
    print("Valid acquired metadata; arrays and custom fields remain borrowed.")

    # Work on a new Dataset container when changing external scientific metadata.
    invalid_site = site.copy(deep=False)
    invalid_site.mf_error.attrs = {"units": "ppb"}
    invalid = merged.with_site_data({"TAC": invalid_site})
    try:
        invalid.validate_scientific_inputs(**choices)
    except ValueError as error:
        print("Expected unit error:", error)
    else:
        raise AssertionError("A numerical-scale mismatch was accepted.")
    assert site.mf_error.attrs["units"] == "umol/mol"


if __name__ == "__main__":
    main()
