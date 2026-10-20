"""Scientific support and alignment contracts at the nested prepare boundary."""

from dataclasses import replace

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.inversion_data import SiteOptions
from openghg_inversions.rhime.nested import (
    _NestedDomainData,
    _inner_measurement_positions,
    align_inner_merged_to_outer_observations,
    mask_outer_merged_for_inner_domain,
)


def _data(lat, lon, *, times=None):
    times = pd.to_datetime(["2019-01-01"] if times is None else times)
    values = da.ones((len(times), len(lat), len(lon)), chunks="auto")
    dataset = xr.Dataset(
        {"fp": (("time", "lat", "lon"), values)},
        coords={"time": times, "lat": lat, "lon": lon},
    )
    return _NestedDomainData(
        site_data={"A": dataset, "B": dataset.copy(deep=False)},
        flux_data={"inventory": dataset.rename({"fp": "flux"})},
        site_options=SiteOptions.from_inputs(sites=["A", "B"], averaging_period=["1h", "1h"]),
    )


@pytest.mark.parametrize("domain", ["outer", "inner"])
def test_nested_rejects_different_site_grids(domain):
    outer = _data([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
    inner = _data([0.5, 1.5], [0.5, 1.5])
    chosen = outer if domain == "outer" else inner
    datasets = dict(chosen.site_data)
    datasets["B"] = datasets["B"].assign_coords(lat=datasets["B"].lat + 0.25)
    chosen = replace(chosen, site_data=datasets)
    with pytest.raises(ValueError, match=f"{domain.capitalize()}.*same indexed lat/lon grid"):
        mask_outer_merged_for_inner_domain(chosen if domain == "outer" else outer,
                                          chosen if domain == "inner" else inner)


def test_shared_inner_extent_masks_each_actual_target_grid_without_computation():
    outer = _data([2.0, 1.0, 0.0], [0.0, 1.0, 2.0])
    inner = _data([0.5, 1.0, 1.5], [0.5, 1.0, 1.5])
    flux = xr.DataArray(
        da.ones((4, 4), chunks=(2, 2)), dims=("lat", "lon"),
        coords={"lat": [0.0, 0.5, 1.5, 2.0], "lon": [0.0, 0.5, 1.5, 2.0]},
    )
    outer = replace(outer, flux_data={"inventory": flux.rename("flux").to_dataset()})
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        masked = mask_outer_merged_for_inner_domain(outer, inner)
    assert not tasks
    for site in outer.sites:
        np.testing.assert_array_equal(masked.site_data[site].fp.compute(),
                                      [[[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
        assert set(outer.site_data[site].fp.data.dask).issubset(masked.site_data[site].fp.data.dask)
        np.testing.assert_array_equal(outer.site_data[site].fp.compute(), np.ones((1, 3, 3)))
    np.testing.assert_array_equal(masked.flux_data["inventory"].flux.compute(),
                                  [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])


def test_unsorted_site_local_alignment_is_reapplied_after_assembly_drops_rows():
    outer = _data([0.0], [0.0], times=["2019-01-01T02:00", "2019-01-01T00:00"])
    inner = _data([0.0], [0.0], times=["2019-01-01T01:10", "2019-01-01T00:10", "2019-01-01T02:10"])
    inner = replace(inner, site_data={site: dataset.assign(
        fp=dataset.fp * xr.DataArray([10, 20, 30], dims="time"))
        for site, dataset in inner.site_data.items()})
    aligned = align_inner_merged_to_outer_observations(outer, inner, time_tolerance="15min")
    for site in outer.sites:
        np.testing.assert_array_equal(aligned.site_data[site].fp.compute().values[:, 0, 0], [30, 20])
    # Assembly can remove different rows at each site. The prepared boundary
    # selects the surviving rows again rather than reusing acquisition positions.
    outer_index = pd.MultiIndex.from_tuples(
        [("B", pd.Timestamp("2019-01-01T00:00")), ("A", pd.Timestamp("2019-01-01T02:00"))],
        names=["site", "time"],
    )
    inner_index = pd.MultiIndex.from_tuples(
        [(site, time) for site in ["A", "B"] for time in aligned.site_data[site].time.values],
        names=["site", "time"],
    )
    assert _inner_measurement_positions(outer_index, inner_index, time_tolerance=None) == [3, 0]
