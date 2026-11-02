"""Independent regression for additive observation-coordinate ownership."""

import dask.array as da
from dask.callbacks import Callback
import pytest
import pytensor
import xarray as xr

from openghg_inversions.models.boundary_corrections import prepare_boundary_correction
from openghg_inversions.models.shared_baseline import prepare_shared_baseline
from test_boundary_correction_priors import operands
from test_shared_baseline import case


@pytest.fixture(autouse=True)
def precision():
    with pytensor.config.change_flags(floatX="float64"):
        yield


def test_additive_preparation_does_not_compute_secondary_observation_coordinates():
    obs, h, prior, prepared = case("additive")
    boundary = prepare_boundary_correction(operands(obs), obs, method="A", level_sd=5.0, pattern_sd=0.7)
    additive = (
        prepared.additive_raw.assign_coords(site=h.site, time=h.time)
        .chunk({"nmeasure": 2})
        .assign_coords(outer_source=prepared.additive_raw.outer_source)
    )
    tasks = []
    with Callback(posttask=lambda key, *_: tasks.append(key)):
        result = prepare_shared_baseline(
            h.copy(data=da.from_array(h.data, chunks=(2, 4))),
            prior,
            xr.full_like(obs, 0.2),
            boundary,
            outer=prior.mean.copy(data=[False, True, True, True]),
            sampled_outer=prior.mean.copy(data=[False, True, True, False]),
            mode="additive",
            delta_sd=5.0,
            additive_design=additive,
            additive_density_sd=prepared.additive_density_sd,
        )
    assert tasks == []
    assert isinstance(result.additive_centered.data, da.Array)
    assert isinstance(additive.site.data, da.Array)
    xr.testing.assert_allclose(result.additive_centered, prepared.additive_centered)
