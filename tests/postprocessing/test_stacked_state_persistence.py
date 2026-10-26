"""Qualified boundary coordinates and gathered flux labels survive persistence."""

from dataclasses import replace

import numpy as np
import xarray as xr

from openghg_inversions.postprocessing import make_outputs
from openghg_inversions.serialization import load_trace, save_trace


def test_boundary_qualification_roundtrip_preserves_canonical_source_blocks(
    tmp_path, fake_source_specific_multisector_basis_functions, multisector_postprocessing_inv_out,
):
    basis = fake_source_specific_multisector_basis_functions()
    index = basis.operator.basis_matrix.get_index("region")
    prior = xr.Dataset(
        {"x": (("chain", "draw", "state"), [[[0.7, 1.3], [1.1, 0.9]]])},
        coords={
            "chain": [2], "draw": [5, 7],
            **xr.Coordinates.from_pandas_multiindex(index, "state"),
            "bc_region": ["north", "south"],
            "bc_state": ("bc_region", [10, 20], {"rhime_component": "bc", "rhime_local_name": "state"}),
        },
    )
    trace = xr.DataTree.from_dict({"prior": prior})
    path = tmp_path / "boundary-qualified-state.nc"
    save_trace(trace, path)
    reloaded = load_trace(path)
    xr.testing.assert_identical(reloaded["prior"].to_dataset(), prior)
    assert reloaded["prior"]["x"].get_index("state").equals(index)
    assert reloaded["prior"].coords["bc_state"].dims == ("bc_region",)
    np.testing.assert_array_equal(reloaded["prior"].coords["bc_state"], [10, 20])
    output = multisector_postprocessing_inv_out(basis)
    output = replace(output, trace=reloaded, model_metadata={
        **output.model_metadata, "variable_roles": {"flux_scale": "x"},
    })
    expected = {"ff-inventory": [[[0.7], [1.1]]], "ocean-inventory": [[[1.3], [0.9]]]}
    for sector in make_outputs._multisector_output_sectors(output):
        selected = make_outputs._sector_scale_trace(output, sector)
        assert selected.x_prior.dims == ("chain", "draw", "region")
        np.testing.assert_array_equal(selected.region, [0])
        np.testing.assert_array_equal(selected.x_prior, expected[sector.flux_source])
