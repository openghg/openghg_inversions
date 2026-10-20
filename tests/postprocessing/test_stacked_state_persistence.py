"""Component-owned coordinates survive trace persistence and sector selection."""

from dataclasses import replace

import numpy as np
import pymc as pm
import xarray as xr

from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.postprocessing import make_outputs
from openghg_inversions.rhime.multisector import build_multisector_model
from openghg_inversions.rhime.specs import FixedErrorSettings, RhimeModelSpec, SectorSpec
from openghg_inversions.serialization import load_trace, save_trace


def test_qualified_gathered_state_roundtrip_preserves_canonical_source_blocks(
    tmp_path, fake_source_specific_multisector_basis_functions, multisector_postprocessing_inv_out,
):
    basis = fake_source_specific_multisector_basis_functions()
    index = basis.operator.basis_matrix.get_index("region")
    h = xr.DataArray(
        [[1., 2.], [3., 4.]], dims=("nmeasure", "state"),
        coords={**xr.Coordinates.from_pandas_multiindex(index, "state"), "nmeasure": [0, 1]},
    )
    inputs = xr.Dataset({
        "H": h, "mf": ("nmeasure", [1., 2.]), "mf_error": ("nmeasure", [.1, .2]),
        "H_bc": xr.DataArray([[1., 2.], [3., 4.]], dims=("nmeasure", "bc_region"),
                             coords={"bc_region": ["north", "south"]}),
    })
    prior = {"pdf": "normal", "mu": 1., "sigma": .2}
    bc_prior = {**prior, "sigma": xr.DataArray(
        [.2, .3], dims="bc_region",
        coords={"bc_region": ["north", "south"], "state": ("bc_region", [10, 20])},
    )}
    spec = RhimeModelSpec(
        "ch4", "EUROPE", (SectorSpec("FF", "ff-inventory", prior, "ff"),
                          SectorSpec("Ocean", "ocean-inventory", prior, "ocean")),
        use_bc=True, bc_prior=bc_prior, likelihood=FixedErrorSettings(),
    )
    model = build_multisector_model(inputs, spec, priors={"x_ff": prior, "x_ocean": prior, "bc": bc_prior}).model
    with model:
        trace = pm.sample_prior_predictive(draws=2, var_names=["x", "bc", "mu", "mu_bc"], random_seed=913)
    registry = get_coord_registry(model)
    assert registry is not None
    trace = restore_inferencedata_coords(trace, registry)
    state_dim = model.named_vars_to_dims["x"][0]
    assert state_dim == "flux_state_vector"
    path = tmp_path / "renamed-state.nc"
    save_trace(trace, path)
    reloaded = load_trace(path)
    restored_index = reloaded["prior"]["x"].get_index(state_dim)
    assert restored_index.names == ["flux_sector", "flux_region_in_source"]
    assert restored_index.tolist() == [("FF", 0), ("Ocean", 0)]
    np.testing.assert_array_equal(reloaded["prior"].coords["flux_source"], index.get_level_values("source"))
    assert reloaded["prior"].coords["bc_state"].dims == ("bc_bc_region",)
    np.testing.assert_array_equal(reloaded["prior"].coords["bc_state"], [10, 20])
    output = multisector_postprocessing_inv_out(basis)
    output = replace(output, trace=reloaded, model_metadata={
        **output.model_metadata, "variable_roles": {"flux_scale": "x"},
    })
    for sector in make_outputs._multisector_output_sectors(output):
        selected = make_outputs._sector_scale_trace(output, sector)
        assert selected.x_prior.dims == ("chain", "draw", "region")
        np.testing.assert_array_equal(selected.region, [0])
        expected = reloaded["prior"]["x"].sel(flux_sector=sector.name)
        np.testing.assert_array_equal(selected.x_prior, expected)
