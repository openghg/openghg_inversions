"""Scientific alignment, reduction and metadata contracts for derived summaries."""

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.postprocessing import stats as module


@pytest.mark.parametrize("axis", ["draw", "point"])
def test_companions_do_not_truncate_derived_coordinates(axis):
    """A companion with fewer labels cannot discard samples or output locations."""
    if axis == "draw":
        operand = xr.DataArray([0.0, 10.0, 20.0], dims="draw", coords={"draw": [10, 20, 30]})
        companions = xr.Dataset({"other": ("draw", [1.0, 3.0])}, coords={"draw": [10, 20]})
    else:
        operand = xr.DataArray(
            np.tile([0.0, 10.0, 20.0], (4, 1)),
            dims=("draw", "point"),
            coords={"point": [10, 20, 30]},
        )
        companions = xr.Dataset(
            {"other": (("draw", "point"), np.ones((4, 1)))}, coords={"point": [10]}
        )
    borrowed = operand.data
    companions_before = companions.copy(deep=True)
    expected = xr.merge(
        [
            module._calculate_stats(operand.to_dataset(name="derived"), ["mean"]),
            module._calculate_stats(companions, ["mean"]),
        ],
        join="outer",
    )

    actual = module.summarize_derived(
        {"value": operand}, lambda block: block.value.to_dataset(name="derived"),
        stats=["mean"], companions=companions,
    )

    xr.testing.assert_identical(actual, expected)
    assert operand.data is borrowed
    xr.testing.assert_identical(companions, companions_before)


@pytest.mark.parametrize("offset_draw_chunk", [64, 100], ids=["matching-samples", "different-samples"])
def test_derived_baseline_preserves_exact_pooled_sample_reductions(offset_draw_chunk):
    """A renamed derived output keeps the native chain-pooling reduction tree."""
    values = np.random.default_rng(19).normal(1950, 200, size=(2, 259, 129)).astype("float32")
    boundary = xr.DataArray(values, dims=("chain", "draw", "nmeasure")).chunk(
        {"chain": 1, "draw": 64, "nmeasure": 64}
    )
    offset = xr.zeros_like(boundary).chunk({"draw": offset_draw_chunk})
    trace = xr.Dataset({"mu_bc_posterior": boundary, "offset_posterior": offset})
    sources = {name: value.data for name, value in trace.data_vars.items()}
    graphs = {name: value.__dask_graph__() for name, value in sources.items()}
    native = trace.assign(mu_bc_posterior=trace.mu_bc_posterior + trace.offset_posterior)
    expected = module._calculate_stats(native, ["mean", "stdev"]).compute()

    actual = module.summarize_derived(
        {"boundary": boundary, "offset": offset},
        lambda block: (block.boundary + block.offset).to_dataset(name="mu_bc_posterior"),
        stats=["mean", "stdev"], companions=trace[["offset_posterior"]],
    )

    xr.testing.assert_identical(actual, expected)
    for name, source in sources.items():
        assert trace[name].data is source
        assert source.__dask_graph__() is graphs[name]


@pytest.mark.parametrize("independent_outputs", [False, True], ids=["unused-operand", "independent-outputs"])
def test_derived_outputs_use_only_their_own_operand_layouts(independent_outputs):
    """Another operand's singleton point chunks cannot change an output's reduction tree."""
    rng = np.random.default_rng(19)
    operands = {
        name: xr.DataArray(
            rng.normal(1950, 200, size=(2, 259, 129)).astype("float32"),
            dims=("chain", "draw", "point"),
        ).chunk({"chain": 1, "draw": 64, "point": point_chunk})
        for name, point_chunk in (("a", 64), ("b", 1))
    }

    def derive(block):
        result = (block.a * 2).to_dataset(name="calibrated_a")
        if independent_outputs:
            result["calibrated_b"] = block.b + 1
        return result

    expected = module._calculate_stats(derive(xr.Dataset(operands)), ["mean", "stdev"]).compute()
    actual = module.summarize_derived(operands, derive, stats=["mean", "stdev"])

    xr.testing.assert_identical(actual, expected)


def test_derived_metadata_and_auxiliary_coordinates_survive_companion_groups():
    """Derived provenance and coordinates survive separate draw-only companions."""
    operand = xr.DataArray(
        np.arange(6 * 300.0).reshape(6, 300),
        dims=("draw", "point"), coords={"point": np.arange(300)},
    )
    companions = xr.Dataset({"other": ("draw", np.arange(6.0))}, attrs={"run": "example"})

    def derive(block):
        result = (block.value + 1).to_dataset(name="adjusted")
        result.attrs["quantity_family"] = "adjusted"
        result.adjusted.attrs["units"] = "ppm"
        return result.assign_coords(cell_class=block.value.point % 2)

    actual = module.summarize_derived(
        {"value": operand}, derive, stats=["mean"], companions=companions,
    )

    assert actual.attrs == {"run": "example", "quantity_family": "adjusted"}
    assert actual.cell_class.dims == ("point",)
    np.testing.assert_array_equal(actual.cell_class.values, np.arange(300) % 2)
    assert actual.indexes["point"].equals(operand.indexes["point"])
    expected = (operand + 1).mean("draw").assign_coords(cell_class=operand.point % 2).rename("adjusted_mean")
    xr.testing.assert_equal(actual.adjusted_mean, expected)
    assert actual.adjusted_mean.attrs["units"] == "ppm"
    assert actual.other_mean.ndim == 0
    assert actual.other_mean.item() == 2.5
    assert companions.attrs == {"run": "example"}


@pytest.mark.parametrize("conflict", ["metadata", "variable", "lower-dimensional-variable"])
def test_derived_and_companion_conflicts_are_rejected(conflict):
    """Conflicting scientific metadata and names cannot be silently overwritten."""
    if conflict == "lower-dimensional-variable":
        operand = xr.DataArray(np.ones((4, 300)), dims=("draw", "point"))
        # Its mean agrees with the derived mean, so ordinary merge conflict
        # handling cannot detect that the names overlap across execution groups.
        companions = xr.Dataset({"adjusted": ("draw", np.full(4, 2.0))})
    else:
        operand = xr.DataArray(np.arange(12.0).reshape(4, 3), dims=("draw", "point"))
        name = "adjusted" if conflict == "variable" else "other"
        companions = (operand * 2).to_dataset(name=name)
    if conflict == "metadata":
        companions.attrs["reference"] = "companion-reference"

    def derive(block):
        result = (block.value + 1).to_dataset(name="adjusted")
        if conflict == "metadata":
            result.attrs["reference"] = "derived-reference"
        return result

    with pytest.raises(ValueError):
        module.summarize_derived({"value": operand}, derive, stats=["mean"], companions=companions)
