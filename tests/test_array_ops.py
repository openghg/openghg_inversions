import numpy as np
import pandas as pd
import pytest
import sparse
import xarray as xr

from openghg_inversions.array_ops import (
    align_to_multi_index_level_values,
    concat_gather_data_arrays,
    concat_gather_datatree,
    concat_gather_datasets,
    select_gathered_data_array,
)


def _make_site_dataset(site: str, *, include_inlet_height: bool) -> xr.Dataset:
    """Create a small site dataset for concat-gather tests."""
    time = xr.DataArray(pd.date_range("2020-01-01", periods=2, freq="1h"), dims="time", name="time")
    data_vars = {
        "mf": xr.DataArray(np.array([1.0, 2.0]), dims="time", coords={"time": time}),
        "mf_error": xr.DataArray(np.array([0.1, 0.2]), dims="time", coords={"time": time}),
    }
    if include_inlet_height:
        data_vars["inlet_height"] = xr.DataArray(np.array([100.0, 100.0]), dims="time", coords={"time": time})

    return xr.Dataset(data_vars).assign_coords(site=site)


def _concat_with_policy(
    datasets: dict[str, xr.Dataset],
    *,
    missing_data_vars: str,
    use_datatree: bool,
) -> xr.Dataset:
    """Run concat-gather via datasets or datatree with the same options."""
    if use_datatree:
        datatree = xr.DataTree.from_dict({key: value for key, value in datasets.items()})
        return concat_gather_datatree(
            datatree,
            key_dim="site",
            ragged_dim="time",
            stack_dim="nmeasure",
            missing_data_vars=missing_data_vars,
        )

    return concat_gather_datasets(
        datasets,
        key_dim="site",
        ragged_dim="time",
        stack_dim="nmeasure",
        missing_data_vars=missing_data_vars,
    )


def test_transpose():
    dums = pd.get_dummies([0] * 3 + [1] * 4 + [2] * 5, dtype=int, sparse=True)
    sparse_dums = sparse.COO.from_scipy_sparse(dums.sparse.to_coo())
    da = xr.DataArray(sparse_dums)
    da.transpose()


def test_align_to_multi_index_level_values_broadcasts_source_to_state():
    """Test aligning coordinate to multi-index level coordinate broadcasts correctly."""
    # da(source, time)
    source = xr.DataArray(["A", "B"], dims="source", name="source")
    time = xr.DataArray(pd.date_range("2020-01-01", periods=3, freq="1h"), dims="time", name="time")

    da = xr.DataArray(
        np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
        dims=("source", "time"),
        coords={"source": source, "time": time},
        name="da",
    )

    # MultiIndex for state: (source, region_in_source) with repetition of source labels
    mi = pd.MultiIndex.from_arrays(
        [["B", "A", "B", "A"], [0, 0, 1, 2]],
        names=["source", "region_in_source"],
    )

    # align to multi index
    state_index = xr.Coordinates.from_pandas_multiindex(mi, "state")["state"]

    res = align_to_multi_index_level_values(
        da,
        multi_index=state_index,
        multi_dim="state",
        level="source",
        other_dim="source",
    )

    # Dims: state, time (source dim is replaced)
    assert res.dims == ("state", "time")

    # The state coordinate should be exactly the MultiIndex we provided
    xr.testing.assert_identical(res["state"], state_index)

    # The original "source" coordinate variable should be dropped to avoid alignment conflicts
    # (it's still available as a level on the MultiIndex)
    # assert "source" not in res.coords

    # Values: each row picks from da at the corresponding state.source
    expected = np.vstack(
        [
            da.sel(source="B").values,
            da.sel(source="A").values,
            da.sel(source="B").values,
            da.sel(source="A").values,
        ]
    )
    np.testing.assert_allclose(res.values, expected)

    expected_da = xr.DataArray(
        expected,
        dims=("state", "time"),
        coords={**xr.Coordinates.from_pandas_multiindex(mi, "state"), **{"time": time}},
        name="da",
    )
    xr.testing.assert_identical(res, expected_da)

    # This multiplication (or any operation that causes alignment) can fail if
    # the coordinate for `other_dim` is not dropped before assigning the MultiIndex.
    # The test here is just that this doesn't cause an error.
    _ = res * expected_da


def test_align_to_multi_index_level_values_with_other_level_as_coord_raises():
    """Test that an error is raised if aligning to a MultiIndex would cause coord conflict."""
    da = xr.DataArray(
        np.arange(12).reshape(3, 4), dims=("a", "b"), coords={"a": np.arange(3), "b": np.arange(4)}
    )

    da_stack = da.stack(c=("a", "b"))
    mi = da_stack.coords["c"]

    with pytest.raises(ValueError):
        _ = align_to_multi_index_level_values(da, multi_index=mi, multi_dim="c", level="a", other_dim="a")


def test_align_to_multi_index_level_values_with_other_level_as_dim_warns():
    """Test that we can align a dataset to one level of multiindex even if it has both levels as dims.

    This should emit a warning that this is potentially confusing.
    """
    da = xr.DataArray(np.arange(12).reshape(3, 4), dims=("a", "b"))

    da_stack = da.stack(c=("a", "b"))
    mi = da_stack.coords["c"]

    with pytest.warns(
        UserWarning,
        match=r"Aligning to MultiIndex level.*\'b\'.*semantically ambiguous",
    ):
        da_aligned = align_to_multi_index_level_values(
            da, multi_index=mi, multi_dim="c", level="a", other_dim="a"
        )

    xr.testing.assert_equal(da_stack.a, da_aligned.a)


def test_select_gathered_data_array_restores_ragged_labels_and_values() -> None:
    """Selecting a gathered key should retain represented values, including NaNs."""
    time = pd.date_range("2020-01-01", periods=2, freq="1h")
    gathered = concat_gather_data_arrays(
        {
            "ff": xr.DataArray(
                [[1.0, np.nan], [2.0, 3.0]],
                dims=("region", "time"),
                coords={"region": [10, 11], "time": time},
            ),
            "ocean": xr.DataArray(
                [[4.0, 5.0]],
                dims=("region", "time"),
                coords={"region": [20], "time": time},
            ),
        },
        key_dim="source",
        ragged_dim="region",
        stack_dim="state",
        join="exact",
    )

    selected = select_gathered_data_array(
        gathered,
        key="ff",
        key_dim="source",
        ragged_dim="region",
        stack_dim="state",
    )

    expected = xr.DataArray(
        [[1.0, np.nan], [2.0, 3.0]],
        dims=("state", "time"),
        coords={"state": [10, 11], "time": time},
    )
    xr.testing.assert_identical(selected, expected)


@pytest.mark.parametrize("use_datatree", [False, True], ids=["datasets", "datatree"])
@pytest.mark.parametrize("order", [("AAA", "BBB"), ("BBB", "AAA")], ids=["extra-first", "extra-second"])
def test_concat_gather_missing_data_vars_error_is_order_independent(
    use_datatree: bool, order: tuple[str, str]
):
    """Mismatched data vars should raise regardless of dataset order."""
    datasets = {
        "AAA": _make_site_dataset("AAA", include_inlet_height=True),
        "BBB": _make_site_dataset("BBB", include_inlet_height=False),
    }
    ordered = {key: datasets[key] for key in order}

    with pytest.raises(ValueError, match="inlet_height"):
        _concat_with_policy(ordered, missing_data_vars="error", use_datatree=use_datatree)


@pytest.mark.parametrize("use_datatree", [False, True], ids=["datasets", "datatree"])
@pytest.mark.parametrize("order", [("AAA", "BBB"), ("BBB", "AAA")], ids=["extra-first", "extra-second"])
def test_concat_gather_missing_data_vars_drop_warns_and_drops(use_datatree: bool, order: tuple[str, str]):
    """Drop mode should keep only shared vars and warn once."""
    datasets = {
        "AAA": _make_site_dataset("AAA", include_inlet_height=True),
        "BBB": _make_site_dataset("BBB", include_inlet_height=False),
    }
    ordered = {key: datasets[key] for key in order}

    with pytest.warns(UserWarning, match="Dropping data variables.*inlet_height"):
        result = _concat_with_policy(ordered, missing_data_vars="drop", use_datatree=use_datatree)

    assert set(result.data_vars) == {"mf", "mf_error"}
    assert "inlet_height" not in result
