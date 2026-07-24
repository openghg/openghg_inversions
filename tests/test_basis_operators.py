from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.operators import (
    BucketBasisOperator,
    MultiSourceBucketBasisOperator,
    BasisOperator,
)

from helpers import basis_function


@pytest.fixture
def basis_func():
    nlat, nlon = 10, 12
    nbasis = 3
    return basis_function(nlat, nlon, nbasis)


@pytest.fixture
def basis_func2():
    nlat, nlon = 10, 12
    nbasis = 7
    return basis_function(nlat, nlon, nbasis)


def test_bucket_basis_operator_roundtrip_datatree(basis_func):
    op = BucketBasisOperator(basis_func, state_dim="state")
    dt = op.to_datatree()
    op2 = BasisOperator.decode_datatree(dt)

    assert isinstance(op2, BucketBasisOperator)
    xr.testing.assert_identical(op2.basis_flat, op.basis_flat)
    xr.testing.assert_identical(op2.basis_matrix, op.basis_matrix)


def test_multisource_basis_operator_roundtrip_datatree(basis_func, basis_func2):
    """Static multisource bases use a labeled source axis with stable ordering."""
    basis = {"B": basis_func, "inventory/anthro": basis_func2}
    op = MultiSourceBucketBasisOperator(basis, state_dim="state")
    dt = op.to_datatree()
    encoded_basis = dt.to_dataset()["basis_flat"]
    op2 = BasisOperator.decode_datatree(dt)

    assert isinstance(op2, MultiSourceBucketBasisOperator)
    assert encoded_basis.source.values.tolist() == ["B", "inventory/anthro"]
    assert "source_order" not in dt.attrs
    assert list(op2.basis_flat) == list(op.basis_flat)

    xr.testing.assert_identical(op2.basis_matrix, op.basis_matrix)

    for k in op.basis_flat:
        xr.testing.assert_identical(op2.basis_flat[k], op.basis_flat[k])


def test_multisource_basis_operator_decodes_legacy_per_source_children(basis_func, basis_func2):
    """The previous per-source-child layout remains readable after write-format revision."""
    legacy = xr.DataTree()
    legacy.attrs.update(
        {
            "schema": BasisOperator.schema,
            "schema_version": BasisOperator.schema_version,
            "kind": MultiSourceBucketBasisOperator.kind,
            "grid_dims": ("lat", "lon"),
            "state_dim": "state",
            "source_dim": "source",
            "region_in_source_dim": "region_in_source",
        }
    )
    legacy["basis_flat"] = xr.DataTree.from_dict(
        {
            "B": xr.Dataset({"basis_flat": basis_func}),
            "A": xr.Dataset({"basis_flat": basis_func2}),
        }
    )

    restored = BasisOperator.decode_datatree(legacy)

    assert isinstance(restored, MultiSourceBucketBasisOperator)
    assert list(restored.basis_flat) == ["B", "A"]
    xr.testing.assert_identical(restored.basis_flat["B"], basis_func.rename("basis_flat"))
    xr.testing.assert_identical(restored.basis_flat["A"], basis_func2.rename("basis_flat"))


def test_multisource_basis_operator_drops_conflicting_array_attrs(
    basis_func,
    basis_func2,
):
    """Serialization retains common attrs and omits conflicting source attrs."""
    basis_func = basis_func.assign_attrs(units="1", provenance="inventory-a")
    basis_func2 = basis_func2.assign_attrs(units="1", provenance="inventory-b")
    operator = MultiSourceBucketBasisOperator(
        {"A": basis_func, "B": basis_func2},
        state_dim="state",
    )

    encoded_basis = operator.to_datatree().to_dataset()["basis_flat"]

    assert encoded_basis.attrs == {"units": "1"}


def test_multisource_basis_operator_reorders_equivalent_grid_indexes(
    basis_func,
    basis_func2,
):
    """Serialization follows coordinate labels instead of array position."""
    reversed_basis = basis_func2.isel(lon=slice(None, None, -1))
    operator = MultiSourceBucketBasisOperator(
        {"A": basis_func, "B": reversed_basis},
        state_dim="state",
    )

    restored = BasisOperator.decode_datatree(operator.to_datatree())

    assert isinstance(restored, MultiSourceBucketBasisOperator)
    xr.testing.assert_identical(
        restored.basis_flat["B"],
        basis_func2.rename("basis_flat"),
    )


def test_multisource_basis_operator_rejects_different_grid_labels(
    basis_func,
    basis_func2,
):
    """Serialization rejects genuinely different source grids."""
    shifted_basis = basis_func2.assign_coords(lon=basis_func2.lon + 0.5)
    operator = MultiSourceBucketBasisOperator(
        {"A": basis_func, "B": shifted_basis},
        state_dim="state",
    )

    with pytest.raises(ValueError, match="same grid coordinate labels"):
        operator.to_datatree()


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_multisource_basis_source_labels_roundtrip_storage(
    tmp_path: Path,
    basis_func: xr.DataArray,
    basis_func2: xr.DataArray,
    suffix: str,
) -> None:
    """Path-significant source labels remain coordinate values in both stores."""
    operator = MultiSourceBucketBasisOperator(
        {"B": basis_func, "inventory/anthro": basis_func2},
        state_dim="state",
    )
    path = tmp_path / f"operator{suffix}"
    tree = operator.to_datatree()
    if suffix == ".nc":
        tree.to_netcdf(path)
    else:
        tree.to_zarr(path, mode="w")

    with xr.open_datatree(path) as stored:
        restored = BasisOperator.decode_datatree(stored.load())

    assert isinstance(restored, MultiSourceBucketBasisOperator)
    assert list(restored.basis_flat) == ["B", "inventory/anthro"]


# --------------------------------------------------------------------------------------
# Canonical dim name test: do not bake in "region"
# --------------------------------------------------------------------------------------
def test_bucket_basis_operator_roundtrip_preserves_default_state_dim():
    basis = basis_function(6, 5, 3)
    op = BucketBasisOperator(basis, state_dim="state")
    dt = op.to_datatree()
    op2 = BasisOperator.decode_datatree(dt)

    assert op2.meta.state_dim == "state"
    assert "state" in op2.basis_matrix.dims


def test_bucket_basis_operator_accepts_zero_based_flat_basis():
    """Legacy HBMCMC output basis labels can still reconstruct a dummy basis."""
    basis = xr.DataArray(
        np.array([[0, 1, 2]], dtype=int),
        dims=("lat", "lon"),
        coords={"lat": [0.0], "lon": [0.0, 1.0, 2.0]},
        name="basis_flat",
    )

    op = BucketBasisOperator(basis, state_dim="state")

    assert list(op.basis_matrix.state.values) == [0, 1, 2]
    assert op.basis_matrix.sizes["state"] == 3


# --------------------------------------------------------------------------------------
# Test interpolation for MultiSourceBucketBasisOperator
# --------------------------------------------------------------------------------------


def _make_multisource_operator_for_broadcast_tests(
    state_dim: str = "state",
) -> MultiSourceBucketBasisOperator:
    """Create a tiny multisource operator with ragged per-source region counts.

    Source A has 2 regions; source B has 1 region.
    """
    basis_a = xr.DataArray(
        np.array([[1, 2], [1, 2]], dtype=int),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="basis_flat",
    )
    basis_b = xr.DataArray(
        np.array([[1, 1], [1, 1]], dtype=int),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="basis_flat",
    )
    return MultiSourceBucketBasisOperator({"A": basis_a, "B": basis_b}, state_dim=state_dim)


def _state_for_operator(
    op: MultiSourceBucketBasisOperator, values: list[float], name: str = "state"
) -> xr.DataArray:
    """Create a state(state) vector matching the operator's gathered MultiIndex order."""
    state_index = op.basis_matrix[op.meta.state_dim]
    if len(values) != state_index.size:
        raise ValueError("values must match gathered state size")

    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=(op.meta.state_dim,),
        coords={op.meta.state_dim: state_index},
        name=name,
    )


def test_multisource_interpolate_broadcasts_weights_source_to_state():
    """weights(source, lat, lon) should broadcast onto gathered state(source, region_in_source)."""
    op = _make_multisource_operator_for_broadcast_tests(state_dim="state")

    # State order is source-major: (A, r1), (A, r2), (B, r1)
    # For basis defined above:
    # - A r1 covers left column, A r2 covers right column, B r1 covers entire grid.
    state = _state_for_operator(op, [10.0, 100.0, 1000.0], name="x")

    # weights differ by source, so broadcasting/alignment matters
    weights = xr.DataArray(
        np.stack([np.ones((2, 2), dtype=float) * 2.0, np.ones((2, 2), dtype=float) * 3.0], axis=0),
        dims=("source", "lat", "lon"),
        coords={"source": ["A", "B"], "lat": [0, 1], "lon": [0, 1]},
        name="weights",
    )

    out = op.interpolate(state, weights=weights)
    assert set(out.dims) == {"lat", "lon"}

    expected = xr.DataArray(
        np.array(
            [
                [10.0 * 2.0 + 1000.0 * 3.0, 100.0 * 2.0 + 1000.0 * 3.0],
                [10.0 * 2.0 + 1000.0 * 3.0, 100.0 * 2.0 + 1000.0 * 3.0],
            ],
            dtype=float,
        ),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name=out.name,
    )

    xr.testing.assert_allclose(out, expected)


def test_multisource_interpolate_broadcasts_weights_source_with_nontrivial_values():
    """Second check: per-source weights and per-region states combine correctly."""
    op = _make_multisource_operator_for_broadcast_tests(state_dim="state")

    # Make A regions different, B different, to ensure ordering is respected.
    state_array = _state_for_operator(op, [1.0, 10.0, 100.0], name="x")

    # A weights=5, B weights=7
    weights = xr.DataArray(
        np.stack([np.ones((2, 2), dtype=float) * 5.0, np.ones((2, 2), dtype=float) * 7.0], axis=0),
        dims=("source", "lat", "lon"),
        coords={"source": ["A", "B"], "lat": [0, 1], "lon": [0, 1]},
        name="weights",
    )

    out = op.interpolate(state_array, weights=weights)

    expected = xr.DataArray(
        np.array(
            [
                [1.0 * 5.0 + 100.0 * 7.0, 10.0 * 5.0 + 100.0 * 7.0],
                [1.0 * 5.0 + 100.0 * 7.0, 10.0 * 5.0 + 100.0 * 7.0],
            ],
            dtype=float,
        ),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name=out.name,
    )

    xr.testing.assert_allclose(out, expected)


def test_multisource_operator_for_source_matches_source_slice_interpolation():
    """Selected source operators reproduce the corresponding multisource interpolation slice."""
    op = _make_multisource_operator_for_broadcast_tests(state_dim="state")

    # Select the two-region source A from the gathered multisource operator.
    selected = op.operator_for_source("A", state_dim="region")
    source_state = xr.DataArray(
        [2.0, 5.0],
        dims=("region",),
        coords={"region": [0, 1]},
        name="x",
    )
    source_weights = xr.DataArray(
        np.array([[3.0, 7.0], [11.0, 13.0]], dtype=float),
        dims=("lat", "lon"),
        coords={"lat": [0, 1], "lon": [0, 1]},
        name="weights",
    )

    selected_out = selected.interpolate(source_state, weights=source_weights)

    multisource_state = _state_for_operator(op, [2.0, 5.0, 0.0], name="x")
    multisource_weights = xr.concat(
        [
            source_weights.expand_dims(source=["A"]),
            xr.zeros_like(source_weights).expand_dims(source=["B"]),
        ],
        dim="source",
    )
    multisource_out = op.interpolate(multisource_state, weights=multisource_weights)

    xr.testing.assert_allclose(selected_out, multisource_out)
