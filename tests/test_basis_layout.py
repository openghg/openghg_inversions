import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.basis.layout import (
    BASIS_GROUP_COORD,
    BASIS_LABEL_DIM,
    BASIS_PARTITION_COORD,
    REGION_IN_PARTITION_COORD,
    BasisLayout,
    BasisPartition,
)
from openghg_inversions.basis.operators import BucketBasisOperator


def _basis_grid(values: list[list[int]], *, name: str = "basis") -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=int),
        dims=("lat", "lon"),
        coords={"lat": [0.0, 1.0], "lon": [10.0, 20.0]},
        name=name,
    )


def _state_metadata(raw_labels: list[int]) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            BASIS_GROUP_COORD: (BASIS_LABEL_DIM, ["outer", "inner"]),
            BASIS_PARTITION_COORD: (BASIS_LABEL_DIM, ["fixed_outer_10", "generated_inner"]),
            REGION_IN_PARTITION_COORD: (BASIS_LABEL_DIM, [10, 1]),
        },
        coords={BASIS_LABEL_DIM: raw_labels},
    )


def test_basis_layout_combines_disjoint_partitions_with_metadata():
    """A layout offsets partition-local labels and records raw-label metadata."""
    inner = _basis_grid([[1, 2], [0, 0]], name="inner")
    outer = _basis_grid([[0, 0], [1, 1]], name="outer")

    result = BasisLayout(
        partitions=(
            BasisPartition(name="inner", labels=inner, group="inner"),
            BasisPartition(name="outer", labels=outer, group="outer"),
        ),
        state_dim="region",
    ).to_flat_basis()

    expected_basis = _basis_grid([[1, 2], [3, 3]])
    expected_metadata = xr.Dataset(
        data_vars={
            BASIS_GROUP_COORD: (BASIS_LABEL_DIM, ["inner", "inner", "outer"]),
            BASIS_PARTITION_COORD: (BASIS_LABEL_DIM, ["inner", "inner", "outer"]),
            REGION_IN_PARTITION_COORD: (BASIS_LABEL_DIM, [1, 2, 1]),
        },
        coords={BASIS_LABEL_DIM: [1, 2, 3]},
        attrs={"state_dim": "region"},
    )

    xr.testing.assert_identical(result.basis_flat, expected_basis)
    xr.testing.assert_identical(result.state_metadata, expected_metadata)


def test_basis_layout_rejects_overlapping_partitions():
    """Partitions must cover disjoint grid cells."""
    first = _basis_grid([[1, 0], [0, 0]], name="first")
    second = _basis_grid([[1, 1], [1, 1]], name="second")

    layout = BasisLayout(
        partitions=(
            BasisPartition(name="first", labels=first, group="first"),
            BasisPartition(name="second", labels=second, group="second"),
        )
    )

    with pytest.raises(ValueError, match="overlaps"):
        layout.to_flat_basis()


def test_basis_layout_rejects_unmapped_grid_cells():
    """Every grid cell must belong to exactly one partition in this core helper."""
    first = _basis_grid([[1, 1], [0, 0]], name="first")
    second = _basis_grid([[0, 0], [1, 0]], name="second")

    layout = BasisLayout(
        partitions=(
            BasisPartition(name="first", labels=first, group="first"),
            BasisPartition(name="second", labels=second, group="second"),
        )
    )

    with pytest.raises(ValueError, match="unmapped"):
        layout.to_flat_basis()


def test_basis_layout_rejects_non_integer_labels():
    """Partition labels must be integer-valued before global relabeling."""
    labels = _basis_grid([[1, 1], [0, 0]], name="labels").astype(float)
    labels.values[0, 0] = 1.5
    other = _basis_grid([[0, 0], [1, 1]], name="other")

    layout = BasisLayout(
        partitions=(
            BasisPartition(name="labels", labels=labels, group="labels"),
            BasisPartition(name="other", labels=other, group="other"),
        )
    )

    with pytest.raises(ValueError, match="integer-valued"):
        layout.to_flat_basis()


def test_basis_layout_rejects_coordinate_mismatch():
    """Partition grids must align exactly before labels can be combined."""
    first = _basis_grid([[1, 1], [0, 0]], name="first")
    second = _basis_grid([[0, 0], [1, 1]], name="second").assign_coords(lat=[0.0, 2.0])

    layout = BasisLayout(
        partitions=(
            BasisPartition(name="first", labels=first, group="first"),
            BasisPartition(name="second", labels=second, group="second"),
        )
    )

    with pytest.raises(ValueError):
        layout.to_flat_basis()


def test_bucket_operator_attaches_metadata_after_region_label_policy():
    """Raw basis labels can differ from final state coordinates."""
    basis = _basis_grid([[10, 20], [10, 20]])
    operator = BucketBasisOperator(
        basis_flat=basis,
        state_dim="region",
        region_labels="range0",
        state_metadata=_state_metadata([10, 20]),
    )

    assert operator.basis_matrix.region.values.tolist() == [0, 1]
    assert operator.basis_matrix[BASIS_GROUP_COORD].values.tolist() == ["outer", "inner"]
    assert operator.basis_matrix[BASIS_PARTITION_COORD].values.tolist() == [
        "fixed_outer_10",
        "generated_inner",
    ]
    assert operator.basis_matrix[REGION_IN_PARTITION_COORD].values.tolist() == [10, 1]


def test_bucket_operator_reorders_raw_label_metadata():
    """Metadata keyed by raw basis labels need not arrive in sorted label order."""
    basis = _basis_grid([[10, 20], [10, 20]])
    metadata = xr.Dataset(
        data_vars={
            BASIS_GROUP_COORD: (BASIS_LABEL_DIM, ["inner", "outer"]),
            BASIS_PARTITION_COORD: (BASIS_LABEL_DIM, ["generated_inner", "fixed_outer_10"]),
            REGION_IN_PARTITION_COORD: (BASIS_LABEL_DIM, [1, 10]),
        },
        coords={BASIS_LABEL_DIM: [20, 10]},
    )
    operator = BucketBasisOperator(
        basis_flat=basis,
        state_dim="region",
        region_labels="range0",
        state_metadata=metadata,
    )

    assert operator.basis_matrix[BASIS_GROUP_COORD].values.tolist() == ["outer", "inner"]
    assert operator.basis_matrix[REGION_IN_PARTITION_COORD].values.tolist() == [10, 1]


def test_bucket_operator_rejects_missing_state_metadata_variable():
    """State metadata must include the standard grouped-layout coordinates."""
    basis = _basis_grid([[10, 20], [10, 20]])
    metadata = _state_metadata([10, 20]).drop_vars(BASIS_GROUP_COORD)

    with pytest.raises(ValueError, match=BASIS_GROUP_COORD):
        BucketBasisOperator(
            basis_flat=basis,
            state_dim="region",
            region_labels="range0",
            state_metadata=metadata,
        )


def test_bucket_operator_rejects_mismatched_raw_label_metadata():
    """Raw-label metadata must cover exactly the basis labels."""
    basis = _basis_grid([[10, 20], [10, 20]])

    with pytest.raises(ValueError, match="basis_label values"):
        BucketBasisOperator(
            basis_flat=basis,
            state_dim="region",
            region_labels="range0",
            state_metadata=_state_metadata([10, 30]),
        )


def test_bucket_operator_rejects_mismatched_final_state_metadata():
    """Metadata already indexed by final state must match the final state coordinate."""
    basis = _basis_grid([[10, 20], [10, 20]])
    metadata = xr.Dataset(
        data_vars={
            BASIS_GROUP_COORD: ("region", ["outer", "inner"]),
            BASIS_PARTITION_COORD: ("region", ["fixed_outer_10", "generated_inner"]),
            REGION_IN_PARTITION_COORD: ("region", [10, 1]),
        },
        coords={"region": [1, 0]},
    )

    with pytest.raises(ValueError, match="does not match"):
        BucketBasisOperator(
            basis_flat=basis,
            state_dim="region",
            region_labels="range0",
            state_metadata=metadata,
        )


def test_bucket_operator_datatree_roundtrip_preserves_state_metadata():
    """Operator DataTree serialization keeps grouped state coordinates."""
    basis = _basis_grid([[10, 20], [10, 20]])
    operator = BucketBasisOperator(
        basis_flat=basis,
        state_dim="region",
        region_labels="range0",
        state_metadata=_state_metadata([10, 20]),
    )

    restored = BucketBasisOperator.from_datatree(operator.to_datatree())

    xr.testing.assert_identical(restored.basis_matrix, operator.basis_matrix)
    xr.testing.assert_identical(restored.state_metadata, operator.state_metadata)


def test_basis_functions_save_load_preserves_state_metadata(tmp_path):
    """BasisFunctions artifacts retain state metadata coordinates on load."""
    basis = _basis_grid([[10, 20], [10, 20]])
    flux = xr.ones_like(basis, dtype=float).rename("flux")
    basis_functions = BasisFunctions.from_flat_basis(
        basis_flat=basis,
        flux=flux,
        operator_kwargs={
            "state_dim": "region",
            "region_labels": "range0",
            "state_metadata": _state_metadata([10, 20]),
        },
    )

    output_file = tmp_path / "basis.nc"
    basis_functions.save(output_file)
    restored = BasisFunctions.load(output_file)

    xr.testing.assert_identical(restored.operator.basis_matrix, basis_functions.operator.basis_matrix)
