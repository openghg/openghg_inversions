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
from openghg_inversions.basis.operators import BasisOperator, BucketBasisOperator


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


def test_basis_layout_requires_explicit_remainder_partition():
    """Layouts do not synthesize an implicit remainder partition."""
    inner = _basis_grid([[1, 2], [0, 0]], name="inner")
    layout = BasisLayout(
        partitions=(BasisPartition(name="generated_inner", labels=inner, group="inner"),),
        state_dim="region",
    )

    with pytest.raises(ValueError, match="unmapped"):
        layout.to_flat_basis()


def test_basis_layout_preserves_explicit_remainder_partition_metadata():
    """A caller-provided remainder partition is recorded like any partition."""
    inner = _basis_grid([[1, 2], [0, 0]], name="inner")
    remainder = _basis_grid([[0, 0], [1, 1]], name="remainder")

    result = BasisLayout(
        partitions=(
            BasisPartition(name="generated_inner", labels=inner, group="inner"),
            BasisPartition(name="explicit_remainder", labels=remainder, group="remainder"),
        ),
        state_dim="region",
    ).to_flat_basis()
    metadata = result.state_metadata

    xr.testing.assert_identical(result.basis_flat, _basis_grid([[1, 2], [3, 3]]))
    assert metadata[BASIS_GROUP_COORD].values.tolist() == ["inner", "inner", "remainder"]
    assert metadata[BASIS_PARTITION_COORD].values.tolist() == [
        "generated_inner",
        "generated_inner",
        "explicit_remainder",
    ]
    assert metadata[REGION_IN_PARTITION_COORD].values.tolist() == [1, 2, 1]
    assert metadata[BASIS_LABEL_DIM].values.tolist() == [1, 2, 3]


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


def test_bucket_operator_rejects_duplicate_raw_label_metadata():
    """Raw-label metadata must identify each basis label exactly once."""
    basis = _basis_grid([[10, 20], [10, 20]])
    metadata = xr.Dataset(
        data_vars={
            BASIS_GROUP_COORD: (BASIS_LABEL_DIM, ["outer-a", "outer-b", "inner"]),
            BASIS_PARTITION_COORD: (
                BASIS_LABEL_DIM,
                ["fixed_outer_a", "fixed_outer_b", "generated_inner"],
            ),
            REGION_IN_PARTITION_COORD: (BASIS_LABEL_DIM, [10, 10, 1]),
        },
        coords={BASIS_LABEL_DIM: [10, 10, 20]},
    )

    with pytest.raises(ValueError, match="basis_label.*unique"):
        BucketBasisOperator(
            basis_flat=basis,
            state_dim="region",
            region_labels="range0",
            state_metadata=metadata,
        )


def test_bucket_operator_rejects_non_numeric_raw_label_metadata():
    """Raw basis-label coordinates must be numeric before state alignment."""
    basis = _basis_grid([[10, 20], [10, 20]])
    metadata = xr.Dataset(
        data_vars={
            BASIS_GROUP_COORD: (BASIS_LABEL_DIM, ["outer", "inner"]),
            BASIS_PARTITION_COORD: (BASIS_LABEL_DIM, ["fixed_outer_10", "generated_inner"]),
            REGION_IN_PARTITION_COORD: (BASIS_LABEL_DIM, [10, 1]),
        },
        coords={BASIS_LABEL_DIM: ["10", "20"]},
    )

    with pytest.raises(ValueError, match="basis_label.*numeric"):
        BucketBasisOperator(
            basis_flat=basis,
            state_dim="region",
            region_labels="range0",
            state_metadata=metadata,
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
    decoded = BasisOperator.decode_datatree(operator.to_datatree())

    xr.testing.assert_identical(restored.basis_matrix, operator.basis_matrix)
    assert restored.state_metadata is not None
    assert operator.state_metadata is not None
    xr.testing.assert_identical(restored.state_metadata, operator.state_metadata)
    assert isinstance(decoded, BucketBasisOperator)
    assert decoded.state_metadata is not None
    xr.testing.assert_identical(decoded.basis_matrix, operator.basis_matrix)
    xr.testing.assert_identical(decoded.state_metadata, operator.state_metadata)


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
    restored_operator = restored.operator
    original_operator = basis_functions.operator

    xr.testing.assert_identical(restored_operator.basis_matrix, original_operator.basis_matrix)
    assert isinstance(restored_operator, BucketBasisOperator)
    assert isinstance(original_operator, BucketBasisOperator)
    assert restored_operator.state_metadata is not None
    assert original_operator.state_metadata is not None
    xr.testing.assert_identical(
        restored_operator.state_metadata,
        original_operator.state_metadata,
    )
    for coord_name in (BASIS_GROUP_COORD, BASIS_PARTITION_COORD, REGION_IN_PARTITION_COORD):
        xr.testing.assert_identical(
            restored_operator.basis_matrix[coord_name],
            original_operator.basis_matrix[coord_name],
        )
