"""Verify the executable grouped basis documentation example."""

from pathlib import Path
from runpy import run_path

import xarray as xr


def test_grouped_basis_documentation_example():
    """The documented layout, selection, and round trip retain their contract."""
    example_path = Path(__file__).parents[1] / "docs" / "examples" / "grouped_basis_layout.py"
    example = run_path(str(example_path))

    result = example["result"]
    matrix = example["matrix"]
    restored_matrix = example["restored_matrix"]

    assert result.basis_flat.values.tolist() == [[1, 2], [3, 3]]
    assert result.state_metadata.basis_label.values.tolist() == [1, 2, 3]
    assert matrix.region.values.tolist() == [0, 1, 2]
    assert matrix.basis_group.values.tolist() == ["inner", "inner", "outer"]
    assert matrix.basis_partition.values.tolist() == [
        "generated_inner",
        "generated_inner",
        "explicit_remainder",
    ]
    assert matrix.region_in_partition.values.tolist() == [1, 2, 1]
    assert example["uncovered_error"] == "BasisLayout partitions leave 2 grid cells unmapped."
    assert example["inner_state"].region.values.tolist() == [0, 1]
    assert example["outer_state"].region.values.tolist() == [2]
    xr.testing.assert_identical(restored_matrix, matrix)
