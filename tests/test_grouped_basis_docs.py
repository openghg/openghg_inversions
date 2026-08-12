"""Verify the executable grouped basis documentation example."""

from pathlib import Path

import xarray as xr


def _jupyter_cells(document: Path) -> list[str]:
    """Extract executable Python cells from a jupyter-sphinx document.

    Args:
        document: Path to the reStructuredText source document.

    Returns:
        The dedented bodies of all ``jupyter-execute`` directives, in source
        order.
    """
    lines = document.read_text(encoding="utf-8").splitlines()
    cells: list[str] = []
    index = 0

    while index < len(lines):
        if lines[index] != ".. jupyter-execute::":
            index += 1
            continue

        index += 1
        while index < len(lines) and not lines[index]:
            index += 1

        cell_lines: list[str] = []
        while index < len(lines) and (not lines[index] or lines[index].startswith("   ")):
            line = lines[index]
            cell_lines.append(line[3:] if line else "")
            index += 1
        cells.append("\n".join(cell_lines).rstrip())

    return cells


def test_grouped_basis_documentation_example() -> None:
    """The notebook cells retain the documented layout and metadata contract."""
    document = Path(__file__).parents[1] / "docs" / "usage" / "grouped_basis_layout.rst"
    cells = _jupyter_cells(document)
    namespace: dict[str, object] = {}

    assert len(cells) == 6
    for cell in cells:
        exec(cell, namespace)

    result = namespace["result"]
    matrix = namespace["matrix"]
    restored_matrix = namespace["restored_matrix"]

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
    assert namespace["uncovered_error"] == "BasisLayout partitions leave 2 grid cells unmapped."
    assert namespace["inner_state"].region.values.tolist() == [0, 1]
    assert namespace["outer_state"].region.values.tolist() == [2]
    xr.testing.assert_identical(restored_matrix, matrix)
