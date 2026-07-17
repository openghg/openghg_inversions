"""Tests for experimental dyadic SLS visualization output."""

from pathlib import Path

import matplotlib
import numpy as np

from openghg_inversions.basis.experimental.dyadic.state import PartitionState
from openghg_inversions.basis.experimental.dyadic.tree import DyadicTree
from openghg_inversions.basis.experimental.dyadic.visualization import (
    SLSVisualizationFrame,
    render_partition_comparison,
    render_search_gif,
)


def _example_states() -> tuple[DyadicTree, PartitionState, PartitionState]:
    """Return a tiny tree and two valid partitions for rendering tests."""
    tree = DyadicTree.from_shape((3, 4))
    initial_state = PartitionState.root(tree)
    best_state = initial_state.split(tree, tree.root_id)
    return tree, initial_state, best_state


def test_visualization_module_uses_agg_backend() -> None:
    """Visualization imports force the non-interactive Agg backend."""
    assert matplotlib.get_backend().lower() == "agg"


def test_render_partition_comparison_writes_nonempty_png(tmp_path: Path) -> None:
    """Static comparison rendering writes a non-empty PNG artifact."""
    tree, initial_state, best_state = _example_states()
    background = np.arange(12, dtype=float).reshape(tree.shape)
    output_path = tmp_path / "nested" / "comparison.png"

    result = render_partition_comparison(
        background,
        tree,
        initial_state,
        best_state,
        initial_score=1.25,
        best_score=2.5,
        output_path=output_path,
        dpi=72,
    )

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_render_search_gif_writes_nonempty_animation(tmp_path: Path) -> None:
    """Animated rendering writes selected states as a non-empty GIF."""
    tree, initial_state, best_state = _example_states()
    background = np.arange(12, dtype=float).reshape(tree.shape)
    frames = (
        SLSVisualizationFrame(
            state=initial_state,
            iteration=0,
            current_score=1.25,
            best_score=1.25,
            temperature=1.0,
            accepted=False,
        ),
        SLSVisualizationFrame(
            state=best_state,
            iteration=3,
            current_score=2.5,
            best_score=2.5,
            temperature=0.1,
            accepted=True,
        ),
    )
    output_path = tmp_path / "search.gif"

    result = render_search_gif(background, tree, frames, output_path, fps=2, dpi=60)

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith((b"GIF87a", b"GIF89a"))


def test_render_search_gif_rejects_unordered_frames(tmp_path: Path) -> None:
    """Animation rejects records that do not advance in iteration order."""
    tree, initial_state, _ = _example_states()
    frames = (
        SLSVisualizationFrame(initial_state, 2, 1.0, 1.0, 0.5, True),
        SLSVisualizationFrame(initial_state, 1, 1.0, 1.0, 0.4, False),
    )

    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        render_search_gif(np.ones(tree.shape), tree, frames, tmp_path / "unordered.gif")
