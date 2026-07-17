"""Static and animated output helpers for experimental dyadic SLS runs.

The visualizations deliberately keep the supplied two-dimensional background
fixed and draw only partition boundaries over it. This avoids implying that
the arbitrary compact integer labels returned by
:meth:`~openghg_inversions.basis.experimental.dyadic.state.PartitionState.to_labels`
have a numerical meaning or stable colour identity between search steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.image import AxesImage

from .state import PartitionState
from .tree import DyadicTree

_SCORE_NAME = "Gaussian benchmark DFS"


@dataclass(frozen=True, slots=True)
class SLSVisualizationFrame:
    """Partition and scalar diagnostics for one selected SLS output frame.

    Attributes:
        state: Current dyadic partition at this recorded search step.
        iteration: Zero-based search iteration.
        current_score: Current value of the plotted search score for ``state``.
        best_score: Best plotted search score found through this iteration.
        cellwise_isotropic_dfs: Optional fixed DFS reference for an independent
            isotropic prior over all grid cells. It is not an upper bound when
            regional covariance is not projected from the cell prior.
        temperature: Non-negative stochastic-local-search temperature.
        accepted: Whether the proposal evaluated at this iteration was accepted.
    """

    state: PartitionState
    iteration: int
    current_score: float
    best_score: float
    temperature: float
    accepted: bool
    cellwise_isotropic_dfs: float | None = None

    def __post_init__(self) -> None:
        """Validate scalar frame diagnostics."""
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative.")
        if not np.isfinite(self.current_score):
            raise ValueError("current_score must be finite.")
        if not np.isfinite(self.best_score):
            raise ValueError("best_score must be finite.")
        if self.best_score < self.current_score:
            raise ValueError("best_score must not be smaller than current_score.")
        if self.cellwise_isotropic_dfs is not None and not np.isfinite(self.cellwise_isotropic_dfs):
            raise ValueError("cellwise_isotropic_dfs must be finite when provided.")
        if not np.isfinite(self.temperature) or self.temperature < 0.0:
            raise ValueError("temperature must be finite and non-negative.")

    @property
    def k(self) -> int:
        """Return the number of active basis regions in this frame."""
        return len(self.state.active)


def render_partition_comparison(
    background: np.ndarray,
    tree: DyadicTree,
    initial_state: PartitionState,
    best_state: PartitionState,
    initial_score: float,
    best_score: float,
    output_path: str | PathLike[str],
    *,
    background_label: str = "Fixed sensitivity background",
    title: str = "Dyadic SLS partition comparison",
    dpi: int = 150,
) -> Path:
    """Render initial and best partitions over the same fixed background.

    Args:
        background: Two-dimensional field to display beneath both partitions.
        tree: Dyadic tree defining the grid and tile boundaries.
        initial_state: Initial partition state.
        best_state: Highest-scoring partition state found by the search.
        initial_score: Gaussian benchmark DFS for ``initial_state``.
        best_score: Gaussian benchmark DFS for ``best_state``.
        output_path: Caller-selected image output path.
        background_label: Colour-bar label for the fixed background field.
        title: Figure title.
        dpi: Positive output resolution in dots per inch.

    Returns:
        Path to the written image.

    Raises:
        ValueError: If inputs are inconsistent or scalar values are invalid.
    """
    values = _validate_background(background, tree)
    initial_state.validate(tree)
    best_state.validate(tree)
    initial_value = _finite_score(initial_score, "initial_score")
    best_value = _finite_score(best_score, "best_score")
    _validate_dpi(dpi)
    path = _prepare_output_path(output_path)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), constrained_layout=True)
    try:
        image = None
        panels = (
            (axes[0], initial_state, "Initial", initial_value),
            (axes[1], best_state, "Best", best_value),
        )
        for axis, state, panel_name, score in panels:
            image = _draw_background(axis, values)
            _draw_partition(axis, state, tree)
            axis.set_title(f"{panel_name}: K={len(state.active)}\n{_SCORE_NAME}={score:.4g}")
        if image is not None:
            figure.colorbar(image, ax=axes, shrink=0.82, label=background_label)
        figure.suptitle(title)
        figure.savefig(path, dpi=dpi)
    finally:
        plt.close(figure)
    return path


def render_search_gif(
    background: np.ndarray,
    tree: DyadicTree,
    frames: Sequence[SLSVisualizationFrame],
    output_path: str | PathLike[str],
    *,
    background_label: str = "Fixed sensitivity background",
    title: str = "Dyadic stochastic local search",
    score_label: str = _SCORE_NAME,
    score_axis_label: str | None = None,
    show_region_count: bool = False,
    fps: int = 4,
    dpi: int = 100,
) -> Path:
    """Render selected SLS records as an animated partition and score trace.

    The caller controls trace size by supplying exactly the records that should
    become animation frames. Integer region labels are never colour mapped;
    only boundaries derived from each state's label array are updated.

    Args:
        background: Fixed two-dimensional field shown in every frame.
        tree: Dyadic tree defining the grid and tile boundaries.
        frames: Non-empty, iteration-ordered records selected by the caller.
        output_path: Caller-selected GIF output path.
        background_label: Colour-bar label for the fixed background field.
        title: Figure title.
        score_label: Label describing current and best score traces.
        score_axis_label: Optional broader y-axis label when plotted reference
            lines are not the same quantity as ``score_label``.
        show_region_count: Whether to add active-region count on a secondary
            axis in the trace panel.
        fps: Positive playback rate in frames per second.
        dpi: Positive output resolution in dots per inch.

    Returns:
        Path to the written GIF.

    Raises:
        ValueError: If inputs are inconsistent, frames are out of order, or
            rendering parameters are invalid.
    """
    values = _validate_background(background, tree)
    frame_records = tuple(frames)
    _validate_frames(frame_records, tree)
    if isinstance(fps, bool) or not isinstance(fps, (int, np.integer)) or fps <= 0:
        raise ValueError("fps must be a positive integer.")
    _validate_dpi(dpi)
    path = _prepare_output_path(output_path)
    if path.suffix.lower() != ".gif":
        raise ValueError("Animated output_path must use the .gif suffix.")

    figure, (partition_axis, score_axis) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.1, 1.0)},
    )
    try:
        image = _draw_background(partition_axis, values)
        figure.colorbar(image, ax=partition_axis, shrink=0.82, label=background_label)
        boundary_collection = _draw_partition(partition_axis, frame_records[0].state, tree)
        diagnostic_text = partition_axis.text(
            0.02,
            0.98,
            "",
            transform=partition_axis.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.6", "alpha": 0.9, "pad": 3.0},
        )

        iterations = np.asarray([frame.iteration for frame in frame_records], dtype=np.int64)
        current_scores = np.asarray([frame.current_score for frame in frame_records], dtype=np.float64)
        best_scores = np.asarray([frame.best_score for frame in frame_records], dtype=np.float64)
        region_counts = np.asarray([frame.k for frame in frame_records], dtype=np.int64)
        cellwise_isotropic_dfs = (
            np.asarray([frame.cellwise_isotropic_dfs for frame in frame_records], dtype=np.float64)
            if frame_records[0].cellwise_isotropic_dfs is not None
            else None
        )
        (current_line,) = score_axis.plot([], [], color="tab:blue", linewidth=1.5, label="Current")
        (best_line,) = score_axis.plot([], [], color="tab:red", linewidth=1.8, label="Best")
        cellwise_isotropic_dfs_line = None
        _configure_score_axis(
            score_axis,
            iterations,
            current_scores,
            best_scores,
            score_axis_label or score_label,
            cellwise_isotropic_dfs=cellwise_isotropic_dfs,
        )
        region_axis = None
        region_line = None
        legend_handles = [current_line, best_line]
        if cellwise_isotropic_dfs is not None:
            cellwise_isotropic_dfs_line = score_axis.axhline(
                float(cellwise_isotropic_dfs[0]),
                color="tab:green",
                linewidth=1.5,
                linestyle=":",
                label="Cellwise-I DFS (not bound)",
            )
            legend_handles.append(cellwise_isotropic_dfs_line)
        if show_region_count:
            region_axis = score_axis.twinx()
            (region_line,) = region_axis.plot(
                [],
                [],
                color="0.3",
                linewidth=1.2,
                linestyle="--",
                label="K",
            )
            _configure_region_axis(region_axis, region_counts)
            legend_handles.append(region_line)
        score_axis.legend(handles=legend_handles, loc="lower right")
        figure.suptitle(title)

        def update(frame_index: int) -> tuple[LineCollection, ...]:
            """Update partition boundaries, diagnostics, and score traces."""
            frame = frame_records[frame_index]
            labels = frame.state.to_labels(tree)
            boundary_collection.set_segments(_boundary_segments(labels))
            current_line.set_data(iterations[: frame_index + 1], current_scores[: frame_index + 1])
            best_line.set_data(iterations[: frame_index + 1], best_scores[: frame_index + 1])
            if region_line is not None:
                region_line.set_data(iterations[: frame_index + 1], region_counts[: frame_index + 1])
            diagnostic_text.set_text(_diagnostic_text(frame, score_label))
            return (boundary_collection,)

        animation = FuncAnimation(
            figure,
            update,
            frames=len(frame_records),
            interval=1000.0 / fps,
            repeat=False,
            blit=False,
        )
        animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(figure)
    return path


def _validate_background(background: np.ndarray, tree: DyadicTree) -> np.ma.MaskedArray:
    """Validate a background field and mask its non-finite values.

    Args:
        background: Candidate two-dimensional fixed background.
        tree: Tree whose shape the field must match.

    Returns:
        Floating masked array with non-finite values hidden.

    Raises:
        ValueError: If the background is not two-dimensional, has the wrong
            shape, or contains no finite values.
    """
    values = np.asarray(background, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("background must be two-dimensional.")
    if values.shape != tree.shape:
        raise ValueError(f"background shape {values.shape} does not match tree shape {tree.shape}.")
    if not np.isfinite(values).any():
        raise ValueError("background must contain at least one finite value.")
    return np.ma.masked_invalid(values)


def _validate_frames(frames: tuple[SLSVisualizationFrame, ...], tree: DyadicTree) -> None:
    """Validate selected animation records against one tree.

    Args:
        frames: Materialized visualization records in playback order.
        tree: Tree against which every partition state is validated.

    Raises:
        ValueError: If no records are supplied or iterations are not strictly
            increasing.
    """
    if not frames:
        raise ValueError("frames must contain at least one record.")
    previous_iteration = -1
    for frame in frames:
        frame.state.validate(tree)
        if frame.iteration <= previous_iteration:
            raise ValueError("frame iterations must be strictly increasing.")
        previous_iteration = frame.iteration
    has_cellwise_isotropic_dfs = [frame.cellwise_isotropic_dfs is not None for frame in frames]
    if any(has_cellwise_isotropic_dfs) and not all(has_cellwise_isotropic_dfs):
        raise ValueError("cellwise_isotropic_dfs must be provided for every frame or no frames.")


def _finite_score(score: float, name: str) -> float:
    """Return a finite score value or raise a named validation error."""
    value = float(score)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _validate_dpi(dpi: int) -> None:
    """Validate a positive output resolution."""
    if dpi < 1:
        raise ValueError("dpi must be positive.")


def _prepare_output_path(output_path: str | PathLike[str]) -> Path:
    """Normalize an output path and create its parent directories."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _draw_background(axis: Axes, background: np.ma.MaskedArray) -> AxesImage:
    """Draw one fixed grid background on an image axis."""
    image = axis.imshow(background, origin="lower", interpolation="nearest", cmap="viridis")
    axis.set_xlabel("Grid column")
    axis.set_ylabel("Grid row")
    axis.set_aspect("equal")
    return image


def _draw_partition(axis: Axes, state: PartitionState, tree: DyadicTree) -> LineCollection:
    """Draw stable grid-edge boundaries for one partition.

    Args:
        axis: Matplotlib image axis receiving the boundaries.
        state: Partition state whose compact labels define adjacent regions.
        tree: Tree defining the state and output grid.

    Returns:
        Mutable line collection suitable for animation updates.
    """
    labels = state.to_labels(tree)
    boundaries = LineCollection(_boundary_segments(labels), colors="white", linewidths=1.2)
    axis.add_collection(boundaries)
    return boundaries


def _boundary_segments(labels: np.ndarray) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Convert label transitions and the grid perimeter to line segments.

    Args:
        labels: Two-dimensional positive integer partition labels.

    Returns:
        Grid-edge line segments separating adjacent unequal labels, including
        the outer grid perimeter.
    """
    rows, cols = labels.shape
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = [
        ((-0.5, -0.5), (cols - 0.5, -0.5)),
        ((cols - 0.5, -0.5), (cols - 0.5, rows - 0.5)),
        ((cols - 0.5, rows - 0.5), (-0.5, rows - 0.5)),
        ((-0.5, rows - 0.5), (-0.5, -0.5)),
    ]
    for row, boundary_col in np.argwhere(labels[:, :-1] != labels[:, 1:]):
        x_coord = float(boundary_col) + 0.5
        segments.append(((x_coord, row - 0.5), (x_coord, row + 0.5)))
    for boundary_row, col in np.argwhere(labels[:-1, :] != labels[1:, :]):
        y_coord = float(boundary_row) + 0.5
        segments.append(((col - 0.5, y_coord), (col + 0.5, y_coord)))
    return segments


def _configure_score_axis(
    axis: Axes,
    iterations: np.ndarray,
    current_scores: np.ndarray,
    best_scores: np.ndarray,
    axis_label: str,
    *,
    cellwise_isotropic_dfs: np.ndarray | None = None,
) -> None:
    """Configure fixed limits and labels for an animated score axis.

    Args:
        axis: Matplotlib axis receiving current and best score traces.
        iterations: Ordered integer iteration values.
        current_scores: Current plotted score values.
        best_scores: Best-so-far plotted score values.
        axis_label: Label for the plotted score/reference scale.
        cellwise_isotropic_dfs: Optional independent-cell isotropic DFS
            reference plotted on the same scale.
    """
    x_min = float(iterations[0])
    x_max = float(iterations[-1])
    x_padding = max((x_max - x_min) * 0.03, 0.5)
    score_arrays = (
        (current_scores, best_scores)
        if cellwise_isotropic_dfs is None
        else (current_scores, best_scores, cellwise_isotropic_dfs)
    )
    all_scores = np.concatenate(score_arrays)
    y_min = float(all_scores.min())
    y_max = float(all_scores.max())
    y_padding = max((y_max - y_min) * 0.08, max(abs(y_min), abs(y_max), 1.0) * 0.02)
    axis.set_xlim(x_min - x_padding, x_max + x_padding)
    axis.set_ylim(y_min - y_padding, y_max + y_padding)
    axis.set_xlabel("Search iteration")
    axis.set_ylabel(axis_label)
    axis.set_title("Score history")
    axis.grid(alpha=0.25)


def _configure_region_axis(axis: Axes, region_counts: np.ndarray) -> None:
    """Configure a fixed secondary axis for active-region count.

    Args:
        axis: Secondary Matplotlib axis receiving the K trace.
        region_counts: Active-region count for every animation frame.
    """
    lower = int(region_counts.min())
    upper = int(region_counts.max())
    padding = max((upper - lower) * 0.08, 1.0)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_ylabel("Active regions (K)", color="0.3")
    axis.tick_params(axis="y", colors="0.3")


def _diagnostic_text(frame: SLSVisualizationFrame, score_label: str) -> str:
    """Format one frame's diagnostics for display without hidden semantics."""
    text = (
        f"Iteration: {frame.iteration}\n"
        f"K: {frame.k}\n"
        f"Temperature: {frame.temperature:.4g}\n"
        f"Accepted: {'yes' if frame.accepted else 'no'}\n"
        f"Current {score_label}: {frame.current_score:.4g}\n"
        f"Best {score_label}: {frame.best_score:.4g}"
    )
    if frame.cellwise_isotropic_dfs is not None:
        text += f"\nCellwise-I DFS (not bound): {frame.cellwise_isotropic_dfs:.4g}"
    return text


__all__ = ["SLSVisualizationFrame", "render_partition_comparison", "render_search_gif"]
