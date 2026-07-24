"""Run the exact fixed-K canonical full-tiling correctness demonstration.

The script enumerates the complete ``2 x 4``, ``K=4`` recursive-bisection
tiling space and assembles an equal-weight edge-flip/resolution-relocation MH
kernel. One fixed interior auxiliary fraction is sufficient to verify the
pointwise additive-alpha cancellation. The script prints stochasticity,
detailed-balance, stationarity, cancellation, and connectivity diagnostics to
stdout; it is an exact oracle, not a Monte Carlo or production inversion run.
"""

from __future__ import annotations

import math

import numpy as np

from openghg_inversions.experimental.rjmcmc.full_tiling import (
    AdditiveAlphaPrior,
    Axis,
    LeafTiling,
    SplitChoice,
    TilingState,
    edge_flip_paths,
    enumerate_tilings,
    propose_edge_flip,
    propose_resolution_relocation,
    relocation_paths,
)


def _four_cell_tiling(*, first_axis: Axis) -> LeafTiling:
    """Construct the same four-cell tiling through one of two split orders.

    Args:
        first_axis: Root split orientation.

    Returns:
        Canonical four-cell leaf tiling.
    """
    tiling = LeafTiling.root((2, 2))
    root = tiling.leaves[0]
    tiling = tiling.split(SplitChoice(root, first_axis))
    second_axis: Axis = "vertical" if first_axis == "horizontal" else "horizontal"
    for leaf in tuple(tiling.leaves):
        tiling = tiling.split(SplitChoice(leaf, second_axis))
    return tiling


def _transition_matrix() -> tuple[np.ndarray, float]:
    """Build the exact prior-only edge-flip/relocation mixture.

    Returns:
        Transition matrix in :func:`enumerate_tilings` order and the maximum
        residual after cancelling allocation, auxiliary, and Jacobian terms.
    """
    shape = (2, 4)
    tilings = enumerate_tilings(shape, 4)
    index = {tiling: position for position, tiling in enumerate(tilings)}
    prior = AdditiveAlphaPrior(
        np.arange(1.0, 9.0).reshape(shape),
        concentration=3.7,
    )
    matrix = np.zeros((len(tilings), len(tilings)))
    cancellation_error = 0.0
    for source_position, tiling in enumerate(tilings):
        state = TilingState(tiling, prior.leaf_alphas(tiling))
        for path in edge_flip_paths(tiling):
            transition = propose_edge_flip(prior, state, path=path, new_fraction=0.37)
            discrete_log_ratio = (
                transition.log_q_reverse_selection - transition.log_q_forward_selection
            )
            cancellation_error = max(
                cancellation_error,
                abs(transition.log_acceptance_ratio - discrete_log_ratio),
            )
            probability = 0.5 * math.exp(transition.log_q_forward_selection)
            acceptance = min(1.0, math.exp(transition.log_acceptance_ratio))
            matrix[
                source_position,
                index[transition.candidate.tiling],
            ] += probability * acceptance
        for path in relocation_paths(tiling):
            transition = propose_resolution_relocation(
                prior,
                state,
                path=path,
                new_fraction=0.37,
            )
            discrete_log_ratio = (
                transition.log_q_reverse_selection - transition.log_q_forward_selection
            )
            cancellation_error = max(
                cancellation_error,
                abs(transition.log_acceptance_ratio - discrete_log_ratio),
            )
            probability = 0.5 * math.exp(transition.log_q_forward_selection)
            acceptance = min(1.0, math.exp(transition.log_acceptance_ratio))
            matrix[
                source_position,
                index[transition.candidate.tiling],
            ] += probability * acceptance
        matrix[source_position, source_position] += 1.0 - matrix[source_position].sum()
    return matrix, cancellation_error


def _is_connected(matrix: np.ndarray) -> bool:
    """Return whether the positive off-diagonal support graph is connected.

    Args:
        matrix: Square transition matrix.

    Returns:
        Whether every state is reachable in the undirected positive support.
    """
    seen = {0}
    pending = [0]
    while pending:
        source = pending.pop()
        for destination in np.flatnonzero(matrix[source] > 0.0):
            if destination != source and int(destination) not in seen:
                seen.add(int(destination))
                pending.append(int(destination))
    return len(seen) == matrix.shape[0]


def main() -> None:
    """Print exact finite-state correctness results to stdout."""
    history_quotient = _four_cell_tiling(first_axis="vertical") == _four_cell_tiling(
        first_axis="horizontal"
    )
    matrix, cancellation_error = _transition_matrix()
    uniform = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    print(f"construction histories quotient to one leaf state: {history_quotient}")
    print(f"unique 2x4 K=4 tilings: {matrix.shape[0]}")
    print(f"maximum row-sum error: {np.max(np.abs(matrix.sum(axis=1) - 1.0)):.3e}")
    print(f"maximum detailed-balance error: {np.max(np.abs(matrix - matrix.T)):.3e}")
    print(f"maximum uniform-stationarity error: {np.max(np.abs(uniform @ matrix - uniform)):.3e}")
    print(f"maximum additive-alpha cancellation error: {cancellation_error:.3e}")
    print(f"move graph connected: {_is_connected(matrix)}")


if __name__ == "__main__":
    main()
