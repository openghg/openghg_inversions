"""Exact geometry and balance tests for the tiny full-tiling oracle."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import combinations
from math import exp, log

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc import full_tiling as full_tiling_module
from openghg_inversions.experimental.rjmcmc.full_tiling import (
    AdditiveAlphaPrior,
    Axis,
    EdgeFlipPath,
    LeafTiling,
    MergeChoice,
    Rectangle,
    RelocationPath,
    SplitChoice,
    TilingState,
    edge_flip_paths,
    enumerate_tilings,
    is_recursive_bisection_tiling,
    merge_choices,
    propose_edge_flip,
    propose_resolution_relocation,
    relocation_path_log_probability,
    relocation_paths,
    split_choices,
)


def _state_with_mass_map(
    tiling: LeafTiling,
    masses: dict[Rectangle, float],
) -> TilingState:
    """Return a state whose masses follow canonical leaf order."""
    return TilingState(tiling, np.array([masses[leaf] for leaf in tiling.leaves]))


def _two_by_two_cell_tiling() -> LeafTiling:
    """Return the canonical four-cell tiling of a two-by-two grid."""
    return LeafTiling(
        (2, 2),
        (
            Rectangle(0, 1, 0, 1),
            Rectangle(0, 1, 1, 2),
            Rectangle(1, 2, 0, 1),
            Rectangle(1, 2, 1, 2),
        ),
    )


def _brute_force_merge_choices(tiling: LeafTiling) -> tuple[MergeChoice, ...]:
    """Return merge choices using the original exhaustive pair scan."""
    choices: set[MergeChoice] = set()
    for first, second in combinations(tiling.leaves, 2):
        if first.row_start == second.row_start and first.row_stop == second.row_stop:
            parent = Rectangle(
                first.row_start,
                first.row_stop,
                min(first.col_start, second.col_start),
                max(first.col_stop, second.col_stop),
            )
            if set(parent.midpoint_children("vertical")) == {first, second}:
                choices.add(MergeChoice(parent, "vertical"))
        if first.col_start == second.col_start and first.col_stop == second.col_stop:
            parent = Rectangle(
                min(first.row_start, second.row_start),
                max(first.row_stop, second.row_stop),
                first.col_start,
                first.col_stop,
            )
            if set(parent.midpoint_children("horizontal")) == {first, second}:
                choices.add(MergeChoice(parent, "horizontal"))
    return tuple(sorted(choices))


@pytest.mark.parametrize(
    ("bounds", "error"),
    [
        ((0.0, 1, 0, 1), TypeError),
        ((False, 1, 0, 1), TypeError),
        ((1, 1, 0, 1), ValueError),
        ((0, 1, 2, 1), ValueError),
    ],
)
def test_rectangle_rejects_invalid_bounds(
    bounds: tuple[object, object, object, object],
    error: type[Exception],
) -> None:
    """Rectangle bounds must be integral and define nonempty intervals."""
    with pytest.raises(error):
        Rectangle(*bounds)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "leaves",
    [
        (Rectangle(0, 2, 0, 1),),
        (Rectangle(0, 2, 0, 2), Rectangle(0, 1, 0, 1)),
        (Rectangle(0, 2, 0, 1), Rectangle(0, 2, 0, 1)),
        (Rectangle(0, 2, 0, 1), Rectangle(0, 2, 1, 3)),
    ],
)
def test_leaf_tiling_rejects_incomplete_overlap_duplicate_and_out_of_bounds(
    leaves: tuple[Rectangle, ...],
) -> None:
    """A tiling must be an exact, nonoverlapping cover of its declared grid."""
    with pytest.raises(ValueError):
        LeafTiling((2, 2), leaves)


def test_inputs_are_owned_and_public_value_objects_are_immutable() -> None:
    """Priors and states must copy arrays and expose immutable public values."""
    weights = np.arange(1.0, 5.0).reshape(2, 2)
    prior = AdditiveAlphaPrior(weights, concentration=7.0)
    weights[...] = -1.0

    np.testing.assert_array_equal(prior.cell_weights, np.arange(1.0, 5.0).reshape(2, 2))
    assert not prior.cell_weights.flags.writeable
    with pytest.raises(ValueError):
        prior.cell_weights[0, 0] = 10.0

    tiling = LeafTiling.root((2, 2))
    masses = np.array([3.0])
    state = TilingState(tiling, masses)
    masses[0] = 8.0

    np.testing.assert_array_equal(state.leaf_masses, [3.0])
    assert not state.leaf_masses.flags.writeable
    with pytest.raises(ValueError):
        state.leaf_masses[0] = 4.0
    with pytest.raises(FrozenInstanceError):
        tiling.shape = (1, 1)  # type: ignore[misc]


def test_state_and_path_catalogues_reject_non_recursive_exact_cover() -> None:
    """A non-recursive cover must not expose a one-way relocation path."""
    non_recursive = LeafTiling(
        (1, 5),
        (
            Rectangle(0, 1, 0, 1),
            Rectangle(0, 1, 1, 2),
            Rectangle(0, 1, 2, 4),
            Rectangle(0, 1, 4, 5),
        ),
    )

    assert not is_recursive_bisection_tiling(non_recursive)
    with pytest.raises(ValueError, match="recursive"):
        TilingState(non_recursive, np.ones(4))
    with pytest.raises(ValueError, match="recursive"):
        edge_flip_paths(non_recursive)
    with pytest.raises(ValueError, match="recursive"):
        relocation_paths(non_recursive)


def test_relative_weight_normalization_is_stable_and_totals_must_be_finite() -> None:
    """Large relative weights normalize safely while unrepresentable inputs fail."""
    prior = AdditiveAlphaPrior(np.full((2, 2), 1e308), concentration=5.0)
    alphas = prior.leaf_alphas(_two_by_two_cell_tiling())

    assert np.all(np.isfinite(alphas))
    assert np.sum(alphas) == pytest.approx(prior.concentration)
    with pytest.raises(ValueError, match="representable"):
        AdditiveAlphaPrior(
            np.array([[np.nextafter(0.0, 1.0), 1e308]]),
            concentration=2.0,
        )
    with pytest.raises(ValueError, match="concentrations"):
        AdditiveAlphaPrior(
            np.ones((1, 2)),
            concentration=np.nextafter(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="total leaf mass"):
        TilingState(
            LeafTiling.root((1, 2)).split(SplitChoice(Rectangle(0, 1, 0, 2), "vertical")),
            np.array([1e308, 1e308]),
        )


def test_beta_density_requires_an_ordered_midpoint_friend_pair() -> None:
    """The public Beta density must reject overlapping or reversed children."""
    prior = AdditiveAlphaPrior(np.ones((2, 2)), concentration=3.0)
    child = Rectangle(0, 1, 0, 1)
    with pytest.raises(ValueError, match="midpoint-friend"):
        prior.log_beta_density((child, child), 0.5)
    left, right = Rectangle(0, 2, 0, 2).midpoint_children("vertical")
    with pytest.raises(ValueError, match="midpoint-friend"):
        prior.log_beta_density((right, left), 0.5)


def test_canonical_tiling_is_independent_of_split_history() -> None:
    """Vertical-first and horizontal-first histories must yield one value."""
    root = LeafTiling.root((2, 2))
    vertical_halves = root.split(SplitChoice(root.leaves[0], "vertical"))
    vertical_then_horizontal = vertical_halves
    for leaf in vertical_halves.leaves:
        vertical_then_horizontal = vertical_then_horizontal.split(SplitChoice(leaf, "horizontal"))

    horizontal_halves = root.split(SplitChoice(root.leaves[0], "horizontal"))
    horizontal_then_vertical = horizontal_halves
    for leaf in horizontal_halves.leaves:
        horizontal_then_vertical = horizontal_then_vertical.split(SplitChoice(leaf, "vertical"))

    assert vertical_then_horizontal == horizontal_then_vertical
    assert vertical_then_horizontal == _two_by_two_cell_tiling()
    assert hash(vertical_then_horizontal) == hash(horizontal_then_vertical)


def test_two_by_two_enumeration_has_exact_unique_counts() -> None:
    """The complete two-by-two catalogues have the known counts at every K."""
    catalogues = [enumerate_tilings((2, 2), k) for k in range(1, 5)]

    assert tuple(map(len, catalogues)) == (1, 2, 4, 1)
    for catalogue in catalogues:
        assert len(catalogue) == len(set(catalogue))
        assert all(tiling.k == catalogue[0].k for tiling in catalogue)


def test_every_split_and_merge_are_geometry_reciprocals() -> None:
    """Each enumerated split must expose the exact inverse friend merge."""
    for k in range(1, 4):
        for source in enumerate_tilings((2, 2), k):
            for split in split_choices(source):
                candidate = source.split(split)
                reverse = MergeChoice(split.leaf, split.axis)

                assert reverse in merge_choices(candidate)
                assert candidate.merge(reverse) == source

    for k in range(2, 5):
        for source in enumerate_tilings((2, 2), k):
            for merge in merge_choices(source):
                candidate = source.merge(merge)
                reverse = SplitChoice(merge.parent, merge.axis)

                assert reverse in split_choices(candidate)
                assert candidate.split(reverse) == source


@pytest.mark.parametrize("shape", [(1, 5), (2, 3), (3, 3)])
def test_merge_choices_matches_brute_force_for_all_tiny_tilings(
    shape: tuple[int, int],
) -> None:
    """Indexed sibling lookup must equal pair scans on every tiny recursive state."""
    for k in range(1, shape[0] * shape[1] + 1):
        for tiling in enumerate_tilings(shape, k):
            assert merge_choices(tiling) == _brute_force_merge_choices(tiling)


def test_merge_choices_matches_brute_force_for_non_recursive_tiling() -> None:
    """Indexed sibling lookup must support arbitrary canonical exact covers."""
    tiling = LeafTiling(
        (3, 5),
        (
            Rectangle(0, 1, 0, 2),
            Rectangle(0, 1, 2, 5),
            Rectangle(1, 3, 0, 1),
            Rectangle(1, 3, 1, 3),
            Rectangle(1, 3, 3, 5),
        ),
    )

    assert not is_recursive_bisection_tiling(tiling)
    assert merge_choices(tiling) == _brute_force_merge_choices(tiling)


def test_merge_choice_cache_reuses_equal_tilings_without_aliasing_distinct_keys() -> None:
    """Equal tilings must share cached tuple work while distinct tilings do not."""
    root = Rectangle(0, 2, 0, 2)
    vertical = LeafTiling((2, 2), root.midpoint_children("vertical"))
    equal_vertical = LeafTiling((2, 2), tuple(reversed(vertical.leaves)))
    horizontal = LeafTiling((2, 2), root.midpoint_children("horizontal"))
    full_tiling_module._cached_merge_choices.cache_clear()

    first = merge_choices(vertical)
    same = merge_choices(equal_vertical)
    distinct = merge_choices(horizontal)
    cache_info = full_tiling_module._cached_merge_choices.cache_info()

    assert same is first
    assert distinct is not first
    assert cache_info.hits == 1
    assert cache_info.misses == 2
    assert cache_info.maxsize == 256


def test_recursive_tiling_cache_reuses_equal_keys_and_separates_distinct_keys() -> None:
    """Recursive-membership checks must share only equal immutable tiling keys."""
    root = Rectangle(0, 2, 0, 2)
    vertical = LeafTiling((2, 2), root.midpoint_children("vertical"))
    equal_vertical = LeafTiling((2, 2), tuple(reversed(vertical.leaves)))
    horizontal = LeafTiling((2, 2), root.midpoint_children("horizontal"))
    full_tiling_module._cached_is_recursive_bisection_tiling.cache_clear()

    assert is_recursive_bisection_tiling(vertical)
    assert is_recursive_bisection_tiling(equal_vertical)
    assert is_recursive_bisection_tiling(horizontal)
    cache_info = full_tiling_module._cached_is_recursive_bisection_tiling.cache_info()

    assert cache_info.hits == 1
    assert cache_info.misses == 2
    assert cache_info.maxsize == 256


def test_cached_tiling_catalogues_retain_public_type_validation() -> None:
    """Public cached-catalogue wrappers must reject non-tiling inputs first."""
    with pytest.raises(TypeError, match="tiling must be a LeafTiling"):
        merge_choices([])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tiling must be a LeafTiling"):
        is_recursive_bisection_tiling([])  # type: ignore[arg-type]


def test_merge_choices_midpoint_checks_scale_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sibling lookup must avoid a quadratic number of pair validations."""
    leaf_count = 128
    tiling = LeafTiling(
        (leaf_count, 1),
        tuple(Rectangle(row, row + 1, 0, 1) for row in range(leaf_count)),
    )
    midpoint_calls = 0
    original_midpoint_children = Rectangle.midpoint_children

    def counted_midpoint_children(
        rectangle: Rectangle,
        axis: Axis,
    ) -> tuple[Rectangle, Rectangle]:
        """Count exact midpoint validations performed during lookup."""
        nonlocal midpoint_calls
        midpoint_calls += 1
        return original_midpoint_children(rectangle, axis)

    monkeypatch.setattr(Rectangle, "midpoint_children", counted_midpoint_children)

    choices = merge_choices(tiling)

    assert len(choices) == leaf_count - 1
    assert midpoint_calls <= 4 * leaf_count


def test_additive_alpha_and_allocation_are_decomposition_independent() -> None:
    """Global cell weights must give identical allocation density by either tree."""
    prior = AdditiveAlphaPrior(
        np.array([[1.0, 2.0], [4.0, 8.0]]),
        concentration=6.5,
    )
    tiling = _two_by_two_cell_tiling()
    state = _state_with_mass_map(
        tiling,
        {
            Rectangle(0, 1, 0, 1): 0.7,
            Rectangle(0, 1, 1, 2): 1.1,
            Rectangle(1, 2, 0, 1): 2.3,
            Rectangle(1, 2, 1, 2): 4.9,
        },
    )
    root = Rectangle(0, 2, 0, 2)
    left, right = root.midpoint_children("vertical")
    top, bottom = root.midpoint_children("horizontal")
    top_left, bottom_left = left.midpoint_children("horizontal")
    top_right, bottom_right = right.midpoint_children("horizontal")
    top_left_again, top_right_again = top.midpoint_children("vertical")
    bottom_left_again, bottom_right_again = bottom.midpoint_children("vertical")

    assert (top_left, top_right, bottom_left, bottom_right) == (
        top_left_again,
        top_right_again,
        bottom_left_again,
        bottom_right_again,
    )
    assert prior.alpha(root) == pytest.approx(prior.concentration)
    assert sum(prior.leaf_alphas(tiling)) == pytest.approx(prior.concentration)
    assert prior.alpha(left) == pytest.approx(prior.alpha(top_left) + prior.alpha(bottom_left))
    assert prior.alpha(top) == pytest.approx(prior.alpha(top_left) + prior.alpha(top_right))

    total = state.total_mass
    left_mass = state.mass(top_left) + state.mass(bottom_left)
    right_mass = state.mass(top_right) + state.mass(bottom_right)
    top_mass = state.mass(top_left) + state.mass(top_right)
    bottom_mass = state.mass(bottom_left) + state.mass(bottom_right)
    vertical_log_density = (
        prior.log_beta_density((left, right), left_mass / total)
        - log(total)
        + prior.log_beta_density(
            (top_left, bottom_left),
            state.mass(top_left) / left_mass,
        )
        - log(left_mass)
        + prior.log_beta_density(
            (top_right, bottom_right),
            state.mass(top_right) / right_mass,
        )
        - log(right_mass)
    )
    horizontal_log_density = (
        prior.log_beta_density((top, bottom), top_mass / total)
        - log(total)
        + prior.log_beta_density(
            (top_left, top_right),
            state.mass(top_left) / top_mass,
        )
        - log(top_mass)
        + prior.log_beta_density(
            (bottom_left, bottom_right),
            state.mass(bottom_left) / bottom_mass,
        )
        - log(bottom_mass)
    )
    direct_log_density = prior.log_mass_allocation_density(state)

    assert vertical_log_density == pytest.approx(direct_log_density, abs=2e-14)
    assert horizontal_log_density == pytest.approx(direct_log_density, abs=2e-14)


def test_edge_flip_has_exact_reverse_and_prior_auxiliary_cancellation() -> None:
    """A two-by-two flip must conserve mass and cancel all continuous terms."""
    prior = AdditiveAlphaPrior(
        np.array([[1.0, 3.0], [2.0, 5.0]]),
        concentration=4.0,
    )
    root = LeafTiling.root((2, 2))
    source_tiling = root.split(SplitChoice(root.leaves[0], "vertical"))
    source = _state_with_mass_map(
        source_tiling,
        {
            Rectangle(0, 2, 0, 1): 2.5,
            Rectangle(0, 2, 1, 2): 6.0,
        },
    )
    path = edge_flip_paths(source_tiling)[0]
    transition = propose_edge_flip(prior, source, path=path, new_fraction=0.37)
    old_fraction = source.mass(path.merge.children[0]) / source.total_mass
    assert isinstance(transition.reverse_path, EdgeFlipPath)
    reverse = propose_edge_flip(
        prior,
        transition.candidate,
        path=transition.reverse_path,
        new_fraction=old_fraction,
    )

    assert transition.valid
    assert transition.reverse_path == EdgeFlipPath(
        MergeChoice(path.merge.parent, path.target_axis),
        path.merge.axis,
    )
    assert transition.candidate.total_mass == pytest.approx(source.total_mass)
    assert transition.log_jacobian == 0.0
    assert transition.delta_log_allocation_prior + (
        transition.log_q_reverse_auxiliary - transition.log_q_forward_auxiliary
    ) == pytest.approx(0.0, abs=2e-14)
    assert transition.log_acceptance_ratio == pytest.approx(0.0, abs=2e-14)
    assert reverse.candidate.tiling == source.tiling
    np.testing.assert_allclose(reverse.candidate.leaf_masses, source.leaf_masses)
    assert reverse.reverse_path == path
    assert reverse.log_acceptance_ratio == pytest.approx(
        -transition.log_acceptance_ratio,
        abs=2e-14,
    )


def test_unequal_mass_relocation_has_reciprocal_jacobian_and_exact_cancellation() -> None:
    """Relocation must reverse pointwise and leave only its selection-degree ratio."""
    prior = AdditiveAlphaPrior(
        np.array([[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]),
        concentration=7.5,
    )
    source_tiling = enumerate_tilings((2, 3), 3)[0]
    path = relocation_paths(source_tiling)[0]
    source = TilingState(source_tiling, np.array([1.2, 3.4, 7.8]))
    source_total = sum(source.mass(child) for child in path.merge.children)
    destination_total = source.mass(path.split.leaf)
    old_fraction = source.mass(path.merge.children[0]) / source_total
    transition = propose_resolution_relocation(
        prior,
        source,
        path=path,
        new_fraction=0.23,
    )
    assert isinstance(transition.reverse_path, RelocationPath)
    reverse = propose_resolution_relocation(
        prior,
        transition.candidate,
        path=transition.reverse_path,
        new_fraction=old_fraction,
    )

    assert transition.valid
    assert source_total != pytest.approx(destination_total)
    assert transition.candidate.total_mass == pytest.approx(source.total_mass)
    assert exp(transition.log_jacobian) == pytest.approx(destination_total / source_total)
    assert exp(transition.log_jacobian + reverse.log_jacobian) == pytest.approx(1.0)
    continuous_terms = (
        transition.delta_log_allocation_prior
        + transition.log_q_reverse_auxiliary
        - transition.log_q_forward_auxiliary
        + transition.log_jacobian
    )
    selection_ratio = transition.log_q_reverse_selection - transition.log_q_forward_selection
    assert continuous_terms == pytest.approx(0.0, abs=3e-14)
    assert transition.log_acceptance_ratio == pytest.approx(selection_ratio, abs=3e-14)
    assert reverse.candidate.tiling == source.tiling
    np.testing.assert_allclose(reverse.candidate.leaf_masses, source.leaf_masses)
    assert reverse.reverse_path == path
    assert reverse.log_acceptance_ratio == pytest.approx(
        -transition.log_acceptance_ratio,
        abs=3e-14,
    )


def test_invalid_paths_and_fractions_are_explicit_self_transitions() -> None:
    """Unavailable paths and unsupported fractions must retain the source object."""
    prior = AdditiveAlphaPrior(np.ones((2, 3)), concentration=3.0)
    source_tiling = enumerate_tilings((2, 3), 3)[0]
    source = TilingState(source_tiling, np.array([1.0, 2.0, 3.0]))
    valid_relocation = relocation_paths(source_tiling)[0]
    unavailable_relocation = RelocationPath(
        MergeChoice(Rectangle(0, 2, 0, 3), "vertical"),
        valid_relocation.split,
    )
    edge_source_tiling = LeafTiling.root((2, 3)).split(SplitChoice(Rectangle(0, 2, 0, 3), "vertical"))
    edge_source = TilingState(edge_source_tiling, np.array([2.0, 4.0]))
    unavailable_edge = EdgeFlipPath(
        MergeChoice(Rectangle(0, 2, 0, 3), "horizontal"),
        "vertical",
    )

    transitions = [
        propose_resolution_relocation(
            prior,
            source,
            path=unavailable_relocation,
            new_fraction=0.4,
        ),
        propose_resolution_relocation(
            prior,
            source,
            path=valid_relocation,
            new_fraction=float("nan"),
        ),
        propose_edge_flip(
            prior,
            edge_source,
            path=unavailable_edge,
            new_fraction=0.4,
        ),
        propose_edge_flip(
            prior,
            edge_source,
            path=edge_flip_paths(edge_source_tiling)[0],
            new_fraction=1.0,
        ),
    ]

    for transition, expected_source in (
        (transitions[0], source),
        (transitions[1], source),
        (transitions[2], edge_source),
        (transitions[3], edge_source),
    ):
        assert not transition.valid
        assert transition.candidate is expected_source
        assert transition.reverse_path is None
        assert transition.reason
        assert transition.log_acceptance_ratio == -np.inf


def test_exhaustive_fixed_k_topology_kernel_is_connected_and_reversible() -> None:
    """The connected 2x3, K=4 mixture must preserve the uniform topology law.

    Prior-matched Beta auxiliaries integrate to one.  Their cancellation is
    pointwise in the fraction, so one arbitrary interior fraction is enough
    to evaluate each discrete path flow exactly.
    """
    shape = (2, 3)
    catalog = enumerate_tilings(shape, 4)
    indices = {tiling: index for index, tiling in enumerate(catalog)}
    prior = AdditiveAlphaPrior(
        np.array([[1.0, 2.0, 3.0], [5.0, 7.0, 11.0]]),
        concentration=9.0,
    )
    matrix = np.zeros((len(catalog), len(catalog)))
    rejected_mass = np.zeros(len(catalog))
    move_weight = 0.5

    assert len(catalog) == 8
    for source_index, tiling in enumerate(catalog):
        source = TilingState(tiling, prior.leaf_alphas(tiling))
        move_catalogues = (
            edge_flip_paths(tiling),
            relocation_paths(tiling),
        )
        for paths in move_catalogues:
            if not paths:
                matrix[source_index, source_index] += move_weight
                rejected_mass[source_index] += move_weight
                continue

            selected_probability = 0.0
            for path in paths:
                if isinstance(path, EdgeFlipPath):
                    transition = propose_edge_flip(
                        prior,
                        source,
                        path=path,
                        new_fraction=0.37,
                    )
                else:
                    transition = propose_resolution_relocation(
                        prior,
                        source,
                        path=path,
                        new_fraction=0.37,
                    )
                assert transition.candidate.tiling in indices
                candidate_index = indices[transition.candidate.tiling]
                forward_probability = move_weight * exp(transition.log_q_forward_selection)
                selected_probability += forward_probability
                acceptance_probability = exp(min(0.0, transition.log_acceptance_ratio))
                matrix[source_index, candidate_index] += forward_probability * acceptance_probability
                rejected_path_probability = forward_probability * (1.0 - acceptance_probability)
                matrix[source_index, source_index] += rejected_path_probability
                rejected_mass[source_index] += rejected_path_probability

                continuous_terms = (
                    transition.delta_log_allocation_prior
                    + transition.log_q_reverse_auxiliary
                    - transition.log_q_forward_auxiliary
                    + transition.log_jacobian
                )
                assert continuous_terms == pytest.approx(0.0, abs=5e-14)
                assert transition.log_acceptance_ratio == pytest.approx(
                    transition.log_q_reverse_selection - transition.log_q_forward_selection,
                    abs=5e-14,
                )
                if isinstance(path, RelocationPath):
                    assert exp(relocation_path_log_probability(tiling, path)) == pytest.approx(
                        exp(transition.log_q_forward_selection),
                        abs=2e-15,
                    )

            assert selected_probability <= move_weight + 2e-15
            unavailable_path_probability = move_weight - selected_probability
            matrix[source_index, source_index] += unavailable_path_probability
            rejected_mass[source_index] += unavailable_path_probability

    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=3e-15)
    assert np.any(rejected_mass > 0.0)
    assert np.all(np.diag(matrix) >= rejected_mass - 2e-15)
    np.testing.assert_allclose(matrix, matrix.T, rtol=2e-13, atol=3e-15)
    support = matrix > 1e-15
    np.testing.assert_array_equal(support, support.T)
    uniform = np.full(len(catalog), 1.0 / len(catalog))
    np.testing.assert_allclose(uniform @ matrix, uniform, rtol=0.0, atol=3e-15)

    reached = {0}
    frontier = [0]
    while frontier:
        source_index = frontier.pop()
        for candidate_index in np.flatnonzero(support[source_index]):
            if candidate_index not in reached:
                reached.add(int(candidate_index))
                frontier.append(int(candidate_index))
    assert reached == set(range(len(catalog)))
