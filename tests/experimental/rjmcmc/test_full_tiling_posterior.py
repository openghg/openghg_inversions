"""Regression tests for likelihood-aware full-tiling posterior operations."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.full_tiling import (
    Rectangle,
    SplitChoice,
    TilingState,
    merge_choices,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingProblem,
    FullTilingPosteriorState,
    PosteriorTransitionTerms,
    build_full_tiling_posterior_state,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
    log_root_total_slice_density,
    propose_fixed_coefficient,
    propose_pair_allocation_refresh,
    propose_posterior_edge_flip,
    propose_posterior_resolution_relocation,
    propose_root_total_refresh,
    rescale_full_tiling_root_total,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    GammaBetaRHIMEAdapterResult,
    gamma_beta_problem_from_rhime_inputs,
)


def _adapter_and_raw() -> tuple[
    GammaBetaRHIMEAdapterResult,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return a shuffled-dimension adapter and its raw forward-model arrays."""
    raw = np.arange(1.0, 49.0).reshape(3, 4, 4)
    outer = np.arange(18.0).reshape(3, 6) / 10.0
    boundary = np.array([3.0, 5.0, 7.0])
    fixed_mean = np.arange(1.0, 7.0) / 4.0
    observations = boundary + raw.sum(axis=(1, 2)) + outer @ fixed_mean
    dataset = xr.Dataset(
        {
            "fp_x_flux": (
                ("lon", "nmeasure", "lat"),
                raw.transpose(2, 0, 1),
            ),
            "mf": ("nmeasure", observations),
            "mf_error": ("nmeasure", np.ones(3)),
            "outer": (("outer_region", "nmeasure"), outer.T),
            "boundary": ("nmeasure", boundary),
        },
        coords={
            "nmeasure": ["a", "b", "c"],
            "lat": np.arange(4) + 50.0,
            "lon": np.arange(4) - 3.0,
            "outer_region": [f"outer-{index}" for index in range(6)],
        },
    )
    weights = xr.DataArray(
        np.arange(1.0, 17.0).reshape(4, 4).T,
        dims=("lon", "lat"),
        coords={"lon": dataset.lon, "lat": dataset.lat},
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=weights,
        k_min=1,
        k_max=16,
        concentration=5.0,
        root_variance=0.3,
        sensitivity_name="fp_x_flux",
        observation_name="mf",
        observation_sd_name="mf_error",
        fixed_design_name="outer",
        fixed_offset_name="boundary",
        fixed_coefficient_prior_mean=fixed_mean,
        fixed_coefficient_prior_sd=np.full(6, 0.4),
    )
    return adapter, raw, outer, boundary


def _problem_state(
    k: int,
    *,
    likelihood_power: float = 1.0,
) -> tuple[FullTilingProblem, FullTilingPosteriorState]:
    """Build a small posterior problem and deterministic prior-mean state."""
    adapter, _, _, _ = _adapter_and_raw()
    base = replace(adapter.problem, likelihood_power=likelihood_power)
    problem = full_tiling_problem_from_gamma_beta_adapter(base, concentration=7.0)
    return problem, initialize_full_tiling_posterior_state(problem, k=k)


def _assert_rebuild_equal(state: FullTilingPosteriorState) -> None:
    """Assert an incremental candidate equals the public full rebuild oracle."""
    rebuilt = build_full_tiling_posterior_state(
        state.problem,
        allocation=state.allocation,
        fixed_coefficients=state.fixed_coefficients,
    )
    assert rebuilt.allocation.tiling == state.allocation.tiling
    for name in (
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        np.testing.assert_allclose(getattr(state, name), getattr(rebuilt, name), rtol=0.0, atol=5e-13)
    for name in (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    ):
        assert getattr(state, name) == pytest.approx(getattr(rebuilt, name), abs=1e-9)


def _valid_relocation(
    problem: FullTilingProblem,
    state: FullTilingPosteriorState,
) -> PosteriorTransitionTerms:
    """Return the first valid relocation from the fixed destination catalogue."""
    for merge in merge_choices(state.allocation.tiling):
        intermediate = state.allocation.tiling.merge(merge)
        for leaf in intermediate.leaves:
            for axis in ("horizontal", "vertical"):
                transition = propose_posterior_resolution_relocation(
                    problem,
                    state,
                    merge_choice=merge,
                    split_choice=SplitChoice(leaf, axis),
                    new_fraction=0.37,
                )
                if transition.valid:
                    return transition
    raise AssertionError("test geometry did not expose a valid relocation")


def test_public_posterior_terms_retain_generic_decomposed_mh_accounting() -> None:
    """Matched builders must not change the exported dataclass constructor."""
    _, state = _problem_state(4)
    generic = PosteriorTransitionTerms(
        candidate=state,
        move="pair_allocation_refresh",
        delta_log_likelihood=1.0,
        delta_log_root_prior=2.0,
        delta_log_allocation_prior=3.0,
        delta_log_fixed_coefficient_prior=4.0,
        log_q_forward_selection=-5.0,
        log_q_forward_auxiliary=-6.0,
        log_q_reverse_selection=-7.0,
        log_q_reverse_auxiliary=-8.0,
        log_jacobian=9.0,
    )

    assert generic.log_target_delta == 10.0
    assert generic.log_acceptance_ratio == 15.0


def test_shuffled_xarray_bridge_closes_inner_boundary_and_outer_prior_mean() -> None:
    """Shuffled labelled inputs retain exact inner, BC, and outer closure."""
    adapter, raw, outer, boundary = _adapter_and_raw()
    problem = full_tiling_problem_from_gamma_beta_adapter(adapter, concentration=7.0)
    state = initialize_full_tiling_posterior_state(problem, k=5)
    fixed_mean = np.arange(1.0, 7.0) / 4.0

    assert problem.shape == (4, 4)
    np.testing.assert_allclose(state.dynamic_prediction, raw.sum(axis=(1, 2)), atol=2e-13)
    np.testing.assert_allclose(state.fixed_prediction, boundary + outer @ fixed_mean, atol=2e-13)
    np.testing.assert_allclose(
        state.prediction,
        raw.sum(axis=(1, 2)) + boundary + outer @ fixed_mean,
        atol=2e-13,
    )


def test_rectangle_design_columns_match_direct_native_matrix_slices() -> None:
    """Lazy rectangle columns equal direct native sensitivity-matrix algebra."""
    adapter, _, _, _ = _adapter_and_raw()
    problem = full_tiling_problem_from_gamma_beta_adapter(adapter, concentration=7.0)
    rectangle = Rectangle(1, 4, 0, 2)
    mass = problem.normalized_nominal_mass[1:4, 0:2]
    native = problem.base.sensitivity.reshape(3, 4, 4)[:, 1:4, 0:2]
    expected = (native * mass[np.newaxis, :, :]).sum(axis=(1, 2)) / mass.sum()

    np.testing.assert_allclose(problem.design_column(rectangle), expected, rtol=0.0, atol=1e-13)
    assert problem.design_column(rectangle) is problem.rectangle_design_column(rectangle)


def test_complete_target_matches_independent_root_share_closed_form() -> None:
    """The target is Gamma plus Dirichlet shares, with no physical-mass Jacobian."""
    problem, source = _problem_state(4)
    root_total = 1.7
    shares = np.array([0.1, 0.2, 0.3, 0.4])
    coefficients = np.array([0.8, 1.1, 1.4, 1.7, 2.0, 2.3])
    state = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(source.allocation.tiling, root_total * shares),
        fixed_coefficients=coefficients,
    )

    gaussian = -0.5 * np.sum(
        (state.residual / problem.observation_sd) ** 2
        + math.log(2.0 * math.pi)
        + 2.0 * np.log(problem.observation_sd)
    )
    shape = problem.base.prior.root_shape
    rate = problem.base.prior.root_rate
    gamma = (
        shape * math.log(rate) - math.lgamma(shape) + (shape - 1.0) * math.log(root_total) - rate * root_total
    )
    alphas = problem.allocation_prior.leaf_alphas(source.allocation.tiling)
    dirichlet = (
        math.lgamma(float(np.sum(alphas)))
        - sum(math.lgamma(float(alpha)) for alpha in alphas)
        + float(np.sum((alphas - 1.0) * np.log(shares)))
    )
    fixed = problem.base.fixed_block
    assert fixed is not None
    log_variances = np.log1p((fixed.coefficient_prior_sd / fixed.coefficient_prior_mean) ** 2)
    log_means = np.log(fixed.coefficient_prior_mean) - 0.5 * log_variances
    lognormal = float(
        np.sum(
            -np.log(coefficients)
            - 0.5 * np.log(2.0 * math.pi * log_variances)
            - (np.log(coefficients) - log_means) ** 2 / (2.0 * log_variances)
        )
    )

    assert state.log_gaussian_likelihood == pytest.approx(gaussian)
    assert state.log_root_prior == pytest.approx(gamma)
    assert state.log_allocation_prior == pytest.approx(dirichlet)
    assert state.log_fixed_coefficient_prior == pytest.approx(lognormal)
    assert state.log_target == pytest.approx(gaussian + gamma + dirichlet + lognormal)

    rescaled = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(source.allocation.tiling, 2.9 * shares),
        fixed_coefficients=coefficients,
    )
    assert rescaled.log_allocation_prior == pytest.approx(dirichlet)


@pytest.mark.parametrize("likelihood_power", [0.0, 0.5, 1.0])
def test_log_root_slice_density_matches_exact_quadratic_and_state_difference(
    likelihood_power: float,
) -> None:
    """The log-root kernel exactly matches powered Gaussian--Gamma algebra."""
    problem, source = _problem_state(4, likelihood_power=likelihood_power)
    first_z = math.log(0.73)
    second_z = math.log(1.91)
    unit_dynamic = source.dynamic_prediction / source.root_total
    fixed_residual = source.fixed_prediction - problem.observations
    weighted_dynamic = unit_dynamic / problem.observation_sd
    weighted_fixed_residual = fixed_residual / problem.observation_sd
    quadratic = float(np.dot(weighted_dynamic, weighted_dynamic))
    linear = float(np.dot(weighted_dynamic, weighted_fixed_residual))
    constant = float(np.dot(weighted_fixed_residual, weighted_fixed_residual))
    assert linear < 0.0
    if likelihood_power > 0.0:
        assert problem.base.prior.root_rate + likelihood_power * linear < 0.0

    gaussian_normalizer = float(
        -np.log(problem.observation_sd).sum() - 0.5 * problem.observations.size * math.log(2.0 * math.pi)
    )
    prior = problem.base.prior

    def expected(z: float) -> float:
        """Evaluate the independent quadratic oracle in log-total coordinates."""
        root_total = math.exp(z)
        gaussian = gaussian_normalizer - 0.5 * (
            quadratic * root_total**2 + 2.0 * linear * root_total + constant
        )
        powered_gaussian = 0.0 if likelihood_power == 0.0 else likelihood_power * gaussian
        return float(
            powered_gaussian
            + prior.root_shape * math.log(prior.root_rate)
            - math.lgamma(prior.root_shape)
            + prior.root_shape * z
            - prior.root_rate * root_total
        )

    first = log_root_total_slice_density(
        problem,
        source,
        log_root_total=first_z,
    )
    second = log_root_total_slice_density(
        problem,
        source,
        log_root_total=second_z,
    )
    assert first == pytest.approx(expected(first_z), rel=0.0, abs=5e-10)
    assert second == pytest.approx(expected(second_z), rel=0.0, abs=5e-10)
    assert second - first == pytest.approx(
        expected(second_z) - expected(first_z),
        rel=0.0,
        abs=5e-10,
    )

    first_state = rescale_full_tiling_root_total(
        problem,
        source,
        new_root_total=math.exp(first_z),
    )
    second_state = rescale_full_tiling_root_total(
        problem,
        source,
        new_root_total=math.exp(second_z),
    )
    scientific_difference = (
        second_state.log_likelihood
        + second_state.log_root_prior
        - first_state.log_likelihood
        - first_state.log_root_prior
    )
    assert second - first == pytest.approx(
        scientific_difference + second_z - first_z,
        rel=0.0,
        abs=5e-10,
    )


def test_prior_only_log_root_slice_density_is_likelihood_independent() -> None:
    """At power zero the log-root kernel ignores predictions and geometry."""
    problem, coarse = _problem_state(1, likelihood_power=0.0)
    fine = initialize_full_tiling_posterior_state(problem, k=6)
    fixed = propose_fixed_coefficient(
        problem,
        fine,
        coefficient_position=2,
        proposed_coefficient=4.2,
    ).candidate
    z = math.log(2.4)
    prior = problem.base.prior
    expected = (
        prior.root_shape * math.log(prior.root_rate)
        - math.lgamma(prior.root_shape)
        + prior.root_shape * z
        - prior.root_rate * math.exp(z)
    )

    assert log_root_total_slice_density(
        problem,
        coarse,
        log_root_total=z,
    ) == pytest.approx(expected)
    assert log_root_total_slice_density(
        problem,
        fixed,
        log_root_total=z,
    ) == pytest.approx(expected)


def test_log_root_slice_density_guards_underflow_and_overflow() -> None:
    """Finite lower-tail log totals survive while upper overflow is rejected."""
    problem, source = _problem_state(4, likelihood_power=0.5)
    lower = log_root_total_slice_density(
        problem,
        source,
        log_root_total=-1000.0,
    )

    assert math.isfinite(lower)
    assert (
        log_root_total_slice_density(
            problem,
            source,
            log_root_total=1000.0,
        )
        == -math.inf
    )
    for value in (math.inf, -math.inf, math.nan, True):
        assert (
            log_root_total_slice_density(
                problem,
                source,
                log_root_total=value,
            )
            == -math.inf
        )


def test_public_root_rescaling_preserves_shares_and_matches_proposal_and_rebuild() -> None:
    """Public root rescaling retains geometry and all state-construction parity."""
    problem, source = _problem_state(5)
    new_root_total = 2.75
    candidate = rescale_full_tiling_root_total(
        problem,
        source,
        new_root_total=new_root_total,
    )
    transition = propose_root_total_refresh(
        problem,
        source,
        new_root_total=new_root_total,
    )

    assert candidate.allocation.tiling is source.allocation.tiling
    assert candidate.root_total == pytest.approx(new_root_total)
    np.testing.assert_allclose(
        candidate.leaf_masses / candidate.root_total,
        source.leaf_masses / source.root_total,
        rtol=0.0,
        atol=3e-17,
    )
    np.testing.assert_array_equal(candidate.fixed_coefficients, source.fixed_coefficients)
    assert transition.valid
    for name in (
        "leaf_masses",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    ):
        np.testing.assert_array_equal(getattr(candidate, name), getattr(transition.candidate, name))
    for name in (
        "log_gaussian_likelihood",
        "log_likelihood",
        "log_root_prior",
        "log_allocation_prior",
        "log_fixed_coefficient_prior",
        "log_target",
    ):
        assert getattr(candidate, name) == getattr(transition.candidate, name)
    _assert_rebuild_equal(candidate)


def test_every_incremental_proposal_candidate_matches_a_full_rebuild() -> None:
    """Every valid proposal family preserves all full-rebuild posterior caches."""
    problem, edge_source = _problem_state(2)
    edge_merge = merge_choices(edge_source.allocation.tiling)[0]
    edge = propose_posterior_edge_flip(
        problem,
        edge_source,
        merge_choice=edge_merge,
        new_fraction=0.31,
    )
    assert edge.valid

    relocation_source = initialize_full_tiling_posterior_state(problem, k=4)
    relocation = _valid_relocation(problem, relocation_source)
    leaves = relocation_source.allocation.tiling.leaves
    pair = propose_pair_allocation_refresh(
        problem,
        relocation_source,
        first_leaf=leaves[0],
        second_leaf=leaves[-1],
        new_fraction=0.42,
    )
    root = propose_root_total_refresh(problem, relocation_source, new_root_total=1.7)
    fixed = propose_fixed_coefficient(
        problem,
        relocation_source,
        coefficient_position=4,
        proposed_coefficient=1.9,
        proposal_stdev=0.2,
    )

    for transition in (edge, relocation, pair, root, fixed):
        assert transition.valid
        _assert_rebuild_equal(transition.candidate)


def test_structural_reciprocals_and_root_gamma_terms_balance_exactly() -> None:
    """Edge, relocation, and root-refresh accounting has exact reverse balance."""
    problem, edge_source = _problem_state(2)
    edge_forward = propose_posterior_edge_flip(
        problem,
        edge_source,
        merge_choice=merge_choices(edge_source.allocation.tiling)[0],
        new_fraction=0.29,
    )
    assert edge_forward.valid
    assert edge_forward.reverse_merge_choice is not None
    edge_reverse = propose_posterior_edge_flip(
        problem,
        edge_forward.candidate,
        merge_choice=edge_forward.reverse_merge_choice,
        new_fraction=edge_source.leaf_masses[0] / edge_source.root_total,
    )
    assert edge_reverse.valid
    assert edge_reverse.candidate.allocation.tiling == edge_source.allocation.tiling
    np.testing.assert_allclose(edge_reverse.candidate.leaf_masses, edge_source.leaf_masses)
    assert edge_forward.log_acceptance_ratio + edge_reverse.log_acceptance_ratio == pytest.approx(0.0)

    relocation_source = initialize_full_tiling_posterior_state(problem, k=4)
    relocation_forward = _valid_relocation(problem, relocation_source)
    assert relocation_forward.reverse_merge_choice is not None
    assert relocation_forward.reverse_split_choice is not None
    reverse_children = relocation_forward.reverse_merge_choice.children
    original_children = relocation_forward.reverse_split_choice.leaf.midpoint_children(
        relocation_forward.reverse_split_choice.axis
    )
    old_fraction = relocation_source.allocation.mass(original_children[0]) / sum(
        relocation_source.allocation.mass(child) for child in original_children
    )
    relocation_reverse = propose_posterior_resolution_relocation(
        problem,
        relocation_forward.candidate,
        merge_choice=relocation_forward.reverse_merge_choice,
        split_choice=relocation_forward.reverse_split_choice,
        new_fraction=old_fraction,
    )
    assert relocation_reverse.valid
    assert relocation_reverse.candidate.allocation.tiling == relocation_source.allocation.tiling
    np.testing.assert_allclose(relocation_reverse.candidate.leaf_masses, relocation_source.leaf_masses)
    assert relocation_forward.log_jacobian == pytest.approx(-relocation_reverse.log_jacobian)
    assert relocation_forward.log_acceptance_ratio + relocation_reverse.log_acceptance_ratio == pytest.approx(
        0.0
    )
    assert reverse_children[0] < reverse_children[1]

    root = propose_root_total_refresh(problem, relocation_source, new_root_total=2.3)
    assert root.valid
    assert (
        root.delta_log_root_prior + root.log_q_reverse_auxiliary - root.log_q_forward_auxiliary
        == pytest.approx(0.0)
    )
    assert root.log_jacobian == 0.0
    assert root.log_acceptance_ratio == pytest.approx(root.delta_log_likelihood)
    assert math.isfinite(root.log_acceptance_ratio)


def test_pair_refresh_accepts_an_arbitrary_nonadjacent_leaf_pair() -> None:
    """Pair refresh is defined for diagonal leaves with no shared boundary."""
    problem, source = _problem_state(4)
    first, second = source.allocation.tiling.leaves[0], source.allocation.tiling.leaves[-1]
    assert first.row_stop < second.row_start

    transition = propose_pair_allocation_refresh(
        problem,
        source,
        first_leaf=first,
        second_leaf=second,
        new_fraction=0.61,
    )

    assert transition.valid
    assert transition.candidate.root_total == pytest.approx(source.root_total)
    _assert_rebuild_equal(transition.candidate)


def test_extreme_positive_mass_ratios_have_finite_reverse_proposal_terms() -> None:
    """Legal sub-ULP mass ratios remain supported in every reverse proposal."""
    problem, source = _problem_state(4, likelihood_power=0.0)
    leaves = source.allocation.tiling.leaves
    extreme_masses = np.full(source.k, 0.5)
    extreme_masses[0] = 0.14790204595
    extreme_masses[-1] = 2.42205857917e-23
    assert extreme_masses[0] / (extreme_masses[0] + extreme_masses[-1]) == 1.0
    pair_source = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(source.allocation.tiling, extreme_masses),
        fixed_coefficients=source.fixed_coefficients,
    )
    pair = propose_pair_allocation_refresh(
        problem,
        pair_source,
        first_leaf=leaves[0],
        second_leaf=leaves[-1],
        new_fraction=0.37,
    )

    assert pair.valid
    assert math.isfinite(pair.log_q_reverse_auxiliary)
    assert pair.log_acceptance_ratio == 0.0
    assert np.all(pair.candidate.leaf_masses > 0.0)
    _assert_rebuild_equal(pair.candidate)

    edge_merge = merge_choices(source.allocation.tiling)[0]
    edge_masses = np.full(source.k, 0.5)
    edge_masses[leaves.index(edge_merge.children[0])] = 1.0
    edge_masses[leaves.index(edge_merge.children[1])] = 2.0**-54
    edge_source = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(source.allocation.tiling, edge_masses),
        fixed_coefficients=source.fixed_coefficients,
    )
    edge = propose_posterior_edge_flip(
        problem,
        edge_source,
        merge_choice=edge_merge,
        new_fraction=0.41,
    )

    assert edge.valid
    assert math.isfinite(edge.log_q_reverse_auxiliary)
    assert edge.log_acceptance_ratio == (edge.log_q_reverse_selection - edge.log_q_forward_selection)
    assert np.all(edge.candidate.leaf_masses > 0.0)
    _assert_rebuild_equal(edge.candidate)

    relocation_source = initialize_full_tiling_posterior_state(problem, k=4)
    relocation_merge = merge_choices(relocation_source.allocation.tiling)[0]
    relocation_leaves = relocation_source.allocation.tiling.leaves
    relocation_masses = np.full(relocation_source.k, 0.5)
    relocation_masses[relocation_leaves.index(relocation_merge.children[0])] = 1.0
    relocation_masses[relocation_leaves.index(relocation_merge.children[1])] = 2.0**-54
    relocation_source = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            relocation_source.allocation.tiling,
            relocation_masses,
        ),
        fixed_coefficients=relocation_source.fixed_coefficients,
    )
    relocation = _valid_relocation(problem, relocation_source)

    assert relocation.valid
    assert math.isfinite(relocation.log_q_reverse_auxiliary)
    assert relocation.log_acceptance_ratio == (
        relocation.log_q_reverse_selection - relocation.log_q_forward_selection
    )
    assert np.all(relocation.candidate.leaf_masses > 0.0)
    _assert_rebuild_equal(relocation.candidate)


def test_unrepresentable_pair_split_is_an_explicit_self_transition() -> None:
    """A positive fraction whose child product underflows cannot abort sampling."""
    problem, initial = _problem_state(2, likelihood_power=0.0)
    smallest = np.nextafter(0.0, 1.0)
    source = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            initial.allocation.tiling,
            np.array([smallest, smallest]),
        ),
        fixed_coefficients=initial.fixed_coefficients,
    )
    first, second = source.allocation.tiling.leaves
    transition = propose_pair_allocation_refresh(
        problem,
        source,
        first_leaf=first,
        second_leaf=second,
        new_fraction=smallest,
    )

    assert not transition.valid
    assert transition.candidate is source
    assert transition.reason == "proposed pair masses are outside representable support"
    assert transition.log_acceptance_ratio == -math.inf


def test_prior_matched_pair_refresh_uses_exact_reduced_acceptance_ratio() -> None:
    """Large finite prior terms cannot contaminate their analytic cancellation."""
    ordinary_problem, _ = _problem_state(2, likelihood_power=0.0)
    problem = FullTilingProblem(
        base=ordinary_problem.base,
        concentration=1.0e305,
    )
    initial = initialize_full_tiling_posterior_state(problem, k=2)
    source = build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(
            initial.allocation.tiling,
            np.array([1.0, np.nextafter(0.0, 1.0)]),
        ),
        fixed_coefficients=initial.fixed_coefficients,
    )
    first, second = source.allocation.tiling.leaves
    transition = propose_pair_allocation_refresh(
        problem,
        source,
        first_leaf=first,
        second_leaf=second,
        new_fraction=0.37,
    )

    assert transition.valid
    assert transition.log_acceptance_ratio == 0.0
    assert math.isfinite(transition.log_q_forward_auxiliary)
    assert math.isfinite(transition.log_q_reverse_auxiliary)
