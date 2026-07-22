"""Tests for inferred-error and shared-hierarchy proposal accounting."""

from __future__ import annotations

from math import log

import numpy as np
import pytest

from openghg_inversions.experimental.rjmcmc.core import (
    InferredOUErrorModel,
    TransDimensionalProblem,
    TransDimensionalState,
    build_state,
    uniform_log_k_prior,
)
from openghg_inversions.experimental.rjmcmc.hierarchy import SharedLognormalHierarchy
from openghg_inversions.experimental.rjmcmc.likelihood import IndependentSiteOUData
from openghg_inversions.experimental.rjmcmc.proposals import (
    TransitionTerms,
    propose_birth,
    propose_correlation_timescale,
    propose_death,
    propose_mismatch_sd,
    propose_shared_hierarchy,
)


def _problem(
    *,
    inferred_error: bool = True,
    hierarchy: bool = True,
) -> TransDimensionalProblem:
    """Return a small target with independently switchable optional layers."""
    observation_sd = np.array([0.4, 0.5, 0.6, 0.7])
    error_model = None
    if inferred_error:
        error_model = InferredOUErrorModel(
            data=IndependentSiteOUData(
                observation_sd=observation_sd,
                observation_time=np.array([0.0, 2.0, 1.0, 5.0]),
                site_index=np.array([0, 0, 1, 1], dtype=np.int64),
                mismatch_group_index=np.array([0, 1, 0, 1], dtype=np.int64),
                site_tau_index=np.array([0, 1], dtype=np.int64),
            ),
            mismatch_sd_prior_lower=np.array([0.1, 0.2]),
            mismatch_sd_prior_upper=np.array([1.1, 2.2]),
            correlation_timescale_prior_lower=np.array([0.5, 1.0]),
            correlation_timescale_prior_upper=np.array([2.5, 5.0]),
        )
    coefficient_hierarchy = None
    if hierarchy:
        coefficient_hierarchy = SharedLognormalHierarchy(
            mean_hyperprior_median=1.0,
            mean_hyperprior_log_sd=0.6,
            sd_hyperprior_median=0.8,
            sd_hyperprior_log_sd=0.4,
        )
    return TransDimensionalProblem(
        observations=np.array([0.2, 0.9, -0.3, 1.1]),
        observation_sd=observation_sd,
        sensitivities=np.array(
            [
                [0.2, 0.4, 0.1, 0.3, 0.5],
                [0.8, 0.1, 0.5, 0.2, 0.3],
                [0.3, 0.6, 0.2, 0.7, 0.1],
                [0.5, 0.2, 0.9, 0.1, 0.4],
            ]
        ),
        grid_coordinates=np.arange(5, dtype=np.float64)[:, None],
        k_min=1,
        k_max=4,
        log_k_prior=uniform_log_k_prior(1, 4),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.7,
        error_model=error_model,
        coefficient_hierarchy=coefficient_hierarchy,
    )


def _state(problem: TransDimensionalProblem) -> TransDimensionalState:
    """Build a valid state for any combination of optional target layers."""
    inferred_error = problem.error_model is not None
    hierarchy = problem.coefficient_hierarchy is not None
    return build_state(
        problem,
        [0, 4],
        [0.8, 1.3],
        mismatch_sd=[0.4, 1.2] if inferred_error else None,
        correlation_timescale=[1.5, 3.0] if inferred_error else None,
        coefficient_prior_mean=1.2 if hierarchy else None,
        coefficient_prior_sd=0.9 if hierarchy else None,
    )


def _assert_pointwise_balance(
    source: TransDimensionalState,
    forward: TransitionTerms,
    reverse: TransitionTerms,
) -> None:
    """Assert reciprocal MH ratios and equal accepted pointwise log flux."""
    assert forward.valid and reverse.valid
    assert reverse.log_acceptance_ratio == pytest.approx(
        -forward.log_acceptance_ratio,
        rel=0.0,
        abs=2e-12,
    )
    log_forward_flux = source.log_target + forward.log_q_forward + min(0.0, forward.log_acceptance_ratio)
    log_reverse_flux = (
        forward.candidate.log_target + reverse.log_q_forward + min(0.0, reverse.log_acceptance_ratio)
    )
    assert log_forward_flux == pytest.approx(log_reverse_flux, rel=0.0, abs=2e-12)


def test_mismatch_update_has_uniform_position_probability_and_balanced_flux() -> None:
    """A mismatch edge must include its group choice and satisfy MH balance."""
    problem = _problem()
    state = _state(problem)
    forward = propose_mismatch_sd(
        problem,
        state,
        mismatch_sd_position=1,
        proposed_mismatch_sd=1.45,
        proposal_stdev=0.3,
    )
    reverse = propose_mismatch_sd(
        problem,
        forward.candidate,
        mismatch_sd_position=1,
        proposed_mismatch_sd=1.2,
        proposal_stdev=0.3,
    )

    assert forward.move == "mismatch_sd"
    assert forward.log_q_forward == pytest.approx(forward.log_q_reverse)
    expected_without_position = -0.5 * ((1.45 - 1.2) / 0.3) ** 2 - log(0.3) - 0.5 * log(2.0 * np.pi)
    assert forward.log_q_forward == pytest.approx(expected_without_position - log(2.0))
    _assert_pointwise_balance(state, forward, reverse)


def test_timescale_update_has_uniform_position_probability_and_balanced_flux() -> None:
    """A timescale edge must include its parameter choice and satisfy balance."""
    problem = _problem()
    state = _state(problem)
    forward = propose_correlation_timescale(
        problem,
        state,
        correlation_timescale_position=0,
        proposed_correlation_timescale=2.1,
        proposal_stdev=0.5,
    )
    reverse = propose_correlation_timescale(
        problem,
        forward.candidate,
        correlation_timescale_position=0,
        proposed_correlation_timescale=1.5,
        proposal_stdev=0.5,
    )

    assert forward.move == "correlation_timescale"
    assert forward.log_q_forward == pytest.approx(forward.log_q_reverse)
    assert reverse.log_q_forward == pytest.approx(forward.log_q_reverse)
    _assert_pointwise_balance(state, forward, reverse)


def test_joint_hierarchy_update_is_symmetric_and_balanced() -> None:
    """A joint eta/zeta Gaussian edge must satisfy pointwise MH balance."""
    problem = _problem()
    state = _state(problem)
    proposed_eta = state.eta + 0.15
    proposed_zeta = state.zeta - 0.08
    forward = propose_shared_hierarchy(
        problem,
        state,
        proposed_eta=proposed_eta,
        proposed_zeta=proposed_zeta,
        eta_proposal_stdev=0.25,
        zeta_proposal_stdev=0.2,
    )
    reverse = propose_shared_hierarchy(
        problem,
        forward.candidate,
        proposed_eta=state.eta,
        proposed_zeta=state.zeta,
        eta_proposal_stdev=0.25,
        zeta_proposal_stdev=0.2,
    )

    assert forward.move == "shared_hierarchy"
    assert forward.log_q_forward == pytest.approx(forward.log_q_reverse)
    assert reverse.log_q_forward == pytest.approx(forward.log_q_reverse)
    _assert_pointwise_balance(state, forward, reverse)


def test_structural_pair_is_balanced_with_ou_and_shared_hierarchy_active() -> None:
    """Optional target factors must remain reciprocal across an RJ structural edge."""
    problem = _problem()
    state = _state(problem)
    forward = propose_birth(
        problem,
        state,
        new_nucleus=2,
        proposed_coefficient=1.1,
        proposal_stdev=0.3,
    )
    remove_position = int(np.flatnonzero(forward.candidate.active_nuclei == 2)[0])
    reverse = propose_death(
        problem,
        forward.candidate,
        remove_position=remove_position,
        proposal_stdev=0.3,
    )

    assert forward.candidate.eta == state.eta
    assert forward.candidate.zeta == state.zeta
    np.testing.assert_array_equal(forward.candidate.mismatch_sd, state.mismatch_sd)
    np.testing.assert_array_equal(
        forward.candidate.correlation_timescale,
        state.correlation_timescale,
    )
    _assert_pointwise_balance(state, forward, reverse)


@pytest.mark.parametrize(
    ("kind", "position", "value"),
    [
        ("mismatch", 0, 0.1),
        ("mismatch", 1, 2.2),
        ("timescale", 0, 0.5),
        ("timescale", 1, 5.0),
    ],
)
def test_bounded_uniform_endpoints_are_valid(
    kind: str,
    position: int,
    value: float,
) -> None:
    """Inclusive prior endpoints must remain valid Gaussian-walk candidates."""
    problem = _problem()
    state = _state(problem)
    if kind == "mismatch":
        transition = propose_mismatch_sd(
            problem,
            state,
            mismatch_sd_position=position,
            proposed_mismatch_sd=value,
            proposal_stdev=0.2,
        )
    else:
        transition = propose_correlation_timescale(
            problem,
            state,
            correlation_timescale_position=position,
            proposed_correlation_timescale=value,
            proposal_stdev=0.2,
        )
    assert transition.valid
    assert np.isfinite(transition.candidate.log_error_model_prior)


@pytest.mark.parametrize(
    ("kind", "position", "value"),
    [
        ("mismatch", 0, 0.1 - 1e-12),
        ("mismatch", 1, 2.2 + 1e-12),
        ("timescale", 0, 0.5 - 1e-12),
        ("timescale", 1, 5.0 + 1e-12),
        ("timescale", 0, np.nan),
    ],
)
def test_out_of_support_error_draws_are_invalid_self_transitions(
    kind: str,
    position: int,
    value: float,
) -> None:
    """Untruncated Gaussian draws beyond prior support must reject safely."""
    problem = _problem()
    state = _state(problem)
    if kind == "mismatch":
        transition = propose_mismatch_sd(
            problem,
            state,
            mismatch_sd_position=position,
            proposed_mismatch_sd=value,
            proposal_stdev=0.2,
        )
    else:
        transition = propose_correlation_timescale(
            problem,
            state,
            correlation_timescale_position=position,
            proposed_correlation_timescale=value,
            proposal_stdev=0.2,
        )
    assert not transition.valid
    assert transition.candidate is state
    assert transition.log_acceptance_ratio == -np.inf
    assert "bounded-uniform" in (transition.reason or "")


def test_parameter_updates_preserve_unrelated_cache_identity() -> None:
    """Optional-parameter candidates must reuse every unaffected state cache."""
    problem = _problem()
    state = _state(problem)
    mismatch = propose_mismatch_sd(
        problem,
        state,
        mismatch_sd_position=0,
        proposed_mismatch_sd=0.6,
        proposal_stdev=0.2,
    ).candidate
    timescale = propose_correlation_timescale(
        problem,
        mismatch,
        correlation_timescale_position=1,
        proposed_correlation_timescale=3.5,
        proposal_stdev=0.4,
    ).candidate
    hierarchy = propose_shared_hierarchy(
        problem,
        timescale,
        proposed_eta=timescale.eta + 0.1,
        proposed_zeta=timescale.zeta - 0.1,
        eta_proposal_stdev=0.2,
        zeta_proposal_stdev=0.2,
    ).candidate

    common_arrays = (
        "nuclei",
        "coefficients",
        "labels",
        "design",
        "fixed_coefficients",
        "dynamic_prediction",
        "fixed_prediction",
        "prediction",
        "residual",
    )
    for name in common_arrays:
        assert getattr(mismatch, name) is getattr(state, name)
        assert getattr(timescale, name) is getattr(state, name)
        assert getattr(hierarchy, name) is getattr(state, name)
    assert mismatch.correlation_timescale is state.correlation_timescale
    assert timescale.mismatch_sd is mismatch.mismatch_sd
    assert hierarchy.mismatch_sd is timescale.mismatch_sd
    assert hierarchy.correlation_timescale is timescale.correlation_timescale


def test_missing_optional_configuration_and_invalid_positions_reject_safely() -> None:
    """Unavailable kernels and malformed indices must return self-transitions."""
    base_problem = _problem(inferred_error=False, hierarchy=False)
    base_state = _state(base_problem)
    unavailable = (
        propose_mismatch_sd(
            base_problem,
            base_state,
            mismatch_sd_position=0,
            proposed_mismatch_sd=0.5,
            proposal_stdev=0.2,
        ),
        propose_correlation_timescale(
            base_problem,
            base_state,
            correlation_timescale_position=0,
            proposed_correlation_timescale=1.0,
            proposal_stdev=0.2,
        ),
        propose_shared_hierarchy(
            base_problem,
            base_state,
            proposed_eta=0.0,
            proposed_zeta=0.0,
            eta_proposal_stdev=0.2,
            zeta_proposal_stdev=0.2,
        ),
    )
    configured_problem = _problem()
    configured_state = _state(configured_problem)
    malformed = (
        propose_mismatch_sd(
            configured_problem,
            configured_state,
            mismatch_sd_position=-1,
            proposed_mismatch_sd=0.5,
            proposal_stdev=0.2,
        ),
        propose_correlation_timescale(
            configured_problem,
            configured_state,
            correlation_timescale_position=True,
            proposed_correlation_timescale=1.0,
            proposal_stdev=0.2,
        ),
        propose_shared_hierarchy(
            configured_problem,
            configured_state,
            proposed_eta=np.inf,
            proposed_zeta=0.0,
            eta_proposal_stdev=0.2,
            zeta_proposal_stdev=0.2,
        ),
    )
    for transition in unavailable + malformed:
        assert not transition.valid
        assert transition.log_acceptance_ratio == -np.inf


@pytest.mark.parametrize("scale", [0.0, -0.1, np.inf, np.nan])
def test_invalid_proposal_scales_raise(scale: float) -> None:
    """Every optional-parameter walk must reject malformed Gaussian scales."""
    problem = _problem()
    state = _state(problem)
    with pytest.raises(ValueError, match="finite and positive"):
        propose_mismatch_sd(
            problem,
            state,
            mismatch_sd_position=0,
            proposed_mismatch_sd=0.5,
            proposal_stdev=scale,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        propose_correlation_timescale(
            problem,
            state,
            correlation_timescale_position=0,
            proposed_correlation_timescale=1.5,
            proposal_stdev=scale,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        propose_shared_hierarchy(
            problem,
            state,
            proposed_eta=state.eta,
            proposed_zeta=state.zeta,
            eta_proposal_stdev=scale,
            zeta_proposal_stdev=0.2,
        )


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_optional_proposals_return_finite_targets_for_each_backend(backend: str) -> None:
    """Both numerical backends must construct valid optional candidates."""
    problem = _problem()
    state = _state(problem)
    transitions = (
        propose_mismatch_sd(
            problem,
            state,
            mismatch_sd_position=0,
            proposed_mismatch_sd=0.6,
            proposal_stdev=0.2,
            backend=backend,  # type: ignore[arg-type]
        ),
        propose_correlation_timescale(
            problem,
            state,
            correlation_timescale_position=1,
            proposed_correlation_timescale=3.5,
            proposal_stdev=0.4,
            backend=backend,  # type: ignore[arg-type]
        ),
        propose_shared_hierarchy(
            problem,
            state,
            proposed_eta=state.eta + 0.1,
            proposed_zeta=state.zeta - 0.1,
            eta_proposal_stdev=0.2,
            zeta_proposal_stdev=0.2,
            backend=backend,  # type: ignore[arg-type]
        ),
    )
    assert all(transition.valid for transition in transitions)
    assert all(np.isfinite(transition.candidate.log_target) for transition in transitions)


def test_numpy_and_numba_optional_proposals_have_parity() -> None:
    """Forced optional draws must yield backend-independent terms and targets."""
    problem = _problem()
    state = _state(problem)
    calls = (
        lambda backend: propose_mismatch_sd(
            problem,
            state,
            mismatch_sd_position=0,
            proposed_mismatch_sd=0.6,
            proposal_stdev=0.2,
            backend=backend,
        ),
        lambda backend: propose_correlation_timescale(
            problem,
            state,
            correlation_timescale_position=1,
            proposed_correlation_timescale=3.5,
            proposal_stdev=0.4,
            backend=backend,
        ),
        lambda backend: propose_shared_hierarchy(
            problem,
            state,
            proposed_eta=state.eta + 0.1,
            proposed_zeta=state.zeta - 0.1,
            eta_proposal_stdev=0.2,
            zeta_proposal_stdev=0.2,
            backend=backend,
        ),
    )
    for call in calls:
        numpy_transition = call("numpy")
        numba_transition = call("numba")
        assert numba_transition.log_target_delta == pytest.approx(
            numpy_transition.log_target_delta,
            rel=0.0,
            abs=2e-12,
        )
        assert numba_transition.log_q_forward == pytest.approx(numpy_transition.log_q_forward)
        assert numba_transition.log_q_reverse == pytest.approx(numpy_transition.log_q_reverse)
