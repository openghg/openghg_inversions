"""Tests for log-root slice updates and full-tiling movement diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from math import exp

import numpy as np
import pytest
import xarray as xr

import openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling as sampling
from openghg_inversions.experimental.rjmcmc.full_tiling_compound_sampling import (
    FullTilingCompoundConfig,
    continue_full_tiling_compound,
    sample_full_tiling_compound,
)
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
    rescale_full_tiling_root_total,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
)


def _problem_state(
    *,
    k: int = 1,
    nominal_weight: np.ndarray | None = None,
):
    """Return a small deterministic posterior problem and initialized state."""
    sensitivity = np.arange(1.0, 49.0).reshape(3, 4, 4)
    fixed_design = np.arange(18.0).reshape(3, 6) / 20.0
    dataset = xr.Dataset(
        {
            "fp_x_flux": (("nmeasure", "lat", "lon"), sensitivity),
            "mf": ("nmeasure", np.zeros(3)),
            "mf_error": ("nmeasure", np.ones(3)),
            "outer": (("nmeasure", "outer_region"), fixed_design),
            "boundary": ("nmeasure", np.array([1.0, 2.0, 3.0])),
        },
        coords={
            "nmeasure": np.arange(3),
            "lat": np.arange(4, dtype=float),
            "lon": np.arange(4, dtype=float),
            "outer_region": np.arange(6),
        },
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=(np.ones((4, 4)) if nominal_weight is None else nominal_weight),
        k_min=1,
        k_max=16,
        concentration=4.0,
        root_variance=0.25,
        fixed_design_name="outer",
        fixed_offset_name="boundary",
        fixed_coefficient_prior_mean=np.ones(6),
        fixed_coefficient_prior_sd=np.full(6, 0.5),
        likelihood_power=0.0,
    )
    problem = full_tiling_problem_from_gamma_beta_adapter(adapter, concentration=4.0)
    return problem, initialize_full_tiling_posterior_state(problem, k=k)


class _ScriptedRng:
    """Minimal random source that fails if the slice sampler overdraws."""

    def __init__(self, values: list[float]) -> None:
        self._values: Iterator[float] = iter(values)
        self.draws = 0

    def random(self) -> float:
        """Return the next scripted open- or closed-unit value."""
        self.draws += 1
        try:
            return next(self._values)
        except StopIteration as error:
            raise AssertionError("slice sampler consumed an unexpected random draw") from error


def test_root_slice_scripted_stepping_out_has_exact_density_and_rng_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finite stepping-out uses the exact scripted bracket and no MH draw."""
    problem, initial = _problem_state()
    source = rescale_full_tiling_root_total(problem, initial, new_root_total=1.0)
    evaluated: list[float] = []

    def quadratic_density(problem_arg, source_arg, *, log_root_total):
        """Record log totals and return a simple finite target."""
        assert problem_arg is problem
        assert source_arg is source
        evaluated.append(log_root_total)
        return -0.5 * log_root_total**2

    monkeypatch.setattr(sampling, "log_root_total_slice_density", quadratic_density)
    rng = _ScriptedRng([exp(-0.5), 0.5, 0.5, 1.0 / 6.0])

    candidate, counters = sampling._draw_root_total_slice(
        problem,
        source,
        width=1.0,
        max_steps=5,
        max_shrink_steps=10,
        rng=rng,
    )

    assert candidate.root_total == pytest.approx(exp(-1.0))
    assert evaluated == pytest.approx([0.0, -0.5, -1.5, 0.5, 1.5, -1.0])
    assert counters.left_steps == 1
    assert counters.right_steps == 1
    assert counters.shrink_draws == 1
    assert counters.log_density_evaluations == 6
    assert rng.draws == 4


def test_root_slice_shrink_guard_raises_instead_of_biasing_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhausting the configured shrink guard raises a clear error."""
    problem, initial = _problem_state()
    source = rescale_full_tiling_root_total(problem, initial, new_root_total=1.0)

    def point_density(problem_arg, source_arg, *, log_root_total):
        """Support only the exact current log total."""
        assert problem_arg is problem
        assert source_arg is source
        return 0.0 if log_root_total == 0.0 else -np.inf

    monkeypatch.setattr(sampling, "log_root_total_slice_density", point_density)
    rng = _ScriptedRng([0.5, 0.5, 0.0, 0.0])

    with pytest.raises(
        RuntimeError,
        match="root_slice_max_shrink_steps=1",
    ):
        sampling._draw_root_total_slice(
            problem,
            source,
            width=1.0,
            max_steps=1,
            max_shrink_steps=1,
            rng=rng,
        )
    assert rng.draws == 4


def test_zero_slice_height_uniform_does_not_accept_zero_density_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero height draw still rejects candidates with log density -inf."""
    problem, initial = _problem_state()
    source = rescale_full_tiling_root_total(problem, initial, new_root_total=1.0)

    def point_density(problem_arg, source_arg, *, log_root_total):
        """Support only the exact current log total."""
        assert problem_arg is problem
        assert source_arg is source
        return 0.0 if log_root_total == 0.0 else -np.inf

    monkeypatch.setattr(sampling, "log_root_total_slice_density", point_density)
    rng = _ScriptedRng([0.0, 0.5, 0.0, 0.0, 0.5])

    candidate, counters = sampling._draw_root_total_slice(
        problem,
        source,
        width=1.0,
        max_steps=1,
        max_shrink_steps=2,
        rng=rng,
    )

    assert candidate.root_total == 1.0
    assert counters.shrink_draws == 2
    assert counters.log_density_evaluations == 3
    assert rng.draws == 5


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        (
            "root_slice_width",
            lambda: FullTilingCompoundConfig(iterations=1, root_slice_width=0.0),
        ),
        (
            "root_slice_max_steps",
            lambda: FullTilingCompoundConfig(iterations=1, root_slice_max_steps=0),
        ),
        (
            "root_slice_max_shrink_steps",
            lambda: FullTilingCompoundConfig(
                iterations=1,
                root_slice_max_shrink_steps=0,
            ),
        ),
    ],
)
def test_root_slice_settings_reject_zero(
    name: str,
    factory: Callable[[], FullTilingCompoundConfig],
) -> None:
    """Every slice tuning value must be strictly positive."""
    with pytest.raises(ValueError, match=name):
        factory()


@pytest.mark.parametrize(
    "root_slice_width",
    [np.bool_(True), np.inf, np.nan, 1.0e308],
)
def test_root_slice_width_rejects_boolean_nonfinite_or_unbounded_extent(
    root_slice_width: object,
) -> None:
    """Slice widths must define a finite stepping-out extent."""
    error = TypeError if isinstance(root_slice_width, np.bool_) else ValueError
    with pytest.raises(error, match="root_slice_width"):
        FullTilingCompoundConfig(
            iterations=1,
            root_slice_width=root_slice_width,
            root_slice_max_steps=100,
        )


def test_diagnostics_do_not_change_trace_state_or_rng() -> None:
    """Opting into timing and movement output leaves scientific replay exact."""
    problem, initial = _problem_state(k=4)
    config = FullTilingCompoundConfig(iterations=28, seed=904)

    ordinary = sample_full_tiling_compound(problem, initial, config)
    diagnosed = sample_full_tiling_compound(
        problem,
        initial,
        config,
        collect_movement_diagnostics=True,
    )

    assert ordinary.movement_diagnostics is None
    assert diagnosed.movement_diagnostics is not None
    for name in ordinary.trace.__dataclass_fields__:
        np.testing.assert_array_equal(
            getattr(ordinary.trace, name),
            getattr(diagnosed.trace, name),
        )
    for name in (
        "leaf_masses",
        "fixed_coefficients",
        "prediction",
        "residual",
    ):
        np.testing.assert_array_equal(
            getattr(ordinary.final_state, name),
            getattr(diagnosed.final_state, name),
        )
    assert ordinary.checkpoint.rng_state == diagnosed.checkpoint.rng_state


def test_diagnostics_clamp_roundoff_in_full_domain_nominal_mass() -> None:
    """A full-domain structural change cannot fail on an ulp mass overshoot."""
    nominal_weight = np.asarray(
        [
            [1.389535653998891, 0.7721530128110184, 4.87184579581855, 3.7447729479550373],
            [1.883916063397523, 0.11041493420998245, 1.0534062637380492, 1.9811672491418417],
            [2.729071866748747, 0.539071510297319, 6.1842847923927495, 0.26702019946585553],
            [0.5160621774555013, 2.547340791318328, 1.050277708857061, 7.406756199749073],
        ],
        dtype=np.float64,
    )
    problem, initial = _problem_state(k=2, nominal_weight=nominal_weight)

    result = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=1, seed=2),
        collect_movement_diagnostics=True,
    )

    diagnostics = result.movement_diagnostics
    assert diagnostics is not None
    assert diagnostics.move.tolist() == ["edge_flip"]
    assert diagnostics.valid.tolist() == [True]
    assert diagnostics.changed_native_cell_count.tolist() == [16]
    assert diagnostics.changed_nominal_mass.tolist() == [1.0]


def test_diagnostics_report_catalogues_movement_and_slice_work() -> None:
    """Per-attempt metrics describe structural, slice, pair, and fixed slots."""
    problem, initial = _problem_state(k=4)
    result = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=14, seed=9),
        collect_movement_diagnostics=True,
    )
    diagnostics = result.movement_diagnostics
    assert diagnostics is not None

    np.testing.assert_array_equal(
        diagnostics.global_transition,
        result.trace.global_transition,
    )
    np.testing.assert_array_equal(diagnostics.move, result.trace.move)
    np.testing.assert_array_equal(diagnostics.valid, result.trace.valid)
    np.testing.assert_array_equal(diagnostics.accepted, result.trace.accepted)
    assert np.all(diagnostics.proposal_elapsed_ns >= 0)
    assert np.all(diagnostics.diagnostic_elapsed_ns >= 0)

    assert diagnostics.move[0] == "resolution_relocation"
    assert diagnostics.source_merge_count[0] == 4
    assert diagnostics.destination_catalogue_size[0] == 6
    assert diagnostics.design_cache_misses[0] == 3
    assert diagnostics.changed_native_cell_count[0] == 12
    assert diagnostics.changed_nominal_mass[0] == pytest.approx(0.75)
    assert diagnostics.allocation_share_l1_displacement[0] > 0.0
    assert diagnostics.standardized_prediction_l2[0] > 0.0

    root = 2
    assert diagnostics.root_abs_displacement[root] > 0.0
    assert diagnostics.root_abs_log_displacement[root] > 0.0
    assert diagnostics.allocation_share_l1_displacement[root] == pytest.approx(0.0)
    assert diagnostics.slice_shrink_draws[root] >= 1
    assert diagnostics.slice_log_density_evaluations[root] >= 2

    assert diagnostics.pair_catalogue_size[3:8].tolist() == [6] * 5
    assert np.all(diagnostics.fixed_position[3:8] == -1)
    assert diagnostics.fixed_position[8:14].tolist() == list(range(6))
    assert np.all(diagnostics.fixed_abs_displacement[8:14] > 0.0)
    assert np.all(diagnostics.fixed_abs_log_displacement[8:14] > 0.0)
    non_root = diagnostics.move != "root_total_slice"
    assert np.all(diagnostics.slice_left_steps[non_root] == 0)
    assert np.all(diagnostics.slice_right_steps[non_root] == 0)
    assert np.all(diagnostics.slice_shrink_draws[non_root] == 0)
    assert np.all(diagnostics.slice_log_density_evaluations[non_root] == 0)


def test_invalid_attempts_have_zero_movement_and_segment_opt_in_is_not_sticky() -> None:
    """Invalid rows are zeroed and diagnostics opt-in does not enter checkpoints."""
    problem, initial = _problem_state(k=1)
    first = sample_full_tiling_compound(
        problem,
        initial,
        FullTilingCompoundConfig(iterations=2, seed=72),
    )
    diagnosed = continue_full_tiling_compound(
        problem,
        first.checkpoint,
        iterations=6,
        collect_movement_diagnostics=True,
    )
    diagnostics = diagnosed.movement_diagnostics
    assert diagnostics is not None
    invalid = ~diagnostics.valid
    assert np.any(invalid)
    for name in (
        "changed_native_cell_count",
        "changed_nominal_mass",
        "standardized_prediction_l2",
        "root_abs_displacement",
        "root_abs_log_displacement",
        "allocation_share_l1_displacement",
        "fixed_abs_displacement",
        "fixed_abs_log_displacement",
        "slice_left_steps",
        "slice_right_steps",
        "slice_shrink_draws",
        "slice_log_density_evaluations",
    ):
        assert np.all(getattr(diagnostics, name)[invalid] == 0)
    assert np.all(diagnostics.fixed_position[invalid] == -1)
    assert not hasattr(diagnosed.checkpoint.kernel_settings, "collect_movement_diagnostics")

    final = continue_full_tiling_compound(
        problem,
        diagnosed.checkpoint,
        iterations=1,
    )
    assert final.movement_diagnostics is None
