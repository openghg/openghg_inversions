"""Focused tests for the fixed-basis PyMC/NumPyro bridge.

PyMC/JAX-dependent model, log-density, precision, and sampler-forwarding
assertions run in fresh subprocesses.  This ensures their pre-import float64
configuration cannot be contaminated by the process-global float32 setting
used by normal RHIME tests.  The parent process retains pure bridge tests and
an independent normalized-density oracle for the constrained ``(T, p, c)``
chart.  Child-only environment overrides and process creation are the only
side effects.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.fixed_basis_nuts import (
    build_fixed_basis_pymc_model,
    fixed_basis_nuts_initvals,
    preflight_fixed_basis_nuts,
    prepare_fixed_basis_nuts,
    require_fixed_basis_nuts_float64,
    sample_fixed_basis_nuts,
)
from openghg_inversions.experimental.rjmcmc.full_tiling import TilingState
from openghg_inversions.experimental.rjmcmc.full_tiling_posterior import (
    FullTilingPosteriorState,
    FullTilingProblem,
    build_full_tiling_posterior_state,
    full_tiling_problem_from_gamma_beta_adapter,
    initialize_full_tiling_posterior_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
)

_THIS_FILE = Path(__file__).resolve()


def _pytensor_flags_with_float64(flags: str) -> str:
    """Return PyTensor flags with an unambiguous float64 test configuration."""
    retained = []
    for item in flags.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name not in {"floatX", "warn_float64"}:
            retained.append(stripped)
    return ",".join(("floatX=float64", "warn_float64=ignore", *retained))


def _run_x64_case(case: str) -> None:
    """Run one PyMC-dependent assertion in a fresh float64 process.

    The copied child environment overrides ``PYTENSOR_FLAGS``,
    ``JAX_ENABLE_X64``, and ``JAX_PLATFORMS`` without changing the parent.

    Args:
        case: Registered isolated assertion name.

    Raises:
        AssertionError: If the child process exits unsuccessfully.
        subprocess.TimeoutExpired: If the child exceeds 180 seconds.
    """
    environment = os.environ.copy()
    environment["PYTENSOR_FLAGS"] = _pytensor_flags_with_float64(environment.get("PYTENSOR_FLAGS", ""))
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, str(_THIS_FILE), case],
        cwd=_THIS_FILE.parents[3],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"isolated fixed-basis NUTS case {case!r} failed\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _problem_state(
    *,
    likelihood_power: float = 1.0,
) -> tuple[FullTilingProblem, FullTilingPosteriorState]:
    """Build a small labelled fixed-basis problem and its prior-mean state.

    Args:
        likelihood_power: Gaussian likelihood multiplier stored by the
            numerical adapter.

    Returns:
        Problem and prior-mean state for three observations, a 4-by-4 inner
        grid, four leaves, and two fixed coefficients.
    """
    sensitivity = np.arange(1.0, 49.0).reshape(3, 4, 4)
    outer = np.arange(6.0).reshape(3, 2) / 5.0
    boundary = np.array([2.0, 3.0, 5.0])
    fixed_mean = np.array([0.8, 1.2])
    observations = boundary + sensitivity.sum(axis=(1, 2)) + outer @ fixed_mean
    dataset = xr.Dataset(
        {
            "fp_x_flux": (
                ("lon", "nmeasure", "lat"),
                sensitivity.transpose(2, 0, 1),
            ),
            "mf": ("nmeasure", observations),
            "mf_error": ("nmeasure", np.array([0.7, 0.8, 0.9])),
            "outer": (("fixed", "nmeasure"), outer.T),
            "boundary": ("nmeasure", boundary),
        },
        coords={
            "nmeasure": ["a", "b", "c"],
            "lat": np.arange(4) + 50.0,
            "lon": np.arange(4) - 3.0,
            "fixed": ["north", "south"],
        },
    )
    nominal_weight = xr.DataArray(
        np.arange(1.0, 17.0).reshape(4, 4).T,
        dims=("lon", "lat"),
        coords={"lon": dataset.lon, "lat": dataset.lat},
    )
    adapter = gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=nominal_weight,
        k_min=4,
        k_max=4,
        concentration=5.0,
        root_variance=0.25,
        likelihood_power=likelihood_power,
        sensitivity_name="fp_x_flux",
        observation_name="mf",
        observation_sd_name="mf_error",
        fixed_design_name="outer",
        fixed_offset_name="boundary",
        fixed_coefficient_prior_mean=fixed_mean,
        fixed_coefficient_prior_sd=np.array([0.3, 0.5]),
    )
    problem = full_tiling_problem_from_gamma_beta_adapter(
        adapter,
        concentration=7.0,
    )
    return problem, initialize_full_tiling_posterior_state(problem, k=4)


def _arbitrary_interior_state(
    problem: FullTilingProblem,
    prior_mean: FullTilingPosteriorState,
) -> FullTilingPosteriorState:
    """Replace continuous coordinates while retaining the fixture topology.

    Args:
        problem: Fixed-basis posterior problem.
        prior_mean: State providing the canonical tiling to retain.

    Returns:
        Non-symmetric state strictly inside every continuous prior support.
    """
    shares = np.array([0.11, 0.19, 0.27, 0.43])
    return build_full_tiling_posterior_state(
        problem,
        allocation=TilingState(prior_mean.allocation.tiling, 1.37 * shares),
        fixed_coefficients=np.array([0.72, 1.41]),
    )


def test_x64_subprocess_flags_override_conflicting_parent_precision() -> None:
    """The isolated runtime must not inherit either parent precision flag."""
    flags = _pytensor_flags_with_float64("floatX=float32,base_compiledir=/tmp/example,warn_float64=raise")

    assert flags == "floatX=float64,warn_float64=ignore,base_compiledir=/tmp/example"


def test_bridge_preserves_canonical_order_shapes_and_forward_model() -> None:
    """The immutable bridge must close the existing fixed-basis prediction."""
    problem, state = _problem_state()
    data = prepare_fixed_basis_nuts(problem, state)
    expected_bounds = np.asarray(
        [
            (leaf.row_start, leaf.row_stop, leaf.col_start, leaf.col_stop)
            for leaf in state.allocation.tiling.leaves
        ]
    )

    assert data.k == 4
    assert data.dynamic_design.shape == (3, 4)
    assert data.fixed_design.shape == (3, 2)
    np.testing.assert_array_equal(data.rectangle_bounds, expected_bounds)
    np.testing.assert_allclose(
        data.dynamic_design,
        np.column_stack([problem.design_column(leaf) for leaf in state.allocation.tiling.leaves]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        data.fixed_offset
        + data.dynamic_design @ state.leaf_masses
        + data.fixed_design @ state.fixed_coefficients,
        state.prediction,
        rtol=0.0,
        atol=5.0e-13,
    )
    for value in (
        data.rectangle_bounds,
        data.observations,
        data.observation_sd,
        data.dynamic_design,
        data.nominal_leaf_share,
        data.dirichlet_alpha,
        data.fixed_design,
        data.fixed_offset,
        data.fixed_coefficient_prior_mean,
        data.fixed_coefficient_prior_sd,
        data.initial_leaf_share,
        data.initial_fixed_coefficient,
    ):
        assert value.flags.owndata
        assert not value.flags.writeable


def test_bridge_rejects_fractional_rectangle_bounds_instead_of_truncating() -> None:
    """Rectangle metadata must contain exact integers before int64 conversion."""
    problem, state = _problem_state()
    data = prepare_fixed_basis_nuts(problem, state)
    fractional = np.asarray(data.rectangle_bounds, dtype=np.float64)
    fractional[0, 0] += 0.5

    with pytest.raises(ValueError, match="exact integer"):
        replace(data, rectangle_bounds=fractional)


def test_target_components_match_independent_closed_form_density_oracle() -> None:
    """A PyMC-independent oracle verifies normalized ``(T, p, c)`` factors.

    The final check excludes an extra ``T**(K-1)`` Jacobian from this
    constrained target chart.
    """
    problem, prior_mean = _problem_state()
    state = _arbitrary_interior_state(problem, prior_mean)
    data = prepare_fixed_basis_nuts(problem, state)
    root_total = float(state.allocation.total_mass)
    leaf_share = np.asarray(state.leaf_masses / root_total)
    fixed_coefficient = np.asarray(state.fixed_coefficients)
    mean_observation = (
        data.fixed_offset
        + data.dynamic_design @ (root_total * leaf_share)
        + data.fixed_design @ fixed_coefficient
    )

    standardized_residual = (data.observations - mean_observation) / data.observation_sd
    expected_likelihood = float(
        -0.5 * data.observations.size * math.log(2.0 * math.pi)
        - np.log(data.observation_sd).sum()
        - 0.5 * np.square(standardized_residual).sum()
    )
    expected_root = (
        data.root_shape * math.log(data.root_rate)
        - math.lgamma(data.root_shape)
        + (data.root_shape - 1.0) * math.log(root_total)
        - data.root_rate * root_total
    )
    expected_share = float(
        math.lgamma(float(data.dirichlet_alpha.sum()))
        - sum(math.lgamma(float(alpha)) for alpha in data.dirichlet_alpha)
        + np.dot(data.dirichlet_alpha - 1.0, np.log(leaf_share))
    )
    lognormal_mu, lognormal_sigma = data.fixed_lognormal_mu_sigma
    expected_fixed_terms = (
        -0.5 * math.log(2.0 * math.pi)
        - np.log(lognormal_sigma)
        - np.log(fixed_coefficient)
        - 0.5 * np.square((np.log(fixed_coefficient) - lognormal_mu) / lognormal_sigma)
    )
    expected_fixed = float(expected_fixed_terms.sum())

    assert state.log_likelihood == pytest.approx(expected_likelihood, rel=2.0e-13, abs=2.0e-13)
    assert state.log_root_prior == pytest.approx(expected_root, rel=2.0e-13, abs=2.0e-13)
    assert state.log_allocation_prior == pytest.approx(expected_share, rel=2.0e-13, abs=2.0e-13)
    assert state.log_fixed_coefficient_prior == pytest.approx(
        expected_fixed,
        rel=2.0e-13,
        abs=2.0e-13,
    )
    expected_target = expected_likelihood + expected_root + expected_share + expected_fixed
    assert state.log_target == pytest.approx(expected_target, rel=2.0e-13, abs=2.0e-13)
    assert state.log_target != pytest.approx(
        expected_target + (data.k - 1) * math.log(root_total),
        rel=2.0e-13,
        abs=2.0e-13,
    )


def test_pymc_model_has_expected_float64_variables_and_coordinates() -> None:
    """The reference model exposes stable scientific variables in float64."""
    _run_x64_case("model")


def _assert_pymc_model_has_expected_float64_variables_and_coordinates() -> None:
    """Assert model metadata inside the isolated float64 process."""
    problem, state = _problem_state()
    data = prepare_fixed_basis_nuts(problem, state)
    model = build_fixed_basis_pymc_model(data)

    assert set(model.coords) == {"observation", "leaf", "fixed"}
    leaf_coord = model.coords["leaf"]
    fixed_coord = model.coords["fixed"]
    assert leaf_coord is not None
    assert fixed_coord is not None
    assert tuple(leaf_coord) == data.leaf_labels
    assert tuple(fixed_coord) == ("fixed_0", "fixed_1")
    assert set(model.named_vars) == {
        "root_total",
        "leaf_share",
        "leaf_mass",
        "leaf_scaling",
        "fixed_coefficient",
        "mean_observation",
        "observed",
    }
    assert {variable.name for variable in model.free_RVs} == {
        "root_total",
        "leaf_share",
        "fixed_coefficient",
    }
    assert {variable.name for variable in model.deterministics} == {
        "leaf_mass",
        "leaf_scaling",
        "mean_observation",
    }
    assert {variable.name for variable in model.observed_RVs} == {"observed"}
    assert all(str(variable.dtype) == "float64" for variable in model.basic_RVs)


@pytest.mark.parametrize("profile", ["prior_mean", "arbitrary"])
def test_compiled_constrained_logp_matches_existing_target(profile: str) -> None:
    """PyMC's no-Jacobian density must equal the existing normalized target."""
    _run_x64_case(f"logp_{profile}")


def _assert_compiled_constrained_logp_matches_existing_target(profile: str) -> None:
    """Assert target parity inside the isolated float64 process.

    Args:
        profile: Either ``"prior_mean"`` or ``"arbitrary"``.

    Raises:
        ValueError: If ``profile`` does not name a supported state.
    """
    if profile not in {"prior_mean", "arbitrary"}:
        raise ValueError("profile must be 'prior_mean' or 'arbitrary'.")
    problem, prior_mean = _problem_state()
    state = prior_mean if profile == "prior_mean" else _arbitrary_interior_state(problem, prior_mean)
    data = prepare_fixed_basis_nuts(problem, state)
    model = build_fixed_basis_pymc_model(data)

    metadata = preflight_fixed_basis_nuts(
        data,
        model,
        initvals=fixed_basis_nuts_initvals(data),
        expected_log_target=state.log_target,
    )

    assert metadata["constrained_log_target"] == pytest.approx(
        state.log_target,
        rel=5.0e-10,
        abs=5.0e-10,
    )
    assert abs(float(metadata["log_target_difference"])) <= float(metadata["log_target_absolute_tolerance"])


def test_bridge_rejects_a_powered_likelihood() -> None:
    """The NUTS reference must not silently sample a tempered target."""
    problem, state = _problem_state(likelihood_power=0.75)

    with pytest.raises(ValueError, match="likelihood_power == 1.0"):
        prepare_fixed_basis_nuts(problem, state)


def test_float64_guard_reports_backend_precision_metadata() -> None:
    """The precision gate reports both PyTensor and JAX float64 settings."""
    _run_x64_case("guard")


def _assert_float64_guard_reports_backend_precision_metadata() -> None:
    """Assert precision metadata inside the isolated float64 process."""
    metadata = require_fixed_basis_nuts_float64()

    assert metadata["pytensor_floatX"] == "float64"
    assert metadata["jax_enable_x64"] is True
    for name in (
        "pymc_version",
        "pytensor_version",
        "jax_version",
        "numpyro_version",
        "arviz_version",
    ):
        assert isinstance(metadata[name], str)
        assert metadata[name]


def test_sampler_wrapper_forwards_exact_numpyro_controls() -> None:
    """The wrapper passes the audited NumPyro controls and constrained start."""
    _run_x64_case("sampler")


def _assert_sampler_wrapper_forwards_exact_numpyro_controls() -> None:
    """Assert sampler forwarding inside the isolated float64 process."""
    problem, state = _problem_state()
    data = prepare_fixed_basis_nuts(problem, state)
    model = build_fixed_basis_pymc_model(data)
    initvals = fixed_basis_nuts_initvals(data)
    captured: dict[str, Any] = {}
    expected = az.from_dict(
        posterior={"root_total": np.ones((2, 2))},
        log_likelihood={"observed": np.zeros((2, 2, 3))},
    )

    def fake_sample(**kwargs: Any) -> az.InferenceData:
        """Capture PyMC arguments while returning a genuine tiny result."""
        captured.update(kwargs)
        return expected

    numpy_seed: Any = np.int64(901)
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr("pymc.sample", fake_sample)
        actual = sample_fixed_basis_nuts(
            model,
            data,
            draws=7,
            tune=11,
            seed=numpy_seed,
            target_accept=0.93,
            chains=2,
            cores=2,
            chain_method="parallel",
            progressbar=False,
            max_tree_depth=9,
            dense_mass=True,
            initvals=initvals,
        )
    finally:
        monkeypatch.undo()

    assert actual is expected
    assert captured["nuts_sampler"] == "numpyro"
    assert captured["random_seed"] == 901
    assert isinstance(captured["random_seed"], int)
    assert captured["initvals"] is initvals
    assert captured["return_inferencedata"] is True
    assert captured["idata_kwargs"] == {"log_likelihood": True}
    assert captured["nuts_sampler_kwargs"] == {
        "jitter": False,
        "chain_method": "parallel",
        "nuts_kwargs": {
            "max_tree_depth": 9,
            "dense_mass": True,
        },
    }
    assert captured["draws"] == 7
    assert captured["tune"] == 11
    assert captured["chains"] == 2
    assert captured["cores"] == 2
    assert captured["target_accept"] == 0.93
    assert captured["progressbar"] is False


_ISOLATED_CASES = {
    "model": _assert_pymc_model_has_expected_float64_variables_and_coordinates,
    "logp_prior_mean": lambda: _assert_compiled_constrained_logp_matches_existing_target("prior_mean"),
    "logp_arbitrary": lambda: _assert_compiled_constrained_logp_matches_existing_target("arbitrary"),
    "guard": _assert_float64_guard_reports_backend_precision_metadata,
    "sampler": _assert_sampler_wrapper_forwards_exact_numpyro_controls,
}


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in _ISOLATED_CASES:
        raise SystemExit("expected one isolated fixed-basis NUTS test case name")
    _ISOLATED_CASES[sys.argv[1]]()
