"""Tests for the explicit RHIME-to-Gamma--Beta coordinate adapter."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.experimental.rjmcmc.gamma_beta_adapter import (
    gamma_beta_problem_from_rhime_inputs,
    initialize_gamma_beta_state,
)
from openghg_inversions.experimental.rjmcmc.gamma_beta_compound_sampling import (
    GammaBetaCompoundConfig,
    sample_gamma_beta_compound,
)
from openghg_inversions.experimental.rjmcmc.retention import RetentionSettings


def _dataset() -> tuple[xr.Dataset, np.ndarray]:
    """Return shuffled-dimension inputs and their canonical sensitivity."""
    canonical = np.arange(1.0, 17.0).reshape(2, 2, 4)
    sensitivity = xr.DataArray(
        canonical.transpose(2, 0, 1),
        dims=("lon", "nmeasure", "lat"),
        coords={
            "nmeasure": ["a", "b"],
            "lat": [50.0, 51.0],
            "lon": [10.0, 20.0, 30.0, 40.0],
        },
    )
    dataset = xr.Dataset(
        {
            "fp_x_flux": sensitivity,
            "mf": xr.DataArray(
                [0.0, 0.0],
                dims="nmeasure",
                coords={"nmeasure": ["a", "b"]},
            ),
            "mf_error": xr.DataArray(
                [1.0, 2.0],
                dims="nmeasure",
                coords={"nmeasure": ["a", "b"]},
            ),
        }
    )
    return dataset, canonical


def _adapter(
    dataset: xr.Dataset,
    nominal_weight: object,
    **kwargs: object,
):
    """Build a common adapter result while keeping tests concise."""
    return gamma_beta_problem_from_rhime_inputs(
        dataset,
        nominal_weight=nominal_weight,  # type: ignore[arg-type]
        k_min=1,
        k_max=6,
        concentration=4.0,
        **kwargs,  # type: ignore[arg-type]
    )


def test_adapter_aligns_shuffled_weights_and_converts_scaling_response() -> None:
    """Shuffled spatial dimensions should yield exact C-order mass sensitivity."""
    dataset, canonical = _dataset()
    canonical_weight = np.arange(1.0, 9.0).reshape(2, 4)
    shuffled_weight = xr.DataArray(
        canonical_weight.T,
        dims=("lon", "lat"),
        coords={"lon": [10.0, 20.0, 30.0, 40.0], "lat": [50.0, 51.0]},
    )

    result = _adapter(dataset, shuffled_weight)
    model_weight = canonical_weight.reshape(-1) / canonical_weight.sum()

    assert result.spatial_shape == (2, 4)
    assert result.weight_normalization_factor == canonical_weight.sum()
    assert result.weights_normalized
    np.testing.assert_array_equal(result.latitudes, [50.0, 51.0])
    np.testing.assert_array_equal(result.longitudes, [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_array_equal(result.nominal_weight, model_weight)
    np.testing.assert_array_equal(
        result.problem.sensitivity * model_weight[np.newaxis, :],
        canonical.reshape(2, -1),
    )


def test_prior_mean_state_closes_all_one_forward_model_with_fixed_terms() -> None:
    """Prior means should reproduce unit scaling plus selected fixed components."""
    dataset, canonical = _dataset()
    weight = np.arange(1.0, 9.0).reshape(2, 4)
    outer_design = np.array([[1.0, 2.0], [3.0, 5.0]])
    offset = np.array([7.0, 11.0])
    dataset["outer"] = xr.DataArray(
        outer_design,
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": ["a", "b"], "outer_region": ["north", "south"]},
    )
    dataset["offset"] = xr.DataArray(
        offset,
        dims="nmeasure",
        coords={"nmeasure": ["a", "b"]},
    )
    fixed_mean = np.array([1.5, 0.5])

    result = _adapter(
        dataset,
        weight,
        fixed_design_name="outer",
        fixed_offset_name="offset",
        fixed_coefficient_prior_mean=fixed_mean,
        fixed_coefficient_prior_sd=[0.25, 0.5],
    )
    state = initialize_gamma_beta_state(result.problem, k=5)

    expected_dynamic = canonical.reshape(2, -1).sum(axis=1)
    expected_fixed = offset + outer_design @ fixed_mean
    np.testing.assert_allclose(state.dynamic_prediction, expected_dynamic, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(state.fixed_prediction, expected_fixed, rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        state.prediction,
        expected_dynamic + expected_fixed,
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_array_equal(state.fixed_coefficients, fixed_mean)


@pytest.mark.parametrize(
    "weight",
    [
        np.array([[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        np.array([[1.0, -1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
        np.array([[1.0, np.nan, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    ],
    ids=("zero", "negative", "nonfinite"),
)
def test_adapter_rejects_nonpositive_or_nonfinite_weights(weight: np.ndarray) -> None:
    """Invalid weights must fail rather than being silently floored."""
    dataset, _ = _dataset()

    with pytest.raises(ValueError, match="strictly positive|finite"):
        _adapter(dataset, weight)


def test_adapter_rejects_misaligned_weight_coordinates() -> None:
    """A labelled weight field must exactly match native spatial coordinates."""
    dataset, _ = _dataset()
    weight = xr.DataArray(
        np.ones((2, 4)),
        dims=("lat", "lon"),
        coords={"lat": [50.0, 52.0], "lon": [10.0, 20.0, 30.0, 40.0]},
    )

    with pytest.raises(ValueError, match="align exactly"):
        _adapter(dataset, weight)


def test_initializer_splits_largest_nominal_mass_with_stable_ties() -> None:
    """Requested K should be reached by deterministic mass-priority splits."""
    dataset, _ = _dataset()
    weight = np.array([[8.0, 1.0, 4.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    result = _adapter(dataset, weight)

    state = initialize_gamma_beta_state(result.problem, k=4)

    assert state.k == 4
    # Reconstruct the required deterministic rule independently.
    expected = state.frontier.root(result.problem.tree)
    while len(expected) < 4:
        eligible = result.problem.tree.splittable_nodes(expected)
        node_id = min(
            eligible,
            key=lambda candidate: (
                -float(result.problem.node_nominal_mass[candidate]),
                candidate,
            ),
        )
        expected = expected.split(result.problem.tree, node_id)
    assert state.frontier == expected
    np.testing.assert_allclose(
        state.active_fractions,
        [
            alpha / (alpha + beta)
            for alpha, beta in (
                result.problem.prior.beta_parameters(node_id)
                for node_id in expected.active_split_nodes(result.problem.tree)
            )
        ],
    )
    assert state.root_total == pytest.approx(1.0)


def test_supplied_k_prior_is_normalized_and_controls_initializer_support() -> None:
    """Explicit K masses should retain only their declared supported counts."""
    dataset, _ = _dataset()
    masses = np.zeros(7)
    masses[2] = 1.0
    masses[4] = 3.0

    result = _adapter(dataset, np.ones((2, 4)), probabilities_by_k=masses)

    np.testing.assert_array_equal(
        result.problem.partition_prior.p_k,
        [0.0, 0.0, 0.25, 0.0, 0.75, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="zero partition-prior"):
        initialize_gamma_beta_state(result.problem, k=3)
    assert initialize_gamma_beta_state(result.problem, k=4).k == 4


def test_unnormalized_weights_preserve_unit_scaling_at_prior_mean() -> None:
    """Disabling normalization should use the supplied total as root mean."""
    dataset, canonical = _dataset()
    weight = np.arange(1.0, 9.0).reshape(2, 4)

    result = _adapter(dataset, weight, normalize_weights=False, root_variance=2.0)
    state = initialize_gamma_beta_state(result.problem, k=3)

    assert not result.weights_normalized
    assert state.root_total == pytest.approx(weight.sum())
    np.testing.assert_allclose(
        state.dynamic_prediction,
        canonical.reshape(2, -1).sum(axis=1),
        rtol=0.0,
        atol=1.0e-13,
    )


def test_adapter_outputs_own_read_only_arrays() -> None:
    """The adapter and initializer must not expose mutable input-backed arrays."""
    dataset, _ = _dataset()
    weight = np.arange(1.0, 9.0).reshape(2, 4)
    dataset["outer"] = xr.DataArray(
        np.ones((2, 1)),
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": ["a", "b"], "outer_region": ["outer"]},
    )
    dataset["offset"] = xr.DataArray(
        np.ones(2),
        dims="nmeasure",
        coords={"nmeasure": ["a", "b"]},
    )

    result = _adapter(
        dataset,
        weight,
        fixed_design_name="outer",
        fixed_offset_name="offset",
        fixed_coefficient_prior_mean=1.0,
        fixed_coefficient_prior_sd=0.5,
    )
    state = initialize_gamma_beta_state(result.problem, k=2)
    assert result.problem.fixed_block is not None

    arrays = [
        result.nominal_weight,
        result.latitudes,
        result.longitudes,
        result.problem.sensitivity,
        result.problem.observations,
        result.problem.observation_sd,
        result.problem.fixed_offset,
        result.problem.fixed_block.design,
        state.active_fractions,
        state.fixed_coefficients,
        state.prediction,
    ]
    assert all(array is not None and not array.flags.writeable for array in arrays)


@pytest.mark.parametrize(
    ("variable_name", "bad_value", "message"),
    [
        ("fp_x_flux", np.nan, "finite"),
        ("mf", np.nan, "finite"),
        ("mf_error", np.nan, "finite"),
        ("mf_error", 0.0, "positive"),
    ],
)
def test_adapter_rejects_malformed_core_numerical_inputs(
    variable_name: str,
    bad_value: float,
    message: str,
) -> None:
    """Nonfinite model inputs and nonpositive errors must fail explicitly."""
    dataset, _ = _dataset()
    dataset[variable_name].values.flat[0] = bad_value

    with pytest.raises(ValueError, match=message):
        _adapter(dataset, np.ones((2, 4)))


@pytest.mark.parametrize(
    ("variable_name", "fixed_design_name", "fixed_offset_name"),
    [
        ("outer", "outer", None),
        ("offset", None, "offset"),
    ],
)
def test_adapter_rejects_nonfinite_explicit_fixed_components(
    variable_name: str,
    fixed_design_name: str | None,
    fixed_offset_name: str | None,
) -> None:
    """Explicit fixed design and offset arrays must contain finite values."""
    dataset, _ = _dataset()
    dataset["outer"] = xr.DataArray(
        [[1.0, np.nan], [2.0, 3.0]],
        dims=("nmeasure", "outer_region"),
        coords={"nmeasure": ["a", "b"], "outer_region": ["one", "two"]},
    )
    dataset["offset"] = xr.DataArray(
        [0.0, np.nan],
        dims="nmeasure",
        coords={"nmeasure": ["a", "b"]},
    )
    kwargs: dict[str, object] = {
        "fixed_design_name": fixed_design_name,
        "fixed_offset_name": fixed_offset_name,
    }
    if fixed_design_name is not None:
        kwargs.update(
            fixed_coefficient_prior_mean=1.0,
            fixed_coefficient_prior_sd=0.5,
        )

    with pytest.raises(ValueError, match=f"{variable_name!r}.*finite"):
        _adapter(dataset, np.ones((2, 4)), **kwargs)


def test_adapter_requires_every_explicitly_selected_variable() -> None:
    """A selected offset or fixed design must exist instead of being guessed."""
    dataset, _ = _dataset()

    with pytest.raises(ValueError, match="YaprioriBC"):
        _adapter(dataset, np.ones((2, 4)), fixed_offset_name="YaprioriBC")
    with pytest.raises(ValueError, match="outer_design"):
        _adapter(
            dataset,
            np.ones((2, 4)),
            fixed_design_name="outer_design",
            fixed_coefficient_prior_mean=1.0,
            fixed_coefficient_prior_sd=0.5,
        )


def test_adapter_problem_runs_one_complete_real_data_style_cycle() -> None:
    """An adapter-built six-outer problem should run the exact 14-slot cycle."""
    dataset, canonical = _dataset()
    dataset["outer_design"] = xr.DataArray(
        np.arange(1.0, 13.0).reshape(2, 6) / 20.0,
        dims=("nmeasure", "outer_region"),
        coords={
            "nmeasure": ["a", "b"],
            "outer_region": np.arange(6),
        },
    )
    dataset["YaprioriBC"] = xr.DataArray(
        [2.0, 3.0],
        dims="nmeasure",
        coords={"nmeasure": ["a", "b"]},
    )
    dataset["mf"] = xr.DataArray(
        canonical.reshape(2, -1).sum(axis=1)
        + dataset["outer_design"].values.sum(axis=1)
        + dataset["YaprioriBC"].values,
        dims="nmeasure",
        coords={"nmeasure": ["a", "b"]},
    )
    result = _adapter(
        dataset,
        np.arange(1.0, 9.0).reshape(2, 4),
        fixed_design_name="outer_design",
        fixed_offset_name="YaprioriBC",
        fixed_coefficient_prior_mean=np.ones(6),
        fixed_coefficient_prior_sd=np.ones(6),
    )
    initial = initialize_gamma_beta_state(result.problem, k=3)

    sampled = sample_gamma_beta_compound(
        result.problem,
        initial,
        GammaBetaCompoundConfig(iterations=14, seed=20260723),
        retention=RetentionSettings(warmup_transitions=0, thin=1),
    )

    assert sampled.checkpoint.kernel_settings.cycle_length == 14
    assert sampled.trace.global_transition.tolist() == list(range(1, 15))
    assert sampled.trace.fixed_coefficients.shape == (15, 6)
    assert sampled.trace.state_transition.tolist() == list(range(15))
    assert np.all(np.isfinite(sampled.trace.log_target))
