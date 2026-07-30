"""Focused tests for the authenticated score-regularized root-flow artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from openghg_inversions.experimental.rjmcmc.aggregation_error_exact_mixture import (  # noqa: E402
    RootResidualSpectrum,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_conditional_mixture import (  # noqa: E402
    ConditionalAllocationMixture,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_low_rank import (  # noqa: E402
    AdditiveDirichletAggregation,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import (  # noqa: E402
    GAMMA_LOG_MASS_CONDITIONING_RULE,
    ScoreRegularizedRootFlow,
)
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_training import (  # noqa: E402
    make_score_regularized_conditional_flow,
)

_HASH_A = "1" * 64
_HASH_B = "2" * 64
_HASH_C = "3" * 64
_ARTIFACT_MAGIC = b"OpenGHG-score-regularized-root-flow-v1\0"


def _replace_metadata_value(
    serialized: bytes,
    name: str,
    value: object,
) -> bytes:
    """Return canonical bytes with one metadata value changed."""
    length_offset = len(_ARTIFACT_MAGIC)
    metadata_length = struct.unpack(
        "<Q",
        serialized[length_offset : length_offset + 8],
    )[0]
    metadata_offset = length_offset + 8
    metadata_end = metadata_offset + metadata_length
    payload = json.loads(serialized[metadata_offset:metadata_end])
    payload[name] = value
    metadata = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _ARTIFACT_MAGIC + struct.pack("<Q", len(metadata)) + metadata + serialized[metadata_end:]


def _spectrum(
    observation_count: int = 2,
    retained_rank: int = 2,
    *,
    basis: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    noise_sd: np.ndarray | None = None,
    eigenvalues: np.ndarray | None = None,
) -> RootResidualSpectrum:
    """Return one small valid complete root spectrum."""
    if basis is None:
        basis = np.eye(observation_count, retained_rank, dtype=np.float64)
    if mean is None:
        mean = np.linspace(0.2, -0.1, observation_count, dtype=np.float64)
    if noise_sd is None:
        noise_sd = np.linspace(0.7, 1.1, observation_count, dtype=np.float64)
    if eigenvalues is None:
        eigenvalues = np.asarray([0.8, 0.25], dtype=np.float64)[:retained_rank]
    total_variance = float(np.sum(eigenvalues))
    return RootResidualSpectrum(
        mean,
        noise_sd,
        basis,
        eigenvalues,
        total_variance=total_variance,
        discarded_variance=0.0,
        requested_retained_variance_fraction=1.0,
        eigenvalue_tolerance=1.0e-14,
        cell_alphas_sha256=_HASH_A,
        design_sha256=_HASH_B,
        noise_sd_sha256=_HASH_C,
    )


def _artifact(
    leading_rank: int,
    *,
    spectrum: RootResidualSpectrum | None = None,
    seed: int = 73,
) -> ScoreRegularizedRootFlow:
    """Return an initialized fitted-artifact-shaped object."""
    context = _spectrum() if spectrum is None else spectrum
    flow = (
        None
        if leading_rank == 0
        else make_score_regularized_conditional_flow(
            leading_rank,
            source_seed=seed,
        )
    )
    return ScoreRegularizedRootFlow(
        context,
        leading_rank,
        43.0,
        43.0,
        flow,
        conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
        initialization_seed=seed,
        source_provenance="score-flow artifact unit test",
    )


@pytest.mark.parametrize("dimension", [1, 2])
def test_flow_likelihood_numerically_normalizes_at_fixed_mass(
    dimension: int,
) -> None:
    """The complete observation-unit density integrates to one."""
    spectrum = _spectrum(
        observation_count=dimension,
        retained_rank=dimension,
    )
    artifact = _artifact(dimension, spectrum=spectrum)
    mass = 1.2
    mean = mass * spectrum.observation_mean_design
    scales = np.sqrt(1.0 + mass * mass * spectrum.eigenvalues)
    if dimension == 1:
        standardized_grid = np.linspace(-10.0, 10.0, 4001)
        observations = (mean[0] + spectrum.noise_sd[0] * scales[0] * standardized_grid)[:, None]
        density = np.exp(artifact.log_likelihood_observation_batch(observations, mass))
        integral = np.trapezoid(density, observations[:, 0])
        assert integral == pytest.approx(1.0, abs=3.0e-5)
    else:
        standardized_grid = np.linspace(-8.0, 8.0, 121)
        first, second = np.meshgrid(
            standardized_grid,
            standardized_grid,
            indexing="ij",
        )
        standardized = np.stack([first.ravel(), second.ravel()], axis=1)
        observations = (
            mean[np.newaxis, :] + standardized * scales[np.newaxis, :] * spectrum.noise_sd[np.newaxis, :]
        )
        density = np.exp(artifact.log_likelihood_observation_batch(observations, mass)).reshape(first.shape)
        integral_second = np.trapezoid(
            density,
            observations[: standardized_grid.size, 1],
            axis=1,
        )
        integral = np.trapezoid(
            integral_second,
            observations[:: standardized_grid.size, 0],
        )
        assert integral == pytest.approx(1.0, abs=2.0e-3)


def test_sample_and_density_are_consistent_and_gaussian_control_has_moments() -> None:
    """Forward draws use the scored density and q=0 has analytic moments."""
    flow_artifact = _artifact(1)
    mass = 0.9
    samples = flow_artifact.sample_observation(
        mass,
        sample_count=128,
        source_seed=109,
    )
    batch_density = flow_artifact.log_likelihood_observation_batch(samples, mass)
    scalar_density = np.asarray([flow_artifact.log_likelihood(row, mass) for row in samples])
    np.testing.assert_allclose(
        batch_density,
        scalar_density,
        rtol=0.0,
        atol=3.0e-13,
    )

    gaussian = _artifact(0)
    gaussian_samples = gaussian.sample_observation(
        mass,
        sample_count=20_000,
        source_seed=211,
    )
    spectrum = gaussian.spectrum
    expected_mean = mass * spectrum.observation_mean_design
    whitened_covariance = (
        np.eye(gaussian.observation_count)
        + spectrum.basis @ np.diag(mass * mass * spectrum.eigenvalues) @ spectrum.basis.T
    )
    expected_covariance = spectrum.noise_sd[:, None] * whitened_covariance * spectrum.noise_sd[None, :]
    np.testing.assert_allclose(
        gaussian_samples.mean(axis=0),
        expected_mean,
        rtol=0.0,
        atol=2.5e-2,
    )
    np.testing.assert_allclose(
        np.cov(gaussian_samples, rowvar=False, ddof=0),
        expected_covariance,
        rtol=0.035,
        atol=0.02,
    )


def test_zero_mass_is_literal_standard_gaussian_without_flow_condition() -> None:
    """T=0 is exact measurement noise and both tau scores vanish."""
    artifact = _artifact(2)
    observation = np.asarray([0.4, -0.7])
    offset = np.asarray([0.1, 0.2])
    residual = (observation - offset) / artifact.spectrum.noise_sd
    expected = (
        -0.5 * (artifact.observation_count * math.log(2.0 * math.pi) + residual @ residual)
        - np.log(artifact.spectrum.noise_sd).sum()
    )
    assert artifact.log_likelihood(
        observation,
        0.0,
        offset=offset,
    ) == pytest.approx(expected, abs=1.0e-15)
    point = np.asarray([0.25, -0.8])
    assert artifact.leading_standardized_partial_log_mass_score(point, 0.0) == 0.0
    np.testing.assert_array_equal(
        artifact.leading_standardized_observation_score(point, 0.0),
        -point,
    )
    assert artifact.leading_fixed_residual_log_mass_score(point, 0.0) == 0.0
    np.testing.assert_array_equal(
        artifact.sample_observation(
            0.0,
            sample_count=8,
            source_seed=51,
            offset=offset,
        ),
        offset
        + np.asarray(
            jax.random.normal(
                jax.random.key(51),
                (8, artifact.observation_count),
                dtype=jnp.float64,
            )
        )
        * artifact.spectrum.noise_sd,
    )


def test_q_zero_and_rank_zero_identical_footprint_limits_are_exact_gaussians() -> None:
    """Gaussian moment closure and a zero aggregation spectrum are literal."""
    gaussian = _artifact(0)
    mass = 1.4
    observation = np.asarray([0.8, -0.2])
    residual = (observation - mass * gaussian.spectrum.observation_mean_design) / gaussian.spectrum.noise_sd
    scales = np.sqrt(1.0 + mass * mass * gaussian.spectrum.eigenvalues)
    expected = (
        -0.5 * (gaussian.observation_count * math.log(2.0 * math.pi) + np.sum((residual / scales) ** 2))
        - np.log(gaussian.spectrum.noise_sd).sum()
        - np.log(scales).sum()
    )
    assert gaussian.log_likelihood(observation, mass) == pytest.approx(
        expected,
        abs=2.0e-15,
    )

    rank_zero_spectrum = _spectrum(
        observation_count=2,
        retained_rank=0,
        basis=np.empty((2, 0), dtype=np.float64),
        mean=np.asarray([1.25, 1.25]),
        noise_sd=np.asarray([0.4, 0.9]),
        eigenvalues=np.empty(0, dtype=np.float64),
    )
    rank_zero = _artifact(0, spectrum=rank_zero_spectrum)
    centered = (observation - mass * rank_zero_spectrum.observation_mean_design) / rank_zero_spectrum.noise_sd
    expected_rank_zero = (
        -0.5 * (2.0 * math.log(2.0 * math.pi) + centered @ centered)
        - np.log(rank_zero_spectrum.noise_sd).sum()
    )
    assert rank_zero.log_likelihood(observation, mass) == pytest.approx(
        expected_rank_zero,
        abs=2.0e-15,
    )
    assert rank_zero.flow is None
    assert rank_zero.leading_fixed_residual_log_mass_score([], mass) == 0.0


def test_one_cell_native_simulator_reaches_exact_rank_zero_gaussian_path() -> None:
    """A real one-cell Dirichlet construction has no allocation residual."""
    design = np.asarray([[1.2], [-0.4]], dtype=np.float64)
    noise_sd = np.asarray([0.5, 0.8], dtype=np.float64)
    aggregation = AdditiveDirichletAggregation(
        np.asarray([3.5], dtype=np.float64),
        design,
        noise_sd,
        np.eye(2, dtype=np.float64),
    )
    spectrum = RootResidualSpectrum.from_aggregation(aggregation)
    assert spectrum.retained_rank == 0
    simulator = ConditionalAllocationMixture.from_aggregation(
        aggregation,
        np.asarray([0], dtype=np.int64),
        sample_count=8,
        source_seed=119,
        source_provenance="one-cell score-flow limit test",
        construction_method="scrambled_sobol_balanced_dirichlet",
    )
    np.testing.assert_array_equal(
        simulator.projected_unit_mass_residual_factors,
        np.zeros((8, 2, 1), dtype=np.float64),
    )

    artifact = _artifact(0, spectrum=spectrum)
    mass = 1.7
    observation = np.asarray([2.3, -0.9], dtype=np.float64)
    centered = (observation - mass * design[:, 0]) / noise_sd
    expected = -0.5 * (2.0 * math.log(2.0 * math.pi) + centered @ centered) - np.log(noise_sd).sum()
    assert artifact.log_likelihood(observation, mass) == pytest.approx(
        expected,
        abs=2e-15,
    )


def test_offset_translation_and_both_batch_evaluators_are_exact() -> None:
    """Offsets translate observations and batched paths replay scalar results."""
    artifact = _artifact(1)
    observation = np.asarray([0.7, -0.5])
    offset = np.asarray([0.2, -0.3])
    masses = np.asarray([0.0, 0.6, 1.5])
    translated = artifact.log_likelihood(
        observation + offset,
        masses[1],
        offset=offset,
    )
    baseline = artifact.log_likelihood(observation, masses[1])
    assert translated == pytest.approx(baseline, abs=3.0e-15)

    expected_mass_batch = np.asarray([artifact.log_likelihood(observation, mass) for mass in masses])
    np.testing.assert_array_equal(
        artifact.log_likelihood_batch(observation, masses),
        expected_mass_batch,
    )
    observations = np.stack([observation, observation + 0.1, observation - 0.2])
    expected_observation_batch = np.asarray([artifact.log_likelihood(row, masses[1]) for row in observations])
    np.testing.assert_allclose(
        artifact.log_likelihood_observation_batch(observations, masses[1]),
        expected_observation_batch,
        rtol=0.0,
        atol=3.0e-13,
    )


def test_permutation_and_eigenvector_sign_invariance_at_likelihood_level() -> None:
    """Representation-only row and eigenvector changes preserve q=0 density."""
    spectrum = _spectrum()
    artifact = _artifact(0, spectrum=spectrum)
    permutation = np.asarray([1, 0])
    changed_basis = spectrum.basis[permutation].copy()
    changed_basis[:, 0] *= -1.0
    transformed_spectrum = _spectrum(
        basis=changed_basis,
        mean=spectrum.observation_mean_design[permutation],
        noise_sd=spectrum.noise_sd[permutation],
        eigenvalues=spectrum.eigenvalues,
    )
    transformed = _artifact(0, spectrum=transformed_spectrum)
    observation = np.asarray([0.9, -0.4])
    offset = np.asarray([0.1, 0.3])
    mass = 1.1
    assert transformed.log_likelihood(
        observation[permutation],
        mass,
        offset=offset[permutation],
    ) == pytest.approx(
        artifact.log_likelihood(observation, mass, offset=offset),
        abs=3.0e-15,
    )


def test_observation_permutation_preserves_positive_rank_flow_likelihood() -> None:
    """Permuting observation-aligned context leaves learned coordinates unchanged."""
    spectrum = _spectrum()
    artifact = _artifact(1, spectrum=spectrum)
    permutation = np.asarray([1, 0])
    transformed_spectrum = _spectrum(
        basis=spectrum.basis[permutation],
        mean=spectrum.observation_mean_design[permutation],
        noise_sd=spectrum.noise_sd[permutation],
        eigenvalues=spectrum.eigenvalues,
    )
    transformed = _artifact(1, spectrum=transformed_spectrum)
    observation = np.asarray([0.9, -0.4])
    offset = np.asarray([0.1, 0.3])
    mass = 1.1
    assert transformed.log_likelihood(
        observation[permutation],
        mass,
        offset=offset[permutation],
    ) == pytest.approx(
        artifact.log_likelihood(observation, mass, offset=offset),
        abs=3.0e-14,
    )


def test_leading_scores_match_direct_jax_derivatives() -> None:
    """Artifact score methods use raw tau and the complete fixed-y chain rule."""
    artifact = _artifact(1)
    mass = 1.3
    leading_residual = np.asarray([0.65])
    eigenvalue = jnp.asarray(artifact.spectrum.eigenvalues[:1])
    raw_tau = jnp.asarray(math.log(mass), dtype=jnp.float64)

    def transformed_log_density(tau: jax.Array) -> jax.Array:
        local_mass = jnp.exp(tau)
        scale = jnp.sqrt(1.0 + local_mass**2 * eigenvalue)
        standardized = jnp.asarray(leading_residual) / scale
        condition = jnp.asarray([(tau - artifact.condition_center) / artifact.condition_scale])
        return artifact.flow.log_prob(standardized, condition) - jnp.log(scale).sum()

    direct_fixed = jax.grad(transformed_log_density)(raw_tau)
    assert artifact.leading_fixed_residual_log_mass_score(
        leading_residual,
        mass,
    ) == pytest.approx(float(direct_fixed), abs=3.0e-14)

    scale = math.sqrt(1.0 + mass * mass * float(eigenvalue[0]))
    standardized = np.asarray([leading_residual[0] / scale])
    condition = jnp.asarray([(raw_tau - artifact.condition_center) / artifact.condition_scale])
    direct_observation = jax.grad(lambda point: artifact.flow.log_prob(point, condition))(
        jnp.asarray(standardized)
    )
    direct_partial = jax.grad(
        lambda tau: artifact.flow.log_prob(
            jnp.asarray(standardized),
            jnp.asarray([(tau - artifact.condition_center) / artifact.condition_scale]),
        )
    )(raw_tau)
    np.testing.assert_allclose(
        artifact.leading_standardized_observation_score(standardized, mass),
        direct_observation,
        rtol=0.0,
        atol=3.0e-14,
    )
    assert artifact.leading_standardized_partial_log_mass_score(
        standardized,
        mass,
    ) == pytest.approx(float(direct_partial), abs=3.0e-14)


@pytest.mark.parametrize("leading_rank", [0, 1, 2])
def test_strict_float64_serialization_replays_exactly(leading_rank: int) -> None:
    """Every spectrum value and fitted float64 leaf survives canonical replay."""
    artifact = _artifact(leading_rank)
    expected_specialization = {
        0: "gaussian-rank-zero",
        1: "masked-autoregressive-rational-quadratic-spline",
        2: "rational-quadratic-spline-coupling",
    }[leading_rank]
    assert (
        artifact.metadata_payload["architecture"]["specialization"]  # type: ignore[index]
        == expected_specialization
    )
    serialized = artifact.to_bytes()
    replay = ScoreRegularizedRootFlow.from_bytes(
        serialized,
        expected_sha256=artifact.sha256,
    )
    assert replay.to_bytes() == serialized
    assert replay.sha256 == hashlib.sha256(serialized).hexdigest()
    np.testing.assert_array_equal(
        replay.spectrum.observation_mean_design,
        artifact.spectrum.observation_mean_design,
    )
    np.testing.assert_array_equal(replay.spectrum.basis, artifact.spectrum.basis)
    assert replay.gamma_shape == 43.0
    assert replay.gamma_rate == 43.0
    assert replay.conditioning_rule_id == GAMMA_LOG_MASS_CONDITIONING_RULE
    assert replay.condition_center == artifact.condition_center
    assert replay.condition_scale == artifact.condition_scale
    assert replay.log_likelihood([0.2, -0.4], 0.8) == artifact.log_likelihood(
        [0.2, -0.4],
        0.8,
    )


@pytest.mark.parametrize("leading_rank", [1, 3])
def test_fresh_subprocess_replays_authenticated_bytes(
    tmp_path: Path,
    leading_rank: int,
) -> None:
    """Template reconstruction and exact bytes are process-independent."""
    if leading_rank == 1:
        spectrum = _spectrum()
        observation = np.asarray((0.2, -0.4))
    else:
        spectrum = _spectrum(
            observation_count=3,
            retained_rank=3,
            eigenvalues=np.asarray((0.8, 0.25, 0.1)),
        )
        observation = np.asarray((0.2, -0.4, 0.1))
    artifact = _artifact(leading_rank, spectrum=spectrum)
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(artifact.to_bytes())
    totals = np.asarray((0.5, 0.8, 1.2))
    canonical = ScoreRegularizedRootFlow.from_bytes(
        artifact_path.read_bytes(),
        expected_sha256=artifact.sha256,
    )
    expected = canonical.log_likelihood_batch(observation, totals)
    script = f"""
from pathlib import Path
import numpy as np
from openghg_inversions.experimental.rjmcmc.aggregation_error_score_flow_artifact import ScoreRegularizedRootFlow
payload = Path({str(artifact_path)!r}).read_bytes()
artifact = ScoreRegularizedRootFlow.from_bytes(
    payload,
    expected_sha256={artifact.sha256!r},
)
assert artifact.to_bytes() == payload
actual = artifact.log_likelihood_batch(
    np.asarray({observation.tolist()!r}, dtype=np.float64),
    np.asarray({totals.tolist()!r}, dtype=np.float64),
)
np.testing.assert_array_equal(
    actual,
    np.asarray({expected.tolist()!r}, dtype=np.float64),
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_malformed_contexts_flows_and_serialized_artifacts_are_rejected() -> None:
    """Construction and authenticated replay fail closed."""
    spectrum = _spectrum()
    flow = make_score_regularized_conditional_flow(1, source_seed=73)
    with pytest.raises(TypeError, match="RootResidualSpectrum"):
        ScoreRegularizedRootFlow(  # type: ignore[arg-type]
            object(),  # pyright: ignore[reportArgumentType]
            1,
            43.0,
            43.0,
            flow,
            conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
            initialization_seed=73,
            source_provenance="bad",
        )
    with pytest.raises(ValueError, match="must not exceed"):
        _artifact(3, spectrum=spectrum)
    with pytest.raises(ValueError, match="flow is required"):
        ScoreRegularizedRootFlow(
            spectrum,
            1,
            43.0,
            43.0,
            None,
            conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
            initialization_seed=73,
            source_provenance="bad",
        )
    with pytest.raises(ValueError, match="must be None"):
        ScoreRegularizedRootFlow(
            spectrum,
            0,
            43.0,
            43.0,
            flow,
            conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
            initialization_seed=73,
            source_provenance="bad",
        )
    with pytest.raises(ValueError, match="event shape"):
        ScoreRegularizedRootFlow(
            spectrum,
            1,
            43.0,
            43.0,
            make_score_regularized_conditional_flow(2, source_seed=73),
            conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
            initialization_seed=73,
            source_provenance="bad",
        )
    with pytest.raises(ValueError, match="rate must be positive"):
        ScoreRegularizedRootFlow(
            spectrum,
            1,
            43.0,
            0.0,
            flow,
            conditioning_rule_id=GAMMA_LOG_MASS_CONDITIONING_RULE,
            initialization_seed=73,
            source_provenance="bad",
        )
    with pytest.raises(ValueError, match="conditioning_rule_id"):
        ScoreRegularizedRootFlow(
            spectrum,
            1,
            43.0,
            43.0,
            flow,
            conditioning_rule_id="empirical-training-moments",
            initialization_seed=73,
            source_provenance="bad",
        )

    artifact = _artifact(1)
    serialized = artifact.to_bytes()
    with pytest.raises(ValueError, match="fingerprint does not match"):
        ScoreRegularizedRootFlow.from_bytes(
            serialized[:-1] + bytes([serialized[-1] ^ 1]),
            expected_sha256=artifact.sha256,
        )
    trailing = serialized + b"x"
    with pytest.raises(ValueError, match="trailing bytes"):
        ScoreRegularizedRootFlow.from_bytes(
            trailing,
            expected_sha256=hashlib.sha256(trailing).hexdigest(),
        )
    wrong_rule = _replace_metadata_value(
        serialized,
        "conditioning_rule_id",
        "empirical-training-moments",
    )
    with pytest.raises(ValueError, match="conditioning rule does not match"):
        ScoreRegularizedRootFlow.from_bytes(
            wrong_rule,
            expected_sha256=hashlib.sha256(wrong_rule).hexdigest(),
        )
    wrong_center = _replace_metadata_value(
        serialized,
        "condition_center",
        artifact.condition_center + 0.01,
    )
    with pytest.raises(ValueError, match="does not replay analytically"):
        ScoreRegularizedRootFlow.from_bytes(
            wrong_center,
            expected_sha256=hashlib.sha256(wrong_center).hexdigest(),
        )
    with pytest.raises(ValueError, match="lower-case SHA-256"):
        ScoreRegularizedRootFlow.from_bytes(
            serialized,
            expected_sha256="not-a-digest",
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: _artifact(1).log_likelihood([0.2], 1.0),
        lambda: _artifact(1).log_likelihood([0.2, 0.3], -1.0),
        lambda: _artifact(1).log_likelihood(
            [0.2, 0.3],
            1.0,
            offset=[0.1],
        ),
        lambda: _artifact(1).leading_standardized_observation_score(
            [0.2, 0.3],
            1.0,
        ),
        lambda: _artifact(1).log_likelihood_batch(
            [0.2, 0.3],
            [[1.0]],
        ),
    ],
)
def test_evaluator_rejects_invalid_shapes_and_values(
    operation: object,
) -> None:
    """Public evaluators validate all scientific arrays and masses."""
    with pytest.raises((TypeError, ValueError)):
        operation()  # type: ignore[operator]
