"""Tests for experimental RJMCMC run profiles and provenance manifests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from openghg_inversions.experimental.rjmcmc.profiles import (
    RUN_MANIFEST_SCHEMA_VERSION,
    InputReference,
    RetentionSettings,
    RunProfile,
    RunProvenance,
    TargetSettings,
)
from openghg_inversions.experimental.rjmcmc.sampling import SamplerConfig


def _target() -> TargetSettings:
    """Return a small non-uniform target declaration."""
    return TargetSettings(
        k_min=2,
        k_max=4,
        k_prior_probabilities=(0.2, 0.5, 0.3),
        coefficient_prior_mean=1,
        coefficient_prior_sd=0.8,
    )


def _sampler(*, seed: int | None = 481, iterations: int = 40) -> SamplerConfig:
    """Return a local-move sampler declaration."""
    return SamplerConfig(
        iterations=iterations,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        seed=seed,
        backend="numba",
        nucleus_move="local",
        local_move_scale=1.4,
    )


def test_manifest_is_complete_json_serializable_and_canonical() -> None:
    """A profile should expose stable scalar settings and sorted input identities."""
    checksum = "A1" * 32
    provenance = RunProvenance(
        code_revision=" 3b463e6 ",
        inputs=(
            InputReference(role="footprints", identifier="name.nc", sha256=checksum),
            InputReference(role="flux", identifier="edgar.nc"),
        ),
    )
    profile = RunProfile(
        name=" checkerboard-local ",
        target=_target(),
        sampler=_sampler(),
        retention=RetentionSettings(warmup_transitions=15, thin=5),
        provenance=provenance,
    )

    manifest = profile.to_manifest()

    assert manifest == {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "profile_name": "checkerboard-local",
        "target": {
            "active_region_count": {
                "minimum": 2,
                "maximum": 4,
                "prior_probabilities": [0.2, 0.5, 0.3],
            },
            "coefficient_prior": {
                "distribution": "lognormal",
                "parameterization": "arithmetic_moments",
                "mean": 1.0,
                "standard_deviation": 0.8,
            },
            "observation_error_model": "independent_gaussian",
        },
        "sampler": {
            "iterations": 40,
            "coefficient_proposal_sd": 0.15,
            "birth_proposal_sd": 0.25,
            "seed": 481,
            "backend": "numba",
            "nucleus_move": "local",
            "local_move_scale": 1.4,
        },
        "retention": {"warmup_transitions": 15, "thin": 5},
        "provenance": {
            "code_revision": "3b463e6",
            "inputs": [
                {"role": "flux", "identifier": "edgar.nc", "sha256": None},
                {
                    "role": "footprints",
                    "identifier": "name.nc",
                    "sha256": checksum.lower(),
                },
            ],
        },
    }
    assert json.loads(profile.to_json()) == manifest
    assert profile.to_json() == profile.to_json()
    assert "NaN" not in profile.to_json()
    assert "Infinity" not in profile.to_json()


def test_input_order_does_not_change_canonical_json() -> None:
    """Equivalent input sets should produce identical JSON regardless of caller order."""
    inputs = (
        InputReference(role="observations", identifier="obs.nc"),
        InputReference(role="flux", identifier="flux.nc"),
    )
    common = {"name": "same-run", "target": _target(), "sampler": _sampler()}

    forward = RunProfile(**common, provenance=RunProvenance(inputs=inputs))
    reverse = RunProfile(**common, provenance=RunProvenance(inputs=tuple(reversed(inputs))))

    assert forward.to_json() == reverse.to_json()


def test_profile_components_are_immutable_and_own_sequence_inputs() -> None:
    """Mutable caller sequences should be copied to tuples by frozen value objects."""
    probabilities = [0.25, 0.75]
    references = [InputReference(role="synthetic", identifier="tiny-v1")]
    target = TargetSettings(
        k_min=1,
        k_max=2,
        k_prior_probabilities=probabilities,  # type: ignore[arg-type]
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=1.0,
    )
    provenance = RunProvenance(inputs=references)  # type: ignore[arg-type]
    probabilities[0] = 1.0
    references.clear()

    assert target.k_prior_probabilities == (0.25, 0.75)
    assert len(provenance.inputs) == 1
    with pytest.raises(FrozenInstanceError):
        target.k_min = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k_min": 0},
        {"k_min": 3, "k_max": 2},
        {"k_min": 1.5},
        {"k_prior_probabilities": (0.5,)},
        {"k_prior_probabilities": (0.2, 0.2, 0.2)},
        {"k_prior_probabilities": (0.5, -0.1, 0.6)},
        {"k_prior_probabilities": (0.5, float("nan"), 0.5)},
        {"k_prior_probabilities": (True, 0.0, 0.0)},
        {"coefficient_prior_mean": 0.0},
        {"coefficient_prior_sd": float("inf")},
        {"observation_error_model": "correlated_gaussian"},
    ],
)
def test_target_settings_reject_malformed_values(kwargs: dict[str, object]) -> None:
    """Target support, probabilities, moments, and likelihood must be explicit and valid."""
    values: dict[str, object] = {
        "k_min": 2,
        "k_max": 4,
        "k_prior_probabilities": (0.2, 0.5, 0.3),
        "coefficient_prior_mean": 1.0,
        "coefficient_prior_sd": 0.8,
    }
    values.update(kwargs)

    with pytest.raises(ValueError):
        TargetSettings(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_transitions": -1},
        {"warmup_transitions": True},
        {"warmup_transitions": 0.5},
        {"thin": 0},
        {"thin": True},
        {"thin": 1.5},
    ],
)
def test_retention_settings_reject_malformed_values(kwargs: dict[str, object]) -> None:
    """Retention selection should use integer transition counts and positive thinning."""
    with pytest.raises(ValueError):
        RetentionSettings(**kwargs)  # type: ignore[arg-type]


def test_retention_settings_select_global_transition_phase() -> None:
    """Warmup and thinning should select completed transitions globally."""
    retention = RetentionSettings(warmup_transitions=5, thin=4)

    retained = [transition for transition in range(23) if retention.retains(transition)]

    assert retained == [5, 9, 13, 17, 21]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"role": " ", "identifier": "x"}, "role"),
        ({"role": "flux", "identifier": ""}, "identifier"),
        ({"role": "flux", "identifier": "x", "sha256": "abc"}, "sha256"),
        ({"role": "flux", "identifier": "x", "sha256": "g" * 64}, "sha256"),
    ],
)
def test_input_reference_rejects_malformed_identity(kwargs: dict[str, object], match: str) -> None:
    """Provenance inputs should have stable non-empty identities and valid checksums."""
    with pytest.raises(ValueError, match=match):
        InputReference(**kwargs)  # type: ignore[arg-type]


def test_provenance_rejects_duplicate_input_identity() -> None:
    """A role/identifier pair should appear at most once in a manifest."""
    reference = InputReference(role="flux", identifier="flux.nc")

    with pytest.raises(ValueError, match="repeat"):
        RunProvenance(inputs=(reference, reference))


@pytest.mark.parametrize(
    ("sampler", "retention", "match"),
    [
        (_sampler(seed=None), RetentionSettings(), "seed"),
        (_sampler(seed=-1), RetentionSettings(), "seed"),
        (_sampler(seed=True), RetentionSettings(), "seed"),
        (_sampler(iterations=4), RetentionSettings(warmup_transitions=5), "warmup"),
    ],
)
def test_profile_rejects_non_reproducible_or_inconsistent_settings(
    sampler: SamplerConfig,
    retention: RetentionSettings,
    match: str,
) -> None:
    """Profiles require an explicit valid seed and warmup inside the sampled chain."""
    with pytest.raises(ValueError, match=match):
        RunProfile(name="invalid", target=_target(), sampler=sampler, retention=retention)
