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
from openghg_inversions.experimental.rjmcmc.sampling import (
    LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE,
    LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE,
    LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE,
)


def _target() -> TargetSettings:
    """Return a small non-uniform target declaration."""
    return TargetSettings(
        k_min=2,
        k_max=4,
        k_prior_probabilities=(0.2, 0.5, 0.3),
        coefficient_prior_mean=1,
        coefficient_prior_sd=0.8,
    )


def _sampler(
    *,
    seed: int | None = 481,
    iterations: int = 40,
    fixed_scale: float | None = None,
) -> SamplerConfig:
    """Return a local-move sampler declaration."""
    return SamplerConfig(
        iterations=iterations,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        fixed_coefficient_proposal_sd=fixed_scale,
        seed=seed,
        backend="numba",
        nucleus_move="local",
        local_move_scale=1.4,
    )


def _ou_target(*, hierarchy: bool = False) -> TargetSettings:
    """Return an OU target with optional shared dynamic-coefficient pooling."""
    hierarchy_settings: dict[str, object] = {}
    if hierarchy:
        hierarchy_settings = {
            "shared_coefficient_hierarchy": True,
            "coefficient_hierarchy_parameterization": "shared_arithmetic_moments_log_state",
            "mean_hyperprior_median": 1.0,
            "mean_hyperprior_log_sd": 0.6,
            "sd_hyperprior_median": 0.8,
            "sd_hyperprior_log_sd": 0.4,
        }
    return TargetSettings(
        k_min=2,
        k_max=4,
        k_prior_probabilities=(0.2, 0.5, 0.3),
        coefficient_prior_mean=1.0,
        coefficient_prior_sd=0.8,
        observation_error_model="independent_site_ou_nugget",
        observation_time_unit="days",
        mismatch_sd_prior_lower=(0.1, 0.2),
        mismatch_sd_prior_upper=(10.0, 12.0),
        correlation_timescale_prior_lower=(0.25,),
        correlation_timescale_prior_upper=(30.0,),
        **hierarchy_settings,  # type: ignore[arg-type]
    )


def _ou_sampler(*, hierarchy: bool = False) -> SamplerConfig:
    """Return a sampler matched to an OU target and optional hierarchy."""
    return SamplerConfig(
        iterations=40,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        fixed_coefficient_proposal_sd=0.12,
        schedule_profile=(
            LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE
            if hierarchy
            else LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE
        ),
        mismatch_sd_proposal_sd=0.11,
        correlation_timescale_proposal_sd=0.7,
        eta_proposal_sd=0.08 if hierarchy else None,
        zeta_proposal_sd=0.09 if hierarchy else None,
        seed=481,
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
            "observation_error_model_settings": None,
            "dynamic_coefficient_hierarchy": None,
        },
        "sampler": {
            "iterations": 40,
            "coefficient_proposal_sd": 0.15,
            "birth_proposal_sd": 0.25,
            "fixed_coefficient_proposal_sd": None,
            "schedule_profile": "default",
            "mismatch_sd_proposal_sd": None,
            "correlation_timescale_proposal_sd": None,
            "eta_proposal_sd": None,
            "zeta_proposal_sd": None,
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


def test_manifest_records_explicit_fixed_coefficient_scale() -> None:
    """A distinct fixed-block proposal scale should be replayable from JSON."""
    profile = RunProfile(
        name="fixed-block",
        target=_target(),
        sampler=_sampler(fixed_scale=0.07),
    )

    sampler_manifest = profile.to_manifest()["sampler"]
    assert isinstance(sampler_manifest, dict)
    assert sampler_manifest["fixed_coefficient_proposal_sd"] == 0.07


@pytest.mark.parametrize("hierarchy", [False, True])
def test_manifest_records_complete_ou_and_optional_hierarchy_settings(hierarchy: bool) -> None:
    """OU and shared-hierarchy manifests should retain every bounded prior and scale."""
    profile = RunProfile(
        name="ou-hierarchy" if hierarchy else "ou",
        target=_ou_target(hierarchy=hierarchy),
        sampler=_ou_sampler(hierarchy=hierarchy),
    )

    manifest = profile.to_manifest()

    target = manifest["target"]
    assert isinstance(target, dict)
    assert target["observation_error_model"] == "independent_site_ou_nugget"
    assert target["observation_error_model_settings"] == {
        "time_unit": "days",
        "mismatch_sd_prior": {
            "distribution": "bounded_uniform",
            "lower": [0.1, 0.2],
            "upper": [10.0, 12.0],
        },
        "correlation_timescale_prior": {
            "distribution": "bounded_uniform",
            "lower": [0.25],
            "upper": [30.0],
        },
    }
    expected_hierarchy = (
        {
            "parameterization": "shared_arithmetic_moments_log_state",
            "includes_fixed_outer_coefficients": False,
            "mean_hyperprior_median": 1.0,
            "mean_hyperprior_log_sd": 0.6,
            "sd_hyperprior_median": 0.8,
            "sd_hyperprior_log_sd": 0.4,
        }
        if hierarchy
        else None
    )
    assert target["dynamic_coefficient_hierarchy"] == expected_hierarchy
    sampler = manifest["sampler"]
    assert isinstance(sampler, dict)
    assert sampler["schedule_profile"] == (
        LUNT_OPPORTUNITY_MATCHED_OU_HIERARCHY_SCHEDULE_PROFILE
        if hierarchy
        else LUNT_OPPORTUNITY_MATCHED_OU_SCHEDULE_PROFILE
    )
    assert sampler["mismatch_sd_proposal_sd"] == 0.11
    assert sampler["correlation_timescale_proposal_sd"] == 0.7
    assert sampler["eta_proposal_sd"] == (0.08 if hierarchy else None)
    assert sampler["zeta_proposal_sd"] == (0.09 if hierarchy else None)
    assert manifest == {
        "schema_version": 2,
        "profile_name": "ou-hierarchy" if hierarchy else "ou",
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
            "observation_error_model": "independent_site_ou_nugget",
            "observation_error_model_settings": target["observation_error_model_settings"],
            "dynamic_coefficient_hierarchy": expected_hierarchy,
        },
        "sampler": {
            "iterations": 40,
            "coefficient_proposal_sd": 0.15,
            "birth_proposal_sd": 0.25,
            "fixed_coefficient_proposal_sd": 0.12,
            "schedule_profile": sampler["schedule_profile"],
            "mismatch_sd_proposal_sd": 0.11,
            "correlation_timescale_proposal_sd": 0.7,
            "eta_proposal_sd": 0.08 if hierarchy else None,
            "zeta_proposal_sd": 0.09 if hierarchy else None,
            "seed": 481,
            "backend": "numba",
            "nucleus_move": "local",
            "local_move_scale": 1.4,
        },
        "retention": {"warmup_transitions": 0, "thin": 1},
        "provenance": {"code_revision": None, "inputs": []},
    }
    assert json.loads(profile.to_json()) == manifest


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
    "overrides",
    [
        {"observation_time_unit": "days"},
        {"mismatch_sd_prior_lower": (0.1,)},
        {"shared_coefficient_hierarchy": True},
        {
            "coefficient_hierarchy_parameterization": "shared_arithmetic_moments_log_state",
        },
    ],
)
def test_independent_target_rejects_ou_or_hierarchy_settings(overrides: dict[str, object]) -> None:
    """Independent nonhierarchical targets should not retain inactive settings."""
    values: dict[str, object] = {
        "k_min": 2,
        "k_max": 4,
        "k_prior_probabilities": (0.2, 0.5, 0.3),
        "coefficient_prior_mean": 1.0,
        "coefficient_prior_sd": 0.8,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        TargetSettings(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"observation_time_unit": None},
        {"observation_time_unit": " "},
        {"mismatch_sd_prior_lower": None},
        {"mismatch_sd_prior_lower": (0.1,)},
        {"mismatch_sd_prior_lower": (0.0, 0.2)},
        {"mismatch_sd_prior_upper": (0.1, 12.0)},
        {"correlation_timescale_prior_upper": (0.25,)},
        {"correlation_timescale_prior_lower": ()},
    ],
)
def test_ou_target_rejects_missing_or_invalid_bounded_priors(overrides: dict[str, object]) -> None:
    """OU declarations should require positive, ordered, shape-matched bounds."""
    values: dict[str, object] = {
        "k_min": 2,
        "k_max": 4,
        "k_prior_probabilities": (0.2, 0.5, 0.3),
        "coefficient_prior_mean": 1.0,
        "coefficient_prior_sd": 0.8,
        "observation_error_model": "independent_site_ou_nugget",
        "observation_time_unit": "days",
        "mismatch_sd_prior_lower": (0.1, 0.2),
        "mismatch_sd_prior_upper": (10.0, 12.0),
        "correlation_timescale_prior_lower": (0.25,),
        "correlation_timescale_prior_upper": (30.0,),
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        TargetSettings(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"coefficient_hierarchy_parameterization": "wrong"},
        {"mean_hyperprior_median": None},
        {"mean_hyperprior_log_sd": 0.0},
        {"sd_hyperprior_median": float("inf")},
        {"sd_hyperprior_log_sd": -0.1},
    ],
)
def test_shared_hierarchy_rejects_incomplete_or_invalid_hyperpriors(
    overrides: dict[str, object],
) -> None:
    """Enabled hierarchy declarations should fully specify positive hyperpriors."""
    values: dict[str, object] = {
        "k_min": 2,
        "k_max": 4,
        "k_prior_probabilities": (0.2, 0.5, 0.3),
        "coefficient_prior_mean": 1.0,
        "coefficient_prior_sd": 0.8,
        "shared_coefficient_hierarchy": True,
        "coefficient_hierarchy_parameterization": "shared_arithmetic_moments_log_state",
        "mean_hyperprior_median": 1.0,
        "mean_hyperprior_log_sd": 0.6,
        "sd_hyperprior_median": 0.8,
        "sd_hyperprior_log_sd": 0.4,
    }
    values.update(overrides)

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


def test_profile_rejects_ou_target_with_14_slot_schedule() -> None:
    """The 14-slot schedule must not silently freeze inferred OU parameters."""
    sampler = SamplerConfig(
        iterations=40,
        coefficient_proposal_sd=0.15,
        birth_proposal_sd=0.25,
        schedule_profile=LUNT_OPPORTUNITY_MATCHED_SCHEDULE_PROFILE,
        seed=481,
    )

    with pytest.raises(ValueError, match="14-slot"):
        RunProfile(name="invalid", target=_ou_target(), sampler=sampler)


def test_profile_rejects_independent_target_with_ou_schedule() -> None:
    """The extended schedules should require the matching inferred target state."""
    with pytest.raises(ValueError, match="16-slot"):
        RunProfile(name="invalid", target=_target(), sampler=_ou_sampler())


def test_profile_rejects_hierarchy_mismatch_for_extended_schedules() -> None:
    """Sixteen- and seventeen-slot profiles should match hierarchy presence exactly."""
    with pytest.raises(ValueError, match="16-slot"):
        RunProfile(
            name="invalid",
            target=_ou_target(hierarchy=True),
            sampler=_ou_sampler(hierarchy=False),
        )
    with pytest.raises(ValueError, match="17-slot"):
        RunProfile(
            name="invalid",
            target=_ou_target(hierarchy=False),
            sampler=_ou_sampler(hierarchy=True),
        )
