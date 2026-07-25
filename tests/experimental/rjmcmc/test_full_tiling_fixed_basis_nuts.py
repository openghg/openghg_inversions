"""Tests for the frozen-input fixed-basis NumPyro NUTS driver."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import arviz as az
import numpy as np
import pytest
import xarray as xr

_DRIVER_PATH = Path(__file__).parents[3] / "examples" / "rjmcmc" / "full_tiling_fixed_basis_nuts.py"


@pytest.fixture(scope="module")
def nuts_driver() -> ModuleType:
    """Load the example driver without invoking its command-line entry point."""
    specification = importlib.util.spec_from_file_location(
        "full_tiling_fixed_basis_nuts",
        _DRIVER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("Could not load the fixed-basis NUTS example.")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_frozen_input(path: Path) -> None:
    """Write a tiny exact-closure native dataset with six labelled outers."""
    sensitivity = np.arange(1.0, 13.0).reshape(3, 2, 2)
    outer = np.arange(18.0).reshape(3, 6) / 8.0
    boundary = np.array([4.0, 5.0, 6.0])
    dataset = xr.Dataset(
        {
            "fp_x_flux": (
                ("lon", "nmeasure", "lat"),
                sensitivity.transpose(2, 0, 1),
            ),
            "mf": (
                "nmeasure",
                boundary + sensitivity.sum(axis=(1, 2)) + outer.sum(axis=1),
            ),
            "mf_error": ("nmeasure", np.ones(3)),
            "nominal_weight": (
                ("lon", "lat"),
                np.full((2, 2), 0.25).T,
            ),
            "outer_design": (("outer_region", "nmeasure"), outer.T),
            "YaprioriBC": ("nmeasure", boundary),
        },
        coords={
            "nmeasure": ["obs-a", "obs-b", "obs-c"],
            "lat": [50.0, 51.0],
            "lon": [-2.0, -1.0],
            "outer_region": [f"region-{index}" for index in range(6)],
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")


def _arguments(
    driver: ModuleType,
    input_path: Path,
    output_path: Path,
    *,
    initialization: str = "prior-mean",
    initialization_seed: int | None = None,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """Return the complete explicit tiny-run CLI contract."""
    digest = driver._sha256_file(input_path)
    values = [
        "--input",
        str(input_path),
        "--output-directory",
        str(output_path),
        "--k",
        "2",
        "--draws",
        "2",
        "--tune",
        "3",
        "--seed",
        "812",
        "--chain-id",
        "tiny-chain",
        "--continuous-initialization",
        initialization,
        "--concentration",
        "3",
        "--root-variance",
        "0.25",
        "--fixed-prior-mean",
        "1",
        "--fixed-prior-sd",
        "1",
        "--target-accept",
        "0.9",
        "--max-tree-depth",
        "8",
        "--no-dense-mass",
        "--input-id",
        "tiny-frozen-native-v1",
        "--code-revision",
        "test-revision",
        "--expected-input-sha256",
        digest,
        "--nominal-weight-policy",
        "positive-native-mass-v1",
    ]
    if initialization_seed is not None:
        values.extend(("--initialization-seed", str(initialization_seed)))
    values.extend(extra)
    return values


def _run_float64_driver(arguments: list[str], *, cache_root: Path) -> dict[str, Any]:
    """Run the real driver in a fresh, explicitly float64 CPU process."""
    environment = os.environ.copy()
    retained_pytensor_flags = []
    for item in environment.get("PYTENSOR_FLAGS", "").split(","):
        stripped = item.strip()
        name = stripped.split("=", 1)[0].strip()
        if stripped and name not in {"floatX", "warn_float64", "base_compiledir"}:
            retained_pytensor_flags.append(stripped)
    environment["PYTENSOR_FLAGS"] = ",".join(
        (
            "floatX=float64",
            "warn_float64=ignore",
            f"base_compiledir={cache_root / 'pytensor'}",
            *retained_pytensor_flags,
        )
    )
    environment["JAX_ENABLE_X64"] = "1"
    environment["JAX_PLATFORMS"] = "cpu"
    environment["MPLCONFIGDIR"] = str(cache_root / "matplotlib")
    environment["XDG_CACHE_HOME"] = str(cache_root / "cache")
    completed = subprocess.run(
        [sys.executable, str(_DRIVER_PATH), *arguments],
        cwd=_DRIVER_PATH.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"isolated fixed-basis NUTS driver failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout)


def _mock_model_preflight(
    driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep pure publication tests independent of process-global PyTensor."""

    def fake_preflight(
        data: Any,
        model: Any,
        *,
        initvals: Any,
        expected_log_target: float,
    ) -> dict[str, object]:
        """Return exact parity metadata without importing the sampler stack."""
        del data, model, initvals
        return {
            "constrained_log_target": expected_log_target,
            "expected_log_target": expected_log_target,
            "log_target_difference": 0.0,
            "log_target_absolute_tolerance": 5.0e-10 * max(1.0, abs(expected_log_target)),
            "model_value_variables_float64": True,
        }

    monkeypatch.setattr(
        driver,
        "_backend_metadata",
        lambda: {
            "pytensor_floatX": "float64",
            "jax_enable_x64": True,
            "jax_backend": "cpu",
            "jax_default_backend": "cpu",
            "jax_devices": [{"platform": "cpu", "device_kind": "test"}],
        },
    )
    monkeypatch.setattr(driver, "build_fixed_basis_pymc_model", lambda data: object())
    monkeypatch.setattr(driver, "preflight_fixed_basis_nuts", fake_preflight)


def _valid_inference_data(data: Any, draws: int) -> az.InferenceData:
    """Return a complete internally consistent fixed-basis trace."""
    root_total = np.ones((1, draws), dtype=np.float64)
    leaf_share = np.broadcast_to(
        np.asarray(data.nominal_leaf_share, dtype=np.float64),
        (1, draws, data.k),
    ).copy()
    leaf_mass = root_total[..., np.newaxis] * leaf_share
    leaf_scaling = leaf_mass / data.nominal_leaf_share
    fixed_coefficient = np.ones(
        (1, draws, data.n_fixed_coefficients),
        dtype=np.float64,
    )
    mean_observation = (
        data.fixed_offset + leaf_mass @ data.dynamic_design.T + fixed_coefficient @ data.fixed_design.T
    )
    residual = (data.observations - mean_observation) / data.observation_sd
    pointwise = -0.5 * residual * residual - np.log(data.observation_sd) - 0.5 * np.log(2.0 * np.pi)
    return az.from_dict(
        posterior={
            "root_total": root_total,
            "leaf_share": leaf_share,
            "leaf_mass": leaf_mass,
            "leaf_scaling": leaf_scaling,
            "fixed_coefficient": fixed_coefficient,
            "mean_observation": mean_observation,
        },
        sample_stats={
            "diverging": np.zeros((1, draws), dtype=bool),
            "n_steps": np.ones((1, draws), dtype=np.int64),
            "tree_depth": np.ones((1, draws), dtype=np.int64),
            "acceptance_rate": np.full((1, draws), 0.9, dtype=np.float64),
            "energy": np.ones((1, draws), dtype=np.float64),
            "lp": -np.ones((1, draws), dtype=np.float64),
            "step_size": np.full((1, draws), 0.1, dtype=np.float64),
        },
        observed_data={"observed": np.asarray(data.observations, dtype=np.float64)},
        log_likelihood={"observed": pointwise},
        coords={
            "leaf": np.asarray(data.leaf_labels),
            "fixed": [f"fixed_{index}" for index in range(data.n_fixed_coefficients)],
            "observation": np.arange(data.observations.size),
        },
        dims={
            "leaf_share": ["leaf"],
            "leaf_mass": ["leaf"],
            "leaf_scaling": ["leaf"],
            "fixed_coefficient": ["fixed"],
            "mean_observation": ["observation"],
            "observed": ["observation"],
        },
    )


@pytest.mark.parametrize(
    ("initialization", "initialization_seed"),
    [("prior-mean", None), ("prior-draw", 913)],
)
def test_dry_run_closes_and_matches_target_without_writing(
    tmp_path: Path,
    nuts_driver: ModuleType,
    initialization: str,
    initialization_seed: int | None,
) -> None:
    """Both audited starts pass closure and exact density parity without output."""
    input_path = tmp_path / f"{initialization}.nc"
    output_path = tmp_path / f"{initialization}-output"
    _write_frozen_input(input_path)
    summary = _run_float64_driver(
        [
            *_arguments(
                nuts_driver,
                input_path,
                output_path,
                initialization=initialization,
                initialization_seed=initialization_seed,
            ),
            "--dry-run",
        ],
        cache_root=tmp_path / f"{initialization}-runtime",
    )

    assert summary["status"] == "dry_run"
    assert summary["closure"] == {
        "mass_coordinate_max_abs_error": 0.0,
        "prior_mean_total_max_abs_error": 0.0,
    }
    assert summary["target"]["absolute_difference"] <= summary["target"]["absolute_tolerance"]
    initialization_record = summary["manifest"]["sampler"]["initialization"]
    assert initialization_record["continuous_profile"] == initialization
    assert initialization_record["continuous_initialization_seed"] == initialization_seed
    if initialization == "prior-draw":
        assert initialization_record["root_total"] != pytest.approx(1.0)
    assert not output_path.exists()


def test_driver_enforces_hash_profile_and_initialization_gates(
    tmp_path: Path,
    nuts_driver: ModuleType,
) -> None:
    """Malformed hash, PARIS profile, and initialization settings fail closed."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "output"
    _write_frozen_input(input_path)
    common = _arguments(nuts_driver, input_path, output_path)

    wrong_hash = list(common)
    wrong_hash[wrong_hash.index("--expected-input-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        nuts_driver.run(nuts_driver.build_parser().parse_args(wrong_hash))

    with pytest.raises(ValueError, match="expected 1382 observations"):
        nuts_driver.run(
            nuts_driver.build_parser().parse_args(
                [
                    *common,
                    "--require-paris-profile",
                    "--expected-outer-labels",
                    ",".join(f"region-{index}" for index in range(6)),
                ]
            )
        )

    with pytest.raises(ValueError, match="requires --initialization-seed"):
        nuts_driver.run(
            nuts_driver.build_parser().parse_args(
                _arguments(
                    nuts_driver,
                    input_path,
                    output_path,
                    initialization="prior-draw",
                )
            )
        )

    with pytest.raises(ValueError, match="likelihood-power must be exactly 1"):
        nuts_driver.run(nuts_driver.build_parser().parse_args([*common, "--likelihood-power", "0.5"]))


def test_completed_bundle_is_create_only_hash_certified_and_has_no_checkpoint(
    tmp_path: Path,
    nuts_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mocked tiny chain publishes a complete hash-certified bundle last."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "output"
    _write_frozen_input(input_path)
    writes: list[str] = []
    original_write_text = nuts_driver._atomic_write_text

    def recording_write_text(path: Path, text: str) -> None:
        """Record durable JSON publication order while retaining real writes."""
        writes.append(path.name)
        original_write_text(path, text)

    def fake_sampler(
        model: Any,
        data: Any,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Return a genuine serializable and scientifically consistent trace."""
        del model
        return _valid_inference_data(data, int(kwargs["draws"]))

    monkeypatch.setattr(nuts_driver, "_atomic_write_text", recording_write_text)
    monkeypatch.setattr(nuts_driver, "sample_fixed_basis_nuts", fake_sampler)
    _mock_model_preflight(nuts_driver, monkeypatch)
    arguments = nuts_driver.build_parser().parse_args(_arguments(nuts_driver, input_path, output_path))

    summary = nuts_driver.run(arguments)

    assert summary["status"] == "complete"
    assert writes[-1] == nuts_driver.COMPLETION_FILENAME
    assert {path.name for path in output_path.iterdir()} == {
        nuts_driver.TRACE_FILENAME,
        nuts_driver.MANIFEST_FILENAME,
        nuts_driver.SUMMARY_FILENAME,
        nuts_driver.COMPLETION_FILENAME,
    }
    assert not any("checkpoint" in path.name.lower() for path in output_path.iterdir())
    completion = json.loads((output_path / nuts_driver.COMPLETION_FILENAME).read_text(encoding="utf-8"))
    assert completion["status"] == "complete"
    assert completion["checkpoint_or_restart_supported"] is False
    for name, expected_digest in completion["sha256"].items():
        assert nuts_driver._sha256_file(output_path / name) == expected_digest
    assert summary["trace_validation"]["in_memory"]["groups"] == [
        "posterior",
        "sample_stats",
        "observed_data",
        "log_likelihood",
    ]
    assert summary["trace_validation"]["reopened_netcdf"] == summary["trace_validation"]["in_memory"]
    written_summary = json.loads((output_path / nuts_driver.SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert written_summary["trace_validation"] == summary["trace_validation"]
    reopened = az.from_netcdf(output_path / nuts_driver.TRACE_FILENAME)
    try:
        posterior: Any = getattr(reopened, "posterior")
        log_likelihood: Any = getattr(reopened, "log_likelihood")
        assert posterior.leaf_share.dtype == np.dtype(np.float64)
        assert log_likelihood.observed.dtype == np.dtype(np.float64)
    finally:
        getattr(reopened, "close")()

    with pytest.raises(FileExistsError, match="already exists"):
        nuts_driver.run(arguments)


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing-variable", "missing required variable 'leaf_mass'"),
        ("nan", "posterior.mean_observation contains non-finite"),
        ("wrong-coordinate", "incorrect 'leaf' coordinate"),
        ("wrong-simplex", "does not lie on the simplex"),
    ],
)
def test_invalid_sampler_trace_fails_before_publication(
    tmp_path: Path,
    nuts_driver: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    message: str,
) -> None:
    """Missing, non-finite, mislabelled, and off-simplex traces fail closed."""
    input_path = tmp_path / "frozen.nc"
    output_path = tmp_path / "output"
    _write_frozen_input(input_path)

    def invalid_sampler(
        model: Any,
        data: Any,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Return one deliberately corrupted otherwise-valid trace."""
        del model
        inference_data = _valid_inference_data(data, int(kwargs["draws"]))
        posterior: Any = getattr(inference_data, "posterior")
        if corruption == "missing-variable":
            setattr(inference_data, "posterior", posterior.drop_vars("leaf_mass"))
        elif corruption == "nan":
            posterior["mean_observation"].values[0, 0, 0] = np.nan
        elif corruption == "wrong-coordinate":
            setattr(
                inference_data,
                "posterior",
                posterior.assign_coords(leaf=[f"wrong-{index}" for index in range(data.k)]),
            )
        elif corruption == "wrong-simplex":
            posterior["leaf_share"].values[0, 0, :] *= 0.75
        else:
            raise AssertionError(f"Unhandled test corruption {corruption!r}.")
        return inference_data

    monkeypatch.setattr(nuts_driver, "sample_fixed_basis_nuts", invalid_sampler)
    _mock_model_preflight(nuts_driver, monkeypatch)
    arguments = nuts_driver.build_parser().parse_args(_arguments(nuts_driver, input_path, output_path))

    with pytest.raises(RuntimeError, match=message):
        nuts_driver.run(arguments)

    assert not output_path.exists()
