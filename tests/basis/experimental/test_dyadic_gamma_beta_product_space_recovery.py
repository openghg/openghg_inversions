"""Tests for the matched Gamma--Beta product-space recovery benchmark."""

from __future__ import annotations

from importlib import util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are creating a TensorVariable with float64 dtype:UserWarning"
)


@pytest.fixture(scope="module")
def example_module() -> ModuleType:
    """Load the executable recovery example as an isolated module."""
    repository_root = Path(__file__).resolve().parents[3]
    script = repository_root / "examples" / "basis" / "dyadic_gamma_beta_product_space_recovery.py"
    module_name = "_test_dyadic_gamma_beta_product_space_recovery"
    specification = util.spec_from_file_location(module_name, script)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load example module from {script}.")
    module = util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def test_case_closes_truth_and_uses_matched_priors(example_module: ModuleType) -> None:
    """Synthetic observations and fixed/latent support should match declarations."""
    case: Any = example_module.build_case(seed=1701)
    unsplit = case.layout.initial_split_mask(1)
    split = case.layout.initial_split_mask(2)

    np.testing.assert_array_equal(case.truth, [0.5, 1.5])
    np.testing.assert_array_equal(
        case.holdout_noiseless,
        np.concatenate((np.full(10, 0.5), np.full(10, 1.5))),
    )
    assert np.isfinite(case.latent_prior(unsplit))
    assert np.isfinite(case.latent_prior(split))
    assert np.isfinite(case.fixed_unsplit_prior(unsplit))
    assert case.fixed_unsplit_prior(split) == -np.inf
    assert case.fixed_split_prior(unsplit) == -np.inf
    assert np.isfinite(case.fixed_split_prior(split))
    assert case.train_target.coordinate_layout.forest is case.forest
    assert case.holdout_target.coordinate_layout.forest is case.forest


def test_cli_contract_prints_serializable_comparison(
    example_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The cheap command path should print the complete benchmark mapping."""
    fit = example_module.GammaBetaFitSummary(
        name="test",
        draws=10,
        tune=5,
        mean_k=1.8,
        split_probability=0.8,
        posterior_mean_field=(0.6, 1.4),
        field_rmse=0.1,
        holdout_prediction_rmse=0.1,
        holdout_log_predictive_density=12.0,
        partition_acceptance_rate=0.2,
        divergence_count=0,
    )
    benchmark = example_module.GammaBetaRecoveryBenchmark(
        latent=fit,
        fixed_true_split=fit,
        fixed_underfit_unsplit=fit,
        latent_matches_fixed_true=True,
        latent_beats_fixed_underfit=True,
    )
    monkeypatch.setattr(example_module, "build_case", lambda seed: object())
    monkeypatch.setattr(example_module, "run_benchmark", lambda *args, **kwargs: benchmark)

    result = example_module.main(["--draws", "10", "--tune", "5", "--indent", "0"])

    assert result == 0
    output = capsys.readouterr().out
    assert '"latent_matches_fixed_true": true' in output
    assert '"latent_beats_fixed_underfit": true' in output


@pytest.mark.slow
def test_latent_fit_matches_oracle_and_beats_underfit(example_module: ModuleType) -> None:
    """Seeded latent K/P should match true fixed P and beat fixed K=1."""
    benchmark: Any = example_module.run_benchmark(
        example_module.build_case(seed=1701),
        draws=500,
        tune=500,
        seed=20260719,
        target_accept=0.9,
    )

    assert benchmark.latent.split_probability > 0.9
    assert benchmark.latent.divergence_count == 0
    assert benchmark.fixed_true_split.divergence_count == 0
    assert benchmark.latent_matches_fixed_true
    assert benchmark.latent_beats_fixed_underfit
    assert benchmark.latent.holdout_prediction_rmse < 0.05
    assert benchmark.latent.field_rmse < 0.05
