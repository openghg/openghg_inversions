"""Integration tests for the executable RHIME customisation example."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from examples.rhime_customisation import likelihoods
from examples.rhime_customisation import runner as custom_runner
from examples.rhime_customisation import run_with_likelihood as short_runner
from examples.standard_handoffs import synthetic_inputs


def test_short_and_full_examples_share_likelihood_and_supported_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both forms select the project likelihood and disable built-in mismatch."""
    config_file = tmp_path / "rhime.ini"
    expected = object()
    seen: dict[str, Any] = {}

    def run_standard(**kwargs: Any) -> Any:
        """Capture the preferred one-call example without executing RHIME."""
        seen.update(kwargs)
        return expected

    monkeypatch.setattr(short_runner, "run", run_standard)
    result = short_runner.run_with_likelihood(
        config_file=config_file,
        output_format="none",
    )

    assert result is expected
    assert short_runner.likelihood_builder is likelihoods.likelihood_builder
    assert custom_runner.likelihood_builder is likelihoods.likelihood_builder
    assert seen == {
        "config_file": config_file,
        "likelihood_builder": custom_runner.likelihood_builder,
        "mismatch_model": None,
        "output_format": "none",
    }

    merged, options, *_ = synthetic_inputs(tmp_path)

    class ReachedLikelihood(Exception):
        pass

    def build(model_inputs, model_spec, **kwargs):
        assert model_spec.likelihood is None
        assert kwargs["likelihood_builder"] is likelihoods.likelihood_builder
        raise ReachedLikelihood

    monkeypatch.setattr(custom_runner, "acquire_merged_data", lambda data_config: merged)
    monkeypatch.setattr(custom_runner, "build_standard_model", build)
    monkeypatch.setattr(
        custom_runner, "sample_rhime_model", lambda *args: pytest.fail("forwarding check reached inference"),
    )
    with pytest.raises(ReachedLikelihood):
        custom_runner.run_custom_rhime(
            **options, mismatch_model="fixed_error", output_format="none", output_name="copied-likelihood",
        )


def test_custom_runner_main_forwards_cli_config_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return the run result after forwarding typed CLI and JSON overrides."""
    config_file = tmp_path / "rhime.ini"
    output_path = tmp_path / "outputs"
    expected_result = object()
    seen: dict[str, Any] = {}

    def run_custom_rhime(*, config_file: str | Path | None, **kwargs: Any) -> Any:
        """Capture the command-line handoff without starting a real inversion."""
        seen["config_file"] = config_file
        seen["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(custom_runner, "run_custom_rhime", run_custom_rhime)

    result = custom_runner.main(
        [
            str(config_file),
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2019-02-01",
            "--output-path",
            str(output_path),
            "--output-name",
            "cli-name",
            "--draws",
            "5",
            "--tune",
            "2",
            "--chains",
            "1",
            "--kwargs",
            '{"draws": 99, "species": "ch4"}',
        ]
    )

    assert result is expected_result
    assert seen == {
        "config_file": config_file,
        "kwargs": {
            "species": "ch4",
            "start_date": "2019-01-01",
            "end_date": "2019-02-01",
            "output_path": output_path,
            "output_name": "cli-name",
            "draws": 5,
            "tune": 2,
            "chains": 1,
        },
    }
