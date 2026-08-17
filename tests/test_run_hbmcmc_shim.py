from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import openghg_inversions.hbmcmc.run_hbmcmc as run_hbmcmc


def _fixedbasis_config(path: Path) -> None:
    path.write_text(
        """
[INPUT.MEASUREMENTS]
species = "ch4"
sites = ["TAC"]
averaging_period = ["1h"]
start_date = "2019-01-01"
end_date = "2019-01-02"

[INPUT.PRIORS]
domain = "EUROPE"
emissions_name = ["total-ukghg-edgar7"]

[MCMC.TYPE]
mcmc_type = "fixed_basis"

[MCMC.PDF]
xprior = {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
bcprior = {"pdf": "normal", "mu": 1.0, "sigma": 1.0}
sigprior = {"pdf": "uniform", "lower": 0.1, "upper": 10.0}

[MCMC.ITERATIONS]
nit = 7
burn = 1
tune = 2

[MCMC.NCHAIN]
nchain = 3

[MCMC.OPTIONS]
verbose = True
sampler_kwargs = {"target_accept": 0.9}
reparameterise_log_normal = False

[MCMC.OUTPUT]
outputpath = "out"
outputname = "legacy_run"
output_format = "hbmcmc"
""",
        encoding="utf-8",
    )


def test_fixedbasis_params_to_rhime_translates_legacy_names(tmp_path: Path) -> None:
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)

    translated = run_hbmcmc.fixedbasis_params_to_rhime(params)

    assert translated["output_path"] == "out"
    assert translated["output_name"] == "legacy_run"
    assert translated["flux_sources"] == ["total-ukghg-edgar7"]
    assert translated["draws"] == 7
    assert translated["burn"] == 1
    assert translated["tune"] == 2
    assert translated["chains"] == 3
    assert translated["progressbar"] is True
    assert translated["sample_kwargs"] == {"target_accept": 0.9}
    assert translated["output_format"] == "legacy"
    assert translated["output_filename_convention"] == "legacy"
    assert translated["save_inversion_output"] is False
    assert "mcmc_type" not in translated
    assert "nit" not in translated
    assert "nchain" not in translated


def test_fixedbasis_params_to_rhime_translates_reparameterise_log_normal(tmp_path: Path) -> None:
    """Legacy lognormal translation warns visibly and updates both priors."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params["reparameterise_log_normal"] = True
    params["xprior"] = {"pdf": "lognormal", "mean": 1.0, "stdev": 2.0}
    params["bcprior"] = {"pdf": "lognormal", "mean": 1.0, "stdev": 1.0}

    with pytest.warns(FutureWarning, match="reparameterise_log_normal"):
        translated = run_hbmcmc.fixedbasis_params_to_rhime(params)

    assert translated["x_prior"]["reparameterise"] is True
    assert translated["bc_prior"]["reparameterise"] is True
    assert "reparameterise_log_normal" not in translated


def test_fixedbasis_params_to_rhime_translates_calculate_min_error(tmp_path: Path) -> None:
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params["calculate_min_error"] = "percentile"

    with pytest.warns(FutureWarning, match="calculate_min_error"):
        translated = run_hbmcmc.fixedbasis_params_to_rhime(params)

    assert translated["min_error"] == "percentile"
    assert "calculate_min_error" not in translated


def test_fixedbasis_params_to_rhime_rejects_non_fixed_basis_mcmc_type(tmp_path: Path) -> None:
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params["mcmc_type"] = "tdmcmc"

    with pytest.raises(ValueError, match="fixed_basis"):
        run_hbmcmc.fixedbasis_params_to_rhime(params)


def test_fixedbasis_params_to_rhime_preserves_paris_compatibility_flag(tmp_path: Path) -> None:
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params.pop("output_format")
    params["paris_postprocessing"] = True

    translated = run_hbmcmc.fixedbasis_params_to_rhime(params)

    assert translated["output_format"] == "paris"
    assert "paris_postprocessing" not in translated


def test_fixedbasis_params_to_rhime_preserves_latest_paris_kwargs(tmp_path: Path) -> None:
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params.pop("output_format")
    params["paris_postprocessing"] = True
    params["paris_postprocessing_kwargs"] = {"template_version": "latest", "inversion_grid": False}

    translated = run_hbmcmc.fixedbasis_params_to_rhime(params)

    assert translated["output_format"] == "paris"
    assert translated["paris_postprocessing_kwargs"] == {
        "template_version": "latest",
        "inversion_grid": False,
    }


def test_fixedbasis_params_to_rhime_forces_legacy_filename_convention(tmp_path: Path) -> None:
    """run_hbmcmc keeps historical filenames even if a RHIME override is supplied."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params["output_filename_convention"] = "rhime"

    translated = run_hbmcmc.fixedbasis_params_to_rhime(params)

    assert translated["output_filename_convention"] == "legacy"


def test_fixedbasis_params_to_rhime_preserves_explicit_inversion_output_save(tmp_path: Path) -> None:
    """run_hbmcmc only suppresses inv_out sidecars when the old config did not request them."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params["save_inversion_output"] = "explicit_inv_out.nc"

    translated = run_hbmcmc.fixedbasis_params_to_rhime(params)

    assert translated["save_inversion_output"] == "explicit_inv_out.nc"


@pytest.mark.rhime_contract
def test_run_hbmcmc_main_routes_to_run_rhime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route legacy CLI parameters through run_rhime and copy the effective config."""
    config_file = tmp_path / "hbmcmc.ini"
    output_path = tmp_path / "outputs"
    _fixedbasis_config(config_file)
    original_config = config_file.read_text(encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake_run_rhime(**kwargs: Any) -> None:
        """Capture translated keyword arguments without running an inversion."""
        seen["run_rhime_kwargs"] = kwargs

    monkeypatch.setattr(run_hbmcmc, "run_rhime", fake_run_rhime)

    run_hbmcmc.main(
        [
            "2020-01-01",
            "2020-02-01",
            "-c",
            str(config_file),
            "--output-path",
            str(output_path),
            "--kwargs",
            '{"nchain": 2}',
        ]
    )

    copied_config = output_path / "CH4_EUROPE_legacy_run_2020-01-01.ini"
    expected_config = (
        original_config.replace('start_date = "2019-01-01"', "start_date = '2020-01-01'")
        .replace('end_date = "2019-01-02"', "end_date = '2020-02-01'")
        .replace('outputpath = "out"', f"outputpath = '{output_path}'")
        .replace("nchain = 3", "nchain = 2")
    )
    assert copied_config.read_text(encoding="utf-8") == expected_config
    assert seen["run_rhime_kwargs"]["start_date"] == "2020-01-01"
    assert seen["run_rhime_kwargs"]["end_date"] == "2020-02-01"
    assert seen["run_rhime_kwargs"]["output_path"] == str(output_path)
    assert seen["run_rhime_kwargs"]["chains"] == 2
    assert seen["run_rhime_kwargs"]["output_format"] == "legacy"
    assert seen["run_rhime_kwargs"]["output_filename_convention"] == "legacy"


def test_run_hbmcmc_legacy_fixedbasis_parser_is_explicit(tmp_path: Path) -> None:
    """The compatibility route is disabled unless its flag is supplied."""
    parser = run_hbmcmc.build_parser(tmp_path / "hbmcmc.ini")

    assert parser.parse_args([]).legacy_fixedbasis is False
    assert parser.parse_args(["--legacy-fixedbasis"]).legacy_fixedbasis is True


@pytest.mark.parametrize("output_format", [None, "hbmcmc", "hbmcmc_postprocessing"])
def test_run_hbmcmc_legacy_fixedbasis_preserves_raw_params_and_selects_true_legacy_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    output_format: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The explicit opt-in calls fixedbasisMCMC with untranslated legacy values."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    if output_format is None:
        config_file.write_text(
            config_file.read_text(encoding="utf-8").replace('output_format = "hbmcmc"\n', ""),
            encoding="utf-8",
        )
    elif output_format != "hbmcmc":
        config_file.write_text(
            config_file.read_text(encoding="utf-8").replace('hbmcmc"', f'{output_format}"'),
            encoding="utf-8",
        )
    events: list[str] = []
    seen: dict[str, Any] = {}

    def fake_copy_config_file(config_file_arg: str, param: dict[str, Any], **command_line: Any) -> None:
        events.append("copy")
        seen["copy_param"] = param

    def fake_fixedbasis_mcmc(**kwargs: Any) -> None:
        events.append("fixedbasis")
        seen["fixedbasis_kwargs"] = kwargs

    def fail_run_rhime(**kwargs: Any) -> None:
        raise AssertionError("The legacy opt-in must not fall back to RHIME.")

    monkeypatch.setattr(run_hbmcmc.output, "copy_config_file", fake_copy_config_file)
    monkeypatch.setattr(run_hbmcmc, "fixedbasisMCMC", fake_fixedbasis_mcmc)
    monkeypatch.setattr(run_hbmcmc, "run_rhime", fail_run_rhime)

    run_hbmcmc.main(["-c", str(config_file), "--legacy-fixedbasis"])

    fixedbasis_kwargs = seen["fixedbasis_kwargs"]
    assert events == ["copy", "fixedbasis"]
    assert seen["copy_param"].get("output_format") == output_format
    assert fixedbasis_kwargs["nit"] == 7
    assert fixedbasis_kwargs["nchain"] == 3
    assert fixedbasis_kwargs["emissions_name"] == ["total-ukghg-edgar7"]
    assert fixedbasis_kwargs["sampler_kwargs"] == {"target_accept": 0.9}
    assert fixedbasis_kwargs["outputpath"] == "out"
    assert fixedbasis_kwargs["outputname"] == "legacy_run"
    assert fixedbasis_kwargs["output_format"] == run_hbmcmc._LEGACY_FIXEDBASIS_OUTPUT_FORMAT
    assert "draws" not in fixedbasis_kwargs
    assert "chains" not in fixedbasis_kwargs
    notice = capsys.readouterr().out
    assert "WARNING: --legacy-fixedbasis SELECTED" in notice
    assert "No automatic fallback to run_rhime" in notice


def test_run_hbmcmc_legacy_fixedbasis_preserves_explicit_modern_legacy_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit output_format=legacy remains the modern fixedbasis adapter mode."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    config_file.write_text(
        config_file.read_text(encoding="utf-8").replace(
            'output_format = "hbmcmc"', 'output_format = "legacy"'
        ),
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    monkeypatch.setattr(run_hbmcmc.output, "copy_config_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_hbmcmc, "fixedbasisMCMC", lambda **kwargs: seen.update(kwargs))

    run_hbmcmc.main(["-c", str(config_file), "--legacy-fixedbasis"])

    assert seen["output_format"] == "legacy"


def test_run_hbmcmc_legacy_fixedbasis_rejects_rhime_only_options_before_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RHIME-only option names fail clearly instead of being silently translated."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)

    def fail_copy_config_file(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Invalid legacy options must fail before config copying.")

    monkeypatch.setattr(run_hbmcmc.output, "copy_config_file", fail_copy_config_file)

    with pytest.raises(ValueError, match=r"--legacy-fixedbasis.*sample_kwargs"):
        run_hbmcmc.main(
            [
                "-c",
                str(config_file),
                "--legacy-fixedbasis",
                "--kwargs",
                '{"sample_kwargs": {"target_accept": 0.95}}',
            ]
        )


def test_run_hbmcmc_legacy_fixedbasis_checks_country_file_before_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The compatibility route validates country_file before filesystem side effects."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    missing_country_file = tmp_path / "missing_country_file.nc"
    config_file.write_text(
        config_file.read_text(encoding="utf-8")
        + f'\n[INPUT.BASIS_CASE]\ncountry_file = "{missing_country_file}"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_hbmcmc.output,
        "copy_config_file",
        lambda *args, **kwargs: pytest.fail("country_file must be checked before config copying"),
    )
    monkeypatch.setattr(
        run_hbmcmc,
        "fixedbasisMCMC",
        lambda **kwargs: pytest.fail("country_file must be checked before execution"),
    )

    with pytest.raises(FileNotFoundError, match="country_file"):
        run_hbmcmc.main(["-c", str(config_file), "--legacy-fixedbasis"])


def test_run_hbmcmc_main_validates_before_copying_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unsupported legacy options fail before output directories or config copies are created."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)

    def fail_copy_config_file(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Config copy should happen only after shim validation.")

    monkeypatch.setattr(run_hbmcmc.output, "copy_config_file", fail_copy_config_file)

    with pytest.raises(ValueError, match="calculate_min_error"):
        run_hbmcmc.main(
            [
                "-c",
                str(config_file),
                "--kwargs",
                '{"calculate_min_error": true}',
            ]
        )


def test_run_hbmcmc_main_validates_rhime_params_before_copying_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unsupported RHIME params fail before output directories or config copies are created."""
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)

    def fail_copy_config_file(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Config copy should happen only after RHIME validation.")

    monkeypatch.setattr(run_hbmcmc.output, "copy_config_file", fail_copy_config_file)

    with pytest.raises(ValueError, match="Unsupported RHIME parameter"):
        run_hbmcmc.main(
            [
                "-c",
                str(config_file),
                "--kwargs",
                '{"unknown_rhime_option": true}',
            ]
        )


def test_run_hbmcmc_main_checks_country_file_before_copying_or_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A configured missing country file fails before output side effects."""
    config_file = tmp_path / "hbmcmc.ini"
    missing_country_file = tmp_path / "missing_country_file.nc"
    _fixedbasis_config(config_file)
    config_file.write_text(
        config_file.read_text(encoding="utf-8")
        + f'\n[INPUT.BASIS_CASE]\ncountry_file = "{missing_country_file}"\n',
        encoding="utf-8",
    )

    def fail_copy_config_file(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Config copy should happen only after country-file validation.")

    def fail_run_rhime(**kwargs: Any) -> None:
        raise AssertionError("RHIME should run only after country-file validation.")

    monkeypatch.setattr(run_hbmcmc.output, "copy_config_file", fail_copy_config_file)
    monkeypatch.setattr(run_hbmcmc, "run_rhime", fail_run_rhime)

    with pytest.raises(FileNotFoundError, match="country_file"):
        run_hbmcmc.main(["-c", str(config_file)])
