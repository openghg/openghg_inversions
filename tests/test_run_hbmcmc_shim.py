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
    assert "mcmc_type" not in translated
    assert "nit" not in translated
    assert "nchain" not in translated


def test_fixedbasis_params_to_rhime_rejects_enabled_legacy_options(tmp_path: Path) -> None:
    config_file = tmp_path / "hbmcmc.ini"
    _fixedbasis_config(config_file)
    params = run_hbmcmc.hbmcmc_extract_param(str(config_file), print_param=False)
    params["reparameterise_log_normal"] = True

    with pytest.raises(ValueError, match="reparameterise_log_normal"):
        run_hbmcmc.fixedbasis_params_to_rhime(params)


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


def test_run_hbmcmc_main_routes_to_run_rhime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_file = tmp_path / "hbmcmc.ini"
    output_path = tmp_path / "outputs"
    _fixedbasis_config(config_file)
    seen: dict[str, Any] = {}

    def fake_copy_config_file(config_file_arg: str, param: dict[str, Any], **command_line: Any) -> None:
        seen["copy_config_file"] = config_file_arg
        seen["copy_param"] = param
        seen["copy_command_line"] = command_line

    def fake_run_rhime(**kwargs: Any) -> None:
        seen["run_rhime_kwargs"] = kwargs

    monkeypatch.setattr(run_hbmcmc.output, "copy_config_file", fake_copy_config_file)
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

    assert seen["copy_config_file"] == str(config_file)
    assert seen["copy_command_line"]["start_date"] == "2020-01-01"
    assert seen["copy_command_line"]["end_date"] == "2020-02-01"
    assert seen["copy_command_line"]["outputpath"] == str(output_path)
    assert seen["run_rhime_kwargs"]["start_date"] == "2020-01-01"
    assert seen["run_rhime_kwargs"]["end_date"] == "2020-02-01"
    assert seen["run_rhime_kwargs"]["output_path"] == str(output_path)
    assert seen["run_rhime_kwargs"]["chains"] == 2
    assert seen["run_rhime_kwargs"]["output_format"] == "legacy"
    assert seen["run_rhime_kwargs"]["output_filename_convention"] == "legacy"
