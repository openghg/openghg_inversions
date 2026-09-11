"""Exercise the packaged end-to-end RHIME tutorial configurations."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.cli import main
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.rhime import (
    RhimeSampler,
    params_from_config,
    run_rhime,
    run_rhime_multisector,
)


_CONFIG_DIRECTORY = files("openghg_inversions.rhime").joinpath("config")
_STANDARD_CONFIG = _CONFIG_DIRECTORY.joinpath("standard_tutorial.ini")
_MULTISECTOR_CONFIG = _CONFIG_DIRECTORY.joinpath("multisector_tutorial.ini")


def _deterministic_trace(model: pm.Model, variable_names: tuple[str, ...]) -> az.InferenceData:
    """Return one labelled posterior draw for tutorial pipeline tests."""
    coords: dict[str, np.ndarray] = {"chain": np.arange(1), "draw": np.arange(1)}
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in variable_names:
        model_dims = model.named_vars_to_dims[name]
        for dim in model_dims:
            coord = model.coords[dim]
            assert coord is not None
            coords[dim] = np.asarray(coord)
        dims = ("chain", "draw", *model_dims)
        variables[name] = (dims, np.ones(tuple(len(coords[dim]) for dim in dims)))
    return az.InferenceData(posterior=xr.Dataset(variables, coords=coords))


def _test_store_overrides(
    tac_ch4_data_args: dict[str, Any],
    tmp_path: Path,
) -> dict[str, Any]:
    """Map public tutorial inputs to the maintained repository fixture store."""
    return {
        "species": tac_ch4_data_args["species"],
        "sites": tac_ch4_data_args["sites"],
        "averaging_period": tac_ch4_data_args["averaging_period"],
        "start_date": tac_ch4_data_args["start_date"],
        "end_date": tac_ch4_data_args["end_date"],
        "inlet": tac_ch4_data_args["inlet"],
        "instrument": tac_ch4_data_args["instrument"],
        "bc_store": tac_ch4_data_args["bc_store"],
        "obs_store": tac_ch4_data_args["obs_store"],
        "footprint_store": tac_ch4_data_args["footprint_store"],
        "emissions_store": tac_ch4_data_args["emissions_store"],
        "domain": tac_ch4_data_args["domain"],
        "fp_model": tac_ch4_data_args["fp_model"],
        "fp_height": tac_ch4_data_args["fp_height"],
        "fp_species": None,
        "met_model": None,
        "flux_sources": tac_ch4_data_args["emissions_name"],
        "bc_input": "cams",
        "basis_output_path": str(tmp_path),
        "output_path": str(tmp_path),
        "reload_merged_data": False,
    }


def test_packaged_tutorial_configs_are_complete_and_distinct() -> None:
    """Both installed resources parse and describe their intended workflow."""
    standard = params_from_config(_STANDARD_CONFIG)
    multisector = params_from_config(_MULTISECTOR_CONFIG)

    assert standard["sites"] == ["MHD", "TAC"]
    assert standard["averaging_period"] == ["4h", "4h"]
    assert standard["start_date"] == "2020-01-01"
    assert standard["end_date"] == "2020-01-08"
    assert standard["inlet"] == ["24m", "185m"]
    assert standard["fp_height"] == ["10m", "185m"]
    assert standard["flux_sources"] == ["edgar-v80-anthropogenic"]
    assert standard["bc_input"] == "camsv22r2_daily"
    assert "bc_basis_directory" not in standard
    assert standard["obs_store"] == "inversions_tutorial_data"
    assert standard["draws"] == 50
    assert standard["output_format"] == "inv_out"
    assert standard["use_bc"] is True
    assert multisector["sites"] == ["MHD", "TAC"]
    assert multisector["flux_sources"] == [
        "edgar-v80-anthropogenic",
        "wetcharts-v131-wetlands",
    ]
    assert multisector["sector_sources"] == {
        "anthropogenic": "edgar-v80-anthropogenic",
        "wetlands": "wetcharts-v131-wetlands",
    }
    assert set(multisector["sector_priors"]) == {"anthropogenic", "wetlands"}
    assert multisector["use_bc"] is True


@pytest.mark.rhime_contract
def test_standard_tutorial_runs_to_persisted_output(
    monkeypatch: pytest.MonkeyPatch,
    tac_ch4_data_args: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Run the documented standard config through real preparation and model build."""
    monkeypatch.setattr(
        RhimeSampler,
        "sample",
        lambda self, model, **kwargs: _deterministic_trace(model, ("x", "mu")),
    )
    result = run_rhime(
        config_file=_STANDARD_CONFIG,
        **_test_store_overrides(tac_ch4_data_args, tmp_path),
    )

    assert result.inv_inputs["H"].dims == ("region", "nmeasure")
    assert result.model_build_result is not None
    assert result.model_build_result.variable_roles["flux_scale"] == "x"
    output_path = Path(result.output_metadata["inversion_output_path"])
    assert output_path.is_file()
    reloaded = InversionOutput.load(output_path)
    assert reloaded.provenance["contract"] == "modern_rhime_inversion_output"
    assert reloaded.run_metadata["split_by_sectors"] is False


@pytest.mark.rhime_contract
def test_multisector_tutorial_runs_to_sector_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tac_ch4_data_args: dict[str, Any],
    tmp_path: Path,
) -> None:
    """Run the documented multisector config with labelled fixture sectors."""
    monkeypatch.setattr(
        RhimeSampler,
        "sample",
        lambda self, model, **kwargs: _deterministic_trace(model, ("x_ff", "x_ocean")),
    )
    overrides = _test_store_overrides(tac_ch4_data_args, tmp_path)
    overrides.update(
        {
            "flux_sources": ["total-ukghg-edgar7", "total-ukghg-edgar7-shuffled"],
            "sector_sources": {
                "FF": "total-ukghg-edgar7",
                "ocean": "total-ukghg-edgar7-shuffled",
            },
            "sector_priors": {
                "FF": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
                "ocean": {"pdf": "normal", "mu": 1.0, "sigma": 1.0},
            },
        }
    )
    result = run_rhime_multisector(config_file=_MULTISECTOR_CONFIG, **overrides)

    assert [sector.name for sector in result.model_spec.sectors] == ["FF", "ocean"]
    assert tuple(result.inv_inputs["H"].source.values) == (
        "total-ukghg-edgar7",
        "total-ukghg-edgar7-shuffled",
    )
    diagnostics = result.outputs["sector_flux_diagnostics"]
    assert {
        "flux_ff_posterior_mean",
        "flux_ocean_posterior_mean",
        "flux_total_posterior_mean",
    }.issubset(diagnostics)
    assert Path(result.output_metadata["sector_flux_diagnostics_path"]).is_file()
    assert Path(result.output_metadata["inversion_output_path"]).is_file()


@pytest.mark.parametrize(
    ("subcommand", "config", "target"),
    [
        ("run-rhime", _STANDARD_CONFIG, "openghg_inversions.rhime.run_rhime"),
        (
            "run-rhime-multisector",
            _MULTISECTOR_CONFIG,
            "openghg_inversions.rhime.run_rhime_multisector",
        ),
    ],
)
def test_tutorial_cli_routes_use_packaged_configs(
    monkeypatch: pytest.MonkeyPatch,
    subcommand: str,
    config: Any,
    target: str,
) -> None:
    """Exercise the exact documented CLI subcommands without sampling twice."""
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        target,
        lambda *, config_file, **kwargs: calls.append((config_file, kwargs)),
    )

    main([subcommand, "--config", str(config), "--output-path", "tutorial-output"])

    assert calls == [(str(config), {"output_path": "tutorial-output"})]
