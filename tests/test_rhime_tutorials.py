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
    default_bc_basis_directory: Path,
) -> dict[str, Any]:
    """Map public tutorial inputs to the maintained repository fixture store."""
    return {
        "bc_store": tac_ch4_data_args["bc_store"],
        "obs_store": tac_ch4_data_args["obs_store"],
        "footprint_store": tac_ch4_data_args["footprint_store"],
        "emissions_store": tac_ch4_data_args["emissions_store"],
        "basis_output_path": str(tmp_path),
        "bc_basis_directory": str(default_bc_basis_directory),
        "output_path": str(tmp_path),
        "reload_merged_data": False,
    }


def test_packaged_tutorial_configs_are_complete_and_distinct() -> None:
    """Both installed resources parse and describe their intended workflow."""
    standard = params_from_config(_STANDARD_CONFIG)
    multisector = params_from_config(_MULTISECTOR_CONFIG)

    assert standard["flux_sources"] == ["total-ukghg-edgar7"]
    assert standard["draws"] == 50
    assert standard["output_format"] == "inv_out"
    assert standard["use_bc"] is True
    assert multisector["flux_sources"] == ["anthropogenic-ch4", "wetlands-ch4"]
    assert multisector["sector_sources"] == {
        "anthropogenic": "anthropogenic-ch4",
        "wetlands": "wetlands-ch4",
    }
    assert set(multisector["sector_priors"]) == {"anthropogenic", "wetlands"}
    assert multisector["use_bc"] is True


@pytest.mark.rhime_contract
def test_standard_tutorial_runs_to_persisted_output(
    monkeypatch: pytest.MonkeyPatch,
    tac_ch4_data_args: dict[str, Any],
    tmp_path: Path,
    default_bc_basis_directory: Path,
) -> None:
    """Run the documented standard config through real preparation and model build."""
    monkeypatch.setattr(
        RhimeSampler,
        "sample",
        lambda self, model, **kwargs: _deterministic_trace(model, ("x", "mu")),
    )
    result = run_rhime(
        config_file=_STANDARD_CONFIG,
        **_test_store_overrides(tac_ch4_data_args, tmp_path, default_bc_basis_directory),
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
    default_bc_basis_directory: Path,
) -> None:
    """Run the documented multisector config with labelled fixture sectors."""
    monkeypatch.setattr(
        RhimeSampler,
        "sample",
        lambda self, model, **kwargs: _deterministic_trace(model, ("x_ff", "x_ocean")),
    )
    overrides = _test_store_overrides(tac_ch4_data_args, tmp_path, default_bc_basis_directory)
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
