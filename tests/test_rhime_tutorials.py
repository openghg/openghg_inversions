"""Exercise the packaged end-to-end RHIME tutorial configurations."""

from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pymc as pm
import pytest
import xarray as xr

from openghg_inversions.cli import main
from openghg_inversions.models.coords import get_coord_registry, restore_inferencedata_coords
from openghg_inversions.postprocessing.inversion_output import InversionOutput
from openghg_inversions.rhime import (
    RhimeSampler,
    run,
)
from openghg_inversions.rhime.ini import read_rhime_ini
from tests.helpers import make_trace


_CONFIG_DIRECTORY = files("openghg_inversions.rhime").joinpath("config")
_STANDARD_CONFIG = _CONFIG_DIRECTORY.joinpath("standard_tutorial.ini")
_MULTISECTOR_CONFIG = _CONFIG_DIRECTORY.joinpath("multisector_tutorial.ini")


def _deterministic_trace(model: pm.Model, variable_names: tuple[str, ...]) -> xr.DataTree:
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
    trace = make_trace(posterior=xr.Dataset(variables, coords=coords))
    registry = get_coord_registry(model)
    assert registry is not None
    return restore_inferencedata_coords(trace, registry)


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
        "flux_store": tac_ch4_data_args["flux_store"],
        "domain": tac_ch4_data_args["domain"],
        "fp_model": tac_ch4_data_args["fp_model"],
        "fp_height": tac_ch4_data_args["fp_height"],
        "fp_species": None,
        "met_model": None,
        "flux_sources": tac_ch4_data_args["flux_sources"],
        "bc_input": "cams",
        "basis_output_path": str(tmp_path / "basis.nc"),
        "output_path": str(tmp_path),
    }


def test_packaged_tutorial_configs_are_complete_and_distinct() -> None:
    """Both installed resources parse and describe their intended workflow."""
    standard = read_rhime_ini(_STANDARD_CONFIG)
    multisector = read_rhime_ini(_MULTISECTOR_CONFIG)

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
    assert standard["mismatch_model"] == "pollution_event"
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
    assert multisector["mismatch_model"] == "pollution_event"
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
        lambda self, model, **kwargs: _deterministic_trace(model, ("x", "mu", "bc")),
    )
    result = run(
        config_file=_STANDARD_CONFIG,
        **_test_store_overrides(tac_ch4_data_args, tmp_path),
    )

    assert result.inv_inputs["H"].dims == ("region", "nmeasure")
    assert isinstance(result, InversionOutput)
    assert result.variable_roles["flux_scale"] == "x"
    output_path = tmp_path / f"standard_tutorial{result.run_metadata['start_date']}_inversion_output.nc"
    assert output_path.is_file()
    reloaded = InversionOutput.load(output_path)
    assert reloaded.provenance == result.provenance
    assert set(reloaded.output_metadata) == {"sampler"}
    np.testing.assert_array_equal(reloaded.trace["posterior"]["x"], result.trace["posterior"]["x"])
    assert reloaded.variable_roles == result.variable_roles
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
        lambda self, model, **kwargs: _deterministic_trace(model, ("x", "mu", "bc")),
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
    result = run(model="multisector", config_file=_MULTISECTOR_CONFIG, **overrides)

    assert isinstance(result, InversionOutput)
    assert [sector["name"] for sector in result.model_metadata["sectors"]] == ["FF", "ocean"]
    assert result.variable_roles["flux_scale"] == "x"
    assert "source" in result.trace_group("posterior")["x"].coords
    assert tuple(result.inv_inputs["H"].source.values) == (
        "total-ukghg-edgar7",
        "total-ukghg-edgar7-shuffled",
    )
    from openghg_inversions.postprocessing.make_outputs import make_sector_flux_outputs

    diagnostics = make_sector_flux_outputs(
        result, stats=["mean"], include_scale_factors=True,
        report_flux_on_inversion_grid=False,
    )
    assert {
        "flux_ff_posterior_mean",
        "flux_ocean_posterior_mean",
        "flux_total_posterior_mean",
    }.issubset(diagnostics)
    stem = f"multisector_tutorial{result.start_date}"
    diagnostics_path = tmp_path / f"{stem}_sector_flux_diagnostics.nc"
    output_path = tmp_path / f"{stem}_inversion_output.nc"
    assert diagnostics_path.is_file()
    xr.testing.assert_allclose(xr.load_dataset(diagnostics_path), diagnostics)
    reloaded = InversionOutput.load(output_path)
    assert reloaded.run_metadata["split_by_sectors"] is True
    assert reloaded.model_metadata["sectors"] == result.model_metadata["sectors"]
    assert reloaded.variable_roles == result.variable_roles
    assert reloaded.provenance == result.provenance
    xr.testing.assert_equal(reloaded.trace_group("posterior"), result.trace_group("posterior"))
    assert set(result.output_metadata) == {"sampler"}


@pytest.mark.parametrize(
    ("subcommand", "config", "target"),
    [
        ("standard", _STANDARD_CONFIG, "openghg_inversions.rhime.standard.run_standard"),
        (
            "multisector",
            _MULTISECTOR_CONFIG,
            "openghg_inversions.rhime.multisector.run_multisector",
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

    main(["run", "--model", subcommand, "--config", str(config), "--output-path", "tutorial-output"])

    assert calls == [(str(config), {"output_path": "tutorial-output"})]
