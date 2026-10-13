"""Runner resource preflight precedes acquisition across public entry points."""

from importlib import import_module
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.rhime.params import RhimeConfig
from tests.test_bc_basis_resources import _basis
from tests.test_staged_workflow import _params


@pytest.mark.parametrize("route", ["standard", "multisector", "staged", "nested", "example"])
@pytest.mark.parametrize("selection", ["files", "missing", "disabled", "nesw"])
def test_bc_resource_loading_precedes_acquisition(tmp_path, monkeypatch, route, selection):
    module = import_module(
        "examples.rhime_customisation.runner" if route == "example"
        else f"openghg_inversions.rhime.{'_standard_stages' if route == 'staged' else route}"
    )
    monkeypatch.chdir(tmp_path)
    options = _params(basis_algorithm="quadtree", output_format="none", use_bc=selection != "disabled")
    if route == "multisector":
        options["flux_sources"] = ["first", "second"]
    paths = [tmp_path / "early.nc", tmp_path / "late.nc"]
    if selection == "files":
        times = pd.date_range("2020-01-01", periods=2)
        for path, time, weight in zip(paths, times, [2, 3], strict=True):
            _basis([time], [weight]).to_netcdf(path)
    if selection != "nesw":
        options["bc_basis_paths"] = paths
    loaded = []
    real_load = module.load_bc_basis

    def load(selected):
        value = real_load(selected)
        loaded.append(value)
        return value

    class Acquired(Exception):
        pass

    def acquire(**kwargs):
        if selection == "files":
            assert len(loaded) == 1
            xr.testing.assert_equal(loaded[0], _basis(pd.date_range("2020-01-01", periods=2), [2, 3]))
        else:
            assert not loaded
        raise Acquired

    monkeypatch.setattr(module, "load_bc_basis", load)
    monkeypatch.setattr(module.RhimeMergedData, "from_options", acquire)

    def run():
        if route in ("staged", "nested"):
            config = RhimeConfig.from_params(options, multisector=False)
            if route == "staged":
                module.prepare_rhime_stage(setup=config, model="standard", output_dir="stage")
            else:
                module.prepare_nested_rhime_inputs(config, inner_domain="6km")
        elif route == "multisector":
            module.run_rhime_multisector(**options)
        elif route in ("example", "custom_basis"):
            module.run_custom_rhime(**options)
        else:
            module.run_rhime(**options)

    with pytest.raises(FileNotFoundError if selection == "missing" else Acquired):
        run()
    assert not Path("absent").exists()
    if selection == "missing":
        assert not Path("stage").exists()


def test_packaged_preparation_defaults_are_independent_of_reporting(tmp_path):
    from openghg_inversions.basis.support import load_basis_support_map, packaged_basis_support_path
    from openghg_inversions.rhime.preparation import resolve_basis_inputs

    config = RhimeConfig.from_params(_params(basis_algorithm="weighted", country_file=tmp_path / "report.nc"),
                                     multisector=False)
    selected = resolve_basis_inputs(config)
    expected = load_basis_support_map(packaged_basis_support_path("landsea", "EUROPE"), "country")
    xr.testing.assert_equal(selected["landsea_classes"], expected)
    assert config.output.country_file == tmp_path / "report.nc"
    assert config.landsea_path is None
    explicit = tmp_path / "classes.nc"
    xr.Dataset({"country": (("lat", "lon"), [[7]])}, coords={"lat": [50.0], "lon": [0.0]}).to_netcdf(explicit)
    from dataclasses import replace

    overridden = resolve_basis_inputs(replace(config, landsea_path=explicit))
    assert overridden["landsea_classes"].item() == 7
    with pytest.raises(FileNotFoundError):
        resolve_basis_inputs(replace(config, landsea_path=tmp_path / "missing.nc"))
    with pytest.raises(FileNotFoundError):
        resolve_basis_inputs(replace(config, domain="NO-SUCH-PACKAGED-DOMAIN"))
    assert resolve_basis_inputs(replace(config, domain="NO-SUCH-PACKAGED-DOMAIN", basis_algorithm="quadtree")) == {
        "landsea_classes": None, "outer_regions": None,
    }
    outer_config = replace(config, domain="EUHROB", basis_algorithm="quadtree", fix_basis_outer_regions=True)
    selected_outer = resolve_basis_inputs(outer_config)["outer_regions"]
    xr.testing.assert_equal(selected_outer, load_basis_support_map(
        packaged_basis_support_path("outer_regions", "EUHROB"), "region"))
    explicit_outer = tmp_path / "outer.nc"
    xr.Dataset({"region": (("lat", "lon"), [[3]])}, coords={"lat": [50.0], "lon": [0.0]}).to_netcdf(explicit_outer)
    assert resolve_basis_inputs(replace(outer_config, outer_regions_path=explicit_outer))["outer_regions"].item() == 3
    with pytest.raises(FileNotFoundError):
        resolve_basis_inputs(replace(outer_config, outer_regions_path=tmp_path / "missing-outer.nc"))
    with pytest.raises(FileNotFoundError):
        resolve_basis_inputs(replace(outer_config, domain="NO-SUCH-PACKAGED-DOMAIN"))


@pytest.mark.parametrize("route", ["standard", "multisector", "staged", "nested", "example", "custom_basis"])
def test_modern_routes_reject_historical_resource_selectors_before_acquisition(monkeypatch, route):
    module = import_module(
        "examples.rhime_customisation.custom_basis_runner" if route == "custom_basis" else
        "examples.rhime_customisation.runner" if route == "example"
        else f"openghg_inversions.rhime.{'stages' if route == 'staged' else route}"
    )
    options = _params(output_format="none", bc_basis_case="NESW", bc_basis_directory=None)

    def unexpected_acquisition(**kwargs):
        pytest.fail("Historical options reached acquisition")

    monkeypatch.setattr("openghg_inversions.inversion_data.RhimeMergedData.from_options", unexpected_acquisition)
    with pytest.raises(ValueError, match="Historical resource selectors.*HBMCMC"):
        if route == "staged":
            module.resolve_stage_setup(options, model="standard")
        elif route == "nested":
            module.run_nested(**options, inner_domain="inner")
        elif route == "multisector":
            module.run_rhime_multisector(**{**options, "flux_sources": ["a", "b"]})
        elif route in ("example", "custom_basis"):
            module.run_custom_rhime(**options)
        else:
            module.run_rhime(**options)
