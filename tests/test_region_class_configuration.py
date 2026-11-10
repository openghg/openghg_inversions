"""Configured region classes for constrained basis generation (#454)."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis.support import load_basis_support_map, packaged_basis_support_path
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.params import DataConfig, RegionClassSelection, RhimeConfig
from openghg_inversions.rhime.preparation import resolve_basis_inputs
from openghg_inversions.rhime.requests import load_params
from tests.test_basis_dataset_inputs import _inputs
from tests.test_staged_workflow import _params

# Two land countries, an ocean code (-1) and a missing cell on the 4x4 test grid.
_COUNTRIES = np.array([
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [1, 3, 3, -1],
    [3, 3, -1, np.nan],
])


def _site_inputs():
    """Return non-uniform site footprints and fluxes on the 4x4 test grid."""
    sites, sources = _inputs()
    gradient = xr.DataArray(np.arange(1.0, 17.0).reshape(4, 4), dims=("lat", "lon"),
                            coords={"lat": sources["a"].lat, "lon": sources["a"].lon})
    return {site: dataset * gradient for site, dataset in sites.items()}, sources


def _write_classes(tmp_path, values=_COUNTRIES, variable="country", name="countries.nc"):
    """Write a class map on a grid one row and column larger than the test grid."""
    _, sources = _inputs()
    lat = np.append(sources["a"].lat.values, 54.0)
    lon = np.append(sources["a"].lon.values, 4.0)
    padded = np.full((5, 5), np.nan)
    padded[:4, :4] = values
    path = tmp_path / name
    xr.Dataset(
        {variable: (("lat", "lon"), padded), "name": ("ncountries", ["A", "B", "C"])},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    return path


def _prepare(basis_inputs, **options):
    """Prepare a basis from the shared synthetic inputs."""
    sites, sources = _site_inputs()
    return prepare_basis(
        site_data=sites, runtime_flux=resolve_runtime_flux(sources), weighting_flux=sources["a"].flux,
        domain="TEST", start_date="2020-01-01", **options, **basis_inputs,
    )


def _direct(path, variable="country", unmapped=-1.0, **options):
    """Prepare the same basis by passing the file's class map directly."""
    with xr.open_dataset(path) as dataset:
        classes = dataset[variable].load()
    return _prepare({}, region_classes=classes.where(classes != unmapped), **options)


def _config(**options):
    """Resolve a standard request with extra data options."""
    return RhimeConfig.from_params(_params(**options), multisector=False)


@pytest.mark.parametrize("variable", ["country", "region"])
def test_configured_class_map_matches_direct_python_basis(tmp_path, variable):
    path = _write_classes(tmp_path, variable=variable)
    config = _config(basis_algorithm="region_constrained", nbasis=5,
                     region_classes={"path": str(path), "unmapped_values": [-1]})
    assert config.data.region_classes == RegionClassSelection(path=path, unmapped_values=(-1,))

    configured = _prepare(resolve_basis_inputs(config.data), basis_algorithm="region_constrained", nbasis=5)
    direct = _direct(path, variable=variable, basis_algorithm="region_constrained", nbasis=5)
    xr.testing.assert_equal(configured.flat_basis(), direct.flat_basis())
    labels = configured.flat_basis().to_numpy()
    assert set(np.unique(labels[(_COUNTRIES == -1) | np.isnan(_COUNTRIES)])) == {0}
    for country in (1, 2, 3):
        assert not set(np.unique(labels[_COUNTRIES == country])) & set(np.unique(labels[_COUNTRIES != country]))


def test_explicit_variable_selects_a_mask_in_a_multivariable_file(tmp_path):
    path = _write_classes(tmp_path, variable="mask")
    with xr.open_dataset(path) as dataset:
        dataset.assign(country=dataset["mask"] * 0).load().to_netcdf(tmp_path / "both.nc")
    config = _config(basis_algorithm="region_constrained", nbasis=5,
                     region_classes={"path": str(tmp_path / "both.nc"), "variable": "mask",
                                     "unmapped_values": [-1]})
    configured = _prepare(resolve_basis_inputs(config.data), basis_algorithm="region_constrained", nbasis=5)
    direct = _direct(path, variable="mask", basis_algorithm="region_constrained", nbasis=5)
    xr.testing.assert_equal(configured.flat_basis(), direct.flat_basis())


def test_ini_configuration_matches_flat_python_configuration(tmp_path):
    path = _write_classes(tmp_path)
    ini = tmp_path / "rhime.ini"
    ini.write_text(
        "[INPUT.BASIS_CASE]\n"
        'basis_algorithm = "region_constrained"\n'
        "nbasis = 5\n"
        f'region_classes = {{"path": "{path}", "unmapped_values": [-1]}}\n'
    )
    from_ini = RhimeConfig.from_params(load_params(config_file=ini, overrides=_params()), multisector=False)
    flat = _config(basis_algorithm="region_constrained", nbasis=5,
                   region_classes=RegionClassSelection(path=path, unmapped_values=(-1,)))
    assert from_ini.data == flat.data


def test_saved_configuration_reproduces_the_class_selection(tmp_path):
    path = _write_classes(tmp_path)
    data = _config(basis_algorithm="region_constrained",
                   region_classes={"path": str(path), "variable": "country", "unmapped_values": [-1]}).data
    restored = DataConfig.from_params(json.loads(json.dumps(data.to_params(), default=str)))
    assert restored == data


def test_omitted_selection_uses_the_domain_land_sea_map():
    resolved = resolve_basis_inputs(_config(basis_algorithm="region_constrained").data)
    packaged = load_basis_support_map(packaged_basis_support_path("landsea", "EUROPE"), "country")
    xr.testing.assert_equal(resolved["region_classes"], packaged)
    masked = resolve_basis_inputs(
        _config(basis_algorithm="region_constrained", region_classes={"unmapped_values": [0]}).data
    )
    xr.testing.assert_equal(masked["region_classes"], packaged.where(packaged != 0))


def test_fixed_outer_regions_with_configured_inner_classes_match_direct_python(tmp_path):
    _, sources = _inputs()
    grid = {"lat": sources["a"].lat, "lon": sources["a"].lon}
    outer = xr.DataArray([[0, 0, 2, 2], [0, 1, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2]], dims=("lat", "lon"),
                         coords=grid, name="region", attrs={"inner_region_label": 1})
    outer_path = tmp_path / "outer.nc"
    outer.to_dataset().to_netcdf(outer_path)
    inner = np.array([[0, 0, 0, 0], [0, 4, 5, 0], [0, 4, 5, 0], [0, 0, 0, 0]], dtype=float)
    path = _write_classes(tmp_path, values=inner)
    config = _config(basis_algorithm="region_constrained", nbasis=2, fix_basis_outer_regions=True,
                     outer_regions_path=str(outer_path), region_classes={"path": str(path)})

    options = dict(basis_algorithm="region_constrained", nbasis=2, fix_outer_regions=True)
    configured = _prepare(resolve_basis_inputs(config.data), **options)
    direct = _direct(path, unmapped=np.nan, outer_regions=outer, **options)
    xr.testing.assert_equal(configured.flat_basis(), direct.flat_basis())
    assert len(np.unique(configured.flat_basis())) == 4


@pytest.mark.parametrize("algorithm", ["weighted", "region_constrained"])
def test_saved_basis_keeps_the_selection_inactive(tmp_path, algorithm):
    xr.Dataset({"basis": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))}).to_netcdf(
        tmp_path / "partition.nc")
    config = _config(basis_partition_path=str(tmp_path / "partition.nc"), basis_algorithm=algorithm,
                     region_classes={"path": str(tmp_path / "absent.nc")})
    assert "region_classes" not in resolve_basis_inputs(config.data)


@pytest.mark.parametrize(("options", "message"), [
    ({"basis_algorithm": "weighted", "region_classes": {}}, "applies only when generating"),
    ({"basis_algorithm": "region_constrained", "region_classes": "classes.nc"}, "a mapping/dict"),
    ({"basis_algorithm": "region_constrained", "region_classes": {"source": "file"}},
     "Unsupported region-class option"),
    ({"basis_algorithm": "region_constrained", "region_classes": {"variable": " "}}, "non-empty name"),
    ({"basis_algorithm": "region_constrained", "region_classes": {"unmapped_values": -1}}, "list of numbers"),
    ({"basis_algorithm": "region_constrained", "region_classes": {"unmapped_values": [float("nan")]}},
     "finite numbers"),
])
def test_invalid_region_class_options_fail_during_configuration(options, message):
    with pytest.raises(ValueError, match=message):
        _config(**options)


@pytest.mark.parametrize(("filename", "variable", "error", "message"), [
    ("missing.nc", None, FileNotFoundError, "Region-class map not found"),
    ("garbage.nc", None, OSError, "Unable to open country file"),
    ("ocean.nc", "mask", KeyError, "mask"),
    ("ocean.nc", None, ValueError, "no mapped cells"),
])
def test_invalid_region_class_files_fail_before_acquisition(tmp_path, monkeypatch, filename, variable,
                                                           error, message):
    from openghg_inversions.rhime import standard

    (tmp_path / "garbage.nc").write_text("not netcdf")
    xr.Dataset({"country": (("lat", "lon"), np.full((2, 2), -1.0))}).to_netcdf(tmp_path / "ocean.nc")
    selection = {"path": str(tmp_path / filename), "unmapped_values": [-1]}
    if variable is not None:
        selection["variable"] = variable
    config = _config(basis_algorithm="region_constrained", region_classes=selection)
    monkeypatch.setattr(standard.MergedData, "from_options", lambda **kwargs: pytest.fail("acquired"))
    with pytest.raises(error, match=message):
        standard.run_standard(config=config)


def test_nested_inner_basis_uses_its_own_region_classes(tmp_path):
    from openghg_inversions.rhime.nested import inner_data_config

    path = _write_classes(tmp_path)
    outer = _config(basis_algorithm="region_constrained", region_classes={"unmapped_values": [0]}).data
    inherited = inner_data_config(outer, inner_domain="6km")
    assert (inherited.basis_algorithm, inherited.region_classes) == ("quadtree", None)

    inner = inner_data_config(
        outer, inner_domain="6km", inner_basis_algorithm="region_constrained", inner_nbasis=5,
        inner_region_classes={"path": str(path), "unmapped_values": [-1]},
    )
    assert inner.region_classes == RegionClassSelection(path=path, unmapped_values=(-1,))
    configured = _prepare(resolve_basis_inputs(inner), basis_algorithm="region_constrained", nbasis=5)
    direct = _direct(path, basis_algorithm="region_constrained", nbasis=5)
    xr.testing.assert_equal(configured.flat_basis(), direct.flat_basis())


@pytest.mark.parametrize(("options", "error", "message"), [
    ({"inner_region_classes": {}}, ValueError, "applies only when generating"),
    ({"inner_basis_algorithm": "region_constrained", "inner_region_classes": {"unmapped": [-1]}},
     ValueError, "Unsupported region-class option"),
    ({"inner_basis_algorithm": "region_constrained", "inner_region_classes": {"path": "absent.nc"}},
     FileNotFoundError, "Region-class map not found"),
])
def test_invalid_inner_region_classes_fail_before_acquisition(tmp_path, monkeypatch, options, error, message):
    from openghg_inversions.rhime import nested

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nested, "acquire_merged_data", lambda *a, **k: pytest.fail("acquired"))
    with pytest.raises(error, match=message):
        nested.run_nested(config=_config(output_format="none"), inner_domain="6km", **options)


def test_direct_selection_requires_resolved_values():
    with pytest.raises(ValueError, match="filesystem path"):
        RegionClassSelection(path="classes.nc")  # type: ignore[arg-type]
    config = _config(basis_algorithm="quadtree")
    with pytest.raises(ValueError, match="applies only when generating"):
        replace(config.data, region_classes=RegionClassSelection(path=Path("/maps/classes.nc")))
