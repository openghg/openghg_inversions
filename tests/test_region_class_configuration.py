"""Configured region classes for constrained basis generation (#454)."""

import json
from dataclasses import replace

import numpy as np
import pytest
import xarray as xr

from openghg_inversions.basis import prepare_basis
from openghg_inversions.basis.support import load_basis_support_map, packaged_basis_support_path
from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.rhime.params import DataConfig, RhimeConfig
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


def _write_classes(tmp_path, values=_COUNTRIES, variable="country_id"):
    """Write a class map on a grid one row and column larger than the test grid."""
    _, sources = _inputs()
    lat = np.append(sources["a"].lat.values, 54.0)
    lon = np.append(sources["a"].lon.values, 4.0)
    padded = np.full((5, 5), np.nan)
    padded[:4, :4] = values
    path = tmp_path / "countries.nc"
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


def _file_options(path, **overrides):
    """Return data options selecting a configured region-class file."""
    return {
        "basis_algorithm": "region_constrained", "nbasis": 5,
        "region_classes_source": "file", "region_classes_path": str(path),
        "region_classes_variable": "country_id", "region_classes_unmapped_values": [-1],
        "region_allocation": "area", "min_regions_per_class": 1, **overrides,
    }


def test_configured_file_classes_match_direct_python_basis(tmp_path):
    path = _write_classes(tmp_path)
    config = RhimeConfig.from_params(_params(**_file_options(path)), multisector=False)

    basis_inputs = resolve_basis_inputs(config.data)
    configured = _prepare(basis_inputs, basis_algorithm="region_constrained", nbasis=5)

    with xr.open_dataset(path) as dataset:
        classes = dataset["country_id"].load()
    direct = _prepare({}, basis_algorithm="region_constrained", nbasis=5,
                      region_classes=classes.where(classes != -1), region_allocation="area")
    xr.testing.assert_equal(configured.flat_basis(), direct.flat_basis())
    labels = configured.flat_basis().to_numpy()
    assert set(np.unique(labels[_COUNTRIES == -1])) | set(np.unique(labels[np.isnan(_COUNTRIES)])) == {0}
    for country in (1, 2, 3):
        assert not set(np.unique(labels[_COUNTRIES == country])) & set(np.unique(labels[_COUNTRIES != country]))


def test_ini_configuration_matches_flat_python_configuration(tmp_path):
    path = _write_classes(tmp_path)
    ini = tmp_path / "rhime.ini"
    ini.write_text(
        "[INPUT.BASIS_CASE]\n"
        'basis_algorithm = "region_constrained"\n'
        "nbasis = 5\n"
        'region_classes_source = "file"\n'
        f'region_classes_path = "{path}"\n'
        'region_classes_variable = "country_id"\n'
        "region_classes_unmapped_values = [-1]\n"
        'region_allocation = "area"\n'
    )
    from_ini = RhimeConfig.from_params(load_params(config_file=ini, overrides=_params()), multisector=False)
    flat = RhimeConfig.from_params(_params(**_file_options(path)), multisector=False)
    assert from_ini.data == flat.data
    xr.testing.assert_equal(resolve_basis_inputs(from_ini.data)["region_classes"],
                            resolve_basis_inputs(flat.data)["region_classes"])


def test_saved_configuration_reproduces_the_class_selection(tmp_path):
    path = _write_classes(tmp_path)
    data = RhimeConfig.from_params(_params(**_file_options(path)), multisector=False).data
    restored = DataConfig.from_params(json.loads(json.dumps(data.to_params(), default=str)))
    assert restored == data


def test_landsea_source_uses_the_domain_land_sea_map():
    config = RhimeConfig.from_params(
        _params(basis_algorithm="region_constrained", region_classes_source="landsea",
                region_classes_unmapped_values=[0]),
        multisector=False,
    )
    resolved = resolve_basis_inputs(config.data)
    packaged = load_basis_support_map(packaged_basis_support_path("landsea", "EUROPE"), "country")
    xr.testing.assert_identical(resolved["region_classes"], packaged.where(packaged != 0))
    assert resolved["region_allocation"] == "weight"
    assert resolved["min_regions_per_class"] == 1


def test_fixed_outer_regions_with_configured_inner_classes_match_direct_python(tmp_path):
    _, sources = _inputs()
    grid = {"lat": sources["a"].lat, "lon": sources["a"].lon}
    outer = xr.DataArray([[0, 0, 2, 2], [0, 1, 1, 2], [0, 1, 1, 2], [0, 0, 2, 2]], dims=("lat", "lon"),
                         coords=grid, name="region", attrs={"inner_region_label": 1})
    outer_path = tmp_path / "outer.nc"
    outer.to_dataset().to_netcdf(outer_path)
    inner = np.array([[0, 0, 0, 0], [0, 4, 5, 0], [0, 4, 5, 0], [0, 0, 0, 0]], dtype=float)
    path = _write_classes(tmp_path, values=inner)
    config = RhimeConfig.from_params(_params(**_file_options(
        path, nbasis=2, fix_basis_outer_regions=True, outer_regions_path=str(outer_path),
        region_classes_unmapped_values=[],
    )), multisector=False)

    options = dict(basis_algorithm="region_constrained", nbasis=2, fix_outer_regions=True)
    configured = _prepare(resolve_basis_inputs(config.data), **options)
    with xr.open_dataset(path) as dataset:
        classes = dataset["country_id"].load()
    direct = _prepare({}, **options, outer_regions=outer, region_classes=classes, region_allocation="area")
    xr.testing.assert_equal(configured.flat_basis(), direct.flat_basis())
    assert len(np.unique(configured.flat_basis())) == 4


def test_saved_basis_inputs_do_not_load_region_classes(tmp_path):
    config = RhimeConfig.from_params(
        _params(basis_partition_path=str(tmp_path / "partition.nc"), basis_algorithm="region_constrained"),
        multisector=False,
    )
    xr.Dataset({"basis": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))}).to_netcdf(
        tmp_path / "partition.nc")
    assert "region_classes" not in resolve_basis_inputs(config.data)


@pytest.mark.parametrize(("options", "message"), [
    ({"basis_algorithm": "region_constrained"}, "requires `region_classes_source`"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "countries"}, "must be 'landsea', 'file'"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "file",
      "region_classes_variable": "country"}, "requires `region_classes_path`"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "file",
      "region_classes_path": "classes.nc"}, "non-empty `region_classes_variable`"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "landsea",
      "region_classes_path": "classes.nc"}, "require `region_classes_source = 'file'`"),
    ({"basis_algorithm": "weighted", "region_classes_source": "landsea"}, "applies only to"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "landsea",
      "region_classes_unmapped_values": -1}, "list of finite numbers"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "landsea",
      "region_classes_unmapped_values": [float("nan")]}, "list of finite numbers"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "landsea",
      "region_allocation": "population"}, "`region_allocation` must be"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "landsea",
      "min_regions_per_class": -1}, "must be non-negative"),
    ({"basis_algorithm": "region_constrained", "region_classes_source": "landsea",
      "min_regions_per_class": 1.5}, "expected an integer"),
])
def test_invalid_region_class_options_fail_during_configuration(options, message):
    with pytest.raises(ValueError, match=message):
        RhimeConfig.from_params(_params(**options), multisector=False)


def _invalid_files(tmp_path):
    """Write region-class files that must fail before acquisition."""
    (tmp_path / "garbage.nc").write_text("not netcdf")
    xr.Dataset({"country_id": (("time", "lat", "lon"), np.ones((1, 2, 2)))}).to_netcdf(tmp_path / "3d.nc")
    xr.Dataset({"country_id": (("lat", "lon"), np.full((2, 2), -1.0))}).to_netcdf(tmp_path / "ocean.nc")
    return tmp_path


@pytest.mark.parametrize(("filename", "variable", "error", "message"), [
    ("missing.nc", "country_id", FileNotFoundError, "Region-class file not found"),
    ("garbage.nc", "country_id", ValueError, "Could not read region-class file"),
    ("ocean.nc", "country", ValueError, r"no variable 'country'; available variables: \['country_id'\]"),
    ("3d.nc", "country_id", ValueError, "two-dimensional numeric map"),
    ("ocean.nc", "country_id", ValueError, "no mapped cells"),
])
def test_invalid_region_class_files_fail_before_acquisition(tmp_path, monkeypatch, filename, variable,
                                                           error, message):
    from openghg_inversions.rhime import standard

    directory = _invalid_files(tmp_path)
    config = RhimeConfig.from_params(_params(**_file_options(
        directory / filename, region_classes_variable=variable)), multisector=False)
    monkeypatch.setattr(standard.MergedData, "from_options", lambda **kwargs: pytest.fail("acquired"))
    with pytest.raises(error, match=message):
        standard.run_standard(config=config)


def test_directly_constructed_configuration_without_classes_fails_clearly():
    config = RhimeConfig.from_params(_params(basis_algorithm="quadtree"), multisector=False)
    with pytest.raises(ValueError, match="requires `region_classes_source`"):
        resolve_basis_inputs(replace(config.data, basis_algorithm="region_constrained"))


def test_nested_inner_basis_uses_its_own_region_classes(tmp_path):
    from openghg_inversions.rhime.nested import inner_data_config

    path = _write_classes(tmp_path)
    outer = RhimeConfig.from_params(
        _params(basis_algorithm="region_constrained", region_classes_source="landsea",
                region_allocation="area", min_regions_per_class=2),
        multisector=False,
    ).data
    inherited = inner_data_config(outer, inner_domain="6km")
    assert inherited.basis_algorithm == "quadtree"
    assert inherited.region_classes_source is None
    assert (inherited.region_allocation, inherited.min_regions_per_class) == ("weight", 1)

    inner = inner_data_config(
        outer, inner_domain="6km", inner_basis_algorithm="region_constrained", inner_nbasis=5,
        inner_region_classes_source="file", inner_region_classes_path=str(path),
        inner_region_classes_variable="country_id", inner_region_classes_unmapped_values=[-1],
        inner_region_allocation="area",
    )
    assert inner.region_classes_path == path
    configured = _prepare(resolve_basis_inputs(inner), basis_algorithm="region_constrained", nbasis=5)
    with xr.open_dataset(path) as dataset:
        classes = dataset["country_id"].load()
    direct = _prepare({}, basis_algorithm="region_constrained", nbasis=5,
                      region_classes=classes.where(classes != -1), region_allocation="area")
    xr.testing.assert_equal(configured.flat_basis(), direct.flat_basis())


@pytest.mark.parametrize(("options", "message"), [
    ({"inner_basis_algorithm": "region_constrained"}, "requires `region_classes_source`"),
    ({"inner_region_classes_source": "landsea"}, "applies only to"),
    ({"inner_basis_algorithm": "region_constrained", "inner_region_classes_source": "file",
      "inner_region_classes_variable": "country"}, "requires `region_classes_path`"),
    ({"inner_basis_algorithm": "region_constrained", "inner_region_classes_source": "landsea",
      "inner_region_allocation": "population"}, "`region_allocation` must be"),
])
def test_invalid_inner_region_class_options_fail_before_acquisition(monkeypatch, options, message):
    from openghg_inversions.rhime import nested

    config = RhimeConfig.from_params(_params(output_format="none"), multisector=False)
    monkeypatch.setattr(nested, "acquire_merged_data", lambda *a, **k: pytest.fail("acquired"))
    with pytest.raises(ValueError, match=message):
        nested.run_nested(config=config, inner_domain="6km", **options)


def test_nested_inner_region_class_file_is_loaded_before_acquisition(tmp_path, monkeypatch):
    from openghg_inversions.rhime import nested

    config = RhimeConfig.from_params(_params(output_format="none"), multisector=False)
    monkeypatch.setattr(nested, "acquire_merged_data", lambda *a, **k: pytest.fail("acquired"))
    with pytest.raises(FileNotFoundError, match="Region-class file not found"):
        nested.run_nested(
            config=config, inner_domain="6km", inner_basis_algorithm="region_constrained",
            inner_region_classes_source="file", inner_region_classes_path=str(tmp_path / "missing.nc"),
            inner_region_classes_variable="country",
        )
