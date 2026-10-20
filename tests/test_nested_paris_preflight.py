"""Nested reporting fails before acquisition, sampling, or product/cache writes."""

from dataclasses import replace
import hashlib

import numpy as np
import pytest
import xarray as xr

from openghg_inversions._spatial_grid import coordinate_sha256
from openghg_inversions.postprocessing import nested_paris_outputs as paris
from openghg_inversions.rhime import nested
from openghg_inversions.rhime.outputs import validate_nested_output_resources
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.specs import RhimeOutputSpec
from tests.test_staged_workflow import _params


@pytest.mark.parametrize("options, match", [
    ({"unexpected": True}, "unexpected"),
    ({"template_version": "unknown"}, "Unsupported PARIS template"),
    ({"template_version": "latest", "time_point": "start"}, "requires midpoint"),
    ({"time_point": "end"}, "time_point"),
    ({"flux_frequency": "0h"}, "positive fixed duration"),
    ({"country_file": "override.nc"}, "country_file"),
    ({"obs_avg_period": "1h"}, "obs_avg_period"),
    ({"inner_domain_label": "override"}, "inner_domain_label"),
])
@pytest.mark.parametrize("route", ["full", "prepared", "output"])
def test_nested_invalid_paris_options_fail_before_work(tmp_path, monkeypatch, options, match, route):
    country = tmp_path / "country.nc"
    country.touch()
    destination = tmp_path / "products"
    choices = dict(options, country_file_cache_dir=tmp_path / "cache")
    config = RhimeConfig.from_params(_params(
        output_format="paris", country_file=country, output_path=destination,
        paris_postprocessing_kwargs=choices,
    ), multisector=False)

    def forbidden(*args, **kwargs):
        pytest.fail("work began before reporting preflight")

    monkeypatch.setattr(nested, "acquire_merged_data", forbidden)
    monkeypatch.setattr(nested, "materialize_nested_inputs", forbidden)
    monkeypatch.setattr(nested, "sample_rhime_model", forbidden)
    monkeypatch.setattr(paris, "paris_concentration_outputs", forbidden)
    with pytest.raises(ValueError, match=match):
        if route == "output":
            nested._write_nested_paris_outputs(object(), config.output)
        else:
            nested.run_nested(
                config=config, inner_domain="6km", prepared=object() if route == "prepared" else None,
            )
    assert not destination.exists()
    assert not (tmp_path / "cache").exists()
    assert choices == dict(options, country_file_cache_dir=tmp_path / "cache")


@pytest.mark.parametrize("kind", ["missing", "directory"])
@pytest.mark.parametrize("route", ["runner", "products"])
def test_nested_explicit_inner_country_requires_file(tmp_path, monkeypatch, kind, route):
    country = tmp_path / "country.nc"
    country.touch()
    inner_country = tmp_path / "missing.nc" if kind == "missing" else tmp_path
    choices = {"inner_country_file": inner_country, "country_file_cache_dir": tmp_path / "cache"}

    def forbidden(*args, **kwargs):
        pytest.fail("products began before inner resource preflight")

    monkeypatch.setattr(paris, "paris_concentration_outputs", forbidden)
    monkeypatch.setattr(nested, "acquire_merged_data", forbidden)
    config = RhimeConfig.from_params(_params(
        output_format="paris", country_file=country, output_path=tmp_path / "products",
        paris_postprocessing_kwargs=choices,
    ), multisector=False)
    with pytest.raises(FileNotFoundError):
        if route == "runner":
            nested.run_nested(config=config, inner_domain="6km")
        else:
            paris.make_nested_paris_products(object(), object(), inner_domain_label="6km", **choices)
    assert not (tmp_path / "products").exists()
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("options", [
    {"template_version": "unknown"},
    {"template_version": "latest", "time_point": "start"},
    {"flux_frequency": "invalid"},
])
def test_direct_nested_products_validate_options_before_products(tmp_path, monkeypatch, options):
    monkeypatch.setattr(paris, "paris_concentration_outputs", lambda *a, **kw: pytest.fail("built products"))
    with pytest.raises(ValueError):
        paris.make_nested_paris_products(
            object(), object(), inner_domain_label="6km", country_file_cache_dir=tmp_path / "cache", **options,
        )
    assert not (tmp_path / "cache").exists()


def test_nested_preflight_leaves_none_output_and_valid_resources_unchanged(tmp_path):
    ignored = {"unexpected": True, "inner_country_file": tmp_path / "missing.nc"}
    spec = RhimeOutputSpec(output_format="none", paris_postprocessing_kwargs=ignored)
    validate_nested_output_resources(spec)
    country = tmp_path / "country.nc"
    country.touch()
    choices = {
        "template_version": "latest", "inner_country_file": country,
        "flux_frequency": "monthly", "inversion_grid": False, "country_selections": None,
        "time_point": "midpoint", "report_mode": True, "country_file_cache_dir": tmp_path / "cache",
    }
    validate_nested_output_resources(replace(
        spec, output_format="paris", country_file=country, output_path=tmp_path / "products",
        paris_postprocessing_kwargs=choices,
    ))
    assert not (tmp_path / "products").exists()
    assert not (tmp_path / "cache").exists()
    assert spec.paris_postprocessing_kwargs == ignored


def test_spatial_digest_preserves_existing_artifact_and_cache_identity():
    coordinates = {
        "lat": xr.DataArray(np.array([50., 51.], dtype="float32"), dims="lat"),
        "lon": xr.DataArray(np.array([-2., -1.], dtype="float64"), dims="lon"),
    }
    previous = hashlib.sha256()
    for name, coordinate in coordinates.items():
        values = coordinate.to_numpy()
        previous.update(name.encode())
        previous.update(str(values.dtype).encode())
        previous.update(str(values.shape).encode())
        previous.update(np.ascontiguousarray(values).tobytes())
    assert coordinate_sha256(**coordinates) == previous.hexdigest()
