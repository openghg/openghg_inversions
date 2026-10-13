"""Exact basis inputs, support-resource preflight and compatibility boundaries."""

from dataclasses import replace
from pathlib import Path

import dask.array as da
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.native_flux import resolve_runtime_flux
from openghg_inversions.array_ops import to_dense
from openghg_inversions.basis import BasisFunctions, prepare_basis
from openghg_inversions.basis.compatibility import make_basis_functions
from openghg_inversions.basis.support import packaged_basis_support_path
from openghg_inversions.rhime.params import RhimeConfig
from openghg_inversions.rhime.preparation import resolve_basis_inputs
from tests.test_basis_dataset_inputs import _inputs
from tests.test_staged_workflow import _params


@pytest.mark.parametrize("suffix", [".nc", ".zarr"])
def test_exact_arbitrary_artifacts_reuse_operator_with_current_flux(tmp_path, suffix):
    sites, sources = _inputs()
    partition = xr.ones_like(sources["a"].flux.isel(time=0, drop=True), dtype=int)
    path = tmp_path / f"unstructured-name{suffix}"
    saved = prepare_basis(site_data=sites, runtime_flux=resolve_runtime_flux(sources), weighting_flux=sources["a"].flux, domain="TEST", start_date="2020-01-01",
                          nbasis=1, basis_partition=partition, output_path=path)
    assert path.exists()
    restored = BasisFunctions.load(path)
    current = {source: dataset.assign(flux=dataset.flux * -3) for source, dataset in sources.items()}
    reused = prepare_basis(site_data=sites, runtime_flux=resolve_runtime_flux(current), weighting_flux=current["a"].flux, domain="TEST", start_date="2020-01-01",
                           nbasis=1, basis_path=path)
    xr.testing.assert_equal(saved.flat_basis(), reused.flat_basis())
    xr.testing.assert_equal(reused.flux, saved.flux * -3)
    xr.testing.assert_equal(restored.flux, saved.flux)
    assert isinstance(current["a"].flux.data, da.Array)


def test_modern_default_and_explicit_legacy_output_have_equal_science(tmp_path):
    sites, sources = _inputs()
    classes = xr.ones_like(sources["a"].flux.isel(time=0, drop=True), dtype=int)
    options = dict(site_data=sites, flux_data=sources, species="ch4", domain="TEST", start_date="2020-01-01",
                   nbasis=2, basis_algorithm="region_constrained", region_classes=classes)
    with pytest.warns(DeprecationWarning, match="make_basis_functions"):
        modern = make_basis_functions(**options, output_path=str(tmp_path / "modern"))
    with pytest.warns(DeprecationWarning, match="make_basis_functions"):
        legacy = make_basis_functions(**options, output_path=str(tmp_path / "legacy"), basis_output_format="legacy")
    xr.testing.assert_equal(modern.flat_basis(), legacy.flat_basis())
    xr.testing.assert_equal(modern.flux, legacy.flux)
    BasisFunctions.load(modern.basis_artifact_path)
    assert Path(legacy.basis_artifact_path).name == "region_constrained_ch4_TEST_2020-01.nc"
    historical = legacy.flat_basis().expand_dims(time=[pd.Timestamp("2019-12-01")])
    historical_path = tmp_path / "historical.nc"
    historical.to_netcdf(historical_path)
    with xr.open_dataset(legacy.basis_artifact_path) as dataset:
        xr.testing.assert_equal(dataset.basis.squeeze("time", drop=True), legacy.flat_basis())
        assert dataset.time.dtype.kind == "M"
        xr.testing.assert_equal(dataset.basis.sel(time=pd.Timestamp("2020-01-01"), drop=True), legacy.flat_basis())
        with xr.open_dataarray(historical_path) as earlier:
            combined = xr.concat([earlier, dataset.basis], dim="time")
            assert combined.time.dtype.kind == "M"
            xr.testing.assert_equal(combined.sel(time=pd.Timestamp("2019-12-01"), drop=True), legacy.flat_basis())
            xr.testing.assert_equal(combined.sel(time=pd.Timestamp("2020-01-01"), drop=True), legacy.flat_basis())


@pytest.mark.parametrize("route", ["standard", "staged", "nested"])
@pytest.mark.parametrize("options", [
    {"basis_path": "absent/arbitrary.nc"},
    {"basis_partition_path": "absent/partition.nc"},
    {"landsea_path": "absent/map.nc"},
    {"outer_regions_path": "absent/outer.nc"},
    {"domain": "NO-PACKAGED-RESOURCE", "basis_algorithm": "weighted"},
])
def test_missing_selected_resource_fails_before_acquisition(tmp_path, monkeypatch, options, route):
    from openghg_inversions.rhime import standard, nested, _standard_stages

    monkeypatch.chdir(tmp_path)
    config = RhimeConfig.from_params(_params(**options), multisector=False)
    monkeypatch.setattr(standard.RhimeMergedData, "from_options", lambda **kwargs: pytest.fail("acquired"))
    with pytest.raises(FileNotFoundError):
        if route == "standard":
            standard.run_rhime(config=config)
        elif route == "nested":
            nested.prepare_nested_rhime_inputs(config, inner_domain="6km")
        else:
            _standard_stages.prepare(config.preparation, output_dir=tmp_path / "stage")
    assert not list(tmp_path.glob("**/prepare-manifest.json"))


def test_config_is_data_free_and_maps_are_resolved_at_application_boundary(tmp_path):
    config = RhimeConfig.from_params(_params(domain="MISSING", landsea_path=tmp_path / "missing.nc"),
                                     multisector=False)
    with pytest.raises(FileNotFoundError):
        resolve_basis_inputs(config.preparation)
    europe = packaged_basis_support_path("landsea", "EUROPE")
    resolved = resolve_basis_inputs(replace(config.preparation, domain="EUROPE", landsea_path=europe))
    assert resolved["landsea_classes"].ndim == 2
    with pytest.raises(FileNotFoundError, match="MISSING"):
        packaged_basis_support_path("landsea", "MISSING")


@pytest.mark.parametrize("output_format", ["basic", "paris"])
@pytest.mark.parametrize("country_file", [None, "absent/reporting.nc"])
def test_basis_preflight_does_not_require_reporting_resources(output_format, country_file):
    config = RhimeConfig.from_params(
        _params(output_format=output_format, country_file=country_file), multisector=False,
    )
    resolve_basis_inputs(config.preparation)


def test_fixed_outer_missing_resolved_map_fails_before_numerical_work(monkeypatch):
    from openghg_inversions.basis import _wrapper

    monkeypatch.setattr(
        _wrapper, "fixed_outer_regions_basis_from_data",
        lambda *args, **kwargs: pytest.fail("constructed basis before validating the outer map"),
    )
    with pytest.raises(ValueError, match="outer_regions must be resolved"):
        prepare_basis(
            site_data={}, runtime_flux=xr.DataArray(), weighting_flux=xr.DataArray(), domain="TEST", start_date="2020-01-01",
            nbasis=1, basis_algorithm="quadtree", fix_outer_regions=True,
        )


def test_conflicting_explicit_basis_inputs_fail_before_acquisition(monkeypatch):
    from openghg_inversions.rhime import standard

    config = RhimeConfig.from_params(_params(basis_path="missing.nc", basis_partition_path="also-missing.nc"),
                                     multisector=False)
    monkeypatch.setattr(standard.RhimeMergedData, "from_options", lambda **kwargs: pytest.fail("acquired"))
    with pytest.raises(ValueError, match="Supply either"):
        standard.run_rhime(config=config)


@pytest.mark.parametrize("outer_backend", ["dask", "sparse_dask"])
def test_legacy_relative_outer_map_resolves_at_compatibility_boundary(tmp_path, outer_backend):
    from openghg_inversions.basis.compatibility import resolve_legacy_outer_path

    sites, sources = _inputs()
    outer = xr.ones_like(sources["a"].flux.isel(time=0, drop=True), dtype=int).rename("region")
    outer = outer.where(outer.lat != outer.lat[0], 0).compute()
    outer.to_dataset().to_netcdf(tmp_path / "custom.nc")
    outer.rename("country").to_dataset().to_netcdf(tmp_path / "country-land-sea_TEST.nc")
    classes = xr.ones_like(outer)
    if outer_backend == "sparse_dask":
        import sparse

        outer = outer.copy(data=da.from_array(sparse.COO.from_numpy(outer.to_numpy()), chunks=(2, 2)))
    else:
        outer = outer.chunk({"lat": 2, "lon": 2})
    borrowed_payload = outer.data
    assert resolve_legacy_outer_path("TEST", "custom.nc", tmp_path) == tmp_path / "custom.nc"
    with pytest.warns(DeprecationWarning):
        legacy = make_basis_functions(
            site_data=sites, flux_data=sources, species="ch4", domain="TEST", start_date="2020-01-01",
            nbasis=1, basis_algorithm="region_constrained", region_classes=classes,
            fix_outer_regions=True, country_directory=str(tmp_path), outer_regions_path="custom.nc",
        )
    modern = prepare_basis(site_data=sites, runtime_flux=resolve_runtime_flux(sources), weighting_flux=sources["a"].flux, domain="TEST", start_date="2020-01-01",
                           nbasis=1, basis_algorithm="region_constrained", region_classes=classes,
                           fix_outer_regions=True, outer_regions=outer)
    xr.testing.assert_equal(legacy.flat_basis(), modern.flat_basis())
    assert outer.data is borrowed_payload
    assert isinstance(outer.data, da.Array)
    from openghg_inversions.hbmcmc.legacy_basis import fixed_outer_regions_basis

    historical_partition = fixed_outer_regions_basis(
        fp_all={**sites, ".flux": sources}, domain="TEST", start_date="2020-01-01",
        basis_algorithm="region_constrained", nbasis=1, region_classes=classes,
        country_directory=str(tmp_path), outer_regions_path="custom.nc",
    )
    xr.testing.assert_equal(historical_partition.squeeze("time", drop=True), modern.flat_basis())
    assert resolve_legacy_outer_path("EUHROB", "intem_region_definition_EUHROB.nc", None).is_file()
    config = RhimeConfig.from_params(_params(domain="TEST", basis_algorithm="quadtree", country_directory=tmp_path,
                                            outer_regions_path="custom.nc", fix_basis_outer_regions=True),
                                     multisector=False)
    # A historical directory selects its map only for algorithms which consume it.
    with pytest.warns(DeprecationWarning):
        resolved = resolve_basis_inputs(config.preparation)
    xr.testing.assert_equal(resolved["outer_regions"], to_dense(outer).compute())
    assert outer.data is borrowed_payload
