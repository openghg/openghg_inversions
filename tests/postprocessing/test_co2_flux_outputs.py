"""Independent draw-level checks of conditional affine CO2 flux products."""

from dataclasses import replace
import json

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import sparse
import xarray as xr

from openghg_inversions.basis.affine_flux_map import AffineFluxMap
from openghg_inversions.basis.affine_flux_map_io import AffineFluxMapArtifact, save
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.postprocessing.co2_flux_outputs import (
    co2_country_flux_outputs,
    co2_native_flux_outputs,
)
from openghg_inversions.postprocessing.countries import Countries
from openghg_inversions.rhime.co2.co2_affine_output import (
    BoundCo2AffineFluxMap,
    import_explicit_affine_flux_map,
    load_and_bind_affine_flux_map,
    prepared_inputs_content_id,
    produce_bucket_affine_flux_map,
)
from test_co2_affine_output import _multisource_prepared, _prepared


def _fixture():
    coords = {"native_source": ["fossil", "biosphere"], "lat": [50.0, 51.0], "lon": [-2.0, -1.0]}
    dims = ("native_source", "lat", "lon")
    mean = xr.DataArray(
        np.array([0.8, 1.4, 1.1, 0.6, 1.2, 0.7, 0.9, 1.8]).reshape(2, 2, 2),
        dims=dims,
        coords=coords,
        attrs={"units": "1"},
    )
    flux = mean.copy(data=np.array([2, 3, 1, 4, -3, -5, -2, -1]).reshape(2, 2, 2))
    flux.attrs = {"units": "mol m-2 s-1"}
    prolongation = xr.DataArray(
        np.array(
            [[0.8, 0.2], [-0.1, 1.1], [0.6, 0.4], [0.2, 0.9], [0.7, 0.3], [0.4, 0.8], [0.9, -0.2], [0.1, 0.7]]
        ).reshape(2, 2, 2, 2),
        dims=(*dims, "region"),
        coords={**coords, "region": ["a", "b"]},
        attrs={"units": "1"},
    )
    reference = xr.DataArray([0.4, 1.7], dims="region", coords={"region": ["a", "b"]}, attrs={"units": "1"})
    mapping = AffineFluxMap(mean, flux, prolongation, state_dim="region")
    bound = BoundCo2AffineFluxMap(
        AffineFluxMapArtifact(mapping, "fixture", {"projection_strategy": "explicit"}, {}), reference
    )
    groups = {}
    for group, values in {
        "posterior": np.array([[[0.9, 1.2], [0.3, 2.0], [1.4, 1.2]], [[1.5, 2.2], [0.8, 0.9], [2.1, 1.0]]]),
        "prior": np.array([[[0.4, 1.7], [0.2, 1.0], [1.2, 2.3], [0.8, 2.0]]]),
    }.items():
        groups[group] = xr.Dataset(
            {
                "retained_draws": xr.DataArray(
                    values,
                    dims=("chain", "draw", "region"),
                    coords={
                        "chain": np.arange(len(values)),
                        "draw": np.arange(values.shape[1]),
                        "region": reference.region,
                    },
                    attrs={"units": "1"},
                )
            }
        )
    groups["/"] = xr.Dataset(
        attrs={
            "rhime_model_metadata": json.dumps({"recipe": "co2", "provenance": "independent affine fixture"}),
            "rhime_variable_roles": json.dumps({"flux_scale": "retained_draws"}),
        }
    )
    trace = xr.DataTree.from_dict(groups)
    countries = Countries(
        xr.Dataset(
            {"country": (("lat", "lon"), [[0, 1], [1, 0]]), "name": ("ncountries", ["FRANCE", "GERMANY"])},
            coords={"lat": coords["lat"], "lon": coords["lon"]},
        )
    )
    return bound, trace, countries


def _oracle(bound, state):
    """Evaluate F[m + U*(alpha-alpha_ref)] directly, without package kernels."""
    mapping = bound.affine_map
    delta = state.values - bound.reference_state.values
    return mapping.flux.values * (
        mapping.native_mean.values + np.einsum("slir,cdr->cdsli", mapping.prolongation.values, delta)
    )


def _assert_summary(result, prefix, draws, dims):
    samples = draws.reshape((-1, *draws.shape[2:]))
    np.testing.assert_allclose(result[f"{prefix}_mean"].transpose(*dims), samples.mean(axis=0), rtol=2e-6)
    np.testing.assert_allclose(result[f"{prefix}_stdev"].transpose(*dims), samples.std(axis=0), rtol=2e-6)
    np.testing.assert_allclose(
        result[f"{prefix}_quantile"].transpose("quantile", *dims),
        np.quantile(samples, [0.159, 0.841], axis=0),
        rtol=2e-6,
    )


def test_native_products_match_explicit_affine_oracle_and_sum_draws_before_statistics():
    bound, trace, _ = _fixture()
    before = trace.copy(deep=True)
    result = co2_native_flux_outputs(trace, bound)
    for group in ("prior", "posterior"):
        draws = _oracle(bound, trace[group].retained_draws)
        _assert_summary(result, f"flux_{group}", draws, ("native_source", "lat", "lon"))
        _assert_summary(result, f"flux_total_{group}", draws.sum(axis=2), ("lat", "lon"))
        # Signed source contributions are correlated; summing marginal uncertainties is wrong.
        assert not np.allclose(
            result[f"flux_total_{group}_stdev"], result[f"flux_{group}_stdev"].sum("native_source")
        )
    assert result.attrs["uncertainty_scope"] == "retained_state_conditional"
    for variable in result.data_vars.values():
        assert variable.attrs["units"] == "mol m-2 s-1"
        assert variable.attrs["uncertainty_scope"] == "retained_state_conditional"
    xr.testing.assert_identical(trace, before)


def test_country_products_compose_before_sampling_and_convert_to_annual_co2_mass(monkeypatch):
    bound, trace, countries = _fixture()
    before = countries.matrix.copy(deep=True)
    oracle = {}
    # Independent country membership, with the public area grid in square metres.
    weights = np.stack([np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])])
    weights = weights * countries.area_grid.values
    for group in ("prior", "posterior"):
        oracle[group] = np.einsum("cdsli,kli->cdsk", _oracle(bound, trace[group].retained_draws), weights)
        oracle[group] *= 44.01 * 365 * 24 * 3600

    def fail_native(*args, **kwargs):
        pytest.fail("Country products must compose the aggregate operator before evaluating draws.")

    original_dot = xr.dot

    def check_dot(*args, **kwargs):
        output = original_dot(*args, **kwargs)
        assert not ({"lat", "lon"} & set(output.dims) and {"chain", "draw"} & set(output.dims))
        return output

    monkeypatch.setattr(BoundCo2AffineFluxMap, "state_to_flux", fail_native)
    monkeypatch.setattr(AffineFluxMap, "state_to_flux", fail_native)
    monkeypatch.setattr(xr, "dot", check_dot)
    result = co2_country_flux_outputs(trace, bound, countries)
    for group, draws in oracle.items():
        _assert_summary(result, f"country_{group}", draws, ("native_source", "country"))
        _assert_summary(result, f"country_total_{group}", draws.sum(axis=2), ("country",))
    assert result.country.values.tolist() == ["FRANCE", "GERMANY"]
    assert "lat" not in result.dims and "lon" not in result.dims
    for variable in result.data_vars.values():
        assert variable.attrs["units"] == "g yr-1"
        assert variable.attrs["uncertainty_scope"] == "retained_state_conditional"
    xr.testing.assert_identical(countries.matrix, before)


@pytest.mark.parametrize("chunk_prolongation", [False, True], ids=["eager-sparse", "dask-sparse"])
def test_sparse_dask_inputs_remain_lazy_and_borrowed(chunk_prolongation):
    bound, trace, countries = _fixture()
    expected_native = co2_native_flux_outputs(trace, bound)
    expected_country = co2_country_flux_outputs(trace, bound, countries)
    mapping = bound.affine_map
    sparse_u = sparse.COO.from_numpy(mapping.prolongation.values)
    lazy_u = mapping.prolongation.copy(
        data=da.from_array(sparse_u, chunks=(1, 1, 2, 2)) if chunk_prolongation else sparse_u
    )
    lazy_map = replace(
        mapping,
        prolongation=lazy_u,
        flux=mapping.flux.chunk({"lat": 1}) if chunk_prolongation else mapping.flux,
    )
    lazy_bound = replace(bound, artifact=replace(bound.artifact, affine_map=lazy_map))
    for group in ("prior", "posterior"):
        trace[group] = trace[group].to_dataset().chunk({"chain": 1, "draw": 1})
    original_u = lazy_u.data
    original_flux = lazy_map.flux.data
    original_state = trace["posterior"].retained_draws.data
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        native = co2_native_flux_outputs(trace, lazy_bound)
        country = co2_country_flux_outputs(trace, lazy_bound, countries)
    assert tasks == []
    assert all(isinstance(variable.data, da.Array) for variable in native.data_vars.values())
    assert all(isinstance(variable.data, da.Array) for variable in country.data_vars.values())
    assert lazy_u.data is original_u
    assert lazy_map.flux.data is original_flux
    assert trace["posterior"].retained_draws.data is original_state
    xr.testing.assert_allclose(native.compute(), expected_native)
    xr.testing.assert_allclose(country.compute(), expected_country)


def test_compatible_flux_and_native_scaling_units_preserve_physical_outputs():
    bound, trace, countries = _fixture()
    mapping = bound.affine_map
    scaled = replace(
        mapping,
        native_mean=(100 * mapping.native_mean).assign_attrs(units="0.01"),
        flux=(1e6 * mapping.flux).assign_attrs(units="umol m-2 s-1"),
    )
    converted = replace(bound, artifact=replace(bound.artifact, affine_map=scaled))
    xr.testing.assert_allclose(
        co2_native_flux_outputs(trace, converted), co2_native_flux_outputs(trace, bound)
    )
    xr.testing.assert_allclose(
        co2_country_flux_outputs(trace, converted, countries),
        co2_country_flux_outputs(trace, bound, countries),
    )


@pytest.mark.parametrize("units", ["kg m-2 s-1", "mol s-1"])
def test_incompatible_reference_flux_units_are_rejected(units):
    bound, trace, countries = _fixture()
    mapping = replace(bound.affine_map, flux=bound.affine_map.flux.assign_attrs(units=units))
    invalid = replace(bound, artifact=replace(bound.artifact, affine_map=mapping))
    with pytest.raises(ValueError, match="flux units"):
        co2_native_flux_outputs(trace, invalid)
    with pytest.raises(ValueError, match="flux units"):
        co2_country_flux_outputs(trace, invalid, countries)


@pytest.mark.parametrize("group", ["prior", "posterior"])
def test_state_labels_and_country_grid_must_match_exactly(group):
    bound, trace, countries = _fixture()
    trace[group] = trace[group].to_dataset().isel(region=[1, 0])
    with pytest.raises(ValueError, match="labels|align|exact"):
        co2_native_flux_outputs(trace, bound)
    with pytest.raises(ValueError, match="labels|align|exact"):
        co2_country_flux_outputs(trace, bound, countries)
    bound, trace, countries = _fixture()
    countries.matrix = countries.matrix.assign_coords(lon=[-1.0, -2.0])
    with pytest.raises(ValueError, match="labels|align|exact"):
        co2_country_flux_outputs(trace, bound, countries)


def test_missing_prior_role_and_linked_recipe_fail_explicitly():
    bound, trace, _ = _fixture()
    del trace["prior"]
    with pytest.raises(ValueError, match="prior"):
        co2_native_flux_outputs(trace, bound)
    _, trace, _ = _fixture()
    trace.attrs["rhime_variable_roles"] = json.dumps({"flux_scale": "absent"})
    with pytest.raises(ValueError, match="flux_scale|absent"):
        co2_native_flux_outputs(trace, bound)
    _, trace, _ = _fixture()
    trace.attrs["rhime_model_metadata"] = json.dumps({"recipe": "co2_o2"})
    with pytest.raises(ValueError, match="co2_o2|CO2.O2|recipe"):
        co2_native_flux_outputs(trace, bound)


def test_saved_prepared_and_explicit_artifact_feed_native_products(tmp_path):
    prepared, native_mean = _prepared()
    prepared_path = tmp_path / "prepared.nc"
    affine_path = tmp_path / "affine.nc"
    prepared.save(prepared_path)
    prolongation = xr.DataArray(
        [[[0.8, 0.2], [-0.1, 1.1]]],
        dims=("lat", "lon", "region"),
        coords={**native_mean.coords, "region": prepared.inv_inputs.region},
        attrs={"units": "1"},
    )
    artifact = import_explicit_affine_flux_map(
        prepared,
        native_mean,
        prepared.basis_functions.flux,
        prolongation,
        prepared_inputs_id=prepared_inputs_content_id(prepared_path),
        reference_state=prepared.inv_inputs.alpha_prior_mean,
    )
    save(artifact, affine_path)
    bound = load_and_bind_affine_flux_map(affine_path, prepared_path)
    _, trace, _ = _fixture()
    for group in ("prior", "posterior"):
        trace[group] = trace[group].to_dataset().assign_coords(region=prepared.inv_inputs.region)
    result = co2_native_flux_outputs(trace, bound)
    for group in ("prior", "posterior"):
        delta = trace[group].retained_draws.values - np.array([0.4, 1.7])
        draws = np.array([-2.0, 3.0]) * (
            np.array([0.8, 1.4]) + np.einsum("ir,cdr->cdi", np.array([[0.8, 0.2], [-0.1, 1.1]]), delta)
        )
        _assert_summary(result, f"flux_total_{group}", draws[:, :, None, :], ("lat", "lon"))
    assert "native_source" not in result.dims
    result.to_netcdf(tmp_path / "flux.nc")
    with xr.open_dataset(tmp_path / "flux.nc") as restored:
        xr.testing.assert_identical(restored, result)


def test_saved_multisource_bucket_preserves_time_and_matches_dense_oracle(tmp_path):
    prepared, native_mean = _multisource_prepared()
    lat = [50.0, 51.0]
    native_mean = native_mean.reindex(lat=lat, method="nearest")
    time_scale = xr.DataArray(
        [1.0, 2.0],
        dims="time",
        coords={"time": np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]")},
    )
    old_basis = prepared.basis_functions
    basis = BasisFunctions.from_multi_source_flat_basis(
        {
            source: flat.reindex(lat=lat, method="nearest")
            for source, flat in old_basis.operator.basis_flat.items()
        },
        (old_basis.flux.reindex(lat=lat, method="nearest") * time_scale).assign_attrs(units="mol m-2 s-1"),
        operator_kwargs={"state_dim": "region"},
    )
    prepared = replace(
        prepared,
        rhime_inputs=RhimePreparedInputs(prepared.inv_inputs, basis, prepared.rhime_inputs.site_metadata),
    )
    prepared_path, affine_path = tmp_path / "prepared.nc", tmp_path / "affine.nc"
    prepared.save(prepared_path)
    artifact = produce_bucket_affine_flux_map(
        prepared, native_mean, prepared_inputs_id=prepared_inputs_content_id(prepared_path)
    )
    save(artifact, affine_path)
    bound = load_and_bind_affine_flux_map(affine_path, prepared_path)
    _, trace, countries = _fixture()
    for group, shape in (("prior", (1, 4, 3)), ("posterior", (2, 3, 3))):
        values = np.arange(np.prod(shape)).reshape(shape) / 10 + np.array([0.4, 1.7, 0.9])
        trace[group] = xr.Dataset(
            {
                "retained_draws": xr.DataArray(
                    values,
                    dims=("chain", "draw", "region"),
                    coords={
                        "chain": np.arange(shape[0]),
                        "draw": np.arange(shape[1]),
                        **prepared.inv_inputs.alpha_prior_mean.coords,
                    },
                    attrs={"units": "1"},
                )
            }
        )
    native = co2_native_flux_outputs(trace, bound)
    country = co2_country_flux_outputs(trace, bound, countries)
    # Each fossil cell uses its own state; both biosphere cells share the third.
    prolongation = np.array([[[1, 0, 0], [0, 1, 0]], [[0, 0, 1], [0, 0, 1]]])
    flux = basis.flux.transpose("source", "lat", "lon", "time").values
    weights = np.stack([np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])]) * countries.area_grid.values
    for group in ("prior", "posterior"):
        delta = trace[group].retained_draws.values - np.array([0.4, 1.7, 0.9])
        scaling = native_mean.values + np.einsum("sir,cdr->cdsi", prolongation, delta)[..., None, :]
        draws = scaling[..., None] * flux
        _assert_summary(native, f"flux_{group}", draws, ("native_source", "lat", "lon", "time"))
        _assert_summary(native, f"flux_total_{group}", draws.sum(axis=2), ("lat", "lon", "time"))
        totals = np.einsum("cdslit,kli->cdskt", draws, weights) * 44.01 * 365 * 24 * 3600
        _assert_summary(country, f"country_{group}", totals, ("native_source", "country", "time"))
        _assert_summary(country, f"country_total_{group}", totals.sum(axis=2), ("country", "time"))
    np.testing.assert_array_equal(native.time, time_scale.time)
    np.testing.assert_array_equal(country.time, time_scale.time)
