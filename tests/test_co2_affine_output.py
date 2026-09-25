"""Binding and producer checks for coherent CO2 affine output reconstruction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions._labelled_matrices import renamed_column_coordinates
from openghg_inversions.basis.affine_flux_map_io import save
from openghg_inversions.basis.basis_functions import BasisFunctions
from openghg_inversions.inversion_data import RhimePreparedInputs
from openghg_inversions.rhime.co2.co2_affine_output import (
    _bind_affine_flux_map,
    import_explicit_affine_flux_map,
    load_and_bind_affine_flux_map,
    prepared_inputs_content_id,
    produce_bucket_affine_flux_map,
)
from openghg_inversions.rhime.co2.co2_preparation import Co2PreparedInputs


def _prepared() -> tuple[Co2PreparedInputs, xr.DataArray]:
    measurements = pd.MultiIndex.from_arrays(
        [["MHD"], pd.to_datetime(["2020-01-01"])], names=("site", "time")
    )
    coords = xr.Coordinates.from_pandas_multiindex(measurements, "nmeasure")
    grid = {"lat": [51.0], "lon": [-2.0, -1.0]}
    basis_flat = xr.DataArray([[1, 2]], dims=("lat", "lon"), coords=grid)
    flux = xr.DataArray(
        [[-2.0, 3.0]],
        dims=("lat", "lon"),
        coords=grid,
        attrs={"units": "mol m-2 s-1"},
    )
    basis = BasisFunctions.from_flat_basis(
        basis_flat,
        flux,
        operator_kwargs={"state_dim": "region"},
    )
    state = basis.operator.basis_matrix["region"]
    inputs = xr.Dataset(
        {
            "H": (("nmeasure", "region"), [[0.3, 0.7]]),
            "alpha_prior_mean": ("region", [0.4, 1.7]),
            "alpha_prior_covariance": (("region", "region_cov"), [[1.0, 0.0], [0.0, 1.0]]),
            "fixed_prior_contribution": ("nmeasure", [400.0]),
            "mf": ("nmeasure", [401.0]),
            "mf_error": ("nmeasure", [0.1]),
            "low_rank_factor": (("nmeasure", "agg_rank"), [[0.2]]),
            "diagonal_residual_variance": ("nmeasure", [0.01]),
        },
        coords={**coords, "region": state, "region_cov": state.values, "agg_rank": [0]},
    )
    for name in ("H", "fixed_prior_contribution", "mf", "mf_error", "low_rank_factor"):
        inputs[name].attrs["units"] = "ppm"
    for name in ("alpha_prior_mean", "alpha_prior_covariance"):
        inputs[name].attrs["units"] = "1"
    inputs["diagonal_residual_variance"].attrs["units"] = "ppm**2"
    site_metadata = xr.Dataset({"averaging_period": ("site", ["1h"])}, coords={"site": ["MHD"]})
    prepared = Co2PreparedInputs(
        RhimePreparedInputs(inputs, basis, site_metadata),
        aggregation_error_mode="low_rank",
        provenance={"projection_strategy": "fixture-restriction"},
    )
    native_mean = xr.DataArray([[0.8, 1.4]], dims=("lat", "lon"), coords=grid, attrs={"units": "1"})
    return prepared, native_mean


def _explicit_bucket(prepared: Co2PreparedInputs, native_mean: xr.DataArray) -> xr.DataArray:
    return xr.DataArray(
        np.eye(2).reshape(1, 2, 2),
        dims=("lat", "lon", "region"),
        coords={"lat": native_mean.lat, "lon": native_mean.lon, "region": prepared.inv_inputs.region},
        attrs={"units": "1"},
    )


def _multisource_prepared() -> tuple[Co2PreparedInputs, xr.DataArray]:
    grid = {"lat": [51.0], "lon": [-2.0, -1.0]}
    fossil = xr.DataArray([[1, 2]], dims=("lat", "lon"), coords=grid)
    bio = xr.DataArray([[1, 1]], dims=("lat", "lon"), coords=grid)
    fluxes = {
        "fossil": xr.DataArray([[-2.0, -3.0]], dims=fossil.dims, coords=grid, attrs={"units": "mol m-2 s-1"}),
        "bio": xr.DataArray([[4.0, 5.0]], dims=bio.dims, coords=grid, attrs={"units": "mol m-2 s-1"}),
    }
    basis = BasisFunctions.from_multi_source_flat_basis(
        {"fossil": fossil, "bio": bio}, fluxes, operator_kwargs={"state_dim": "region"}
    )
    state_index = basis.operator.basis_matrix.indexes["region"]
    mean = xr.DataArray([0.4, 1.7, 0.9], dims="region", coords={"region": state_index}, attrs={"units": "1"})
    covariance = xr.DataArray(
        np.eye(3),
        dims=("region", "region_cov"),
        coords={**mean.coords, **renamed_column_coordinates(mean, row_dim="region", column_dim="region_cov")},
        attrs={"units": "1"},
    )
    observations = pd.MultiIndex.from_arrays(
        [["MHD"], pd.to_datetime(["2020-01-01"])], names=("site", "time")
    )
    observation_coords = xr.Coordinates.from_pandas_multiindex(observations, "nmeasure")
    inputs = xr.Dataset(
        {
            "H": xr.DataArray(
                [[0.3, 0.7, 0.1]],
                dims=("nmeasure", "region"),
                coords={**observation_coords, **mean.coords},
                attrs={"units": "ppm"},
            ),
            "alpha_prior_mean": mean,
            "alpha_prior_covariance": covariance,
            "fixed_prior_contribution": ("nmeasure", [400.0]),
            "mf": ("nmeasure", [401.0]),
            "mf_error": ("nmeasure", [0.1]),
            "low_rank_factor": (("nmeasure", "agg_rank"), [[0.2]]),
            "diagonal_residual_variance": ("nmeasure", [0.01]),
        },
        coords={**observation_coords, "agg_rank": [0]},
    )
    for name in ("fixed_prior_contribution", "mf", "mf_error", "low_rank_factor"):
        inputs[name].attrs["units"] = "ppm"
    inputs["diagonal_residual_variance"].attrs["units"] = "ppm**2"
    prepared = Co2PreparedInputs(
        RhimePreparedInputs(
            inputs,
            basis,
            xr.Dataset({"averaging_period": ("site", ["1h"])}, coords={"site": ["MHD"]}),
        ),
        aggregation_error_mode="low_rank",
        provenance={"projection_strategy": "bucket-preserving"},
    )
    native_mean = xr.DataArray(
        [[[0.8, 1.4]], [[1.2, 0.7]]],
        dims=("native_source", "lat", "lon"),
        coords={"native_source": ["fossil", "bio"], **grid},
        attrs={"units": "1"},
    )
    return prepared, native_mean


def test_bucket_producer_reuses_operator_and_authoritative_reference() -> None:
    """Bucket output follows the exact centred equation with signed flux."""
    prepared, native_mean = _prepared()
    artifact = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="fixture")
    assert artifact.affine_map.prolongation is prepared.basis_functions.operator
    bound = _bind_affine_flux_map(artifact, prepared, prepared_inputs_id="fixture")
    state = xr.DataArray(
        [[0.9, 1.2], [0.3, 2.0]],
        dims=("draw", "region"),
        coords={"draw": [0, 1], "region": prepared.inv_inputs.region},
        attrs={"units": "1"},
    )
    native = bound.state_to_native(state).transpose("lat", "lon", "draw")
    expected = np.asarray([[[1.3, 0.7], [0.9, 1.7]]])
    np.testing.assert_allclose(native, expected)
    np.testing.assert_allclose(
        bound.state_to_flux(state).transpose("lat", "lon", "draw"),
        prepared.basis_functions.flux.values[..., None] * expected,
    )
    assert native.attrs["uncertainty_scope"] == "retained_state_conditional"


def test_multisource_bucket_producer_renames_flux_source_without_padding(tmp_path) -> None:
    prepared, native_mean = _multisource_prepared()
    assert prepared.basis_functions.flux.dims[0] == "source"
    prepared_path = tmp_path / "multisource-prepared.nc"
    reconstruction_path = tmp_path / "multisource-affine.nc"
    prepared.save(prepared_path)
    artifact = produce_bucket_affine_flux_map(
        prepared, native_mean, prepared_inputs_id=prepared_inputs_content_id(prepared_path)
    )
    assert artifact.affine_map.flux.dims[0] == "native_source"
    save(artifact, reconstruction_path)
    bound = load_and_bind_affine_flux_map(reconstruction_path, prepared_path)
    reconstructed = bound.state_to_native(prepared.inv_inputs["alpha_prior_mean"])
    xr.testing.assert_allclose(reconstructed, native_mean.rename("native_scaling"))
    assert "source" not in reconstructed.dims
    np.testing.assert_allclose(
        bound.state_to_flux(prepared.inv_inputs["alpha_prior_mean"]),
        native_mean * artifact.affine_map.flux,
    )


def test_supplied_restriction_import_keeps_exact_non_bucket_prolongation(tmp_path) -> None:
    """Verification Games-style supplied U* need not equal the bucket lift."""
    prepared, native_mean = _prepared()
    prolongation = xr.DataArray(
        [[[0.8, 0.2], [-0.1, 1.1]]],
        dims=("lat", "lon", "region"),
        coords={"lat": native_mean.lat, "lon": native_mean.lon, "region": prepared.inv_inputs.region},
        attrs={"units": "1"},
    )
    prepared_path = tmp_path / "prepared.nc"
    reconstruction_path = tmp_path / "explicit.nc"
    prepared.save(prepared_path)
    identity = prepared_inputs_content_id(prepared_path)
    artifact = import_explicit_affine_flux_map(
        prepared,
        native_mean,
        prepared.basis_functions.flux,
        prolongation,
        prepared_inputs_id=identity,
        reference_state=prepared.inv_inputs["alpha_prior_mean"],
        reconstruction_provenance={"restriction": "verification-games-fixture"},
    )
    assert artifact.affine_map.prolongation is prolongation
    save(artifact, reconstruction_path)
    bound = load_and_bind_affine_flux_map(reconstruction_path, prepared_path)
    assert bound.artifact.affine_map.representation == "explicit"
    xr.testing.assert_identical(bound.artifact.affine_map.prolongation, prolongation.rename("prolongation"))
    assert bound.artifact.reconstruction_provenance["restriction"] == "verification-games-fixture"
    state = xr.DataArray(
        [1.4, 1.2],
        dims="region",
        coords={"region": prepared.inv_inputs.region},
        attrs={"units": "1"},
    )
    expected = np.asarray([[1.5, 0.75]])
    np.testing.assert_allclose(bound.state_to_native(state), expected)
    np.testing.assert_allclose(bound.state_to_flux(state), expected * prepared.basis_functions.flux.values)
    assert not np.array_equal(prolongation.values.reshape(2, 2), np.eye(2))


def test_binding_rejects_identity_projection_coordinates_and_flux_units() -> None:
    prepared, native_mean = _prepared()
    artifact = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id="fixture")
    with pytest.raises(ValueError, match="content identity"):
        _bind_affine_flux_map(artifact, prepared, prepared_inputs_id="stale")

    altered = Co2PreparedInputs(
        prepared.rhime_inputs,
        prepared.aggregation_error_mode,
        provenance={"projection_strategy": "different"},
    )
    with pytest.raises(ValueError, match="projection provenance"):
        _bind_affine_flux_map(artifact, altered, prepared_inputs_id="fixture")
    with pytest.raises(ValueError, match="projection provenance"):
        _bind_affine_flux_map(
            replace(
                artifact,
                projection_provenance={"projection_strategy": "fixture-restriction", "unverified": "other"},
            ),
            prepared,
            prepared_inputs_id="fixture",
        )

    with pytest.raises(ValueError, match="labels"):
        produce_bucket_affine_flux_map(
            prepared, native_mean.assign_coords(lon=[-1.0, -2.0]), prepared_inputs_id="fixture"
        )
    with pytest.raises(ValueError, match="flux units"):
        import_explicit_affine_flux_map(
            prepared,
            native_mean,
            prepared.basis_functions.flux.assign_attrs(units="kg"),
            _explicit_bucket(prepared, native_mean),
            prepared_inputs_id="fixture",
        )
    with pytest.raises(ValueError, match="flux dimensions"):
        import_explicit_affine_flux_map(
            prepared,
            native_mean,
            prepared.basis_functions.flux.expand_dims(time=["2020-01"]),
            _explicit_bucket(prepared, native_mean),
            prepared_inputs_id="fixture",
        )


def test_import_rejects_conflicting_reference_state() -> None:
    prepared, native_mean = _prepared()
    prolongation = _explicit_bucket(prepared, native_mean)
    with pytest.raises(ValueError, match="Imported reference_state"):
        import_explicit_affine_flux_map(
            prepared,
            native_mean,
            prepared.basis_functions.flux,
            prolongation,
            prepared_inputs_id="fixture",
            reference_state=prepared.inv_inputs["alpha_prior_mean"] + 1,
        )


@pytest.mark.parametrize("suffix", ["nc", "zarr"])
def test_public_load_and_bind_checks_saved_prepared_content(tmp_path, suffix: str) -> None:
    prepared, native_mean = _prepared()
    prepared_path = tmp_path / f"prepared.{suffix}"
    reconstruction_path = tmp_path / f"affine.{suffix}"
    prepared.save(prepared_path)
    identity = prepared_inputs_content_id(prepared_path)
    artifact = produce_bucket_affine_flux_map(prepared, native_mean, prepared_inputs_id=identity)
    save(artifact, reconstruction_path)
    bound = load_and_bind_affine_flux_map(reconstruction_path, prepared_path)
    assert bound.artifact.prepared_inputs_id == identity
    np.testing.assert_allclose(bound.state_to_native(prepared.inv_inputs["alpha_prior_mean"]), native_mean)
    changed_inputs = prepared.inv_inputs.copy(deep=True)
    changed_inputs["mf"].data[0] = 402.0
    changed = Co2PreparedInputs(
        RhimePreparedInputs(changed_inputs, prepared.basis_functions, prepared.site_metadata),
        prepared.aggregation_error_mode,
        prepared.provenance,
    )
    changed_path = tmp_path / f"changed-prepared.{suffix}"
    changed.save(changed_path)
    with pytest.raises(ValueError, match="content identity"):
        load_and_bind_affine_flux_map(reconstruction_path, changed_path)
