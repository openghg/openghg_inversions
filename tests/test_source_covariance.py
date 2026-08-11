"""Tests for ordered independent-source native covariance blocks."""

from dataclasses import is_dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    project_native_covariance,
)
from openghg_inversions.basis.operators import MultiSourceBucketBasisOperator
from openghg_inversions.native_covariance import SeparableExponentialCovariance
from openghg_inversions.source_covariance import IndependentSourceCovariance


def _problem():
    """Return a small non-lexically ordered gathered-source problem."""
    latitude = xr.DataArray(
        [-1.0, 0.5],
        dims="lat",
        coords={"lat": [-1.0, 0.5]},
        attrs={"units": "degrees_north"},
    )
    longitude = xr.DataArray(
        [10.0, 11.0],
        dims="lon",
        coords={"lon": [10.0, 11.0]},
        attrs={"units": "degrees_east"},
    )
    components = {
        "z-source": SeparableExponentialCovariance(
            latitude,
            longitude,
            sigma=1.6,
            latitude_correlation_length=0.7,
            longitude_correlation_length=2.2,
        ),
        "a-source": SeparableExponentialCovariance(latitude, longitude, sigma=0.7),
    }
    covariance = IndependentSourceCovariance(components)
    native_sensitivity = xr.DataArray(
        np.arange(24, dtype=float).reshape(3, 2, 2, 2) / 9.0,
        dims=("observation", "native_source", "lat", "lon"),
        coords={
            "observation": ["obs-1", "obs-2", "obs-3"],
            "native_source": ["z-source", "a-source"],
            "lat": latitude,
            "lon": longitude,
        },
    )
    return covariance, native_sensitivity


def _dense_block_diagonal(covariance: IndependentSourceCovariance) -> np.ndarray:
    """Construct the small dense independent-source oracle."""
    blocks = []
    for source in covariance.source_labels:
        component = covariance.source_covariances[source]
        lat = component.latitude.values
        lon = component.longitude.values
        k_lat = np.exp(-np.abs(lat[:, None] - lat[None, :]) / component.latitude_correlation_length)
        k_lon = np.exp(-np.abs(lon[:, None] - lon[None, :]) / component.longitude_correlation_length)
        blocks.append(component.sigma**2 * np.kron(k_lat, k_lon))
    result = np.zeros((sum(block.shape[0] for block in blocks),) * 2)
    offset = 0
    for block in blocks:
        stop = offset + block.shape[0]
        result[offset:stop, offset:stop] = block
        offset = stop
    return result


def _basis(covariance: IndependentSourceCovariance) -> MultiSourceBucketBasisOperator:
    """Construct the gathered bucket basis used only by Slice C product tests."""
    component = covariance.source_covariances["z-source"]
    latitude = component.latitude
    longitude = component.longitude
    return MultiSourceBucketBasisOperator(
        {
            "z-source": xr.DataArray(
                [[1, 1], [2, 2]],
                dims=("lat", "lon"),
                coords={"lat": latitude, "lon": longitude},
            ),
            "a-source": xr.DataArray(
                [[1, 1], [1, 1]],
                dims=("lat", "lon"),
                coords={"lat": latitude, "lon": longitude},
            ),
        },
        state_dim="state",
    )


def test_independent_source_action_and_serialization_preserve_order() -> None:
    """Block action, solve, and serialization retain configured non-lexical order."""
    covariance, native_sensitivity = _problem()
    rhs = native_sensitivity.transpose("native_source", "lat", "lon", "observation")
    dense = _dense_block_diagonal(covariance)
    matrix = rhs.values.reshape(8, 3)

    applied = covariance.apply(rhs)
    solved = covariance.solve(rhs)
    restored = IndependentSourceCovariance.from_dataset(covariance.to_dataset())

    assert covariance.source_labels == ("z-source", "a-source")
    np.testing.assert_allclose(applied.values.reshape(8, 3), dense @ matrix)
    np.testing.assert_allclose(solved.values.reshape(8, 3), np.linalg.solve(dense, matrix))
    xr.testing.assert_allclose(restored.apply(rhs), applied)
    component = restored.source_covariances["a-source"]
    replaced = SeparableExponentialCovariance(
        component.latitude,
        component.longitude,
        sigma=component.sigma,
        correlation_length=3.0,
        class_labels=component.class_labels,
    )
    assert replaced.latitude_correlation_length == 3.0
    assert replaced.longitude_correlation_length == 3.0


def test_constructor_validates_every_block_type_before_inspecting_dimensions() -> None:
    """Invalid blocks raise ``TypeError`` before any native dimensions are read."""
    covariance, _ = _problem()
    valid = covariance.source_covariances["z-source"]
    invalid = cast(SeparableExponentialCovariance, object())

    with pytest.raises(TypeError, match="bad.*SeparableExponentialCovariance"):
        IndependentSourceCovariance({"bad": invalid})

    with pytest.raises(TypeError, match="bad.*SeparableExponentialCovariance"):
        IndependentSourceCovariance(
            {"valid": valid, "bad": invalid},
            source_dim="lat",
        )

    with pytest.raises(ValueError, match="reserved"):
        IndependentSourceCovariance({"valid": valid}, source_dim="sigma")


def test_source_action_is_a_slotted_identity_object_with_read_only_configuration() -> None:
    """The computational action avoids dataclass replacement semantics."""
    covariance, _ = _problem()
    equivalent, _ = _problem()

    assert not is_dataclass(covariance)
    assert not hasattr(covariance, "__dict__")
    assert covariance != equivalent
    assert isinstance(hash(covariance), int)
    with pytest.raises(AttributeError):
        covariance.source_dim = "source"  # type: ignore[misc]


def test_source_schema_rejects_reserved_spatial_dimension() -> None:
    """Spatial dimensions cannot overwrite source configuration variables."""
    latitude = xr.DataArray([0.0, 1.0], dims="sigma")
    longitude = xr.DataArray([10.0, 11.0], dims="lon")
    component = SeparableExponentialCovariance(latitude, longitude)

    with pytest.raises(ValueError, match="spatial dimension.*reserved"):
        IndependentSourceCovariance({"source": component})


def test_source_blocks_require_identical_coordinate_metadata() -> None:
    """Source serialization cannot silently choose one block's grid metadata."""
    covariance, _ = _problem()
    original = covariance.source_covariances["a-source"]
    changed_latitude = original.latitude.copy(deep=True)
    changed_latitude.attrs["description"] = "different grid metadata"
    changed = SeparableExponentialCovariance(changed_latitude, original.longitude)

    with pytest.raises(ValueError, match="grid differs"):
        IndependentSourceCovariance({"first": original, "second": changed})


def test_action_rejects_numeric_source_label_matching_only_after_string_coercion() -> None:
    """A numeric source label cannot masquerade as the configured string label."""
    covariance, native_sensitivity = _problem()
    component = covariance.source_covariances["z-source"]
    numeric_named_covariance = IndependentSourceCovariance({"1": component})
    rhs = native_sensitivity.isel(native_source=[0]).assign_coords(native_source=[1])

    with pytest.raises(ValueError, match="source labels/order"):
        numeric_named_covariance.apply(rhs)


@pytest.mark.parametrize("spatial_dim", ["lat", "lon"])
def test_deserialization_rejects_missing_spatial_coordinates(spatial_dim: str) -> None:
    """Missing serialized spatial labels raise a clear validation error."""
    covariance, _ = _problem()
    dataset = covariance.to_dataset().drop_indexes(spatial_dim).drop_vars(spatial_dim)

    with pytest.raises(ValueError, match="missing latitude or longitude coordinates"):
        IndependentSourceCovariance.from_dataset(dataset)


def test_source_serialization_preserves_class_label_identity() -> None:
    """Class-blocked source round trips retain typed labels, name, and attributes."""
    covariance, _ = _problem()
    original_component = covariance.source_covariances["z-source"]
    class_labels = xr.DataArray(
        [[1, 2], [1, 2]],
        dims=("lat", "lon"),
        coords={"lat": original_component.latitude, "lon": original_component.longitude},
        name="surface_class",
        attrs={"description": "typed test classes", "version": np.int64(2)},
    )
    blocked_component = SeparableExponentialCovariance(
        original_component.latitude,
        original_component.longitude,
        sigma=original_component.sigma,
        latitude_correlation_length=original_component.latitude_correlation_length,
        longitude_correlation_length=original_component.longitude_correlation_length,
        class_labels=class_labels,
    )
    blocked = IndependentSourceCovariance(
        {
            "z-source": blocked_component,
            "a-source": covariance.source_covariances["a-source"],
        }
    )

    restored = IndependentSourceCovariance.from_dataset(blocked.to_dataset())

    restored_labels = restored.source_covariances["z-source"].class_labels
    assert restored_labels is not None
    xr.testing.assert_identical(restored_labels, class_labels.assign_attrs(version=2))


def test_source_serialization_coordinates_do_not_alias_component_state() -> None:
    """Mutating an export cannot invalidate a component's cached factors."""
    covariance, _ = _problem()
    component = covariance.source_covariances["z-source"]
    original_latitude = component.latitude.to_numpy().copy()

    dataset = covariance.to_dataset()
    dataset.coords["lat"].values[0] = 999.0

    np.testing.assert_array_equal(component.latitude, original_latitude)


def test_source_deserialization_rejects_contradictory_implicit_length() -> None:
    """Fallback-derived source lengths must match their serialized fallback."""
    covariance, _ = _problem()
    dataset = covariance.to_dataset()
    dataset["latitude_correlation_length"].loc[{covariance.source_dim: "a-source"}] = 99.0

    with pytest.raises(ValueError, match="Implicit latitude.*contradicts"):
        IndependentSourceCovariance.from_dataset(dataset)


def test_source_operations_preserve_auxiliary_source_coordinates() -> None:
    """Block dispatch restores semantic coordinates attached to the source axis."""
    covariance, native_sensitivity = _problem()
    rhs = native_sensitivity.transpose("native_source", "lat", "lon", "observation").assign_coords(
        inventory=("native_source", ["bottom-up", "top-down"])
    )

    applied = covariance.apply(rhs)
    solved = covariance.solve(rhs)

    xr.testing.assert_identical(applied.coords["inventory"], rhs.coords["inventory"])
    xr.testing.assert_identical(solved.coords["inventory"], rhs.coords["inventory"])


def test_deserialization_rejects_unknown_class_label_encoding() -> None:
    """Source covariance schema readers reject unknown typed-label codecs."""
    covariance, _ = _problem()
    dataset = covariance.to_dataset()
    dataset.attrs["class_label_encoding"] = "future_codec"

    with pytest.raises(ValueError, match="class-label encoding"):
        IndependentSourceCovariance.from_dataset(dataset)


def test_deserialization_rejects_contradictory_source_blocked_state() -> None:
    """A source flag cannot silently discard an existing encoded class grid."""
    covariance, _ = _problem()
    component = covariance.source_covariances["z-source"]
    labels = xr.DataArray(
        [[1, 2], [1, 2]],
        dims=("lat", "lon"),
        coords={"lat": component.latitude, "lon": component.longitude},
    )
    blocked = IndependentSourceCovariance(
        {
            "z-source": SeparableExponentialCovariance(
                component.latitude,
                component.longitude,
                sigma=component.sigma,
                correlation_length=component.correlation_length,
                latitude_correlation_length=component.latitude_correlation_length,
                longitude_correlation_length=component.longitude_correlation_length,
                class_labels=labels,
            ),
            "a-source": covariance.source_covariances["a-source"],
        }
    )
    dataset = blocked.to_dataset()
    dataset["class_blocked"].loc[{blocked.source_dim: "z-source"}] = False

    with pytest.raises(ValueError, match="contradicts"):
        IndependentSourceCovariance.from_dataset(dataset)


def test_gathered_source_products_match_block_diagonal_dense_oracle() -> None:
    """Ragged source states keep insertion order and zero cross-source covariance."""
    covariance, native_sensitivity = _problem()
    basis = _basis(covariance)
    products = project_native_covariance(
        covariance=covariance,
        basis_operator=basis,
        native_sensitivity=native_sensitivity,
        observation_dim="observation",
        observation_batch_size=1,
    )
    dense_b = _dense_block_diagonal(covariance)
    h = native_sensitivity.values.reshape(3, 8)

    base = basis.basis_matrix.transpose("lat", "lon", "state").compute()
    base_values = np.asarray(base.data.todense())
    state_sources = np.asarray(base.coords["source"].values)
    u = np.zeros((2, 2, 2, base.sizes["state"]))
    for source_index, source in enumerate(covariance.source_labels):
        u[source_index] = base_values * (state_sources == source)[None, None, :]
    u = u.reshape(8, base.sizes["state"])
    c_alpha = np.linalg.inv(u.T @ np.linalg.solve(dense_b, u))
    pi = c_alpha @ np.linalg.solve(dense_b, u).T

    assert products.state_covariance.coords["source"].values.tolist() == [
        "z-source",
        "z-source",
        "a-source",
    ]
    np.testing.assert_allclose(products.restriction.values.reshape(3, 8), pi, atol=1e-11)
    np.testing.assert_allclose(products.state_covariance, c_alpha, atol=1e-11)
    np.testing.assert_allclose(products.effective_observation_operator, h @ u, atol=1e-11)
    np.testing.assert_allclose(
        products.observation_state_cross_covariance,
        h @ dense_b @ pi.T,
        atol=1e-11,
    )
    np.testing.assert_allclose(products.native_observation_covariance, h @ dense_b @ h.T, atol=1e-11)
    np.testing.assert_allclose(pi @ u, np.eye(3), atol=1e-11)


def test_gathered_source_product_datatree_persists_multiindex(tmp_path: Path) -> None:
    """Ragged state labels and covariance configuration survive NetCDF persistence."""
    covariance, native_sensitivity = _problem()
    observation_index = pd.MultiIndex.from_arrays(
        [["observed-a", "observed-b", "observed-c"], [1, 2, 3]],
        names=("source", "observation_id"),
    )
    native_sensitivity = native_sensitivity.drop_indexes("observation").drop_vars("observation")
    native_sensitivity = native_sensitivity.assign_coords(
        xr.Coordinates.from_pandas_multiindex(observation_index, "observation")
    )
    native_sensitivity = native_sensitivity.assign_coords(
        source_cov=("observation", ["aux-a", "aux-b", "aux-c"])
    )
    products = project_native_covariance(
        covariance=covariance,
        basis_operator=_basis(covariance),
        native_sensitivity=native_sensitivity,
        observation_dim="observation",
    )
    path = tmp_path / "source-products.nc"

    products.to_datatree().to_netcdf(path, engine="h5netcdf")
    with xr.open_datatree(path, engine="h5netcdf") as stored:
        restored = NativeCovarianceProducts.from_datatree(stored.load())

    assert restored.state_covariance.coords["source"].values.tolist() == [
        "z-source",
        "z-source",
        "a-source",
    ]
    assert restored.covariance_configuration is not None
    assert products.covariance_configuration is not None
    xr.testing.assert_identical(restored.covariance_configuration, products.covariance_configuration)
    xr.testing.assert_allclose(restored.state_covariance, products.state_covariance)
    np.testing.assert_array_equal(
        restored.native_observation_covariance.coords["observation_source"],
        native_sensitivity.coords["source"],
    )
    np.testing.assert_array_equal(
        restored.native_observation_covariance.coords["observation_source_cov"],
        native_sensitivity.coords["source_cov"],
    )
    assert restored.native_observation_covariance.indexes["observation"].names == [
        "observation_source",
        "observation_id",
    ]


def test_gathered_source_product_requires_multiindex_restoration_metadata() -> None:
    """Expanded ragged-state coordinates cannot load silently as a plain index."""
    from openghg_inversions.serialization import MULTIINDEX_DIMS_ATTR

    covariance, native_sensitivity = _problem()
    tree = project_native_covariance(
        covariance=covariance,
        basis_operator=_basis(covariance),
        native_sensitivity=native_sensitivity,
        observation_dim="observation",
    ).to_datatree()
    root = tree.to_dataset(inherit=False).copy(deep=True)
    root.attrs.pop(MULTIINDEX_DIMS_ATTR)
    malformed = xr.DataTree(root)
    malformed["covariance_configuration"] = xr.DataTree(
        cast(xr.DataTree, tree["covariance_configuration"]).to_dataset(inherit=False).copy(deep=True)
    )

    with pytest.raises(ValueError, match="restore MultiIndex"):
        NativeCovarianceProducts.from_datatree(malformed)
