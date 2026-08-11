"""Tests for covariance products that preserve bucket prolongation semantics."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openghg_inversions.basis.covariance_products import (
    NativeCovarianceProducts,
    PreserveBucketProlongation,
    RetainedProjection,
    RetainedProjectionStrategy,
    project_native_covariance,
)
from openghg_inversions.basis.operators import BucketBasisOperator
from openghg_inversions.native_covariance import SeparableExponentialCovariance


def _problem() -> tuple[
    SeparableExponentialCovariance,
    BucketBasisOperator,
    xr.DataArray,
    np.ndarray,
]:
    """Return a labelled small-grid problem and its dense native covariance."""
    latitude = xr.DataArray(
        [-1.0, 0.5, 2.0],
        dims="lat",
        coords={"lat": [-1.0, 0.5, 2.0]},
        attrs={"units": "degrees_north"},
    )
    longitude = xr.DataArray(
        [10.0, 11.0],
        dims="lon",
        coords={"lon": [10.0, 11.0]},
        attrs={"units": "degrees_east"},
    )
    covariance = SeparableExponentialCovariance(
        latitude=latitude,
        longitude=longitude,
        sigma=1.4,
        correlation_length=0.9,
    )
    basis_flat = xr.DataArray(
        [[1, 1], [1, 2], [2, 2]],
        dims=("lat", "lon"),
        coords={"lat": latitude, "lon": longitude},
        name="basis_flat",
    )
    basis_operator = BucketBasisOperator(basis_flat, state_dim="state")
    native_sensitivity = xr.DataArray(
        np.array(
            [
                [[0.2, -0.1], [0.4, 0.7], [0.3, -0.2]],
                [[-0.3, 0.8], [0.1, -0.4], [0.9, 0.5]],
                [[0.6, 0.2], [-0.7, 0.3], [0.2, 0.1]],
            ]
        ),
        dims=("observation", "lat", "lon"),
        coords={
            "observation": ["MHD-1", "TAC-1", "RGL-1"],
            "lat": latitude,
            "lon": longitude,
        },
        name="native_sensitivity",
    )
    k_lat = np.exp(-np.abs(latitude.values[:, np.newaxis] - latitude.values[np.newaxis, :]) / 0.9)
    k_lon = np.exp(-np.abs(longitude.values[:, np.newaxis] - longitude.values[np.newaxis, :]) / 0.9)
    dense_b = 1.4**2 * np.kron(k_lat, k_lon)
    return covariance, basis_operator, native_sensitivity, dense_b


def _dense_operators(
    basis_operator: BucketBasisOperator,
    dense_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense-oracle prolongation, restriction, and retained covariance."""
    matrix = basis_operator.basis_matrix.transpose("lat", "lon", "state").compute()
    u = np.asarray(matrix.data.todense()).reshape(6, 2)
    precision_u = np.linalg.solve(dense_b, u)
    c_alpha = np.linalg.inv(u.T @ precision_u)
    restriction = c_alpha @ precision_u.T
    return u, restriction, c_alpha


def _project(
    covariance: SeparableExponentialCovariance,
    basis_operator: BucketBasisOperator,
    native_sensitivity: xr.DataArray,
    **options,
) -> NativeCovarianceProducts:
    """Project the shared problem with its canonical observation dimension."""
    return project_native_covariance(
        covariance=covariance,
        basis_operator=basis_operator,
        native_sensitivity=native_sensitivity,
        observation_dim="observation",
        **options,
    )


def test_preserve_bucket_strategy_derives_compatible_restriction() -> None:
    """The derived restriction satisfies Pi_U U_bucket = identity with stable labels."""
    covariance, basis_operator, _, dense_b = _problem()
    u, expected_pi, _ = _dense_operators(basis_operator, dense_b)

    projection = PreserveBucketProlongation().projection(
        covariance,
        basis_operator.basis_matrix,
        native_dims=covariance.native_dims,
        state_dim="state",
    )

    assert projection.strategy == "preserve_bucket_prolongation"
    assert projection.prolongation.dims == ("lat", "lon", "state")
    assert projection.restriction.dims == ("state", "lat", "lon")
    assert projection.restriction.state.values.tolist() == [0, 1]
    np.testing.assert_allclose(
        projection.restriction.values.reshape(2, 6), expected_pi, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        projection.restriction.values.reshape(2, 6) @ u,
        np.eye(2),
        rtol=1e-11,
        atol=1e-11,
    )


def test_products_match_dense_oracle_and_coherent_forward_model() -> None:
    """C_alpha, H_alpha, H B Pi.T, and H B H.T match dense coherent oracles."""
    covariance, basis_operator, h, dense_b = _problem()
    u, expected_pi, expected_c_alpha = _dense_operators(basis_operator, dense_b)
    h_matrix = h.values.reshape(3, 6)

    products = _project(covariance, basis_operator, h)

    expected_h_alpha = h_matrix @ u
    expected_hb_pi_t = h_matrix @ dense_b @ expected_pi.T
    expected_hbht = h_matrix @ dense_b @ h_matrix.T
    np.testing.assert_allclose(products.state_covariance, expected_c_alpha, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        products.effective_observation_operator, expected_h_alpha, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        products.observation_state_cross_covariance,
        expected_hb_pi_t,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(products.native_observation_covariance, expected_hbht, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(
        products.observation_state_cross_covariance,
        expected_h_alpha @ expected_c_alpha,
        rtol=1e-11,
        atol=1e-11,
    )
    assert products.state_covariance.dims == ("state", "state_cov")
    assert products.effective_observation_operator.dims == ("observation", "state")
    assert products.observation_state_cross_covariance.dims == ("observation", "state_cov")
    assert products.native_observation_covariance.dims == ("observation", "observation_cov")
    assert products.native_observation_covariance.observation.values.tolist() == [
        "MHD-1",
        "TAC-1",
        "RGL-1",
    ]
    assert products.native_observation_covariance.observation_cov.values.tolist() == [
        "MHD-1",
        "TAC-1",
        "RGL-1",
    ]


def test_covariance_natural_prolongation_is_the_bucket_prolongation() -> None:
    """The compatible Pi makes B Pi.T C_alpha^-1 exactly equal U_bucket."""
    covariance, basis_operator, h, dense_b = _problem()
    u, _, _ = _dense_operators(basis_operator, dense_b)
    products = _project(covariance, basis_operator, h)
    pi = products.restriction.values.reshape(2, 6)

    u_star = dense_b @ pi.T @ np.linalg.inv(products.state_covariance.values)

    np.testing.assert_allclose(u_star, u, rtol=1e-11, atol=1e-11)


def test_dense_and_diagonal_products_are_batch_invariant() -> None:
    """Execution batch size changes neither covariance products nor their identity."""
    covariance, basis_operator, h, _ = _problem()

    full_batch = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="dense",
        observation_batch_size=64,
    )
    single_observation_batches = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="dense",
        observation_batch_size=1,
    )
    diagonal = _project(
        covariance,
        basis_operator,
        h,
        observation_covariance="diagonal",
        observation_batch_size=2,
    )

    for batched, full in (
        (single_observation_batches.restriction, full_batch.restriction),
        (single_observation_batches.prolongation, full_batch.prolongation),
        (single_observation_batches.state_covariance, full_batch.state_covariance),
        (
            single_observation_batches.effective_observation_operator,
            full_batch.effective_observation_operator,
        ),
        (
            single_observation_batches.observation_state_cross_covariance,
            full_batch.observation_state_cross_covariance,
        ),
        (
            single_observation_batches.native_observation_covariance,
            full_batch.native_observation_covariance,
        ),
    ):
        xr.testing.assert_allclose(batched, full)
    assert single_observation_batches.content_identity == full_batch.content_identity
    assert diagonal.native_observation_covariance.dims == ("observation",)
    expected_diagonal = xr.DataArray(
        np.diag(full_batch.native_observation_covariance.values),
        dims="observation",
        coords={"observation": h.observation},
        name="native_observation_covariance",
    )
    np.testing.assert_allclose(diagonal.native_observation_covariance, expected_diagonal)


def test_products_round_trip_with_labels_and_identity() -> None:
    """Dataset serialization preserves all blocks, labels, strategy, and shared identity."""
    covariance, basis_operator, h, _ = _problem()
    products = _project(
        covariance,
        basis_operator,
        h,
        observation_batch_size=2,
    )

    restored = NativeCovarianceProducts.from_dataset(products.to_dataset())

    xr.testing.assert_identical(restored.restriction, products.restriction)
    xr.testing.assert_identical(restored.state_covariance, products.state_covariance)
    xr.testing.assert_identical(
        restored.effective_observation_operator, products.effective_observation_operator
    )
    xr.testing.assert_identical(
        restored.observation_state_cross_covariance,
        products.observation_state_cross_covariance,
    )
    xr.testing.assert_identical(
        restored.native_observation_covariance, products.native_observation_covariance
    )
    assert restored.strategy == products.strategy
    assert restored.content_identity == products.content_identity
    assert len(products.content_identity) == 64
    assert all(
        array.attrs["content_identity"] == products.content_identity
        for array in (
            products.restriction,
            products.state_covariance,
            products.effective_observation_operator,
            products.observation_state_cross_covariance,
            products.native_observation_covariance,
        )
    )

    restored_tree = NativeCovarianceProducts.from_datatree(products.to_datatree())
    assert restored_tree.covariance_configuration is not None
    assert products.covariance_configuration is not None
    xr.testing.assert_identical(
        restored_tree.covariance_configuration,
        products.covariance_configuration,
    )
    assert restored_tree.basis_provenance == products.basis_provenance
    assert restored_tree.prolongation.attrs["content_identity"] == products.content_identity


def test_content_identity_changes_with_covariance_configuration() -> None:
    """Kernel amplitudes and class maps participate in the stable product identity."""
    covariance, basis_operator, h, _ = _problem()
    baseline = _project(covariance, basis_operator, h)
    rescaled = _project(
        SeparableExponentialCovariance(
            covariance.latitude,
            covariance.longitude,
            sigma=2.0,
            correlation_length=covariance.correlation_length,
        ),
        basis_operator,
        h,
    )
    classes = xr.DataArray(
        [[0, 1], [0, 1], [1, 1]],
        dims=("lat", "lon"),
        coords={"lat": covariance.latitude, "lon": covariance.longitude},
    )
    blocked = _project(
        SeparableExponentialCovariance(
            covariance.latitude,
            covariance.longitude,
            sigma=covariance.sigma,
            correlation_length=covariance.correlation_length,
            class_labels=classes,
        ),
        basis_operator,
        h,
    )

    assert baseline.content_identity != rescaled.content_identity
    assert baseline.content_identity != blocked.content_identity


def test_product_deserialization_rejects_mixed_content_identity() -> None:
    """A block copied from another artifact cannot masquerade as coherent output."""
    covariance, basis_operator, h, _ = _problem()
    dataset = _project(covariance, basis_operator, h).to_dataset()
    dataset["state_covariance"].attrs["content_identity"] = "0" * 64

    with pytest.raises(ValueError, match="content identity"):
        NativeCovarianceProducts.from_dataset(dataset)


def test_projection_strategy_must_return_covariance_natural_prolongation() -> None:
    """Custom strategies cannot return a Pi/U pair that violates B Pi.T = U C_alpha."""
    covariance, basis_operator, h, _ = _problem()

    class IncoherentStrategy:
        """Perturb an otherwise compatible prolongation to violate the invariant."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a deliberately incoherent restriction/prolongation pair."""
            valid = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            return RetainedProjection(valid.restriction, 2.0 * valid.prolongation, "incoherent")

    for sigma in (covariance.sigma, 1e-8):
        scaled = SeparableExponentialCovariance(
            covariance.latitude,
            covariance.longitude,
            sigma=sigma,
            correlation_length=covariance.correlation_length,
        )
        with pytest.raises(ValueError, match="incoherent|B Pi"):
            _project(scaled, basis_operator, h, strategy=IncoherentStrategy())


def test_projection_strategy_protocol_accepts_structural_implementations() -> None:
    """A compatible strategy is accepted without inheriting from the public protocol."""
    covariance, basis_operator, h, _ = _problem()

    class DelegatingStrategy:
        """Delegate projection construction while exposing a distinct strategy name."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a coherent projection using the established bucket strategy."""
            projection = PreserveBucketProlongation().projection(
                covariance,
                basis_prolongation,
                native_dims=native_dims,
                state_dim=state_dim,
            )
            return RetainedProjection(
                projection.restriction,
                projection.prolongation,
                "delegating_strategy",
            )

    strategy: RetainedProjectionStrategy = DelegatingStrategy()

    products = _project(covariance, basis_operator, h, strategy=strategy)

    assert products.strategy == "delegating_strategy"


def test_projection_strategy_must_return_full_rank_restriction() -> None:
    """A formally compatible zero Pi/U pair is rejected because C_alpha is singular."""
    covariance, basis_operator, h, _ = _problem()

    class SingularStrategy:
        """Return a zero restriction and prolongation with the expected dimensions."""

        def projection(self, covariance, basis_prolongation, *, native_dims, state_dim):
            """Return a deliberately singular retained projection."""
            return RetainedProjection(
                xr.zeros_like(basis_prolongation.transpose(state_dim, *native_dims)),
                xr.zeros_like(basis_prolongation),
                "singular",
            )

    with pytest.raises(ValueError, match="positive definite|full rank"):
        _project(covariance, basis_operator, h, strategy=SingularStrategy())


def test_content_identity_includes_auxiliary_semantic_coordinates() -> None:
    """Relabelling an auxiliary observation coordinate changes artifact identity."""
    covariance, basis_operator, h, _ = _problem()
    first = _project(
        covariance,
        basis_operator,
        h.assign_coords(network=("observation", ["ICOS", "DECC", "DECC"])),
    )
    second = _project(
        covariance,
        basis_operator,
        h.assign_coords(network=("observation", ["DECC", "DECC", "DECC"])),
    )

    assert first.content_identity != second.content_identity


def test_redundant_bucket_states_are_rejected() -> None:
    """Linearly dependent retained coordinates fail instead of being pseudoinverted."""
    covariance, basis_operator, _, _ = _problem()
    prolongation = basis_operator.basis_matrix.compute()
    redundant = xr.concat(
        [
            prolongation.sel(state=0),
            prolongation.sel(state=0),
        ],
        dim=xr.IndexVariable("state", ["west", "west-copy"]),
    ).transpose("lat", "lon", "state")

    with pytest.raises(ValueError, match="positive definite|redundant|rank"):
        PreserveBucketProlongation().projection(
            covariance,
            redundant,
            native_dims=covariance.native_dims,
            state_dim="state",
        )


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        pytest.param(
            lambda h: h.assign_coords(lon=h.lon[::-1]),
            "coordinate|align|longitude|lon",
            id="misordered-native-coordinate",
        ),
        pytest.param(
            lambda h: h.where(h.observation != "TAC-1"),
            "finite|NaN",
            id="non-finite-sensitivity",
        ),
    ],
)
def test_products_reject_invalid_native_sensitivity(transform, message: str) -> None:
    """Product construction rejects misaligned or non-finite native sensitivity."""
    covariance, basis_operator, h, _ = _problem()

    with pytest.raises(ValueError, match=message):
        _project(covariance, basis_operator, transform(h))


def test_class_blocked_products_match_dense_oracle() -> None:
    """The compatible restriction and all products remain coherent with class masks."""
    covariance, basis_operator, h, dense_unblocked = _problem()
    labels = xr.DataArray(
        [["land", "sea"], ["land", "sea"], ["sea", "sea"]],
        dims=("lat", "lon"),
        coords={"lat": covariance.latitude, "lon": covariance.longitude},
    )
    covariance = SeparableExponentialCovariance(
        covariance.latitude,
        covariance.longitude,
        sigma=covariance.sigma,
        correlation_length=covariance.correlation_length,
        class_labels=labels,
    )
    flat_labels = labels.values.reshape(-1)
    dense_b = dense_unblocked * (flat_labels[:, None] == flat_labels[None, :])
    u, expected_pi, expected_c = _dense_operators(basis_operator, dense_b)
    h_matrix = h.values.reshape(3, 6)

    products = _project(covariance, basis_operator, h)

    np.testing.assert_allclose(products.restriction.values.reshape(2, 6), expected_pi, atol=2e-10)
    np.testing.assert_allclose(products.state_covariance, expected_c, atol=2e-10)
    np.testing.assert_allclose(products.effective_observation_operator, h_matrix @ u, atol=2e-10)
    np.testing.assert_allclose(
        products.observation_state_cross_covariance,
        h_matrix @ dense_b @ expected_pi.T,
        atol=2e-10,
    )
    np.testing.assert_allclose(
        products.native_observation_covariance,
        h_matrix @ dense_b @ h_matrix.T,
        atol=2e-10,
    )


def test_multiindex_observation_labels_are_preserved_on_both_matrix_axes() -> None:
    """Dense observation products retain rich row labels and suffixed column metadata."""
    covariance, basis_operator, h, _ = _problem()
    observation_index = pd.MultiIndex.from_tuples(
        [("MHD", "2020-01-01"), ("TAC", "2020-01-02"), ("RGL", "2020-01-03")],
        names=("site", "date"),
    )
    h = h.drop_indexes("observation").drop_vars("observation")
    h = h.assign_coords(xr.Coordinates.from_pandas_multiindex(observation_index, "observation"))

    products = _project(covariance, basis_operator, h)

    assert products.native_observation_covariance.coords["site"].values.tolist() == [
        "MHD",
        "TAC",
        "RGL",
    ]
    assert products.native_observation_covariance.coords["site_cov"].values.tolist() == [
        "MHD",
        "TAC",
        "RGL",
    ]
    assert products.native_observation_covariance.coords["date_cov"].values.tolist() == [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
    ]
